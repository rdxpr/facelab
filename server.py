"""FaceLab — a multi-model benchmarking harness for classroom attendance.

Detectors and recognizers are selected independently, scans are cached so
threshold sweeps run in milliseconds, and ground-truth datasets drive automatic
per-model threshold calibration.

Run it with the project's own virtualenv:

    ./run.sh
    # or: .venv/bin/python server.py

Open http://127.0.0.1:8010
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
from datetime import UTC, datetime

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from facelab import benchmark as bench
from facelab import cache as cache_store
from facelab import datasets as dataset_store
from facelab import face as engine
from facelab import matching
from facelab.config import (
    KNOWN_DETECTORS,
    KNOWN_PACKS,
    KNOWN_RECOGNIZERS,
    STATIC_DIR,
    default_detector_for_pack,
    default_recognizer_for_pack,
    detector_registry,
    recognizer_registry,
    settings,
)

MAX_IMAGE_BYTES = 12 * 1024 * 1024

_state_lock = threading.RLock()
_model_lock = threading.Lock()
_people: dict[int, dict] = {}
_next_number = 1


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class RenameRequest(BaseModel):
    name: str


class ReEmbedRequest(BaseModel):
    recognizer_ids: list[str] = []


class BenchmarkRunRequest(BaseModel):
    dataset_ids: list[str]
    detectors: list[str] = []
    recognizers: list[str] = []
    det_sizes: list[int] = []
    tiling_modes: list[str] = []
    det_thresh: float = 0.35
    nms_thresh: float = 0.45
    flip_aug: bool = True
    lowres_aug: bool = False
    with_quality: bool = True
    with_pose: bool = False
    variant: str | None = None
    accept_threshold: float | None = None
    review_threshold: float = 0.34
    min_margin: float = 0.04
    assignment_mode: str = "hungarian"
    template_strategy: str = "top_k_mean"
    top_k: int = 2
    fusion_mode: str = "k_of_n_voting"
    k: int = 2
    min_quality: float = 0.0
    force_fresh: bool = False
    label_unknown: bool = False


class SweepRequest(BaseModel):
    dataset_ids: list[str] = []
    detector: str = ""
    recognizer: str = ""
    det_size: int = 0
    tiling_mode: str = "none"
    det_thresh: float = 0.35
    nms_thresh: float = 0.45
    flip_aug: bool = True
    lowres_aug: bool = False
    with_quality: bool = True
    with_pose: bool = False
    variant: str | None = None
    accept_threshold: float = 0.42
    review_threshold: float = 0.34
    min_margin: float = 0.04
    assignment_mode: str = "hungarian"
    template_strategy: str = "top_k_mean"
    top_k: int = 2
    fusion_mode: str = "k_of_n_voting"
    k: int = 2
    min_quality: float = 0.0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def require_token(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None),
) -> None:
    """Guard mutating endpoints when ``FACELAB_API_TOKEN`` is set."""
    expected = settings.api_token
    if not expected:
        return
    provided = x_api_token
    if not provided and authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()
    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


def _variant_for(flip_aug: bool, lowres_aug: bool) -> str:
    if lowres_aug:
        return "lowres_aug"
    return "flip_avg" if flip_aug else "raw"


# --------------------------------------------------------------------------- #
# Persisted roster
# --------------------------------------------------------------------------- #
def _encode_embedding(entry: dict) -> dict:
    encoded: dict[str, object] = {}
    for name in ("raw", "flip_avg", "lowres_aug"):
        vector = entry.get(name)
        if vector is not None:
            encoded[name] = np.asarray(vector, dtype=np.float32).tolist()
    encoded["norm_magnitude"] = float(entry.get("norm_magnitude") or 0.0)
    return encoded


def _decode_embedding(entry: dict) -> dict:
    decoded: dict[str, object] = {}
    for name in ("raw", "flip_avg", "lowres_aug"):
        vector = entry.get(name)
        if vector is not None:
            decoded[name] = np.asarray(vector, dtype=np.float32)
    decoded["norm_magnitude"] = float(entry.get("norm_magnitude") or 0.0)
    return decoded


def _save_state() -> None:
    payload = {
        "version": 3,
        "nextNumber": _next_number,
        "students": [
            {
                "id": person["id"],
                "number": person["number"],
                "name": person["name"],
                "createdAt": person["createdAt"],
                "photos": [
                    {
                        "id": photo["id"],
                        "filename": photo["filename"],
                        "sha256": photo.get("sha256") or "",
                        "qualityScore": photo.get("quality_score") or 0.0,
                        "qualityNote": photo.get("quality_note") or "",
                        "headPose": photo.get("head_pose"),
                        "createdAt": photo["createdAt"],
                        "embeddings": {
                            rid: _encode_embedding(entry)
                            for rid, entry in (photo.get("embeddings") or {}).items()
                        },
                    }
                    for photo in person["photos"]
                ],
            }
            for person in sorted(_people.values(), key=lambda item: item["number"])
        ],
    }
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    temp = settings.students_file.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(settings.students_file)


def _load_legacy_people(payload: dict) -> dict[int, dict]:
    """Migrate the pre-registry ``people.json`` (v1/v2) roster."""
    people: dict[int, dict] = {}
    for person in payload.get("people", []):
        photos = []
        for photo in person.get("photos", []):
            raw_bytes = photo.get("embedding")
            if raw_bytes is None:
                continue
            vector = np.frombuffer(base64.b64decode(raw_bytes), dtype=np.float32).copy()
            recognizer_id = default_recognizer_for_pack(
                photo.get("model") or settings.default_model
            )
            photos.append(
                {
                    "id": photo.get("id") or uuid.uuid4().hex[:10],
                    "filename": photo.get("filename"),
                    "sha256": "",
                    "quality_score": 0.0,
                    "quality_note": photo.get("quality") or "",
                    "head_pose": None,
                    "createdAt": photo.get("createdAt") or _now(),
                    "embeddings": {
                        recognizer_id: {
                            "raw": vector,
                            "norm_magnitude": float(np.linalg.norm(vector)),
                        }
                    },
                }
            )
        people[int(person["id"])] = {
            "id": int(person["id"]),
            "number": int(person["number"]),
            "name": person.get("name") or f"Student {int(person['number']):02d}",
            "createdAt": person.get("createdAt") or _now(),
            "photos": photos,
        }
    return people


def _load_state() -> None:
    global _next_number
    path = settings.students_file
    try:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            people: dict[int, dict] = {}
            for person in payload.get("students", []):
                photos = []
                for photo in person.get("photos", []):
                    photos.append(
                        {
                            "id": photo["id"],
                            "filename": photo.get("filename"),
                            "sha256": photo.get("sha256") or "",
                            "quality_score": float(photo.get("qualityScore") or 0.0),
                            "quality_note": photo.get("qualityNote") or "",
                            "head_pose": photo.get("headPose"),
                            "createdAt": photo.get("createdAt") or _now(),
                            "embeddings": {
                                rid: _decode_embedding(entry)
                                for rid, entry in (photo.get("embeddings") or {}).items()
                            },
                        }
                    )
                people[int(person["id"])] = {
                    "id": int(person["id"]),
                    "number": int(person["number"]),
                    "name": person.get("name"),
                    "createdAt": person.get("createdAt") or _now(),
                    "photos": photos,
                }
            _people.update(people)
            _next_number = int(payload.get("nextNumber") or (max(people, default=0) + 1))
            return

        legacy = settings.legacy_people_file
        if legacy.is_file():
            payload = json.loads(legacy.read_text(encoding="utf-8"))
            people = _load_legacy_people(payload)
            _people.update(people)
            _next_number = int(payload.get("nextNumber") or (max(people, default=0) + 1))
            print(f"[facelab] migrated {len(people)} students from {legacy}")
            _save_state()
    except Exception as exc:  # a corrupt local file should not brick the tool
        broken = path.with_suffix(f".broken-{int(time.time())}.json")
        try:
            path.replace(broken)
        except OSError:
            pass
        print(f"[facelab] could not read {path}: {exc}; starting empty")


# --------------------------------------------------------------------------- #
# Public shapes
# --------------------------------------------------------------------------- #
def _public_person(person: dict) -> dict:
    recognizer_usage: dict[str, int] = {}
    for photo in person["photos"]:
        for recognizer_id in (photo.get("embeddings") or {}):
            recognizer_usage[recognizer_id] = recognizer_usage.get(recognizer_id, 0) + 1
    return {
        "id": person["id"],
        "number": person["number"],
        "name": person["name"],
        "createdAt": person["createdAt"],
        "templates": len(person["photos"]),
        "recognizers": recognizer_usage,
        "photos": [
            {
                "id": photo["id"],
                "url": f"/media/{photo['filename']}",
                "quality": photo.get("quality_note") or "",
                "qualityScore": photo.get("quality_score") or 0.0,
                "headPose": photo.get("head_pose"),
                "createdAt": photo["createdAt"],
                "recognizers": sorted((photo.get("embeddings") or {}).keys()),
            }
            for photo in person["photos"]
        ],
    }


def _recognizer_usage() -> dict[str, dict]:
    usage: dict[str, dict] = {}
    for person in _people.values():
        for photo in person["photos"]:
            for recognizer_id in (photo.get("embeddings") or {}):
                entry = usage.setdefault(recognizer_id, {"templates": 0, "people": set()})
                entry["templates"] += 1
                entry["people"].add(person["id"])
    return {
        name: {"templates": value["templates"], "people": len(value["people"])}
        for name, value in usage.items()
    }


def _student_snapshot() -> dict[int, dict]:
    with _state_lock:
        return {
            person["id"]: {
                "id": person["id"],
                "number": person["number"],
                "name": person["name"],
                "photos": [dict(photo) for photo in person["photos"]],
            }
            for person in _people.values()
        }


def _registry_state() -> dict:
    payload = engine.registry_payload()
    usage = _recognizer_usage()
    for entry in payload["recognizers"]:
        use = usage.get(entry["id"], {"templates": 0, "people": 0})
        entry["templates"] = use["templates"]
        entry["people"] = use["people"]
    return payload


def _packs_payload() -> list[dict]:
    installed = {item["name"]: item for item in engine.available_models()}
    usage = _recognizer_usage()
    packs = []
    names = sorted(set(KNOWN_PACKS) | set(installed))
    for name in names:
        info = installed.get(name, {})
        recognizer_id = default_recognizer_for_pack(name)
        use = usage.get(recognizer_id, {"templates": 0, "people": 0})
        packs.append(
            {
                "name": name,
                "files": info.get("files", []),
                "fileCount": info.get("fileCount", 0),
                "sizeBytes": info.get("sizeBytes", 0),
                "installed": name in installed,
                "default": name == settings.default_model,
                "recognizer": recognizer_id,
                "templates": use["templates"],
                "people": use["people"],
            }
        )
    return packs


def _models_payload() -> dict:
    registry = _registry_state()
    packs = _packs_payload()
    return {
        "models": packs,  # legacy key
        "packs": packs,
        "detectors": registry["detectors"],
        "recognizers": registry["recognizers"],
        "defaultModel": settings.default_model,
        "defaultDetector": settings.default_detector,
        "defaultRecognizer": settings.default_recognizer,
        "providers": settings.provider_list,
        "modelsDir": str(settings.models_dir),
    }


def _settings_payload() -> dict:
    return {
        "threshold": settings.face_similarity_threshold,
        "acceptThreshold": settings.accept_threshold,
        "reviewThreshold": settings.review_threshold,
        "minMargin": settings.min_margin,
        "detSize": settings.face_scan_det_size,
        "enrollDetSize": settings.face_det_size,
        "detThresh": 0.35,
        "nmsThresh": 0.45,
        "tilingMode": settings.tiling_mode,
        "assignmentMode": settings.assignment_mode,
        "templateStrategy": settings.template_strategy,
        "templateTopK": settings.template_top_k,
        "fusionMode": settings.multi_frame_fusion,
        "fusionK": settings.fusion_k,
        "minFaceQuality": settings.min_face_quality,
        "benchmarkDetSizes": settings.benchmark_det_size_list,
        "benchmarkMaxRows": settings.benchmark_max_rows,
        "maxFrames": settings.face_scan_max_frames,
        "maxPhotosPerUpload": settings.max_photos_per_upload,
        "maxPhotosPerPerson": settings.max_photos_per_person,
        "defaultModel": settings.default_model,
        "defaultDetector": settings.default_detector,
        "defaultRecognizer": settings.default_recognizer,
        "modelsDir": str(settings.models_dir),
        "env": settings.env,
        "ephemeral": settings.ephemeral,
        "dataDir": str(settings.data_dir),
        "providers": settings.provider_list,
        "authRequired": bool(settings.api_token),
    }


def _public_state() -> dict:
    with _state_lock:
        people = [_public_person(person) for person in _people.values()]
        usage = _recognizer_usage()
    people.sort(key=lambda item: item["number"])
    total_templates = sum(person["templates"] for person in people)
    return {
        "people": people,
        "students": people,
        "nextNumber": _next_number,
        "totals": {"people": len(people), "templates": total_templates},
        "modelUsage": usage,
        "settings": _settings_payload(),
        "registry": _registry_state(),
    }


# --------------------------------------------------------------------------- #
# Scan results on disk
# --------------------------------------------------------------------------- #
def _result_path(scan_id: str) -> "os.PathLike[str] | None":
    if not scan_id or any(char not in "0123456789abcdef-" for char in scan_id.lower()):
        return None
    return settings.results_dir / f"{scan_id}.json"


def _save_result(result: dict) -> None:
    _prune_results(keep=settings.results_to_keep - 1)
    path = settings.results_dir / f"{result['scanId']}.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temp.replace(path)


def _prune_results(keep: int | None = None) -> None:
    keep = settings.results_to_keep if keep is None else keep
    if not settings.results_dir.exists():
        return
    files = [path for path in settings.results_dir.glob("*.json")]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for stale in files[keep:]:
        stale.unlink(missing_ok=True)


def _prune_scans(keep: int | None = None) -> None:
    keep = settings.scans_to_keep if keep is None else keep
    if not settings.scans_dir.exists():
        return
    folders = [path for path in settings.scans_dir.iterdir() if path.is_dir()]
    folders.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for stale in folders[keep:]:
        shutil.rmtree(stale, ignore_errors=True)


def _result_summary(result: dict) -> dict:
    summary = result.get("summary", {})
    frames = result.get("frames", [])
    thumb = None
    if frames:
        best = summary.get("bestIndex", 0)
        best = best if isinstance(best, int) and 0 <= best < len(frames) else 0
        thumb = frames[best].get("annotatedUrl") or frames[best].get("imageUrl")
    return {
        "scanId": result.get("scanId"),
        "createdAt": result.get("createdAt"),
        "model": result.get("model"),
        "detectorId": result.get("detectorId"),
        "recognizerId": result.get("recognizerId"),
        "threshold": summary.get("acceptThreshold"),
        "detSize": summary.get("detSize"),
        "frames": summary.get("frames"),
        "facesDetected": summary.get("facesDetected"),
        "matchedCount": summary.get("matchedCount"),
        "unknown": summary.get("unknown"),
        "peopleCount": summary.get("peopleCount"),
        "templateCount": summary.get("templateCount"),
        "elapsedMs": summary.get("elapsedMs"),
        "warnings": len(result.get("warnings", [])),
        "thumbUrl": thumb,
    }


def _list_results(model: str | None = None) -> list[dict]:
    if not settings.results_dir.exists():
        return []
    items = []
    for path in settings.results_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if model and data.get("model") != model and data.get("recognizerId") != model:
            continue
        items.append(_result_summary(data))
    items.sort(key=lambda item: (item.get("createdAt") or "", item.get("scanId") or ""), reverse=True)
    return items


# --------------------------------------------------------------------------- #
# Imaging helpers
# --------------------------------------------------------------------------- #
def _write_jpeg_bgr(image_bgr: np.ndarray, relative: str, quality: int = 92) -> str:
    path = settings.data_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode an image for display")
    path.write_bytes(buffer.tobytes())
    return relative


def _write_jpeg_rgb(image_rgb: np.ndarray, relative: str, quality: int = 92) -> str:
    return _write_jpeg_bgr(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), relative, quality)


def _decode(data: bytes, label: str) -> np.ndarray:
    if not data:
        raise HTTPException(status_code=422, detail=f"{label}: empty file")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"{label}: larger than 12 MB")
    try:
        return engine.image_to_bgr(data)
    except engine.FaceEngineUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"{label}: could not read image ({exc})") from exc


def _require_detector(detector_id: str) -> None:
    if detector_id not in KNOWN_DETECTORS:
        raise HTTPException(status_code=422, detail=f"Unknown detector '{detector_id}'")
    ok, reason = engine.detector_availability(detector_id)
    if not ok:
        raise HTTPException(status_code=422, detail=f"Detector '{detector_id}' is unavailable: {reason}")


def _require_recognizer(recognizer_id: str) -> None:
    if recognizer_id not in KNOWN_RECOGNIZERS:
        raise HTTPException(status_code=422, detail=f"Unknown recognizer '{recognizer_id}'")
    ok, reason = engine.recognizer_availability(recognizer_id)
    if not ok:
        raise HTTPException(status_code=422, detail=f"Recognizer '{recognizer_id}' is unavailable: {reason}")


_MANIFEST_CACHE: dict[str, str] | None = None


def _manifest_hashes() -> dict[str, str]:
    """Map weight filename -> SHA-256 from the last fetch_models.py manifest."""
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is not None:
        return _MANIFEST_CACHE
    out: dict[str, str] = {}
    path = settings.models_manifest_file
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for entry in data.get("models", []):
                name = os.path.basename(entry.get("file", ""))
                if name:
                    out[name] = entry.get("sha256", "")
        except Exception:
            out = {}
    _MANIFEST_CACHE = out
    return out


def _weight_sha256(entry: dict) -> str | None:
    """SHA-256 of the ONNX file a registry entry resolves to, if recorded."""
    hashes = _manifest_hashes()
    for candidate in entry.get("file_candidates", []):
        name = os.path.basename(candidate)
        if name in hashes:
            return hashes[name]
    return None


def _legacy_ids(model: str | None, detector_id: str | None, recognizer_id: str | None) -> tuple[str, str]:
    if model:
        if not engine.model_exists(model):
            raise HTTPException(
                status_code=422,
                detail=f"Model pack '{model}' is not installed. Run scripts/fetch_models.py --model {model}.",
            )
        detector_id = detector_id or default_detector_for_pack(model)
        recognizer_id = recognizer_id or default_recognizer_for_pack(model)
    return (
        detector_id or settings.default_detector,
        recognizer_id or settings.default_recognizer,
    )


# --------------------------------------------------------------------------- #
# Inference / caching
# --------------------------------------------------------------------------- #
def _analyze_or_cache(
    image_bgr: np.ndarray,
    data: bytes,
    *,
    detector_id: str,
    recognizer_id: str,
    det_params: dict,
    flip_aug: bool,
    force_fresh: bool = False,
) -> tuple[dict, bool]:
    image_sha = cache_store.sha256_bytes(data)
    key = cache_store.cache_key(image_sha, detector_id, det_params, recognizer_id, flip_aug)
    cached = None if force_fresh else cache_store.load_analysis(key)
    if cached is not None:
        cached["cache_key"] = key
        return cached, True

    payload = engine.analyze_frame(
        image_bgr,
        detector_id=detector_id,
        recognizer_id=recognizer_id,
        det_size=int(det_params["det_size"][0]),
        det_thresh=float(det_params["det_thresh"]),
        nms_thresh=float(det_params["nms_thresh"]),
        tiling_mode=det_params.get("tiling_mode", "none"),
        tile_size=int(det_params.get("tile_size", 640)),
        tile_overlap=float(det_params.get("tile_overlap", 0.25)),
        flip_aug=flip_aug,
        lowres_aug=bool(det_params.get("lowres_aug", False)),
        with_quality=bool(det_params.get("with_quality", True)),
        with_pose=bool(det_params.get("with_pose", True)),
        crops_dir=settings.crops_dir,
        crop_prefix=key,
    )
    payload["cache_key"] = key
    payload["image_sha256"] = image_sha
    cache_store.save_analysis(payload)
    return payload, False


def _pick_detector(preferred: str | None = None) -> str:
    for candidate in [preferred, settings.default_detector, *KNOWN_DETECTORS.keys()]:
        if candidate and engine.detector_availability(candidate)[0]:
            return candidate
    raise engine.FaceEngineUnavailable(
        "No detector weights are installed. Run scripts/fetch_models.py."
    )


def _embed_reference(
    image_bgr: np.ndarray, data: bytes, recognizer_ids: list[str], detector_id: str
) -> dict:
    """Run one detection and embed the aligned face with every recognizer."""
    try:
        result = engine.embed_reference(
            image_bgr,
            detector_id=detector_id,
            recognizer_ids=recognizer_ids,
            det_size=settings.face_det_size,
            det_thresh=0.5,
            nms_thresh=0.45,
            flip_aug=True,
            lowres_aug=True,
            with_quality=True,
            with_pose=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    embeddings = {
        recognizer_id: {
            "raw": np.asarray(entry["raw"], dtype=np.float32),
            "flip_avg": (
                np.asarray(entry["flip_avg"], dtype=np.float32)
                if entry.get("flip_avg") is not None
                else None
            ),
            "lowres_aug": (
                np.asarray(entry["lowres_aug"], dtype=np.float32)
                if entry.get("lowres_aug") is not None
                else None
            ),
            "norm_magnitude": float(entry.get("norm_magnitude") or 0.0),
        }
        for recognizer_id, entry in result["embeddings"].items()
    }
    return {
        "id": uuid.uuid4().hex[:10],
        "sha256": cache_store.sha256_bytes(data),
        "quality_score": float(result.get("quality_score") or 0.0),
        "quality_note": (
            "Face is small in the frame; move closer for a stronger embedding."
            if float(result.get("face_height_px") or 0.0) < 50
            else "Face detected."
        ),
        "head_pose": result.get("head_pose"),
        "createdAt": _now(),
        "embeddings": embeddings,
    }


def _process_reference_photos(
    raw: list[tuple[bytes, str | None]], recognizer_ids: list[str]
) -> list[tuple[np.ndarray, dict]]:
    for recognizer_id in recognizer_ids:
        _require_recognizer(recognizer_id)
    detector_id = _pick_detector()
    processed: list[tuple[np.ndarray, dict]] = []
    with _model_lock:
        for data, filename in raw:
            label = filename or "photo"
            image_bgr = _decode(data, label)
            try:
                record = _embed_reference(image_bgr, data, recognizer_ids, detector_id)
            except HTTPException:
                raise
            except engine.FaceEngineUnavailable as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"{label}: detection failed ({exc})") from exc
            record.setdefault("quality_note", "")
            processed.append((image_bgr, record))
    return processed


def _create_person_sync(
    name: str, recognizer_ids: list[str], raw: list[tuple[bytes, str | None]]
) -> dict:
    global _next_number
    if not raw:
        raise HTTPException(status_code=422, detail="Add at least one face photo")
    if len(raw) > settings.max_photos_per_upload:
        raise HTTPException(
            status_code=422, detail=f"Up to {settings.max_photos_per_upload} photos per upload"
        )
    processed = _process_reference_photos(raw, recognizer_ids)

    with _state_lock:
        number = _next_number
        _next_number += 1
        clean_name = (name or "").strip()[:80] or f"Student {number:02d}"
        person = {
            "id": number,
            "number": number,
            "name": clean_name,
            "createdAt": _now(),
            "photos": [],
        }
        for image_bgr, record in processed:
            photo_id = record["id"]
            relative = _write_jpeg_bgr(image_bgr, f"faces/p{number}/{photo_id}.jpg")
            record["filename"] = relative
            person["photos"].append(record)
        _people[person["id"]] = person
        _save_state()
        return _public_person(person)


def _add_photos_sync(
    person_id: int, recognizer_ids: list[str], raw: list[tuple[bytes, str | None]]
) -> dict:
    with _state_lock:
        person = _people.get(person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="Student not found")
        remaining = settings.max_photos_per_person - len(person["photos"])
    if remaining <= 0:
        raise HTTPException(
            status_code=422,
            detail=f"This student already has {settings.max_photos_per_person} photos",
        )
    if len(raw) > remaining:
        raise HTTPException(status_code=422, detail=f"{remaining} more photo(s) fit for this student")

    processed = _process_reference_photos(raw, recognizer_ids)

    with _state_lock:
        person = _people.get(person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="Student not found")
        for image_bgr, record in processed:
            photo_id = record["id"]
            relative = _write_jpeg_bgr(image_bgr, f"faces/p{person['number']}/{photo_id}.jpg")
            record["filename"] = relative
            person["photos"].append(record)
        _save_state()
        return _public_person(person)


def _reembed_sync(recognizer_ids: list[str]) -> dict:
    """Backfill embeddings for recognizers added after a student was enrolled.

    Photos are grouped so each one is detected once and embedded with every
    recognizer it is still missing, instead of re-detecting per recognizer.
    """
    registry = recognizer_registry()
    targets = [rid for rid in (recognizer_ids or list(registry.keys())) if rid in registry]
    targets = [rid for rid in targets if engine.recognizer_availability(rid)[0]]
    if not targets:
        raise HTTPException(status_code=422, detail="No available recognizers to embed with")
    detector_id = _pick_detector()

    results = {rid: {"updated": 0, "skipped": 0} for rid in targets}
    errors: list[str] = []
    with _state_lock, _model_lock:
        for person in _people.values():
            for photo in person["photos"]:
                existing = photo.get("embeddings") or {}
                missing = [
                    rid for rid in targets if existing.get(rid, {}).get("raw") is None
                ]
                for recognizer_id in targets:
                    if recognizer_id not in missing:
                        results[recognizer_id]["skipped"] += 1
                if not missing:
                    continue
                path = settings.data_dir / (photo.get("filename") or "")
                image_bgr = cv2.imread(str(path)) if path.is_file() else None
                if image_bgr is None:
                    for recognizer_id in missing:
                        results[recognizer_id]["skipped"] += 1
                    errors.append(f"missing reference image for photo {photo['id']}")
                    continue
                try:
                    record = _embed_reference(
                        image_bgr, path.read_bytes(), missing, detector_id
                    )
                except HTTPException as exc:
                    for recognizer_id in missing:
                        results[recognizer_id]["skipped"] += 1
                    errors.append(f"{photo['id']}: {exc.detail}")
                    continue
                entry = photo.setdefault("embeddings", {})
                for recognizer_id in missing:
                    entry[recognizer_id] = record["embeddings"][recognizer_id]
                    results[recognizer_id]["updated"] += 1
                if not photo.get("quality_score"):
                    photo["quality_score"] = record["quality_score"]
                if not photo.get("head_pose"):
                    photo["head_pose"] = record["head_pose"]
        _save_state()
    return {"results": results, "errors": errors[:20], "state": _public_state()}


def _available_recognizer_ids() -> list[str]:
    return [
        rid for rid in recognizer_registry() if engine.recognizer_availability(rid)[0]
    ]


def _resolve_recognizer_ids(requested: list[str] | None) -> list[str]:
    """Enrolment targets every installed recognizer unless callers name some."""
    requested = [rid for rid in (requested or []) if rid]
    if requested:
        for recognizer_id in requested:
            _require_recognizer(recognizer_id)
        return requested
    available = _available_recognizer_ids()
    if not available:
        raise HTTPException(
            status_code=503,
            detail="No recognizer weights are installed. Run scripts/fetch_models.py.",
        )
    return available


# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #
def _scan_det_params(
    det_size: int, det_thresh: float, nms_thresh: float, tiling_mode: str,
    lowres_aug: bool, with_quality: bool, with_pose: bool,
) -> dict:
    return cache_store.analysis_params(
        det_size=det_size,
        det_thresh=det_thresh,
        nms_thresh=nms_thresh,
        tiling_mode=tiling_mode,
        lowres_aug=lowres_aug,
        with_quality=with_quality,
        with_pose=with_pose,
    )


def _serialize_candidates(candidates: list[dict], snapshot: dict[int, dict]) -> list[dict]:
    return [
        {
            "studentId": candidate["student_id"],
            "name": (snapshot.get(candidate["student_id"]) or {}).get("name", "?"),
            "score": candidate["score"],
        }
        for candidate in candidates
    ]


def _serialize_face(decision: dict, det: dict, snapshot: dict[int, dict]) -> dict:
    student_id = decision.get("matched_student_id")
    person = snapshot.get(student_id) if student_id is not None else None
    return {
        "box": [round(float(value), 1) for value in det["bbox"]],
        "detScore": round(float(det.get("det_score") or 0.0), 4),
        "status": decision.get("status", "unmatched"),
        "matched": decision.get("status") == "present",
        "personId": student_id,
        "number": person["number"] if person else None,
        "name": person["name"] if person else "Unknown",
        "confidence": round(float(decision.get("score") or 0.0), 4),
        "margin": round(float(decision.get("margin") or 0.0), 4),
        "sizeBucket": det.get("size_bucket"),
        "faceHeightPx": det.get("face_height_px"),
        "quality": det.get("quality_score"),
        "qualitySource": det.get("quality_source"),
        "headPose": det.get("head_pose"),
        "cropUrl": (
            f"/media/{det['aligned_crop_path']}" if det.get("aligned_crop_path") else None
        ),
        "landmarksCrop": det.get("landmarks_crop"),
        "landmarks5pt": det.get("landmarks_5pt"),
        "topCandidates": _serialize_candidates(
            decision.get("top_candidates", []), snapshot
        ),
    }


def _run_scan_sync(
    raw: list[tuple[bytes, str | None]],
    *,
    detector_id: str,
    recognizer_id: str,
    det_size: int,
    det_thresh: float,
    nms_thresh: float,
    tiling_mode: str,
    accept_threshold: float,
    review_threshold: float,
    min_margin: float,
    assignment_mode: str,
    template_strategy: str,
    top_k: int,
    fusion_mode: str,
    fusion_k: int,
    flip_aug: bool,
    lowres_aug: bool,
    with_quality: bool,
    with_pose: bool,
    min_quality: float,
    label_unknown: bool = False,
) -> dict:
    _require_detector(detector_id)
    _require_recognizer(recognizer_id)
    started = time.perf_counter()
    warnings: list[str] = []
    det_params = _scan_det_params(
        det_size, det_thresh, nms_thresh, tiling_mode, lowres_aug, with_quality, with_pose
    )
    variant = _variant_for(flip_aug, lowres_aug)

    try:
        engine.warm_up(
            detector_id, recognizer_id, det_size, tiling_mode, det_thresh, nms_thresh
        )
    except engine.FaceEngineUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    decoded: list[dict] = []
    total_decode_ms = 0.0
    with _model_lock:
        for index, (data, filename) in enumerate(raw):
            label = filename or f"frame {index + 1}"
            decode_started = time.perf_counter()
            try:
                image_bgr = _decode(data, label)
            except HTTPException as exc:
                warnings.append(str(exc.detail))
                continue
            decode_ms = (time.perf_counter() - decode_started) * 1000
            try:
                payload, cached = _analyze_or_cache(
                    image_bgr, data,
                    detector_id=detector_id, recognizer_id=recognizer_id,
                    det_params=det_params, flip_aug=flip_aug,
                )
            except engine.FaceEngineUnavailable as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except Exception as exc:
                warnings.append(f"{label}: {exc}")
                continue
            height, width = image_bgr.shape[:2]
            total_decode_ms += decode_ms
            decoded.append(
                {
                    "index": index,
                    "sourceName": filename,
                    "bgr": image_bgr,
                    "width": width,
                    "height": height,
                    "detections": payload["detections"],
                    "timings": payload["timings_ms"],
                    "cache_key": payload.get("cache_key"),
                    "cached": cached,
                    "decode_ms": round(decode_ms, 3),
                }
            )

        if not decoded:
            raise HTTPException(status_code=422, detail="No readable frames. " + " ".join(warnings))

        snapshot = _student_snapshot()
        gallery = bench.build_gallery(snapshot, recognizer_id, variant)
        enrolled_ids = set(gallery.keys())

        fusion_frames = [
            {
                "frame_id": f"frame_{order + 1}",
                "width": item["width"],
                "height": item["height"],
                "detections": [
                    {
                        "embedding": bench.select_embedding(det, variant),
                        "bbox": det["bbox"],
                        "quality": det.get("quality_score") or 0.0,
                    }
                    for det in item["detections"]
                ],
            }
            for order, item in enumerate(decoded)
        ]

        match_started = time.perf_counter()
        fused = matching.fuse_multi_frame(
            fusion_frames,
            gallery,
            mode=fusion_mode,
            accept_threshold=accept_threshold,
            review_threshold=review_threshold,
            min_margin=min_margin,
            assignment_mode=assignment_mode,
            template_strategy=template_strategy,
            top_k=top_k,
            k=fusion_k,
            min_quality=min_quality,
        )
        match_ms = (time.perf_counter() - match_started) * 1000

        scan_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        annotate_started = time.perf_counter()
        frames_payload = []
        for order, item in enumerate(decoded):
            decisions = fused["per_frame"][order]
            faces_payload = [
                _serialize_face(decision, item["detections"][det_index], snapshot)
                for det_index, decision in enumerate(decisions)
            ]
            frame_rel = _write_jpeg_bgr(item["bgr"], f"scans/{scan_id}/frame-{order}.jpg")
            matches_for_annotation = [
                {
                    "bbox": face["box"],
                    "status": face["status"],
                    "student_name": face["name"],
                    "confidence": face["confidence"],
                    "matched": face["matched"],
                }
                for face in faces_payload
            ]
            annotated = engine.annotate_classroom_photo(
                item["bgr"], matches_for_annotation, label_unknown=label_unknown
            )
            annotated_rel = _write_jpeg_rgb(
                annotated, f"scans/{scan_id}/frame-{order}-annotated.jpg"
            )
            frames_payload.append(
                {
                    "order": order,
                    "sourceIndex": item["index"],
                    "sourceName": item["sourceName"],
                    "width": item["width"],
                    "height": item["height"],
                    "imageUrl": f"/media/{frame_rel}",
                    "annotatedUrl": f"/media/{annotated_rel}",
                    "cacheKey": item["cache_key"],
                    "cached": item["cached"],
                    "timings": {**item["timings"], "decode_ms": item["decode_ms"]},
                    "faces": faces_payload,
                }
            )
        annotate_ms = (time.perf_counter() - annotate_started) * 1000

        roster_payload = []
        student_status = fused["students"]
        for student_id, person in sorted(snapshot.items(), key=lambda kv: kv[1]["number"]):
            record = student_status.get(student_id, {"status": "absent", "score": 0.0, "frames_hit": 0})
            templates = sum(
                1 for photo in person["photos"] if (photo.get("embeddings") or {}).get(recognizer_id)
            )
            roster_payload.append(
                {
                    "personId": student_id,
                    "number": person["number"],
                    "name": person["name"],
                    "status": record["status"],
                    "confidence": record["score"],
                    "framesHit": record["frames_hit"],
                    "enrolled": student_id in enrolled_ids,
                    "photoUrl": (
                        f"/media/{person['photos'][0]['filename']}"
                        if person["photos"]
                        else None
                    ),
                    "templates": templates,
                }
            )

    present_count = sum(1 for row in roster_payload if row["status"] == "present")
    unknown = max(
        (sum(1 for face in frame["faces"] if face["status"] == "unmatched") for frame in frames_payload),
        default=0,
    )
    faces_detected = max((len(frame["faces"]) for frame in frames_payload), default=0)
    best_index = 0
    best_present = -1
    for order, frame in enumerate(frames_payload):
        count = sum(1 for face in frame["faces"] if face["status"] == "present")
        if count > best_present:
            best_present = count
            best_index = order

    detect_ms = sum(float(item["timings"].get("detect_ms") or 0.0) for item in decoded)
    align_ms = sum(float(item["timings"].get("align_embed_ms") or 0.0) for item in decoded)
    quality_ms = sum(float(item["timings"].get("quality_pose_ms") or 0.0) for item in decoded)
    elapsed_ms = (time.perf_counter() - started) * 1000

    result = {
        "scanId": scan_id,
        "createdAt": _now(),
        "model": recognizer_id,
        "detectorId": detector_id,
        "recognizerId": recognizer_id,
        "people": roster_payload,
        "frames": frames_payload,
        "summary": {
            "model": recognizer_id,
            "detectorId": detector_id,
            "recognizerId": recognizer_id,
            "detSize": det_size,
            "detThresh": det_thresh,
            "nmsThresh": nms_thresh,
            "tilingMode": tiling_mode,
            "acceptThreshold": round(accept_threshold, 4),
            "reviewThreshold": round(review_threshold, 4),
            "minMargin": round(min_margin, 4),
            "assignmentMode": assignment_mode,
            "templateStrategy": template_strategy,
            "topK": top_k,
            "minQuality": min_quality,
            "fusionMode": fusion_mode,
            "fusionK": fusion_k,
            "variant": variant,
            "flipAug": flip_aug,
            "lowresAug": lowres_aug,
            "labelUnknown": label_unknown,
            "detParams": det_params,
            "frames": len(decoded),
            "uploadedFrames": len(raw),
            "facesDetected": faces_detected,
            "unknown": unknown,
            "matchedCount": present_count,
            "needsReview": sum(1 for row in roster_payload if row["status"] == "needs_review"),
            "peopleCount": len(roster_payload),
            "templateCount": sum(row["templates"] for row in roster_payload),
            "bestIndex": best_index,
            "cachedFrames": sum(1 for item in decoded if item["cached"]),
            "elapsedMs": round(elapsed_ms),
            "timings": {
                "decode_ms": round(total_decode_ms, 3),
                "detect_ms": round(detect_ms, 3),
                "align_embed_ms": round(align_ms, 3),
                "quality_pose_ms": round(quality_ms, 3),
                "match_ms": round(match_ms, 3),
                "annotate_ms": round(annotate_ms, 3),
                "total_ms": round(elapsed_ms, 3),
            },
        },
        "warnings": warnings,
    }
    _prune_scans()
    cache_store.prune_cache()
    _save_result(result)
    return result


def _model_info_sync(name: str | None = None, detector_id: str | None = None,
                     recognizer_id: str | None = None) -> dict:
    if detector_id or recognizer_id:
        detector_id = detector_id or settings.default_detector
        recognizer_id = recognizer_id or settings.default_recognizer
        started = time.perf_counter()
        _require_detector(detector_id)
        _require_recognizer(recognizer_id)
        try:
            engine.warm_up(detector_id, recognizer_id, settings.face_scan_det_size, "none")
        except engine.FaceEngineUnavailable as exc:
            return {"ready": False, "error": str(exc), "detectorId": detector_id,
                    "recognizerId": recognizer_id}
        return {
            "ready": True,
            "detectorId": detector_id,
            "recognizerId": recognizer_id,
            "detector": KNOWN_DETECTORS[detector_id],
            "recognizer": KNOWN_RECOGNIZERS[recognizer_id],
            "providers": settings.provider_list,
            "loadMs": round((time.perf_counter() - started) * 1000),
            "python": sys.version.split()[0],
        }

    name = name or settings.default_model
    started = time.perf_counter()
    path = engine.model_path(name)
    if not engine.model_exists(name):
        return {
            "ready": False,
            "error": f"Model pack '{name}' is not installed at {path}.",
            "name": name,
            "path": str(path),
            "installed": False,
        }
    try:
        engine.get_face_analyzer(name)
    except engine.FaceEngineUnavailable as exc:
        return {"ready": False, "error": str(exc), "name": name, "path": str(path), "installed": True}
    except Exception as exc:  # pragma: no cover
        return {
            "ready": False,
            "error": f"Model could not be loaded: {exc}",
            "name": name,
            "path": str(path),
            "installed": True,
        }
    return {
        "ready": True,
        "installed": True,
        "name": name,
        "path": str(path),
        "providers": settings.provider_list,
        "enrollDetSize": settings.face_det_size,
        "scanDetSize": settings.face_scan_det_size,
        "threshold": settings.face_similarity_threshold,
        "loadMs": round((time.perf_counter() - started) * 1000),
        "python": sys.version.split()[0],
        "insightface": getattr(sys.modules.get("insightface"), "__version__", "unknown"),
    }


# --------------------------------------------------------------------------- #
# Benchmark + sweep
# --------------------------------------------------------------------------- #
def _resolve_benchmark_config(request: BenchmarkRunRequest) -> dict:
    registry = engine.registry_payload()
    available_detectors = [item["id"] for item in registry["detectors"] if item["available"]]
    available_recognizers = [item["id"] for item in registry["recognizers"] if item["available"]]
    detectors = [item for item in (request.detectors or available_detectors) if item in available_detectors]
    recognizers = [item for item in (request.recognizers or available_recognizers) if item in available_recognizers]
    det_sizes = [int(size) for size in (request.det_sizes or settings.benchmark_det_size_list) if 320 <= int(size) <= 2048]
    tiling_modes = request.tiling_modes or ["none"]
    if not detectors:
        raise HTTPException(status_code=422, detail="No available detectors to benchmark")
    if not recognizers:
        raise HTTPException(status_code=422, detail="No available recognizers to benchmark")
    combos = [
        (detector, recognizer, size, tiling)
        for detector in detectors
        for recognizer in recognizers
        for size in det_sizes
        for tiling in tiling_modes
    ][: settings.benchmark_max_rows]
    return {
        "detectors": detectors,
        "recognizers": recognizers,
        "det_sizes": det_sizes,
        "tiling_modes": tiling_modes,
        "combos": combos,
    }


def _combo_slug(index: int, detector_id: str, recognizer_id: str, det_size: int, tiling: str) -> str:
    raw = f"{detector_id}-{recognizer_id}-{det_size}-{tiling}".lower()
    return f"{index:02d}-" + re.sub(r"[^a-z0-9]+", "-", raw).strip("-")


def _benchmark_frames(
    dataset_inputs: list, evaluations: list, snapshot: dict[int, dict]
) -> list[dict]:
    output = []
    for (frames, _present, _boxes), evaluation in zip(dataset_inputs, evaluations):
        for frame, decisions in zip(frames, evaluation["per_frame"]):
            faces = [
                _serialize_face(decision, det, snapshot)
                for det, decision in zip(frame["detections"], decisions)
            ]
            for face in faces:
                # Source-space landmarks are unused by the benchmark viewer;
                # dropping them keeps the run JSON compact.
                face.pop("landmarks5pt", None)
            output.append(
                {
                    "datasetId": frame["dataset_id"],
                    "frameId": frame["frame_id"],
                    "width": frame["img_w"],
                    "height": frame["img_h"],
                    "imageUrl": f"/media/datasets/{frame['dataset_id']}/{frame['filename']}",
                    "annotatedUrl": None,
                    "cached": frame["cached"],
                    "faces": faces,
                }
            )
    return output


def _render_benchmark_preview(
    image_bgr: np.ndarray, faces: list[dict], relative: str, label_unknown: bool
) -> str | None:
    """Burn one annotated preview JPEG for a combo, downscaled to stay legible."""
    try:
        height, width = image_bgr.shape[:2]
        scale = min(1.0, settings.benchmark_preview_max_width / max(1, width))
        if scale < 1.0:
            image_bgr = cv2.resize(
                image_bgr,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        matches = [
            {
                "bbox": [value * scale for value in face["box"]],
                "status": face["status"],
                "student_name": face["name"],
                "confidence": face["confidence"],
                "matched": face["matched"],
            }
            for face in faces
        ]
        annotated = engine.annotate_classroom_photo(image_bgr, matches, label_unknown=label_unknown)
        _write_jpeg_rgb(annotated, relative, quality=85)
        return f"/media/{relative}"
    except Exception as exc:  # a preview must never fail the run
        print(f"[facelab] benchmark preview failed: {exc}")
        return None


def _run_benchmark_sync(request: BenchmarkRunRequest) -> dict:
    if not request.dataset_ids:
        raise HTTPException(status_code=422, detail="Pick at least one dataset")
    manifests = {}
    for dataset_id in request.dataset_ids:
        manifest = dataset_store.load_manifest(dataset_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
        manifests[dataset_id] = manifest

    config = _resolve_benchmark_config(request)
    if not config["combos"]:
        raise HTTPException(status_code=422, detail="Benchmark grid is empty")

    variant = request.variant or _variant_for(request.flip_aug, request.lowres_aug)
    snapshot = _student_snapshot()
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    rows = []
    warmups: dict[tuple, float] = {}

    with _model_lock:
        for combo_index, (detector_id, recognizer_id, det_size, tiling_mode) in enumerate(
            config["combos"]
        ):
            warm_key = (detector_id, recognizer_id, det_size, tiling_mode)
            if warm_key not in warmups:
                try:
                    info = engine.warm_up(
                        detector_id, recognizer_id, det_size, tiling_mode,
                        request.det_thresh, request.nms_thresh,
                    )
                    warmups[warm_key] = float(info.get("warmup_ms") or 0.0)
                except engine.FaceEngineUnavailable as exc:
                    rows.append(
                        {
                            "detector": detector_id,
                            "recognizer": recognizer_id,
                            "detSize": det_size,
                            "tilingMode": tiling_mode,
                            "error": str(exc),
                        }
                    )
                    continue
            warmup_ms = warmups.get(warm_key, 0.0)
            combo_started = time.perf_counter()
            cached_frames = 0
            first_bgr = None

            gallery = bench.build_gallery(snapshot, recognizer_id, variant)
            det_params = cache_store.analysis_params(
                det_size=det_size,
                det_thresh=request.det_thresh,
                nms_thresh=request.nms_thresh,
                tiling_mode=tiling_mode,
                lowres_aug=request.lowres_aug,
                with_quality=request.with_quality,
                with_pose=request.with_pose,
            )

            dataset_inputs: list[tuple[list[dict], list[int], list[dict]]] = []
            timings: list[dict] = []
            expected_faces = 0
            try:
                for dataset_id, manifest in manifests.items():
                    gt = manifest.get("ground_truth", {})
                    present_ids = [int(value) for value in gt.get("present_student_ids", [])]
                    dataset_boxes = gt.get("annotated_boxes", [])
                    expected_faces += int(
                        gt.get("expected_total_faces") or len(manifest.get("frames", []))
                    )
                    dataset_frames: list[dict] = []
                    for frame in manifest["frames"][: settings.benchmark_max_frames]:
                        path = dataset_store.dataset_dir(dataset_id) / frame["filename"]
                        if not path.is_file():
                            continue
                        data = path.read_bytes()
                        image_bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if image_bgr is None:
                            continue
                        payload, cached = _analyze_or_cache(
                            image_bgr, data,
                            detector_id=detector_id, recognizer_id=recognizer_id,
                            det_params=det_params, flip_aug=request.flip_aug,
                            force_fresh=request.force_fresh,
                        )
                        cached_frames += int(cached)
                        if first_bgr is None:
                            first_bgr = image_bgr
                        dataset_frames.append(
                            {
                                "frame_id": frame["frame_id"],
                                "width": frame["width"],
                                "height": frame["height"],
                                "detections": payload["detections"],
                                "timings": payload["timings_ms"],
                                "dataset_id": dataset_id,
                                "filename": frame["filename"],
                                "img_w": int(image_bgr.shape[1]),
                                "img_h": int(image_bgr.shape[0]),
                                "cached": cached,
                            }
                        )
                        timings.append(payload["timings_ms"])
                    if dataset_frames:
                        dataset_inputs.append((dataset_frames, present_ids, dataset_boxes))
            except engine.FaceEngineUnavailable as exc:
                rows.append(
                    {
                        "detector": detector_id,
                        "recognizer": recognizer_id,
                        "detSize": det_size,
                        "tilingMode": tiling_mode,
                        "error": str(exc),
                    }
                )
                continue

            if not dataset_inputs:
                rows.append(
                    {
                        "detector": detector_id,
                        "recognizer": recognizer_id,
                        "detSize": det_size,
                        "tilingMode": tiling_mode,
                        "error": "no readable frames in the selected datasets",
                    }
                )
                continue

            curve = bench.threshold_curve_datasets(
                [(frames, present_ids) for frames, present_ids, _boxes in dataset_inputs],
                gallery,
                variant=variant,
                review_threshold=request.review_threshold,
                min_quality=request.min_quality,
                min_margin=request.min_margin,
                assignment_mode=request.assignment_mode,
                template_strategy=request.template_strategy,
                top_k=request.top_k,
                fusion_mode=request.fusion_mode,
                k=request.k,
            )
            points = bench.pick_operating_points(curve)
            chosen_threshold = request.accept_threshold or (
                points["best_f1"]["threshold"] if points["best_f1"] else request.review_threshold
            )
            review_used = min(request.review_threshold, chosen_threshold)
            match_started = time.perf_counter()
            evaluations = [
                bench.evaluate_frames(
                    frames,
                    gallery,
                    present_ids,
                    variant=variant,
                    accept_threshold=chosen_threshold,
                    review_threshold=review_used,
                    min_margin=request.min_margin,
                    assignment_mode=request.assignment_mode,
                    template_strategy=request.template_strategy,
                    top_k=request.top_k,
                    fusion_mode=request.fusion_mode,
                    k=request.k,
                    min_quality=request.min_quality,
                    annotated_boxes=dataset_boxes,
                )
                for frames, present_ids, dataset_boxes in dataset_inputs
            ]
            match_ms = (time.perf_counter() - match_started) * 1000
            per_frame_match = round(match_ms / max(1, len(timings)), 3)
            # Copy rather than mutate: cached timing dicts are shared with the cache.
            timings = [{**item, "match_ms": per_frame_match} for item in timings]
            attendance = bench.aggregate_attendance(evaluations)
            detection = bench.aggregate_detection(
                [evaluation["detection"] for evaluation in evaluations]
            )
            faces_detected = sum(
                evaluation["faces_detected"] for evaluation in evaluations
            )

            frames_meta = _benchmark_frames(dataset_inputs, evaluations, snapshot)
            if frames_meta and first_bgr is not None:
                slug = _combo_slug(combo_index, detector_id, recognizer_id, det_size, tiling_mode)
                frames_meta[0]["annotatedUrl"] = _render_benchmark_preview(
                    first_bgr, frames_meta[0]["faces"],
                    f"benchmarks/{run_id}/{slug}.jpg", request.label_unknown,
                )
            first_bgr = None
            latency = bench.latency_percentiles(timings)
            wall_ms = (time.perf_counter() - combo_started) * 1000

            rows.append(
                {
                    "detector": detector_id,
                    "recognizer": recognizer_id,
                    "detSize": det_size,
                    "tilingMode": tiling_mode,
                    "variant": variant,
                    "providers": settings.provider_list,
                    "detectorSha256": _weight_sha256(KNOWN_DETECTORS.get(detector_id, {})),
                    "recognizerSha256": _weight_sha256(KNOWN_RECOGNIZERS.get(recognizer_id, {})),
                    "acceptThreshold": chosen_threshold,
                    "reviewThreshold": review_used,
                    "minMargin": request.min_margin,
                    "assignmentMode": request.assignment_mode,
                    "templateStrategy": request.template_strategy,
                    "fusionMode": request.fusion_mode,
                    "metrics": attendance,
                    "meanPresentSim": attendance["mean_present_sim"],
                    "maxImpostorSim": attendance["max_impostor_sim"],
                    "simMargin": round(
                        attendance["mean_present_sim"] - attendance["max_impostor_sim"], 4
                    ),
                    "tauF1": points["best_f1"]["threshold"] if points["best_f1"] else None,
                    "bestF1": points["best_f1"]["f1"] if points["best_f1"] else 0.0,
                    "far0": points["far0"],
                    "prec98": points["prec98"],
                    "detection": detection,
                    "detectedBuckets": detection["detected_buckets"],
                    "facesDetected": faces_detected,
                    "expectedFaces": expected_faces,
                    "latency": latency,
                    "fps": latency.get("frames_per_sec", 0.0),
                    "matchMs": round(match_ms, 3),
                    "wallMs": round(wall_ms, 3),
                    "cachedFrames": cached_frames,
                    "frameCount": len(timings),
                    "annotatedUrl": frames_meta[0]["annotatedUrl"] if frames_meta else None,
                    "frames": frames_meta,
                    "warmupMs": round(warmup_ms, 3),
                    "curve": curve,
                }
            )

    run = {
        "runId": run_id,
        "createdAt": _now(),
        "config": {
            "datasetIds": request.dataset_ids,
            "detectors": config["detectors"],
            "recognizers": config["recognizers"],
            "detSizes": config["det_sizes"],
            "tilingModes": config["tiling_modes"],
            "detThresh": request.det_thresh,
            "nmsThresh": request.nms_thresh,
            "variant": variant,
            "flipAug": request.flip_aug,
            "lowresAug": request.lowres_aug,
            "withQuality": request.with_quality,
            "withPose": request.with_pose,
            "forceFresh": request.force_fresh,
            "labelUnknown": request.label_unknown,
            "providers": settings.provider_list,
        },
        "rows": rows,
    }
    cache_store.prune_cache()
    _save_benchmark(run)
    return run


def _save_benchmark(run: dict) -> None:
    path = settings.benchmarks_dir / f"{run['runId']}.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(run, indent=2), encoding="utf-8")
    temp.replace(path)
    files = sorted(settings.benchmarks_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in files[settings.benchmarks_to_keep:]:
        stale.unlink(missing_ok=True)
        shutil.rmtree(settings.benchmarks_dir / stale.stem, ignore_errors=True)


def _list_benchmarks() -> list[dict]:
    if not settings.benchmarks_dir.exists():
        return []
    items = []
    for path in settings.benchmarks_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        items.append(
            {
                "runId": data.get("runId"),
                "createdAt": data.get("createdAt"),
                "rows": len(data.get("rows", [])),
                "config": data.get("config"),
            }
        )
    items.sort(key=lambda item: item.get("createdAt") or "", reverse=True)
    return items


def _sweep_dataset(
    dataset_ids: list[str],
    detector_id: str,
    recognizer_id: str,
    det_size: int,
    det_params: dict,
    flip_aug: bool,
    variant: str,
    accept_threshold: float,
    review_threshold: float,
    min_margin: float,
    assignment_mode: str,
    template_strategy: str,
    top_k: int,
    fusion_mode: str,
    fusion_k: int,
    min_quality: float,
) -> dict:
    snapshot = _student_snapshot()
    gallery = bench.build_gallery(snapshot, recognizer_id, variant)
    per_dataset = []
    total_tp = total_fp = total_fn = 0
    for dataset_id in dataset_ids:
        manifest = dataset_store.load_manifest(dataset_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
        frames, missing = bench.load_cached_frames(
            dataset_id,
            detector_id=detector_id,
            recognizer_id=recognizer_id,
            det_params=det_params,
            flip_aug=flip_aug,
        )
        if missing or not frames:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Dataset '{dataset_id}' has no cached inference for this model "
                    f"({missing} frame(s) missing). Run the benchmark first."
                ),
            )
        gt = manifest.get("ground_truth", {})
        evaluation = bench.evaluate_frames(
            frames,
            gallery,
            gt.get("present_student_ids", []),
            variant=variant,
            accept_threshold=accept_threshold,
            review_threshold=review_threshold,
            min_margin=min_margin,
            assignment_mode=assignment_mode,
            template_strategy=template_strategy,
            top_k=top_k,
            fusion_mode=fusion_mode,
            k=fusion_k,
            min_quality=min_quality,
            annotated_boxes=gt.get("annotated_boxes", []),
        )
        total_tp += evaluation["attendance"]["tp"]
        total_fp += evaluation["attendance"]["fp"]
        total_fn += evaluation["attendance"]["fn"]
        per_dataset.append(
            {
                "datasetId": dataset_id,
                "name": manifest.get("name"),
                "attendance": evaluation["attendance"],
                "detection": evaluation["detection"],
                "students": {str(k): v for k, v in evaluation["students"].items()},
                "faces": [
                    [
                        _serialize_face(decision, det, snapshot)
                        for det, decision in zip(frame["detections"], decisions)
                    ]
                    for frame, decisions in zip(frames, evaluation["per_frame"])
                ],
            }
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "attendance": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "datasets": per_dataset,
        "params": {
            "detector": detector_id,
            "recognizer": recognizer_id,
            "detSize": det_size,
            "variant": variant,
            "acceptThreshold": accept_threshold,
            "reviewThreshold": review_threshold,
            "minMargin": min_margin,
            "assignmentMode": assignment_mode,
            "templateStrategy": template_strategy,
            "fusionMode": fusion_mode,
            "fusionK": fusion_k,
        },
    }


def _run_sweep_sync(request: SweepRequest) -> dict:
    if not request.dataset_ids:
        raise HTTPException(status_code=422, detail="Pick at least one dataset")
    detector_id = request.detector or settings.default_detector
    recognizer_id = request.recognizer or settings.default_recognizer
    _require_detector(detector_id)
    _require_recognizer(recognizer_id)
    det_size = request.det_size or settings.face_scan_det_size
    det_params = cache_store.analysis_params(
        det_size=det_size,
        det_thresh=request.det_thresh,
        nms_thresh=request.nms_thresh,
        tiling_mode=request.tiling_mode,
        lowres_aug=request.lowres_aug,
        with_quality=request.with_quality,
        with_pose=request.with_pose,
    )
    variant = request.variant or _variant_for(request.flip_aug, request.lowres_aug)
    return _sweep_dataset(
        request.dataset_ids, detector_id, recognizer_id, det_size, det_params,
        request.flip_aug, variant, request.accept_threshold, request.review_threshold,
        request.min_margin, request.assignment_mode, request.template_strategy,
        request.top_k, request.fusion_mode, request.k, request.min_quality,
    )


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="FaceLab", version="0.3.0")

if settings.cors_origin_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials="*" not in settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.mount("/media/faces", StaticFiles(directory=str(settings.faces_dir)), name="faces")
app.mount("/media/scans", StaticFiles(directory=str(settings.scans_dir)), name="scans")
app.mount("/media/crops", StaticFiles(directory=str(settings.crops_dir)), name="crops")
app.mount("/media/datasets", StaticFiles(directory=str(settings.datasets_dir)), name="datasets")
app.mount("/media/benchmarks", StaticFiles(directory=str(settings.benchmarks_dir)), name="benchmarks")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "defaultModel": settings.default_model,
        "defaultDetector": settings.default_detector,
        "defaultRecognizer": settings.default_recognizer,
        "providers": settings.provider_list,
        "models": [item["name"] for item in engine.available_models()],
    }


@app.get("/api/state")
def state() -> dict:
    return _public_state()


@app.get("/api/registry")
def registry() -> dict:
    return _registry_state()


@app.get("/api/models")
def models() -> dict:
    return _models_payload()


@app.get("/api/model")
async def model(
    name: str | None = None,
    detector: str | None = None,
    recognizer: str | None = None,
) -> dict:
    return await run_in_threadpool(_model_info_sync, name, detector, recognizer)


@app.get("/api/cache")
def cache_stats() -> dict:
    files = list(settings.cache_dir.glob("*.json")) if settings.cache_dir.exists() else []
    return {
        "cacheDir": str(settings.cache_dir),
        "entries": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "crops": len(list(settings.crops_dir.glob("*.jpg"))) if settings.crops_dir.exists() else 0,
    }


# --------------------------------------------------------------------------- #
# Students / enrollment
# --------------------------------------------------------------------------- #
@app.post("/api/people")
async def create_person(
    images: list[UploadFile] = File(...),
    name: str = Form(""),
    recognizer: str = Form(""),
    recognizers: list[str] = Form([]),
    _: None = Depends(require_token),
) -> dict:
    raw = [(await image.read(), image.filename) for image in images]
    targets = _resolve_recognizer_ids([*recognizers, recognizer])
    person = await run_in_threadpool(_create_person_sync, name, targets, raw)
    return {"person": person, "state": _public_state()}


@app.post("/api/people/{person_id}/photos")
async def add_photos(
    person_id: int,
    images: list[UploadFile] = File(...),
    recognizer: str = Form(""),
    recognizers: list[str] = Form([]),
    _: None = Depends(require_token),
) -> dict:
    raw = [(await image.read(), image.filename) for image in images]
    targets = _resolve_recognizer_ids([*recognizers, recognizer])
    person = await run_in_threadpool(_add_photos_sync, person_id, targets, raw)
    return {"person": person, "state": _public_state()}


@app.post("/api/students/re-embed")
async def re_embed(payload: ReEmbedRequest, _: None = Depends(require_token)) -> dict:
    return await run_in_threadpool(_reembed_sync, payload.recognizer_ids)


@app.patch("/api/people/{person_id}")
def rename_person(person_id: int, payload: RenameRequest, _: None = Depends(require_token)) -> dict:
    clean_name = payload.name.strip()[:80]
    if not clean_name:
        raise HTTPException(status_code=422, detail="Name cannot be empty")
    with _state_lock:
        person = _people.get(person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="Student not found")
        person["name"] = clean_name
        _save_state()
        public = _public_person(person)
    return {"person": public, "state": _public_state()}


@app.delete("/api/people/{person_id}/photos/{photo_id}")
def delete_photo(person_id: int, photo_id: str, _: None = Depends(require_token)) -> dict:
    with _state_lock:
        person = _people.get(person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="Student not found")
        photo = next((item for item in person["photos"] if item["id"] == photo_id), None)
        if photo is None:
            raise HTTPException(status_code=404, detail="Photo not found")
        person["photos"].remove(photo)
        try:
            (settings.data_dir / photo["filename"]).unlink(missing_ok=True)
        except (OSError, TypeError):
            pass
        _save_state()
        public = _public_person(person)
    return {"person": public, "state": _public_state()}


@app.delete("/api/people/{person_id}")
def delete_person(person_id: int, _: None = Depends(require_token)) -> dict:
    with _state_lock:
        person = _people.pop(person_id, None)
        if person is None:
            raise HTTPException(status_code=404, detail="Student not found")
        shutil.rmtree(settings.faces_dir / f"p{person['number']}", ignore_errors=True)
        _save_state()
    return {"ok": True, "state": _public_state()}


# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #
@app.post("/api/scan")
async def scan(
    images: list[UploadFile] = File(...),
    model: str = Form(""),
    detector: str = Form(""),
    recognizer: str = Form(""),
    threshold: float | None = Form(None),
    accept_threshold: float | None = Form(None, alias="acceptThreshold"),
    review_threshold: float | None = Form(None, alias="reviewThreshold"),
    min_margin: float | None = Form(None, alias="minMargin"),
    det_size: int | None = Form(None, alias="detSize"),
    det_thresh: float | None = Form(None, alias="detThresh"),
    nms_thresh: float | None = Form(None, alias="nmsThresh"),
    tiling_mode: str = Form("", alias="tilingMode"),
    assignment_mode: str = Form("", alias="assignmentMode"),
    template_strategy: str = Form("", alias="templateStrategy"),
    top_k: int | None = Form(None, alias="topK"),
    fusion_mode: str = Form("", alias="fusionMode"),
    fusion_k: int | None = Form(None, alias="fusionK"),
    flip_aug: bool = Form(True, alias="flipAug"),
    lowres_aug: bool = Form(False, alias="lowresAug"),
    with_quality: bool = Form(True, alias="withQuality"),
    with_pose: bool = Form(True, alias="withPose"),
    min_quality: float | None = Form(None, alias="minQuality"),
    label_unknown: bool = Form(False, alias="labelUnknown"),
    _: None = Depends(require_token),
) -> dict:
    if not images:
        raise HTTPException(status_code=422, detail="Add at least one classroom frame")
    if len(images) > settings.face_scan_max_frames:
        raise HTTPException(
            status_code=422, detail=f"Up to {settings.face_scan_max_frames} frames per scan"
        )
    detector_id, recognizer_id = _legacy_ids(model or None, detector or None, recognizer or None)
    det_size = det_size or settings.face_scan_det_size
    if not 320 <= det_size <= 2048:
        raise HTTPException(status_code=422, detail="Detector size must be 320-2048")
    accept = accept_threshold if accept_threshold is not None else (
        threshold if threshold is not None else settings.accept_threshold
    )
    for name, value in (("acceptThreshold", accept), ("reviewThreshold", review_threshold),
                        ("minMargin", min_margin)):
        if value is not None and not 0.0 <= value <= 1.0:
            raise HTTPException(status_code=422, detail=f"{name} must be between 0 and 1")
    raw = [(await image.read(), image.filename) for image in images]
    return await run_in_threadpool(
        _run_scan_sync,
        raw,
        detector_id=detector_id,
        recognizer_id=recognizer_id,
        det_size=det_size,
        det_thresh=det_thresh if det_thresh is not None else 0.35,
        nms_thresh=nms_thresh if nms_thresh is not None else 0.45,
        tiling_mode=tiling_mode or settings.tiling_mode,
        accept_threshold=accept,
        review_threshold=(
            review_threshold if review_threshold is not None else settings.review_threshold
        ),
        min_margin=min_margin if min_margin is not None else settings.min_margin,
        assignment_mode=assignment_mode or settings.assignment_mode,
        template_strategy=template_strategy or settings.template_strategy,
        top_k=top_k if top_k is not None else settings.template_top_k,
        fusion_mode=fusion_mode or settings.multi_frame_fusion,
        fusion_k=fusion_k if fusion_k is not None else settings.fusion_k,
        flip_aug=flip_aug,
        lowres_aug=lowres_aug,
        with_quality=with_quality,
        with_pose=with_pose,
        min_quality=min_quality if min_quality is not None else settings.min_face_quality,
        label_unknown=label_unknown,
    )


@app.get("/api/results")
def results(model: str | None = None) -> dict:
    return {"results": _list_results(model)}


@app.get("/api/results/{scan_id}")
def get_result(scan_id: str) -> dict:
    path = _result_path(scan_id)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.delete("/api/results/{scan_id}")
def delete_result(scan_id: str, _: None = Depends(require_token)) -> dict:
    path = _result_path(scan_id)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    path.unlink(missing_ok=True)
    shutil.rmtree(settings.scans_dir / scan_id, ignore_errors=True)
    return {"ok": True, "results": _list_results()}


@app.post("/api/results/{scan_id}/sweep")
def sweep_result(
    scan_id: str, payload: SweepRequest, _: None = Depends(require_token)
) -> dict:
    """Re-score a saved scan from its cache when the threshold sliders move."""
    path = _result_path(scan_id)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    for label, value in (
        ("accept_threshold", payload.accept_threshold),
        ("review_threshold", payload.review_threshold),
        ("min_margin", payload.min_margin),
    ):
        if not 0.0 <= value <= 1.0:
            raise HTTPException(status_code=422, detail=f"{label} must be between 0 and 1")
    result = json.loads(path.read_text(encoding="utf-8"))
    summary = result.get("summary", {})
    recognizer_id = payload.recognizer or result.get("recognizerId") or settings.default_recognizer
    _require_recognizer(recognizer_id)
    variant = payload.variant or summary.get("variant") or _variant_for(
        summary.get("flipAug", True), summary.get("lowresAug", False)
    )
    snapshot = _student_snapshot()
    gallery = bench.build_gallery(snapshot, recognizer_id, variant)

    frames = []
    missing = 0
    for frame in result.get("frames", []):
        key = frame.get("cacheKey")
        cached = cache_store.load_analysis(key) if key else None
        if cached is None:
            missing += 1
            continue
        frames.append(
            {
                "frame_id": f"frame_{frame['order'] + 1}",
                "width": frame["width"],
                "height": frame["height"],
                "detections": cached["detections"],
            }
        )
    if missing or not frames:
        raise HTTPException(
            status_code=409,
            detail=f"Cached inference for this scan is missing ({missing} frame(s)). Re-run the scan.",
        )

    # Attendance ground truth is unknown for ad-hoc scans, so report the
    # per-student decisions and per-face statuses only.
    fusion_frames = bench._fusion_frames(frames, variant)
    fused = matching.fuse_multi_frame(
        fusion_frames,
        gallery,
        mode=payload.fusion_mode,
        accept_threshold=payload.accept_threshold,
        review_threshold=payload.review_threshold,
        min_margin=payload.min_margin,
        assignment_mode=payload.assignment_mode,
        template_strategy=payload.template_strategy,
        top_k=payload.top_k,
        k=payload.k,
        min_quality=payload.min_quality,
    )
    faces = [
        [
            _serialize_face(decision, det, snapshot)
            for det, decision in zip(frame["detections"], decisions)
        ]
        for frame, decisions in zip(frames, fused["per_frame"])
    ]
    return {
        "students": {str(k): v for k, v in fused["students"].items()},
        "faces": faces,
        "matchedCount": sum(1 for record in fused["students"].values() if record["status"] == "present"),
        "needsReview": sum(1 for record in fused["students"].values() if record["status"] == "needs_review"),
        "unknown": max(
            (sum(1 for face in frame if face["status"] == "unmatched") for frame in faces),
            default=0,
        ),
        "params": payload.model_dump(),
    }


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
@app.get("/api/datasets")
def list_datasets() -> dict:
    return {"datasets": dataset_store.list_datasets()}


@app.post("/api/datasets")
async def create_dataset(
    images: list[UploadFile] = File(...),
    name: str = Form(""),
    present_ids: str = Form("", alias="presentIds"),
    boxes: str = Form(""),
    notes: str = Form(""),
    dataset_id: str = Form("", alias="datasetId"),
    _: None = Depends(require_token),
) -> dict:
    if not images:
        raise HTTPException(status_code=422, detail="Add at least one classroom frame")
    if len(images) > settings.face_scan_max_frames:
        raise HTTPException(
            status_code=422,
            detail=f"Up to {settings.face_scan_max_frames} frames per dataset",
        )
    try:
        present_list = json.loads(present_ids) if present_ids else []
        box_list = json.loads(boxes) if boxes else []
        assert isinstance(present_list, list) and isinstance(box_list, list)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid ground-truth JSON: {exc}") from exc

    raw = [(await image.read(), image.filename) for image in images]
    try:
        manifest = await run_in_threadpool(
            dataset_store.create_dataset,
            name,
            raw,
            present_list,
            box_list,
            notes,
            dataset_id or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"dataset": manifest, "datasets": dataset_store.list_datasets()}


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str) -> dict:
    manifest = dataset_store.load_manifest(dataset_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return manifest


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str, _: None = Depends(require_token)) -> dict:
    if not dataset_store.delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"ok": True, "datasets": dataset_store.list_datasets()}


# --------------------------------------------------------------------------- #
# Benchmarking
# --------------------------------------------------------------------------- #
@app.post("/api/benchmark/run")
async def benchmark_run(payload: BenchmarkRunRequest, _: None = Depends(require_token)) -> dict:
    return await run_in_threadpool(_run_benchmark_sync, payload)


@app.post("/api/benchmark/sweep")
async def benchmark_sweep(payload: SweepRequest, _: None = Depends(require_token)) -> dict:
    return await run_in_threadpool(_run_sweep_sync, payload)


@app.get("/api/benchmarks")
def list_benchmarks() -> dict:
    return {"benchmarks": _list_benchmarks()}


@app.get("/api/benchmarks/{run_id}")
def get_benchmark(run_id: str) -> dict:
    if not run_id or any(char not in "0123456789abcdef-" for char in run_id.lower()):
        raise HTTPException(status_code=404, detail="Benchmark not found")
    path = settings.benchmarks_dir / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Reset
# --------------------------------------------------------------------------- #
@app.post("/api/reset")
def reset(_: None = Depends(require_token)) -> dict:
    global _next_number
    with _state_lock:
        _people.clear()
        _next_number = 1
        _save_state()
    for folder in (
        settings.faces_dir,
        settings.scans_dir,
        settings.results_dir,
        settings.crops_dir,
        settings.cache_dir,
        settings.datasets_dir,
        settings.benchmarks_dir,
    ):
        shutil.rmtree(folder, ignore_errors=True)
    settings.ensure_dirs()
    return {"ok": True, "state": _public_state()}


def main() -> None:
    import uvicorn

    default_host = os.environ.get("FACELAB_HOST") or os.environ.get("HOST")
    default_port = os.environ.get("FACELAB_PORT") or os.environ.get("PORT")
    if default_host is None:
        default_host = "0.0.0.0" if default_port else "127.0.0.1"

    parser = argparse.ArgumentParser(description="FaceLab face-model benchmark harness")
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=int(default_port or 8010))
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    print(f"[facelab] env:       {settings.env}")
    print(f"[facelab] models:    {settings.models_dir}")
    print(f"[facelab] data:      {settings.data_dir}")
    print(f"[facelab] providers: {settings.provider_list}")
    print(f"[facelab] detector:  {settings.default_detector}")
    print(f"[facelab] recognizer:{settings.default_recognizer}")
    print(f"[facelab] open:      http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


_load_state()

if __name__ == "__main__":
    main()
