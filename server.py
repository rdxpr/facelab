"""FaceLab — a standalone harness for testing face-detection models.

FaceLab runs the InsightFace pipeline locally, lets you enroll reference photos
per model pack, scan classroom frames, and keeps every scan result as JSON so
the parent Attendyrc project can reuse it.

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
import shutil
import sys
import threading
import time
import uuid
from datetime import UTC, datetime

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from facelab import face as engine
from facelab.config import STATIC_DIR, settings

MAX_IMAGE_BYTES = 12 * 1024 * 1024

_state_lock = threading.RLock()
_model_lock = threading.Lock()
_people: dict[int, dict] = {}
_next_number = 1


class RenameRequest(BaseModel):
    name: str


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _photo_model(photo: dict) -> str:
    """Legacy photos predate the model field; they came from the default pack."""
    return photo.get("model") or settings.default_model


# --------------------------------------------------------------------------- #
# Persisted roster
# --------------------------------------------------------------------------- #
def _save_state() -> None:
    payload = {
        "version": 2,
        "nextNumber": _next_number,
        "people": [
            {
                "id": person["id"],
                "number": person["number"],
                "name": person["name"],
                "createdAt": person["createdAt"],
                "photos": [
                    {
                        "id": photo["id"],
                        "filename": photo["filename"],
                        "quality": photo["quality"],
                        "createdAt": photo["createdAt"],
                        "model": _photo_model(photo),
                        "embedding": base64.b64encode(
                            np.asarray(photo["embedding"], dtype=np.float32).tobytes()
                        ).decode("ascii"),
                    }
                    for photo in person["photos"]
                ],
            }
            for person in sorted(_people.values(), key=lambda item: item["number"])
        ],
    }
    temp = settings.people_file.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(settings.people_file)


def _load_state() -> None:
    global _next_number
    path = settings.people_file
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        people: dict[int, dict] = {}
        for person in payload.get("people", []):
            photos = []
            for photo in person.get("photos", []):
                raw = base64.b64decode(photo["embedding"])
                photos.append(
                    {
                        "id": photo["id"],
                        "filename": photo["filename"],
                        "quality": photo.get("quality") or "",
                        "createdAt": photo.get("createdAt") or _now(),
                        "model": photo.get("model") or settings.default_model,
                        "embedding": np.frombuffer(raw, dtype=np.float32).copy(),
                    }
                )
            people[int(person["id"])] = {
                "id": int(person["id"]),
                "number": int(person["number"]),
                "name": person["name"],
                "createdAt": person.get("createdAt") or _now(),
                "photos": photos,
            }
        _people.update(people)
        _next_number = int(payload.get("nextNumber") or (max(people, default=0) + 1))
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
    models: dict[str, int] = {}
    for photo in person["photos"]:
        name = _photo_model(photo)
        models[name] = models.get(name, 0) + 1
    return {
        "id": person["id"],
        "number": person["number"],
        "name": person["name"],
        "createdAt": person["createdAt"],
        "templates": len(person["photos"]),
        "models": models,
        "photos": [
            {
                "id": photo["id"],
                "url": f"/media/{photo['filename']}",
                "quality": photo["quality"],
                "createdAt": photo["createdAt"],
                "model": _photo_model(photo),
            }
            for photo in person["photos"]
        ],
    }


def _model_usage() -> dict[str, dict]:
    usage: dict[str, dict] = {}
    for person in _people.values():
        for photo in person["photos"]:
            name = _photo_model(photo)
            entry = usage.setdefault(name, {"templates": 0, "people": set()})
            entry["templates"] += 1
            entry["people"].add(person["id"])
    return {
        name: {"templates": value["templates"], "people": len(value["people"])}
        for name, value in usage.items()
    }


def _models_payload() -> dict:
    installed = {item["name"]: item for item in engine.available_models()}
    usage = _model_usage()
    models = []
    for name in sorted(set(installed) | set(usage)):
        info = installed.get(name, {})
        use = usage.get(name, {"templates": 0, "people": 0})
        models.append(
            {
                "name": name,
                "files": info.get("files", []),
                "fileCount": info.get("fileCount", 0),
                "sizeBytes": info.get("sizeBytes", 0),
                "loaded": info.get("loaded", False),
                "installed": name in installed,
                "default": name == settings.default_model,
                "templates": use["templates"],
                "people": use["people"],
            }
        )
    return {"models": models, "defaultModel": settings.default_model, "modelsDir": str(settings.models_dir)}


def _public_state() -> dict:
    with _state_lock:
        people = [_public_person(person) for person in _people.values()]
        usage = _model_usage()
    people.sort(key=lambda item: item["number"])
    total_templates = sum(person["templates"] for person in people)
    return {
        "people": people,
        "nextNumber": _next_number,
        "totals": {"people": len(people), "templates": total_templates},
        "modelUsage": usage,
        "settings": {
            "threshold": settings.face_similarity_threshold,
            "detSize": settings.face_scan_det_size,
            "enrollDetSize": settings.face_det_size,
            "maxFrames": settings.face_scan_max_frames,
            "maxPhotosPerUpload": settings.max_photos_per_upload,
            "maxPhotosPerPerson": settings.max_photos_per_person,
            "defaultModel": settings.default_model,
            "modelsDir": str(settings.models_dir),
        },
    }


# --------------------------------------------------------------------------- #
# Scan results on disk
# --------------------------------------------------------------------------- #
def _result_path(scan_id: str) -> "os.PathLike[str] | None":
    # Guard against path traversal from the {scan_id} path parameter.
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
        "threshold": summary.get("threshold"),
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
        if model and data.get("model") != model:
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


def _require_installed_model(model: str) -> None:
    if not engine.model_exists(model):
        raise HTTPException(
            status_code=422,
            detail=f"Model pack '{model}' is not installed. Add its ONNX files to {settings.models_dir / model} or run scripts/fetch_models.py.",
        )


# --------------------------------------------------------------------------- #
# Enrollment
# --------------------------------------------------------------------------- #
def _process_reference_photos(
    raw: list[tuple[bytes, str | None]], model: str
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    processed: list[tuple[np.ndarray, np.ndarray, str]] = []
    with _model_lock:
        for data, filename in raw:
            label = filename or "photo"
            image_bgr = _decode(data, label)
            try:
                ok, face_obj, message = engine.validate_registration_face(
                    image_bgr, model_name=model
                )
            except engine.FaceEngineUnavailable as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=422, detail=f"{label}: detection failed ({exc})"
                ) from exc
            if not ok or face_obj is None:
                raise HTTPException(status_code=422, detail=f"{label}: {message}")
            processed.append((image_bgr, engine.extract_embedding(face_obj), message))
    return processed


def _create_person_sync(
    name: str, model: str, raw: list[tuple[bytes, str | None]]
) -> dict:
    global _next_number
    if not raw:
        raise HTTPException(status_code=422, detail="Add at least one face photo")
    if len(raw) > settings.max_photos_per_upload:
        raise HTTPException(
            status_code=422,
            detail=f"Up to {settings.max_photos_per_upload} photos per upload",
        )
    _require_installed_model(model)
    processed = _process_reference_photos(raw, model)

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
        for image_bgr, embedding, message in processed:
            photo_id = uuid.uuid4().hex[:10]
            relative = _write_jpeg_bgr(image_bgr, f"faces/p{number}/{photo_id}.jpg")
            person["photos"].append(
                {
                    "id": photo_id,
                    "filename": relative,
                    "quality": message,
                    "createdAt": _now(),
                    "model": model,
                    "embedding": embedding,
                }
            )
        _people[person["id"]] = person
        _save_state()
        return _public_person(person)


def _add_photos_sync(
    person_id: int, model: str, raw: list[tuple[bytes, str | None]]
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
        raise HTTPException(
            status_code=422, detail=f"{remaining} more photo(s) fit for this student"
        )
    _require_installed_model(model)
    processed = _process_reference_photos(raw, model)

    with _state_lock:
        person = _people.get(person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="Student not found")
        for image_bgr, embedding, message in processed:
            photo_id = uuid.uuid4().hex[:10]
            relative = _write_jpeg_bgr(
                image_bgr, f"faces/p{person['number']}/{photo_id}.jpg"
            )
            person["photos"].append(
                {
                    "id": photo_id,
                    "filename": relative,
                    "quality": message,
                    "createdAt": _now(),
                    "model": model,
                    "embedding": embedding,
                }
            )
        _save_state()
        return _public_person(person)


# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #
def _serialize_face(match: dict, width: int, height: int, person: dict | None) -> dict:
    face = match["face"]
    x1, y1, x2, y2 = (float(value) for value in face.bbox)
    det_score = float(getattr(face, "det_score", 0.0))
    return {
        "box": [
            round(max(0.0, x1), 1),
            round(max(0.0, y1), 1),
            round(min(float(width), x2), 1),
            round(min(float(height), y2), 1),
        ],
        "detScore": round(det_score, 4),
        "matched": bool(match["matched"]),
        "personId": match["student_id"],
        "number": person["number"] if person else None,
        "name": match["student_name"],
        "confidence": round(float(match["confidence"] or 0.0), 4),
    }


def _run_scan_sync(
    raw: list[tuple[bytes, str | None]],
    model: str,
    threshold: float,
    det_size: int,
) -> dict:
    _require_installed_model(model)
    started = time.perf_counter()
    warnings: list[str] = []
    decoded: list[dict] = []

    with _model_lock:
        for index, (data, filename) in enumerate(raw):
            label = filename or f"frame {index + 1}"
            try:
                image_bgr = _decode(data, label)
                faces = engine.detect_faces(image_bgr, det_size=det_size, model_name=model)
            except HTTPException as exc:
                warnings.append(str(exc.detail))
                continue
            except engine.FaceEngineUnavailable as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except Exception as exc:
                warnings.append(f"{label}: {exc}")
                continue
            height, width = image_bgr.shape[:2]
            decoded.append(
                {
                    "index": index,
                    "sourceName": filename,
                    "bgr": image_bgr,
                    "width": width,
                    "height": height,
                    "faces": faces,
                }
            )

        if not decoded:
            raise HTTPException(
                status_code=422, detail="No readable frames. " + " ".join(warnings)
            )

        with _state_lock:
            people_snapshot = {
                person["id"]: {
                    "id": person["id"],
                    "number": person["number"],
                    "name": person["name"],
                    "photos": person["photos"],
                }
                for person in _people.values()
            }

        # Only templates enrolled for this model pack are comparable.
        registered_rows: list[tuple[int, str, str | None, bytes]] = []
        for person in people_snapshot.values():
            for photo in person["photos"]:
                if _photo_model(photo) != model:
                    continue
                registered_rows.append(
                    (
                        person["id"],
                        person["name"],
                        f"#{person['number']:02d}",
                        photo["embedding"].tobytes(),
                    )
                )

        # A student is on this model's roster only if they have a template for it.
        enrolled_ids = {row[0] for row in registered_rows}
        roster_rows = [
            (person["id"], person["name"], f"#{person['number']:02d}")
            for person in people_snapshot.values()
            if person["id"] in enrolled_ids
        ]

        # Drop templates whose embedding dimension does not match this model's
        # output, so a mixed roster can never break the similarity matrix.
        detected_dim = 0
        for item in decoded:
            if item["faces"]:
                detected_dim = len(engine.extract_embedding(item["faces"][0]))
                break
        if detected_dim:
            mismatched = [row for row in registered_rows if len(row[3]) // 4 != detected_dim]
            if mismatched:
                warnings.append(
                    f"Ignored {len(mismatched)} template(s) with a different embedding size for model '{model}'."
                )
                registered_rows = [
                    row for row in registered_rows if len(row[3]) // 4 == detected_dim
                ]
                kept = {row[0] for row in registered_rows}
                roster_rows = [row for row in roster_rows if row[0] in kept]

        frame_faces = [item["faces"] for item in decoded]
        _best_matches, roster, summary = engine.match_frames_to_students(
            frame_faces, registered_rows, roster_rows, threshold
        )
        per_frame_matches = [
            engine.match_faces_to_students(faces, registered_rows, roster_rows, threshold)[0]
            for faces in frame_faces
        ]

        scan_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        frames_payload = []
        for order, (item, matches) in enumerate(zip(decoded, per_frame_matches)):
            frame_rel = _write_jpeg_bgr(item["bgr"], f"scans/{scan_id}/frame-{order}.jpg")
            annotated = engine.annotate_classroom_photo(item["bgr"], matches)
            annotated_rel = _write_jpeg_rgb(
                annotated, f"scans/{scan_id}/frame-{order}-annotated.jpg"
            )
            faces_payload = []
            for match in matches:
                person = people_snapshot.get(match["student_id"])
                faces_payload.append(
                    _serialize_face(match, item["width"], item["height"], person)
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
                    "faces": faces_payload,
                }
            )

        best_index = int(summary["bestIndex"])
        roster_payload = []
        for row in roster:
            person = people_snapshot.get(row["student_id"])
            if person is None:
                continue
            primary = next(
                (photo for photo in person["photos"] if _photo_model(photo) == model), None
            )
            roster_payload.append(
                {
                    "personId": person["id"],
                    "number": person["number"],
                    "name": person["name"],
                    "status": row["status"],
                    "confidence": row["confidence"],
                    "photoUrl": f"/media/{primary['filename']}" if primary else None,
                    "templates": sum(
                        1 for photo in person["photos"] if _photo_model(photo) == model
                    ),
                }
            )

    result = {
        "scanId": scan_id,
        "createdAt": _now(),
        "model": model,
        "people": roster_payload,
        "frames": frames_payload,
        "summary": {
            "model": model,
            "threshold": round(threshold, 4),
            "detSize": det_size,
            "frames": len(decoded),
            "uploadedFrames": len(raw),
            "facesDetected": int(summary["facesDetected"]),
            "unknown": int(summary["unknown"]),
            "matchedCount": len(summary["matched"]),
            "peopleCount": len(roster_payload),
            "templateCount": len(registered_rows),
            "bestIndex": best_index,
            "elapsedMs": round((time.perf_counter() - started) * 1000),
        },
        "warnings": warnings,
    }
    _prune_scans()
    _save_result(result)
    return result


def _model_info_sync(model: str) -> dict:
    started = time.perf_counter()
    path = engine.model_path(model)
    if not engine.model_exists(model):
        return {
            "ready": False,
            "error": f"Model pack '{model}' is not installed at {path}.",
            "name": model,
            "path": str(path),
            "installed": False,
        }
    try:
        engine.get_face_analyzer(model)
    except engine.FaceEngineUnavailable as exc:
        return {
            "ready": False,
            "error": str(exc),
            "name": model,
            "path": str(path),
            "installed": True,
        }
    except Exception as exc:  # pragma: no cover - surfaced to the UI
        return {
            "ready": False,
            "error": f"Model could not be loaded: {exc}",
            "name": model,
            "path": str(path),
            "installed": True,
        }
    return {
        "ready": True,
        "installed": True,
        "name": model,
        "path": str(path),
        "providers": ["CPUExecutionProvider"],
        "enrollDetSize": settings.face_det_size,
        "scanDetSize": settings.face_scan_det_size,
        "threshold": settings.face_similarity_threshold,
        "loadMs": round((time.perf_counter() - started) * 1000),
        "python": sys.version.split()[0],
        "insightface": getattr(sys.modules.get("insightface"), "__version__", "unknown"),
    }


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="FaceLab", version="0.2.0")

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
        "models": [item["name"] for item in engine.available_models()],
    }


@app.get("/api/state")
def state() -> dict:
    return _public_state()


@app.get("/api/models")
def models() -> dict:
    return _models_payload()


@app.get("/api/model")
async def model(name: str | None = None) -> dict:
    return await run_in_threadpool(_model_info_sync, name or settings.default_model)


@app.post("/api/people")
async def create_person(
    images: list[UploadFile] = File(...),
    name: str = Form(""),
    model: str = Form(""),
) -> dict:
    raw = []
    for image in images:
        raw.append((await image.read(), image.filename))
    person = await run_in_threadpool(
        _create_person_sync, name, model or settings.default_model, raw
    )
    return {"person": person, "state": _public_state()}


@app.post("/api/people/{person_id}/photos")
async def add_photos(
    person_id: int,
    images: list[UploadFile] = File(...),
    model: str = Form(""),
) -> dict:
    raw = []
    for image in images:
        raw.append((await image.read(), image.filename))
    person = await run_in_threadpool(
        _add_photos_sync, person_id, model or settings.default_model, raw
    )
    return {"person": person, "state": _public_state()}


@app.patch("/api/people/{person_id}")
def rename_person(person_id: int, payload: RenameRequest) -> dict:
    clean_name = payload.name.strip()[:80]
    if not clean_name:
        raise HTTPException(status_code=422, detail="Name cannot be empty")
    with _state_lock:
        person = _people.get(person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="Student not found")
        person["name"] = clean_name
        _save_state()
    return {"person": _public_person(person), "state": _public_state()}


@app.delete("/api/people/{person_id}/photos/{photo_id}")
def delete_photo(person_id: int, photo_id: str) -> dict:
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
        except OSError:
            pass
        _save_state()
    return {"person": _public_person(person), "state": _public_state()}


@app.delete("/api/people/{person_id}")
def delete_person(person_id: int) -> dict:
    with _state_lock:
        person = _people.pop(person_id, None)
        if person is None:
            raise HTTPException(status_code=404, detail="Student not found")
        shutil.rmtree(settings.faces_dir / f"p{person['number']}", ignore_errors=True)
        _save_state()
    return {"ok": True, "state": _public_state()}


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
def delete_result(scan_id: str) -> dict:
    path = _result_path(scan_id)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    path.unlink(missing_ok=True)
    shutil.rmtree(settings.scans_dir / scan_id, ignore_errors=True)
    return {"ok": True, "results": _list_results()}


@app.post("/api/reset")
def reset() -> dict:
    global _next_number
    with _state_lock:
        _people.clear()
        _next_number = 1
        _save_state()
    shutil.rmtree(settings.faces_dir, ignore_errors=True)
    shutil.rmtree(settings.scans_dir, ignore_errors=True)
    shutil.rmtree(settings.results_dir, ignore_errors=True)
    settings.ensure_dirs()
    return {"ok": True, "state": _public_state()}


@app.post("/api/scan")
async def scan(
    images: list[UploadFile] = File(...),
    model: str = Form(""),
    threshold: float | None = Form(None),
    det_size: int | None = Form(None, alias="detSize"),
) -> dict:
    if not images:
        raise HTTPException(status_code=422, detail="Add at least one classroom frame")
    if len(images) > settings.face_scan_max_frames:
        raise HTTPException(
            status_code=422, detail=f"Up to {settings.face_scan_max_frames} frames per scan"
        )
    value = settings.face_similarity_threshold if threshold is None else threshold
    if not 0.0 <= value <= 1.0:
        raise HTTPException(status_code=422, detail="Threshold must be between 0 and 1")
    size = settings.face_scan_det_size if det_size is None else det_size
    if not 320 <= size <= 2048:
        raise HTTPException(status_code=422, detail="Detector size must be 320-2048")

    raw = []
    for image in images:
        raw.append((await image.read(), image.filename))
    return await run_in_threadpool(
        _run_scan_sync, raw, model or settings.default_model, value, size
    )


def main() -> None:
    import uvicorn

    default_host = os.environ.get("FACELAB_HOST") or os.environ.get("HOST")
    default_port = os.environ.get("FACELAB_PORT") or os.environ.get("PORT")
    if default_host is None:
        # Hugging Face Spaces / most PaaS expose the service on PORT and need
        # the server on 0.0.0.0.
        default_host = "0.0.0.0" if default_port else "127.0.0.1"

    parser = argparse.ArgumentParser(description="FaceLab face-model playground")
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=int(default_port or 8010))
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    print(f"[facelab] env:     {settings.env}")
    print(f"[facelab] models:  {settings.models_dir}")
    print(f"[facelab] data:    {settings.data_dir}")
    print(f"[facelab] default: {settings.default_model}")
    print(f"[facelab] open:    http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


_load_state()

if __name__ == "__main__":
    main()
