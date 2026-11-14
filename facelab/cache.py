"""Persistent inference cache.

Neural-network inference is the expensive part of a classroom sweep. This module
stores, per ``(image, detector, detection params, recognizer, flip_aug)`` tuple,
every detected face's geometry, aligned crop path, quality metrics, and raw
embedding. Once the artifact exists, threshold/margin/assignment/fusion sweeps
run in pure NumPy without touching ONNX Runtime again.

Vectors are stored as plain JSON lists (readable and diff-friendly) rather than
compressed blobs, matching the schema in the specification guide.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from facelab.config import settings

CACHE_VERSION = 1


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def analysis_params(
    det_size: int,
    det_thresh: float = 0.35,
    nms_thresh: float = 0.45,
    tiling_mode: str = "none",
    tile_size: int = 640,
    tile_overlap: float = 0.25,
    lowres_aug: bool = False,
    with_quality: bool = True,
    with_pose: bool = True,
) -> dict:
    """Build the analysis parameter dict shared by the scan/benchmark paths.

    Two callers must build this identically or their cache keys will not match.
    """
    return {
        "det_size": [int(det_size), int(det_size)],
        "det_thresh": float(det_thresh),
        "nms_thresh": float(nms_thresh),
        "tiling_mode": tiling_mode,
        "tile_size": int(tile_size),
        "tile_overlap": float(tile_overlap),
        "lowres_aug": bool(lowres_aug),
        "with_quality": bool(with_quality),
        "with_pose": bool(with_pose),
    }


def _param_fingerprint(
    image_sha256: str,
    detector_id: str,
    det_params: dict,
    recognizer_id: str,
    flip_aug: bool,
) -> dict:
    return {
        "v": CACHE_VERSION,
        "image_sha256": image_sha256,
        "detector_id": detector_id,
        "det_size": list(det_params.get("det_size") or []),
        "det_thresh": round(float(det_params.get("det_thresh", 0.5)), 4),
        "nms_thresh": round(float(det_params.get("nms_thresh", 0.45)), 4),
        "tiling_mode": det_params.get("tiling_mode", "none"),
        "tile_size": int(det_params.get("tile_size", 640)),
        "tile_overlap": round(float(det_params.get("tile_overlap", 0.25)), 3),
        "lowres_aug": bool(det_params.get("lowres_aug", False)),
        "with_quality": bool(det_params.get("with_quality", True)),
        "with_pose": bool(det_params.get("with_pose", True)),
        "recognizer_id": recognizer_id,
        "flip_aug": bool(flip_aug),
    }


def cache_key(
    image_sha256: str,
    detector_id: str,
    det_params: dict,
    recognizer_id: str,
    flip_aug: bool = True,
) -> str:
    """Deterministic, filesystem-safe key for one inference artifact."""
    fingerprint = _param_fingerprint(
        image_sha256, detector_id, det_params, recognizer_id, flip_aug
    )
    digest = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    det_slug = detector_id.replace(":", "-").replace(".", "")
    rec_slug = recognizer_id.replace(":", "-").replace(".", "")
    return f"{image_sha256[:12]}_{det_slug}_{rec_slug}_{digest}"


def cache_path(key: str) -> Path:
    return settings.cache_dir / f"{key}.json"


def has_analysis(key: str) -> bool:
    return cache_path(key).is_file()


def save_analysis(payload: dict) -> Path:
    path = cache_path(payload["cache_key"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(_encode(payload)), encoding="utf-8")
    temp.replace(path)
    return path


def load_analysis(key: str) -> dict | None:
    path = cache_path(key)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        try:
            path.touch(exist_ok=True)
        except OSError:
            pass
        return _decode(data)
    except Exception:
        return None


def prune_cache(keep: int | None = None) -> int:
    """Delete the oldest cached artifacts and their aligned crop JPEGs."""
    keep = settings.cache_entries_to_keep if keep is None else keep
    if not settings.cache_dir.exists():
        return 0
    files = sorted(
        settings.cache_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for stale in files[keep:]:
        for crop in settings.crops_dir.glob(f"{stale.stem}_det*.jpg"):
            crop.unlink(missing_ok=True)
        stale.unlink(missing_ok=True)
    return max(0, len(files) - keep)


def _encode(payload: dict) -> dict:
    """Convert numpy arrays to JSON-safe lists, leaving everything else alone."""
    encoded: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "detections":
            encoded[key] = [_encode_detection(det) for det in value]
        elif isinstance(value, np.ndarray):
            encoded[key] = value.tolist()
        elif isinstance(value, dict):
            encoded[key] = _encode(value)
        else:
            encoded[key] = value
    return encoded


def _encode_detection(det: dict) -> dict:
    out: dict[str, Any] = {}
    for key, value in det.items():
        if key == "crop":
            continue  # the aligned crop lives on disk, not in the artifact
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def _decode(payload: dict) -> dict:
    decoded: dict[str, Any] = dict(payload)
    decoded["detections"] = [_decode_detection(det) for det in payload.get("detections", [])]
    return decoded


def _decode_detection(det: dict) -> dict:
    out: dict[str, Any] = dict(det)
    for key in ("bbox", "embedding", "embedding_flip_avg", "embedding_lowres_aug"):
        value = out.get(key)
        if isinstance(value, list):
            out[key] = np.asarray(value, dtype=np.float32)
    return out
