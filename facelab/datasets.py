"""Persistent ground-truth datasets for repeatable evaluation.

A dataset is a folder of classroom frames plus a manifest that records the
students who were actually present and (optionally) annotated face boxes with
their row bucket. Benchmark runs and threshold sweeps read from here, so a
parameter change never requires re-uploading images.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import UTC, datetime

import cv2
import numpy as np

from facelab.cache import sha256_bytes
from facelab.config import settings

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def valid_dataset_id(dataset_id: str) -> bool:
    return bool(dataset_id) and bool(_ID_PATTERN.match(dataset_id))


def dataset_dir(dataset_id: str):
    return settings.datasets_dir / dataset_id


def manifest_path(dataset_id: str):
    return dataset_dir(dataset_id) / "manifest.json"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "dataset").strip().lower()).strip("-")
    return (slug or "dataset")[:40]


def create_dataset(
    name: str,
    frames: list[tuple[bytes, str | None]],
    present_student_ids: list[int] | None = None,
    annotated_boxes: list[dict] | None = None,
    notes: str = "",
    dataset_id: str | None = None,
) -> dict:
    """Write a new dataset folder and return its manifest."""
    if not frames:
        raise ValueError("A dataset needs at least one frame")

    identifier = dataset_id or f"{_slug(name)}-{uuid.uuid4().hex[:6]}"
    if not valid_dataset_id(identifier):
        raise ValueError("Invalid dataset id")

    target = dataset_dir(identifier)
    if target.exists():
        raise ValueError(f"Dataset '{identifier}' already exists")
    frames_dir = target / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        frame_records = []
        for index, (data, filename) in enumerate(frames):
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Frame {filename or index + 1} could not be decoded")
            height, width = image.shape[:2]
            extension = ".jpg"
            if filename and "." in filename:
                extension = "." + filename.rsplit(".", 1)[-1].lower()
                if extension not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                    extension = ".jpg"
            frame_id = f"frame_{index + 1:02d}"
            stored_name = f"{frame_id}{extension}"
            (frames_dir / stored_name).write_bytes(data)
            frame_records.append(
                {
                    "frame_id": frame_id,
                    "filename": f"frames/{stored_name}",
                    "original_name": filename,
                    "sha256": sha256_bytes(data),
                    "width": int(width),
                    "height": int(height),
                }
            )

        boxes = []
        for box in annotated_boxes or []:
            try:
                bbox = [float(value) for value in box["bbox"][:4]]
            except (KeyError, TypeError, IndexError, ValueError):
                continue
            height_px = bbox[3] - bbox[1]
            row = box.get("row_bucket")
            if not row:
                row = "large" if height_px >= 55 else ("medium" if height_px >= 30 else "small")
            boxes.append(
                {
                    "frame_id": box.get("frame_id") or frame_records[0]["frame_id"],
                    "bbox": bbox,
                    "student_id": box.get("student_id"),
                    "occlusion": box.get("occlusion") or "none",
                    "row_bucket": row,
                }
            )

        present_count = len(present_student_ids or [])
        expected_faces = len(boxes) or (present_count * len(frame_records)) or len(frame_records)
        manifest = {
            "dataset_id": identifier,
            "datasetId": identifier,
            "name": (name or identifier).strip()[:120] or identifier,
            "notes": notes.strip()[:500],
            "createdAt": _now(),
            "frames": frame_records,
            "ground_truth": {
                "present_student_ids": [int(value) for value in (present_student_ids or [])],
                "expected_total_faces": int(expected_faces),
                "annotated_boxes": boxes,
            },
        }
        manifest_path(identifier).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    _prune_datasets()
    return manifest


def load_manifest(dataset_id: str) -> dict | None:
    if not valid_dataset_id(dataset_id):
        return None
    path = manifest_path(dataset_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_datasets() -> list[dict]:
    if not settings.datasets_dir.exists():
        return []
    items = []
    for folder in settings.datasets_dir.iterdir():
        if not folder.is_dir():
            continue
        manifest = load_manifest(folder.name)
        if manifest is None:
            continue
        gt = manifest.get("ground_truth", {})
        items.append(
            {
                "dataset_id": manifest["dataset_id"],
                "datasetId": manifest["dataset_id"],
                "name": manifest.get("name"),
                "createdAt": manifest.get("createdAt"),
                "frames": len(manifest.get("frames", [])),
                "present": len(gt.get("present_student_ids", [])),
                "annotatedBoxes": len(gt.get("annotated_boxes", [])),
                "thumbUrl": _thumb_url(manifest),
            }
        )
    items.sort(key=lambda item: (item.get("createdAt") or ""), reverse=True)
    return items


def _thumb_url(manifest: dict) -> str | None:
    frames = manifest.get("frames", [])
    if not frames:
        return None
    return f"/media/datasets/{manifest['dataset_id']}/{frames[0]['filename']}"


def delete_dataset(dataset_id: str) -> bool:
    if not valid_dataset_id(dataset_id):
        return False
    target = dataset_dir(dataset_id)
    if not target.exists():
        return False
    shutil.rmtree(target, ignore_errors=True)
    return True


def frame_paths(dataset_id: str, manifest: dict | None = None) -> list[tuple[str, "object"]]:
    manifest = manifest or load_manifest(dataset_id)
    if manifest is None:
        return []
    base = dataset_dir(dataset_id)
    return [(frame["frame_id"], base / frame["filename"]) for frame in manifest["frames"]]


def _prune_datasets(keep: int | None = None) -> None:
    keep = settings.datasets_to_keep if keep is None else keep
    if not settings.datasets_dir.exists():
        return
    folders = [path for path in settings.datasets_dir.iterdir() if path.is_dir()]
    folders.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for stale in folders[keep:]:
        shutil.rmtree(stale, ignore_errors=True)
