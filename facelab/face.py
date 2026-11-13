"""FaceLab face engine.

Vendored from the Attendyrc backend (``app/ml/face.py``) so this tool can run
independently. The detection, embedding, matching, and annotation behavior is
kept identical; the only additions are a per-model analyzer cache and a model
pack registry.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from facelab.config import settings


class FaceEngineUnavailable(RuntimeError):
    """Raised when a model pack or InsightFace cannot be loaded."""


_ENGINE_LOCK = threading.RLock()
_ANALYZERS: dict[str, object] = {}


def model_path(name: str | None = None) -> Path:
    return settings.models_dir / (name or settings.default_model)


def available_models() -> list[dict]:
    """Describe every model pack found under ``models/``."""
    found: list[dict] = []
    if not settings.models_dir.exists():
        return found
    for path in sorted(settings.models_dir.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        onnx_files = sorted(path.glob("*.onnx"))
        if not onnx_files:
            continue
        size_bytes = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        found.append(
            {
                "name": path.name,
                "path": str(path),
                "files": [item.name for item in onnx_files],
                "fileCount": len(onnx_files),
                "sizeBytes": size_bytes,
                "loaded": path.name in _ANALYZERS,
                "default": path.name == settings.default_model,
            }
        )
    return found


def model_exists(name: str | None = None) -> bool:
    path = model_path(name)
    return path.is_dir() and any(path.glob("*.onnx"))


def _open_rgb(data: bytes) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")


def get_face_analyzer(model_name: str | None = None):
    """Load and cache an InsightFace analyzer for the process lifetime.

    One analyzer is kept per model pack, so switching between models in the UI
    never re-reads weights that were already loaded.
    """
    name = model_name or settings.default_model
    path = model_path(name)
    if not path.is_dir() or not any(path.glob("*.onnx")):
        raise FaceEngineUnavailable(
            f"Model pack '{name}' was not found at {path}. "
            "Run `python scripts/fetch_models.py --model "
            f"{name}` or add the ONNX files to that directory."
        )

    with _ENGINE_LOCK:
        analyzer = _ANALYZERS.get(name)
        if analyzer is not None:
            return analyzer
        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:  # pragma: no cover - depends on local install
            raise FaceEngineUnavailable(
                "InsightFace is not installed. Run `uv sync` in this project and try again."
            ) from exc

        try:
            analyzer = FaceAnalysis(
                name=str(path),
                root=str(settings.models_dir),
                providers=["CPUExecutionProvider"],
                allowed_modules=settings.allowed_module_list or None,
            )
            analyzer.prepare(
                ctx_id=0, det_size=(settings.face_det_size, settings.face_det_size)
            )
        except Exception as exc:
            raise FaceEngineUnavailable(f"Model '{name}' could not be loaded: {exc}") from exc
        _ANALYZERS[name] = analyzer
        return analyzer


def image_to_bgr(image_input) -> np.ndarray:
    """Convert bytes, a file-like object, a PIL image, or an array to BGR."""
    if image_input is None:
        raise ValueError("Image input is None")

    if isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3 and image_input.shape[2] == 3:
            return cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
        return image_input

    if hasattr(image_input, "read"):
        image_input.seek(0)
        bytes_data = image_input.read()
        pil_img = _open_rgb(bytes_data)
    elif isinstance(image_input, bytes):
        pil_img = _open_rgb(image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def detect_faces(
    image_bgr: np.ndarray,
    det_size: int | None = None,
    model_name: str | None = None,
):
    """Detect faces with the selected model pack.

    ``det_size`` only changes the detector's input resolution; the model itself
    is loaded once per pack. Larger sizes recover small, far-away faces in a
    classroom frame at the cost of a slower pass.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []
    with _ENGINE_LOCK:
        analyzer = get_face_analyzer(model_name)
        size = det_size or settings.face_det_size
        detector = getattr(analyzer, "det_model", None)
        current = getattr(detector, "input_size", None)
        if not current or tuple(current) != (size, size):
            analyzer.prepare(ctx_id=0, det_size=(size, size))
        return analyzer.get(image_bgr)


def validate_registration_face(
    image_bgr: np.ndarray, model_name: str | None = None
) -> tuple[bool, object, str]:
    """Registration photos must contain exactly one clear face."""
    if image_bgr is None or image_bgr.size == 0:
        return False, None, "Invalid or empty image provided."

    faces = detect_faces(image_bgr, model_name=model_name)
    if len(faces) == 0:
        return False, None, "No face detected. Ask the student to face the camera in good light."
    if len(faces) > 1:
        return (
            False,
            None,
            f"Multiple faces ({len(faces)}) detected. Use exactly one student per photo.",
        )

    face = faces[0]
    bbox = face.bbox.astype(int)
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if width < 50 or height < 50:
        return True, face, "Face is small in the frame; move closer for a stronger embedding."
    return True, face, "Face detected."


def extract_embedding(face_obj) -> np.ndarray:
    if face_obj is None or getattr(face_obj, "embedding", None) is None:
        raise ValueError("Face object does not contain an embedding")
    emb = np.asarray(face_obj.embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    return np.asarray(embedding, dtype=np.float32).tobytes()


def bytes_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def match_faces_to_students(
    detected_faces: list,
    registered_embeddings_rows: list[tuple[int, str, str | None, bytes]],
    all_class_students: list[tuple[int, str, str | None]],
    threshold: float | None = None,
) -> tuple[list[dict], list[dict]]:
    """Match detected faces to registered students with greedy disambiguation.

    Returns ``(matches, roster)``. ``matches`` is per detected face; ``roster``
    is per enrolled student, defaulting to Absent.
    """
    threshold = threshold if threshold is not None else settings.face_similarity_threshold

    student_lookup = {
        student_id: {
            "student_id": student_id,
            "name": name,
            "roll_number": roll,
            "status": "Absent",
            "confidence": None,
        }
        for student_id, name, roll in all_class_students
    }

    if not detected_faces:
        return [], sorted(student_lookup.values(), key=lambda row: row["name"])

    detected_embeddings = np.array(
        [extract_embedding(face) for face in detected_faces], dtype=np.float32
    )

    if not registered_embeddings_rows:
        matches = [
            {
                "face": face,
                "matched": False,
                "student_id": None,
                "student_name": "Unknown",
                "roll_number": None,
                "confidence": 0.0,
            }
            for face in detected_faces
        ]
        return matches, sorted(student_lookup.values(), key=lambda row: row["name"])

    registered_embeddings = []
    meta = []
    for student_id, name, roll, blob in registered_embeddings_rows:
        emb = (
            bytes_to_embedding(blob)
            if isinstance(blob, (bytes, bytearray, memoryview))
            else np.asarray(blob, dtype=np.float32)
        )
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        registered_embeddings.append(emb)
        meta.append((student_id, name, roll))

    matrix = np.asarray(registered_embeddings, dtype=np.float32)
    similarities = np.dot(detected_embeddings, matrix.T)

    candidates = [
        (face_index, student_index, float(similarities[face_index, student_index]))
        for face_index in range(similarities.shape[0])
        for student_index in range(similarities.shape[1])
    ]
    candidates.sort(key=lambda item: item[2], reverse=True)

    assigned_faces: dict[int, tuple[int, str, str | None, float]] = {}
    assigned_students: set[int] = set()
    for face_index, student_index, score in candidates:
        if score < threshold:
            break
        student_id, name, roll = meta[student_index]
        if face_index not in assigned_faces and student_id not in assigned_students:
            assigned_faces[face_index] = (student_id, name, roll, score)
            assigned_students.add(student_id)

    matches = []
    for face_index, face in enumerate(detected_faces):
        if face_index in assigned_faces:
            student_id, name, roll, score = assigned_faces[face_index]
            matches.append(
                {
                    "face": face,
                    "matched": True,
                    "student_id": student_id,
                    "student_name": name,
                    "roll_number": roll,
                    "confidence": round(score, 4),
                }
            )
            if student_id in student_lookup:
                student_lookup[student_id]["status"] = "Present"
                student_lookup[student_id]["confidence"] = round(score, 4)
        else:
            best = float(np.max(similarities[face_index])) if similarities.shape[1] else 0.0
            matches.append(
                {
                    "face": face,
                    "matched": False,
                    "student_id": None,
                    "student_name": "Unknown",
                    "roll_number": None,
                    "confidence": round(best, 4),
                }
            )

    roster = sorted(
        student_lookup.values(), key=lambda row: (row["status"] != "Present", row["name"])
    )
    return matches, roster


def match_frames_to_students(
    detected_frames: list[list],
    registered_embeddings_rows: list[tuple[int, str, str | None, bytes]],
    all_class_students: list[tuple[int, str, str | None]],
    threshold: float | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Match a burst of frames of the same scene and merge the results.

    Each frame is matched on its own. A student keeps the best similarity seen
    across every frame, so a face that is only clear in one frame still counts.
    ``unknown`` is the largest number of unrecognised faces seen in any single
    frame (the same guest appears in every frame, so we do not sum them), and
    ``facesDetected`` is the busiest frame's face count.

    Returns ``(matches, roster, summary)`` where ``matches`` are the per-face
    matches from the frame that recognised the most students (the frame we also
    annotate), and ``summary`` carries the merged results plus ``bestIndex``.
    """
    threshold = threshold if threshold is not None else settings.face_similarity_threshold

    frames = [frame for frame in detected_frames if frame is not None]
    best_matches: list[dict] = []
    best_roster: list[dict] = []
    best_index = -1
    best_recognised = -1
    merged: dict[int, float] = {}
    faces_detected = 0
    unknown = 0

    for index, faces in enumerate(frames):
        matches, roster = match_faces_to_students(
            faces, registered_embeddings_rows, all_class_students, threshold
        )
        faces_detected = max(faces_detected, len(faces))
        unknown = max(unknown, sum(1 for match in matches if not match["matched"]))
        recognised = 0
        for match in matches:
            if match["matched"]:
                recognised += 1
                student_id = match["student_id"]
                merged[student_id] = max(merged.get(student_id, 0.0), match["confidence"])
        if recognised > best_recognised:
            best_recognised = recognised
            best_matches = matches
            best_roster = roster
            best_index = index

    if not frames:
        best_matches, best_roster = match_faces_to_students(
            [], registered_embeddings_rows, all_class_students, threshold
        )

    roster_by_id = {row["student_id"]: dict(row) for row in best_roster}
    for student_id, confidence in merged.items():
        row = roster_by_id.get(student_id)
        if row is not None:
            row["status"] = "Present"
            row["confidence"] = confidence
    roster = sorted(
        roster_by_id.values(), key=lambda row: (row["status"] != "Present", row["name"])
    )

    summary = {
        "matched": merged,
        "unknown": unknown,
        "facesDetected": faces_detected,
        "frames": len(frames),
        "bestIndex": best_index,
    }
    return best_matches, roster, summary


def annotate_classroom_photo(image_bgr: np.ndarray, matches: list[dict]) -> np.ndarray:
    """Draw green boxes for matched faces and red boxes for unknown faces."""
    annotated = image_bgr.copy()
    height, width = annotated.shape[:2]
    diagonal = np.sqrt(height**2 + width**2)
    font_scale = max(0.45, min(1.2, diagonal / 2500.0))
    thickness = max(1, int(round(diagonal / 1200.0)))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for match in matches:
        face = match.get("face")
        if face is None or not hasattr(face, "bbox"):
            continue
        x1, y1, x2, y2 = face.bbox.astype(int)
        if match["matched"]:
            color = (0, 200, 0)
            label = f"{match['student_name']} ({match['confidence']:.2f})"
        else:
            color = (0, 0, 220)
            label = f"Unknown ({match['confidence']:.2f})"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        tag_y1 = max(0, y1 - text_height - 8)
        cv2.rectangle(annotated, (x1, tag_y1), (x1 + text_width + 6, y1), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 3, y1 - 4),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)