"""FaceLab face engine.

Splits the old monolithic InsightFace analyzer into an independent **detector**
and **recognizer** registry, adds Slicing-Aided Hyper Inference (SAHI) for
back-row faces, per-stage timing, aligned-crop saving, and optional
quality/head-pose predictors from UniFace.

Public entry points used by the API:

* :func:`get_detector` / :func:`get_recognizer` — cached ONNX models.
* :func:`analyze_frame` — detect + align + embed (+ quality/pose) one frame.
* :func:`detect_with_sahi` — tiling wrapper for small faces.
* :func:`warm_up` — excluded-from-metrics first pass.
* :func:`registry_payload` — availability + license table for the UI.

Legacy model-pack helpers (``get_face_analyzer``, ``model_exists``, ...) are
retained so older callers keep working.
"""

from __future__ import annotations

import inspect
import io
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from facelab import matching
from facelab.config import (
    KNOWN_DETECTORS,
    KNOWN_RECOGNIZERS,
    detector_registry,
    recognizer_registry,
    settings,
)


class FaceEngineUnavailable(RuntimeError):
    """Raised when a model pack, library, or weight file cannot be loaded."""


_ENGINE_LOCK = threading.RLock()
_ANALYZERS: dict[str, object] = {}
_MAX_DETECTORS = 6
_DETECTORS: "OrderedDict[tuple, BaseDetector]" = OrderedDict()
_RECOGNIZERS: dict[tuple, "BaseRecognizer"] = {}
_QUALITY: dict[str, Any] = {}
_POSE: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# Path / availability helpers
# --------------------------------------------------------------------------- #
def model_path(name: str | None = None) -> Path:
    return settings.models_dir / (name or settings.default_model)


def _candidate_paths(entry: dict) -> list[Path]:
    paths: list[Path] = []
    pack = entry.get("pack")
    for name in entry.get("file_candidates", []):
        paths.append(settings.models_dir / name)
        if pack:
            paths.append(settings.models_dir / pack / Path(name).name)
    return paths


def resolve_model_file(entry: dict) -> Path | None:
    for path in _candidate_paths(entry):
        if path.is_file():
            return path
    return None


def _uniface_module():
    try:
        import uniface  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on local install
        raise FaceEngineUnavailable(
            "The UniFace package is required for this model. Install it with "
            '`uv pip install "uniface[cpu]"` (or `[gpu]` on a CUDA host).'
        ) from exc
    return __import__("uniface")


def _uniface_enum(value: str):
    constants = __import__("uniface.constants", fromlist=["*"])
    import enum as _enum

    for name in dir(constants):
        candidate = getattr(constants, name)
        if not (isinstance(candidate, type) and issubclass(candidate, _enum.Enum)):
            continue
        # Skip the bare `Enum` base and any memberless class: calling them with a
        # value raises TypeError rather than the ValueError we look for.
        if candidate is _enum.Enum or not list(candidate):
            continue
        try:
            return candidate(value)
        except ValueError:
            continue
    raise FaceEngineUnavailable(f"UniFace has no weights entry named '{value}'.")


def _file_model_availability(entry: dict) -> tuple[bool, str]:
    path = resolve_model_file(entry)
    if path is None:
        candidates = ", ".join(str(item) for item in _candidate_paths(entry))
        return False, f"weights not found (looked for: {candidates})"
    return True, ""


def _uniface_weight_cached(value: str | None) -> Path | None:
    """Path to a cached UniFace weight file, or None if it isn't downloaded.

    UniFace resolves weights lazily on first use, so constructing a model can
    kick off a multi-hundred-megabyte download from inside a request. Checking
    the cache (``~/.uniface/models``) directly means a model only counts as
    ready once its weights are actually on disk.
    """
    if not value:
        return None
    try:
        from uniface.model_store import get_cache_dir
    except Exception:
        return None
    cache = Path(get_cache_dir())
    if not cache.is_dir():
        return None
    matches = sorted(cache.glob(f"{value}.*"))
    return matches[0] if matches else None


def _uniface_cached_weight(entry: dict) -> Path | None:
    return _uniface_weight_cached(entry.get("uniface_weights"))


def _uniface_availability(entry: dict) -> tuple[bool, str]:
    try:
        _uniface_module()
    except FaceEngineUnavailable as exc:
        return False, str(exc)
    if _uniface_cached_weight(entry) is None:
        value = entry.get("uniface_weights")
        return False, (
            f"weights not downloaded yet ({value}); run "
            f"`scripts/fetch_models.py --uniface {value}`"
        )
    return True, ""


def detector_availability(detector_id: str) -> tuple[bool, str]:
    entry = KNOWN_DETECTORS.get(detector_id)
    if entry is None:
        return False, "unknown detector id"
    if entry["library"] == "uniface":
        return _uniface_availability(entry)
    return _file_model_availability(entry)


def recognizer_availability(recognizer_id: str) -> tuple[bool, str]:
    entry = KNOWN_RECOGNIZERS.get(recognizer_id)
    if entry is None:
        return False, "unknown recognizer id"
    if entry["library"] == "uniface":
        return _uniface_availability(entry)
    return _file_model_availability(entry)


def registry_payload() -> dict:
    detectors = []
    for detector_id, entry in detector_registry().items():
        ok, reason = detector_availability(detector_id)
        detectors.append({**entry, "available": ok, "reason": reason})
    recognizers = []
    for recognizer_id, entry in recognizer_registry().items():
        ok, reason = recognizer_availability(recognizer_id)
        recognizers.append({**entry, "available": ok, "reason": reason})
    return {
        "detectors": detectors,
        "recognizers": recognizers,
        "providers": settings.provider_list,
    }


# --------------------------------------------------------------------------- #
# Working with images
# --------------------------------------------------------------------------- #
def _open_rgb(data: bytes) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")


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


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    bbox: np.ndarray
    score: float
    kps: np.ndarray | None = None

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


def _box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def _box_intersection(a: np.ndarray, b: np.ndarray) -> float:
    width = min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))
    height = min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1]))
    return max(0.0, width) * max(0.0, height)


def merge_detections(
    primary: list[Detection], extra: list[Detection], iou_threshold: float = 0.45
) -> list[Detection]:
    """Add detections from a second pass without replacing better geometry.

    A lower-resolution pass can recover a face the primary scale missed, but it
    also returns coarser boxes for faces already found. Keep the primary box
    unless the extra box is substantially larger and mostly contains it (the
    primary pass clipped the face).
    """
    merged = list(primary)
    for det in extra:
        area = _box_area(det.bbox)
        best_iou, best_iou_index = 0.0, None
        best_contain, best_contain_index = 0.0, None
        for index, kept in enumerate(merged):
            intersection = _box_intersection(det.bbox, kept.bbox)
            kept_area = _box_area(kept.bbox)
            union = area + kept_area - intersection
            if union > 0 and intersection / union > best_iou:
                best_iou, best_iou_index = intersection / union, index
            if kept_area > 0 and intersection / kept_area > best_contain:
                best_contain, best_contain_index = intersection / kept_area, index
        if (
            best_contain_index is not None
            and best_contain > 0.8
            and area > _box_area(merged[best_contain_index].bbox) * 1.3
        ):
            merged[best_contain_index] = det
            continue
        if best_iou_index is not None and best_iou >= iou_threshold:
            continue
        merged.append(det)
    return merged


def select_dominant_face(detections: list[Detection]) -> Detection:
    """Return the most prominent detection: largest bbox, ties on score.

    Used by enrolment so a portrait with a stray background face still resolves
    to the subject instead of failing the upload.
    """
    return max(
        detections,
        key=lambda det: (
            float((det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1])),
            float(det.score),
        ),
    )


class BaseDetector:
    id: str = ""
    label: str = ""
    det_size: tuple[int, int] = (640, 640)

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        raise NotImplementedError

    def warm_up(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detect(image)


class InsightFaceDetector(BaseDetector):
    """SCRFD detector from the InsightFace model zoo (5-point keypoints)."""

    #: A face taller than this fraction of the frame overflows the detector's
    #: anchor scales at the requested input size: SCRFD then returns nothing or a
    #: clipped fragment. A cheap low-resolution recovery pass fixes close-ups.
    CLOSEUP_FRACTION = 0.5
    RECOVERY_DET_SIZE = 128

    def __init__(self, detector_id: str, entry: dict, det_size: tuple[int, int],
                 det_thresh: float, nms_thresh: float, providers: list[str]):
        from insightface.model_zoo.scrfd import SCRFD

        path = resolve_model_file(entry)
        if path is None:
            raise FaceEngineUnavailable(
                f"Detection weights for {detector_id} were not found in {settings.models_dir}."
            )
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=providers)
        self.model = SCRFD(model_file=str(path), session=session)
        self.model.prepare(
            ctx_id=0, input_size=det_size, det_thresh=det_thresh, nms_thresh=nms_thresh
        )
        # SCRFD keeps these on the model, and some zoo packs reset them in
        # prepare(); pin them so the requested thresholds actually take effect.
        self.model.det_thresh = float(det_thresh)
        self.model.nms_thresh = float(nms_thresh)
        if not getattr(self.model, "use_kps", True):
            raise FaceEngineUnavailable(
                f"{detector_id} has no 5-point keypoint head, so it cannot feed "
                "recognition alignment. Pair it with a landmark predictor or pick "
                "a *_bnkps variant."
            )
        self.id = detector_id
        self.label = entry.get("label", detector_id)
        self.det_size = det_size
        self.det_thresh = det_thresh
        self.nms_thresh = nms_thresh

    def _detect_at(self, image_bgr: np.ndarray, size: int) -> list[Detection]:
        boxes, kps = self.model.detect(image_bgr, input_size=(size, size))
        detections: list[Detection] = []
        for index in range(len(boxes)):
            detections.append(
                Detection(
                    bbox=np.asarray(boxes[index, :4], dtype=np.float32),
                    score=float(boxes[index, 4]),
                    kps=np.asarray(kps[index], dtype=np.float32) if kps is not None else None,
                )
            )
        return detections

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        detections = self._detect_at(image_bgr, int(self.det_size[0]))
        frame_height = float(image_bgr.shape[0] or 1)
        largest = max((det.height for det in detections), default=0.0)
        recovery = min(self.RECOVERY_DET_SIZE, int(self.det_size[0]) - 1)
        if recovery >= 64 and (
            not detections or largest > self.CLOSEUP_FRACTION * frame_height
        ):
            # The primary pass found nothing or a face that no longer fits the
            # input grid, so re-run cheaply at a small input size where a
            # close-up is back inside the detector's supported scale range.
            detections = merge_detections(
                detections, self._detect_at(image_bgr, recovery)
            )
        return detections


class UnifaceDetector(BaseDetector):
    """RetinaFace / YOLOv8-Face / SCRFD detectors from UniFace."""

    _CLASS_HINTS = (
        ("retinaface", "RetinaFace"),
        ("yolov8", "YOLOv8Face"),
        ("yolov5", "YOLOv5Face"),
        ("scrfd", "SCRFD"),
        ("centerface", "CenterFace"),
    )

    def __init__(self, detector_id: str, entry: dict, det_size: tuple[int, int],
                 det_thresh: float, nms_thresh: float, providers: list[str]):
        module = __import__("uniface.detection", fromlist=["*"])
        weights_value = entry["uniface_weights"]
        class_name = next(
            (name for hint, name in self._CLASS_HINTS if hint in weights_value), None
        )
        if class_name is None or not hasattr(module, class_name):
            raise FaceEngineUnavailable(f"UniFace has no detector class for {detector_id}.")
        cls = getattr(module, class_name)
        signature = inspect.signature(cls.__init__).parameters.get("input_size")
        default = signature.default if signature is not None else (640, 640)
        kwargs = {
            "model_name": _uniface_enum(weights_value),
            "confidence_threshold": float(det_thresh),
            "nms_threshold": float(nms_thresh),
            "providers": providers,
        }
        if isinstance(default, int):
            kwargs["input_size"] = int(det_size[0])
        else:
            kwargs["input_size"] = tuple(det_size)
        self.model = cls(**kwargs)
        self.id = detector_id
        self.label = entry.get("label", detector_id)
        self.det_size = det_size

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        faces = self.model.detect(image_bgr)
        detections: list[Detection] = []
        for face in faces:
            bbox = np.asarray(face.bbox, dtype=np.float32)
            landmarks = getattr(face, "landmarks", None)
            if landmarks is None:
                raise FaceEngineUnavailable(
                    f"{self.id} returned no landmarks, so faces cannot be aligned."
                )
            detections.append(
                Detection(
                    bbox=bbox,
                    score=float(getattr(face, "confidence", 0.0)),
                    kps=np.asarray(landmarks, dtype=np.float32).reshape(5, 2),
                )
            )
        return detections


class OpenCVYuNetDetector(BaseDetector):
    """YuNet from OpenCV Zoo; commercial-safe and very light."""

    def __init__(self, detector_id: str, entry: dict, det_size: tuple[int, int],
                 det_thresh: float, nms_thresh: float, providers: list[str] | None = None):
        path = resolve_model_file(entry)
        if path is None:
            raise FaceEngineUnavailable(
                f"YuNet weights were not found in {settings.models_dir}."
            )
        self.model = cv2.FaceDetectorYN_create(
            str(path), "", (int(det_size[0]), int(det_size[1])),
            score_threshold=float(det_thresh), nms_threshold=float(nms_thresh),
        )
        self.id = detector_id
        self.label = entry.get("label", detector_id)
        self.det_size = det_size

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        height, width = image_bgr.shape[:2]
        # Scale the *input* to the requested detector size and map results back.
        # The previous code reset the input to the full frame on every call, so
        # ``detSize`` had no effect and benchmark columns for YuNet were identical.
        scale = min(1.0, float(self.det_size[0]) / max(1, width, height))
        if scale < 1.0:
            resized = cv2.resize(
                image_bgr,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            resized = image_bgr
        resized_height, resized_width = resized.shape[:2]
        self.model.setInputSize((resized_width, resized_height))
        _, faces = self.model.detect(resized)
        detections: list[Detection] = []
        if faces is None:
            return detections
        for row in faces:
            x, y, w, h = (float(value) / scale for value in row[:4])
            landmarks = (row[4:14].astype(np.float32).reshape(5, 2)) / scale
            detections.append(
                Detection(
                    bbox=np.array([x, y, x + w, y + h], dtype=np.float32),
                    score=float(row[14]),
                    kps=landmarks,
                )
            )
        return detections


def _build_detector(detector_id: str, det_size: tuple[int, int], det_thresh: float,
                    nms_thresh: float) -> BaseDetector:
    entry = KNOWN_DETECTORS.get(detector_id)
    if entry is None:
        raise FaceEngineUnavailable(f"Unknown detector '{detector_id}'.")
    providers = settings.provider_list
    library = entry["library"]
    if library == "insightface":
        return InsightFaceDetector(detector_id, entry, det_size, det_thresh, nms_thresh, providers)
    if library == "uniface":
        return UnifaceDetector(detector_id, entry, det_size, det_thresh, nms_thresh, providers)
    if library == "opencv":
        return OpenCVYuNetDetector(detector_id, entry, det_size, det_thresh, nms_thresh)
    raise FaceEngineUnavailable(f"Detector library '{library}' is not supported.")


def get_detector(detector_id: str | None = None, det_size: int | tuple[int, int] | None = None,
                 det_thresh: float = 0.35, nms_thresh: float = 0.45) -> BaseDetector:
    """Load (and cache) a detector for the given resolution and thresholds."""
    detector_id = detector_id or settings.default_detector
    if isinstance(det_size, int):
        size = (det_size, det_size)
    elif det_size is None:
        size = (settings.face_scan_det_size, settings.face_scan_det_size)
    else:
        size = (int(det_size[0]), int(det_size[1]))
    key = (detector_id, size, round(float(det_thresh), 4), round(float(nms_thresh), 4))
    with _ENGINE_LOCK:
        detector = _DETECTORS.get(key)
        if detector is None:
            detector = _build_detector(detector_id, size, det_thresh, nms_thresh)
            _DETECTORS[key] = detector
            while len(_DETECTORS) > _MAX_DETECTORS:
                _DETECTORS.popitem(last=False)
        else:
            _DETECTORS.move_to_end(key)
        return detector


# --------------------------------------------------------------------------- #
# Recognizers
# --------------------------------------------------------------------------- #
class BaseRecognizer:
    id: str = ""
    label: str = ""
    embedding_dim: int = 512

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def warm_up(self) -> None:
        self.embed([np.zeros((112, 112, 3), dtype=np.uint8)])


class InsightFaceRecognizer(BaseRecognizer):
    """ArcFaceONNX recognizer used for the InsightFace / custom-ONNX families."""

    def __init__(self, recognizer_id: str, entry: dict, providers: list[str], path: Path):
        from insightface.model_zoo.arcface_onnx import ArcFaceONNX

        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=providers)
        self.model = ArcFaceONNX(model_file=str(path), session=session)
        self.id = recognizer_id
        self.label = entry.get("label", recognizer_id)
        self.embedding_dim = int(entry.get("embedding_dim", 512))

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        features = self.model.get_feat(list(crops))
        return np.asarray(features, dtype=np.float32)


class UnifaceRecognizer(BaseRecognizer):
    _CLASS_HINTS = (
        ("adaface", "AdaFace"),
        ("edgeface", "EdgeFace"),
        ("sphere", "SphereFace"),
        ("arcface", "ArcFace"),
        ("mobileface", "MobileFace"),
    )

    def __init__(self, recognizer_id: str, entry: dict, providers: list[str]):
        module = __import__("uniface.recognition", fromlist=["*"])
        weights_value = entry["uniface_weights"]
        class_name = next(
            (name for hint, name in self._CLASS_HINTS if hint in weights_value), None
        )
        if class_name is None or not hasattr(module, class_name):
            raise FaceEngineUnavailable(f"UniFace has no recognizer class for {recognizer_id}.")
        cls = getattr(module, class_name)
        self.model = cls(model_name=_uniface_enum(weights_value), providers=providers)
        self.id = recognizer_id
        self.label = entry.get("label", recognizer_id)
        self.embedding_dim = int(entry.get("embedding_dim", 512))

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        vectors = []
        for crop in crops:
            vector = np.asarray(self.model.get_embedding(crop, None), dtype=np.float32).reshape(-1)
            vectors.append(vector)
        if not vectors:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.vstack(vectors)


def _build_recognizer(recognizer_id: str) -> BaseRecognizer:
    entry = KNOWN_RECOGNIZERS.get(recognizer_id)
    if entry is None:
        raise FaceEngineUnavailable(f"Unknown recognizer '{recognizer_id}'.")
    providers = settings.provider_list
    if entry["library"] == "uniface":
        return UnifaceRecognizer(recognizer_id, entry, providers)
    path = resolve_model_file(entry)
    if path is None:
        raise FaceEngineUnavailable(
            f"Weights for recognizer '{recognizer_id}' were not found. "
            f"Looked in {settings.models_dir}. Run scripts/fetch_models.py."
        )
    return InsightFaceRecognizer(recognizer_id, entry, providers, path)


def get_recognizer(recognizer_id: str | None = None) -> BaseRecognizer:
    recognizer_id = recognizer_id or settings.default_recognizer
    key = (recognizer_id, tuple(settings.provider_list))
    with _ENGINE_LOCK:
        recognizer = _RECOGNIZERS.get(key)
        if recognizer is None:
            recognizer = _build_recognizer(recognizer_id)
            _RECOGNIZERS[key] = recognizer
        return recognizer


# --------------------------------------------------------------------------- #
# Auxiliary predictors (quality / head pose)
# --------------------------------------------------------------------------- #
def get_quality_predictor():
    predictor_id = settings.quality_predictor
    if not predictor_id:
        return None
    with _ENGINE_LOCK:
        if predictor_id in _QUALITY:
            return _QUALITY[predictor_id]
        value = predictor_id.split(":", 1)[-1]
        if _uniface_weight_cached(value) is None:
            # Weights aren't cached; the norm-based fallback keeps working and we
            # avoid blocking a request on a lazy multi-hundred-MB download.
            _QUALITY[predictor_id] = None
            return None
        try:
            module = __import__("uniface.quality", fromlist=["*"])
            model = module.EDifFIQA(model_name=_uniface_enum(value), providers=settings.provider_list)
        except Exception as exc:
            _QUALITY[predictor_id] = None
            print(f"[facelab] quality predictor unavailable ({exc})")
            return None
        _QUALITY[predictor_id] = model
        return model


def get_head_pose_predictor():
    predictor_id = settings.head_pose_predictor
    if not predictor_id:
        return None
    with _ENGINE_LOCK:
        if predictor_id in _POSE:
            return _POSE[predictor_id]
        value = predictor_id.split(":", 1)[-1]
        if value in ("6d", "headpose_6d"):
            value = "headpose_mobilenetv2"
        if _uniface_weight_cached(value) is None:
            _POSE[predictor_id] = None
            return None
        try:
            module = __import__("uniface.headpose", fromlist=["*"])
            model = module.HeadPose(model_name=_uniface_enum(value), providers=settings.provider_list)
        except Exception as exc:
            _POSE[predictor_id] = None
            print(f"[facelab] head-pose predictor unavailable ({exc})")
            return None
        _POSE[predictor_id] = model
        return model


def _quality_score(model, crop: np.ndarray) -> tuple[float | None, str]:
    if model is not None:
        try:
            result = model.score_aligned(crop)
            return float(result.score), "ediffiqa"
        except Exception:
            pass
    # No model-independent fallback. The pre-L2 embedding norm depends on the
    # recognizer (AdaFace/SphereFace return ~1.0 for a perfectly good crop where
    # ArcFace returns ~20), so the old norm-based score reported 0.0 for valid
    # faces and silently rejected them whenever ``min_quality`` was set.
    return None, "unavailable"


def _head_pose(model, crop: np.ndarray) -> dict | None:
    if model is None:
        return None
    try:
        result = model.estimate(crop)
        return {
            "pitch": round(float(result.pitch), 2),
            "yaw": round(float(result.yaw), 2),
            "roll": round(float(result.roll), 2),
        }
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Slicing-Aided Hyper Inference
# --------------------------------------------------------------------------- #
def generate_tiles(image_height: int, image_width: int, tile_size: int = 640,
                   overlap: float = 0.25, upper_band_only: bool = False) -> list[tuple[int, int, int, int]]:
    y_max_limit = int(image_height * 0.70) if upper_band_only else image_height
    stride = max(1, int(tile_size * (1.0 - overlap)))
    tiles = []

    y_starts = list(range(0, max(1, y_max_limit - tile_size + stride), stride))
    x_starts = list(range(0, max(1, image_width - tile_size + stride), stride))

    for y0 in y_starts:
        y1 = min(y0 + tile_size, image_height)
        y0_adj = max(0, y1 - tile_size)
        for x0 in x_starts:
            x1 = min(x0 + tile_size, image_width)
            x0_adj = max(0, x1 - tile_size)
            tiles.append((x0_adj, y0_adj, x1, y1))

    return list(dict.fromkeys(tiles))


def nms_boxes_and_landmarks(boxes: np.ndarray, scores: np.ndarray, landmarks: np.ndarray,
                            iou_threshold: float = 0.45) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0, 5, 2))

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return boxes[keep], scores[keep], landmarks[keep]


def _touches_internal_tile_edge(
    box: np.ndarray,
    tile: tuple[int, int, int, int],
    shape: tuple[int, int],
    margin: float,
) -> bool:
    """True when a detection is clipped by a tile border that is not an image border."""
    x0, y0, x1, y1 = tile
    image_height, image_width = shape
    bx0, by0, bx2, by2 = (float(value) for value in box[:4])
    if x0 > 0 and bx0 <= x0 + margin:
        return True
    if y0 > 0 and by0 <= y0 + margin:
        return True
    if x1 < image_width and bx2 >= x1 - margin:
        return True
    if y1 < image_height and by2 >= y1 - margin:
        return True
    return False


def _merge_tiled_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    landmarks: np.ndarray,
    truncated: np.ndarray,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NMS that never lets a tile-clipped fragment beat a complete detection.

    Tile detections that touch an internal tile border are ranked after every
    complete detection, and a clipped box mostly contained by a complete one is
    dropped even when their IoU is below the NMS threshold (heavy clipping
    shrinks IoU, so score-only NMS used to keep the fragment and discard the
    full-frame box, feeding clipped landmarks into alignment).
    """
    order = sorted(
        range(len(scores)), key=lambda index: (bool(truncated[index]), -float(scores[index]))
    )
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    keep: list[int] = []
    for index in order:
        drop = False
        for kept in keep:
            intersection = _box_intersection(boxes[index], boxes[kept])
            union = areas[index] + areas[kept] - intersection
            if union > 0 and intersection / union > iou_threshold:
                drop = True
                break
            if (
                truncated[index]
                and not truncated[kept]
                and areas[index] > 0
                and intersection / areas[index] > 0.9
            ):
                drop = True
                break
        if not drop:
            keep.append(index)
    return boxes[keep], scores[keep], landmarks[keep]


def detect_with_sahi(image: np.ndarray, detect_fn, tile_size: int = 640, overlap: float = 0.25,
                     iou_threshold: float = 0.45, tiling_mode: str = "sahi_upper_band"
                     ) -> list[Detection]:
    """Detect on the full frame plus overlapping tiles, then merge with NMS.

    ``tiling_mode`` is ``none`` (full frame only), ``sahi_upper_band`` (tiles the
    back two-thirds of the room), or ``sahi_full``. Tile detections clipped by an
    internal tile border are ranked below complete detections during the merge.
    """
    if tiling_mode in ("none", None, ""):
        return detect_fn(image)

    height, width = image.shape[:2]
    upper_band_only = tiling_mode == "sahi_upper_band"
    margin = max(2.0, tile_size * 0.02)

    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_landmarks: list[np.ndarray] = []
    truncated: list[bool] = []

    base = detect_fn(image)
    for det in base:
        all_boxes.append(det.bbox[None, :])
        all_scores.append(np.array([det.score], dtype=np.float32))
        all_landmarks.append(
            (det.kps if det.kps is not None else np.zeros((5, 2), dtype=np.float32))[None, :, :]
        )
        truncated.append(False)

    for x0, y0, x1, y1 in generate_tiles(
        height, width, tile_size=tile_size, overlap=overlap, upper_band_only=upper_band_only
    ):
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        for det in detect_fn(crop):
            box = det.bbox.copy()
            box[[0, 2]] += x0
            box[[1, 3]] += y0
            kps = det.kps.copy() if det.kps is not None else np.zeros((5, 2), dtype=np.float32)
            kps[:, 0] += x0
            kps[:, 1] += y0
            all_boxes.append(box[None, :])
            all_scores.append(np.array([det.score], dtype=np.float32))
            all_landmarks.append(kps[None, :, :])
            truncated.append(
                _touches_internal_tile_edge(box, (x0, y0, x1, y1), (height, width), margin)
            )

    if not all_boxes:
        return []

    merged_boxes, merged_scores, merged_kps = _merge_tiled_detections(
        np.vstack(all_boxes),
        np.concatenate(all_scores),
        np.vstack(all_landmarks),
        np.asarray(truncated, dtype=bool),
        iou_threshold=iou_threshold,
    )
    return [
        Detection(bbox=merged_boxes[i], score=float(merged_scores[i]), kps=merged_kps[i])
        for i in range(len(merged_boxes))
    ]


# --------------------------------------------------------------------------- #
# Full-frame analysis
# --------------------------------------------------------------------------- #
def align_face(
    image_bgr: np.ndarray, kps: np.ndarray, size: int = 112, antialias: bool = True
) -> np.ndarray:
    """Warp a face to ``size``×``size`` with anti-aliasing for large faces.

    InsightFace's ``norm_crop`` point-samples the source with a plain bilinear
    warp. That is fine for back-row faces (little downscaling) but aliases
    high-frequency detail when a front-row face is reduced several times to
    112 px. Blur the source ROI with a Gaussian sized to the reduction factor
    before warping; the transform itself is unchanged.
    """
    from insightface.utils import face_align

    kps = np.asarray(kps, dtype=np.float32)
    matrix = face_align.estimate_norm(kps, image_size=size)
    scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
    if not antialias or scale >= 0.5:
        return cv2.warpAffine(image_bgr, matrix, (size, size), borderValue=0.0)

    sigma = max(0.5, 0.5 * (1.0 / max(scale, 1e-6) - 1.0))
    inverse = cv2.invertAffineTransform(matrix)
    corners = np.array(
        [[[0.0, 0.0], [size, 0.0], [0.0, size], [size, size]]], dtype=np.float32
    )
    source = cv2.transform(corners, inverse).reshape(-1, 2)
    pad = int(np.ceil(3 * sigma)) + 1
    height, width = image_bgr.shape[:2]
    x0 = max(0, int(np.floor(source[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(source[:, 1].min())) - pad)
    x1 = min(width, int(np.ceil(source[:, 0].max())) + pad)
    y1 = min(height, int(np.ceil(source[:, 1].max())) + pad)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return cv2.warpAffine(image_bgr, matrix, (size, size), borderValue=0.0)

    roi = cv2.GaussianBlur(image_bgr[y0:y1, x0:x1], (0, 0), sigma)
    local = matrix.copy()
    # M maps global coordinates; the ROI is offset by (x0, y0).
    local[0, 2] += local[0, 0] * x0 + local[0, 1] * y0
    local[1, 2] += local[1, 0] * x0 + local[1, 1] * y0
    return cv2.warpAffine(roi, local, (size, size), borderValue=0.0)


def _project_landmarks_to_crop(kps: np.ndarray, size: int = 112) -> np.ndarray | None:
    """Map 5-point landmarks from the source image into ``size``×``size`` crop space.

    Mirrors the affine transform ``face_align.norm_crop`` applies, so the UI can
    overlay the landmarks on the saved aligned crop for alignment diagnosis.
    """
    try:
        from insightface.utils import face_align

        matrix = face_align.estimate_norm(np.asarray(kps, dtype=np.float32), image_size=size)
        homogenous = np.hstack(
            [np.asarray(kps, dtype=np.float32), np.ones((len(kps), 1), dtype=np.float32)]
        )
        return (homogenous @ matrix.T).astype(np.float32)
    except Exception:
        return None


def lowres_augment(crop: np.ndarray, tiny: int = 24, size: int = 112) -> np.ndarray:
    """Simulate a back-row face: downsample to ``tiny`` then bicubic upscale."""
    down = cv2.resize(crop, (tiny, tiny), interpolation=cv2.INTER_AREA)
    return cv2.resize(down, (size, size), interpolation=cv2.INTER_CUBIC)


def _embed_crop(recognizer: "BaseRecognizer", crop: np.ndarray) -> np.ndarray:
    return np.asarray(recognizer.embed([crop]), dtype=np.float32)[0]


def embed_variants(
    recognizer: "BaseRecognizer",
    crop: np.ndarray,
    *,
    flip_aug: bool = True,
    lowres_aug: bool = True,
) -> dict[str, np.ndarray | None]:
    """Embed one aligned crop into every stored variant with one definition.

    ``raw`` is the clean embedding, ``flip_avg`` averages it with its mirror, and
    ``lowres_aug`` is the flip-averaged *degraded* embedding used as a low-res
    gallery template. Classroom probes are never degraded
    (:func:`facelab.benchmark.select_embedding` keeps using the clean vector), so
    this helper exists to make the gallery-side variant unambiguous: enrolment
    and any future caller now produce the same thing under the same name instead
    of a clean/degraded blend versus a pure degraded vector.
    """
    raw = _embed_crop(recognizer, crop)
    flip_avg = None
    if flip_aug:
        flip_avg = matching.flip_averaged(raw, _embed_crop(recognizer, cv2.flip(crop, 1)))
    lowres = None
    if lowres_aug:
        degraded = lowres_augment(crop)
        lowres = _embed_crop(recognizer, degraded)
        if flip_aug:
            lowres = matching.flip_averaged(
                lowres, _embed_crop(recognizer, lowres_augment(cv2.flip(crop, 1)))
            )
    return {"raw": raw, "flip_avg": flip_avg, "lowres_aug": lowres}


def embed_reference(
    image_bgr: np.ndarray,
    *,
    detector_id: str,
    recognizer_ids: list[str],
    det_size: int | None = None,
    det_thresh: float = 0.5,
    nms_thresh: float = 0.45,
    flip_aug: bool = True,
    lowres_aug: bool = True,
    with_quality: bool = True,
    with_pose: bool = True,
) -> dict:
    """Embed one reference photo with *every* recognizer from a single detection.

    Enrolment is recognizer-agnostic, so we detect and align the face once and
    then run each recognizer over the same 112×112 crop. That keeps enrolment
    cheap as the recognizer collection grows (adding a model costs one embed,
    not another full detection pass).

    Returns a cache-shaped record with an ``embeddings`` mapping keyed by
    recognizer id. Raises :class:`ValueError` when no face is found; when the
    photo also contains background faces, the dominant (largest) one is used.
    """
    detector_id = detector_id or settings.default_detector
    det_size = det_size or settings.face_det_size
    recognizers = [get_recognizer(rid) for rid in recognizer_ids]
    if not recognizers:
        raise ValueError("No recognizers requested for enrolment.")

    detector = get_detector(detector_id, det_size, det_thresh, nms_thresh)
    detections = detect_with_sahi(
        image_bgr, lambda img: detector.detect(img), tiling_mode="none"
    )
    candidates = [det for det in detections if det.kps is not None]
    if not candidates:
        raise ValueError("No face detected. Ask the student to face the camera in good light.")

    # Reference photos are usually portraits, but casual shots often include
    # background faces. Enrol the dominant (largest) face rather than rejecting
    # the upload; break ties on detector score.
    det = select_dominant_face(candidates)
    crop = align_face(image_bgr, det.kps.astype(np.float32), 112)
    bbox = det.bbox.astype(np.float32)
    height_px = float(bbox[3] - bbox[1])

    embeddings: dict[str, dict] = {}
    for recognizer in recognizers:
        variants = embed_variants(
            recognizer, crop, flip_aug=flip_aug, lowres_aug=lowres_aug
        )
        embeddings[recognizer.id] = {
            **variants,
            "norm_magnitude": round(float(np.linalg.norm(variants["raw"])), 4),
        }

    primary = recognizers[0]
    quality_score: float | None = None
    quality_source = "unavailable"
    head_pose = None
    if with_quality:
        score, source = _quality_score(get_quality_predictor(), crop)
        quality_score = None if score is None else round(score, 4)
        quality_source = source
    if with_pose:
        head_pose = _head_pose(get_head_pose_predictor(), crop)

    return {
        "face_height_px": round(height_px, 2),
        "quality_score": quality_score,
        "quality_source": quality_source,
        "head_pose": head_pose,
        "embeddings": embeddings,
    }


def analyze_frame(
    image_bgr: np.ndarray,
    *,
    detector_id: str | None = None,
    recognizer_id: str | None = None,
    det_size: int | None = None,
    det_thresh: float = 0.35,
    nms_thresh: float = 0.45,
    tiling_mode: str = "none",
    tile_size: int = 640,
    tile_overlap: float = 0.25,
    nms_iou: float = 0.45,
    flip_aug: bool = True,
    lowres_aug: bool = False,
    with_quality: bool = True,
    with_pose: bool = True,
    crops_dir: Path | None = None,
    crop_prefix: str = "",
) -> dict:
    """Detect, align, embed, and score every face in one frame.

    Returns a cache-shaped payload. Embeddings are raw (un-normalized) vectors.
    ``lowres_aug`` only affects enrolment templates: classroom probes are always
    embedded from the clean aligned crop (see :func:`embed_variants`).
    """
    detector_id = detector_id or settings.default_detector
    recognizer_id = recognizer_id or settings.default_recognizer
    det_size = det_size or settings.face_scan_det_size

    detector = get_detector(detector_id, det_size, det_thresh, nms_thresh)
    recognizer = get_recognizer(recognizer_id)

    timings: dict[str, float] = {}

    started = time.perf_counter()
    detection_list = detect_with_sahi(
        image_bgr,
        lambda img: detector.detect(img),
        tile_size=tile_size,
        overlap=tile_overlap,
        iou_threshold=nms_iou,
        tiling_mode=tiling_mode,
    )
    timings["detect_ms"] = round((time.perf_counter() - started) * 1000, 3)

    started = time.perf_counter()
    crops: list[np.ndarray] = []
    payloads: list[dict] = []
    for det_index, det in enumerate(detection_list):
        bbox = det.bbox.astype(np.float32)
        height_px = float(bbox[3] - bbox[1])
        crop = None
        crop_rel = None
        landmarks_crop = None
        if det.kps is not None:
            kps = det.kps.astype(np.float32)
            crop = align_face(image_bgr, kps, 112)
            landmarks_crop = _project_landmarks_to_crop(kps, 112)
        if crop is not None:
            crops.append(crop)
            if crops_dir is not None:
                name = f"{crop_prefix or 'face'}_det{det_index}.jpg"
                target = crops_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(target), crop)
                crop_rel = str(target.relative_to(settings.data_dir))
        payloads.append(
            {
                "det_index": det_index,
                "crop_index": (len(crops) - 1) if crop is not None else None,
                "bbox": [round(float(v), 2) for v in bbox],
                "face_height_px": round(height_px, 2),
                "size_bucket": matching.size_bucket(height_px),
                "det_score": round(float(det.score), 4),
                "landmarks_5pt": det.kps.tolist() if det.kps is not None else None,
                "landmarks_crop": (
                    landmarks_crop.tolist() if landmarks_crop is not None else None
                ),
                "aligned_crop_path": crop_rel,
                "embedding": None,
                "embedding_flip_avg": None,
                "embedding_lowres_aug": None,
                "embedding_norm_mag": 0.0,
                "quality_score": None,
                "quality_source": "unavailable",
                "head_pose": None,
                "crop": crop,
            }
        )

    if crops:
        raw = recognizer.embed(crops)
        raw = np.asarray(raw, dtype=np.float32)
        indexed = [
            (payload, payload["crop_index"])
            for payload in payloads
            if payload["crop_index"] is not None
        ]
        for payload, crop_index in indexed:
            vector = raw[crop_index]
            payload["embedding"] = vector
            payload["embedding_norm_mag"] = round(float(np.linalg.norm(vector)), 4)
        if flip_aug:
            flipped = [cv2.flip(crop, 1) for crop in crops]
            mirror = np.asarray(recognizer.embed(flipped), dtype=np.float32)
            for payload, crop_index in indexed:
                payload["embedding_flip_avg"] = matching.flip_averaged(
                    raw[crop_index], mirror[crop_index]
                )
        # lowres_aug is a *gallery-side* augmentation: classroom probes are
        # already low-resolution and select_embedding() scores them with the
        # clean vector, so embedding a degraded probe here cost a full extra
        # recognition pass per frame and stored a vector that was never read.
        # The field stays in the schema (as None) for cache/UI compatibility.
    timings["align_embed_ms"] = round((time.perf_counter() - started) * 1000, 3)

    started = time.perf_counter()
    if with_quality or with_pose:
        quality_model = get_quality_predictor() if with_quality else None
        pose_model = get_head_pose_predictor() if with_pose else None
        for payload in payloads:
            crop = payload.get("crop")
            if crop is None:
                continue
            if with_quality:
                score, source = _quality_score(quality_model, crop)
                payload["quality_score"] = None if score is None else round(score, 4)
                payload["quality_source"] = source
            if with_pose:
                payload["head_pose"] = _head_pose(pose_model, crop)
    timings["quality_pose_ms"] = round((time.perf_counter() - started) * 1000, 3)

    for payload in payloads:
        payload.pop("crop", None)
        payload.pop("crop_index", None)

    return {
        "detector_id": detector_id,
        "recognizer_id": recognizer_id,
        "det_params": {
            "det_size": [int(det_size), int(det_size)],
            "det_thresh": float(det_thresh),
            "nms_thresh": float(nms_thresh),
            "tiling_mode": tiling_mode,
            "tile_size": int(tile_size),
            "tile_overlap": float(tile_overlap),
        },
        "timings_ms": timings,
        "detections": payloads,
    }


def warm_up(detector_id: str | None = None, recognizer_id: str | None = None,
            det_size: int | None = None, tiling_mode: str = "none",
            det_thresh: float = 0.35, nms_thresh: float = 0.45) -> dict:
    """One throwaway inference pass so benchmark timings exclude lazy loading."""
    started = time.perf_counter()
    image = np.full((480, 640, 3), 127, dtype=np.uint8)
    try:
        analyze_frame(
            image,
            detector_id=detector_id,
            recognizer_id=recognizer_id,
            det_size=det_size,
            det_thresh=det_thresh,
            nms_thresh=nms_thresh,
            tiling_mode=tiling_mode,
            with_quality=False,
            with_pose=False,
        )
    except FaceEngineUnavailable:
        raise
    except Exception:
        pass
    return {"warmup_ms": round((time.perf_counter() - started) * 1000, 3)}


# --------------------------------------------------------------------------- #
# Legacy InsightFace pack API (kept for older callers and model management)
# --------------------------------------------------------------------------- #
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


def get_face_analyzer(model_name: str | None = None):
    """Legacy monolithic InsightFace analyzer (one per pack, process lifetime)."""
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
                providers=settings.provider_list,
                allowed_modules=settings.allowed_module_list or None,
            )
            analyzer.prepare(
                ctx_id=0, det_size=(settings.face_det_size, settings.face_det_size)
            )
        except Exception as exc:
            raise FaceEngineUnavailable(f"Model '{name}' could not be loaded: {exc}") from exc
        _ANALYZERS[name] = analyzer
        return analyzer


# Annotation colours by decision status (present / needs_review / unmatched).
STATUS_COLORS = {
    "present": (0, 200, 0),
    "needs_review": (0, 165, 255),
    "unmatched": (0, 0, 220),
    "unknown": (0, 0, 220),
}


def annotate_classroom_photo(
    image_bgr: np.ndarray, matches: list[dict], *, label_unknown: bool = False
) -> np.ndarray:
    """Draw a box per face: green present, amber review, red unmatched.

    Unmatched faces get a red box only by default (``label_unknown=False``); turn
    it on to also burn an "Unknown" tag. Matched faces always carry a name.
    """
    annotated = image_bgr.copy()
    height, width = annotated.shape[:2]
    diagonal = np.sqrt(height**2 + width**2)
    font_scale = max(0.45, min(1.2, diagonal / 2500.0))
    thickness = max(1, int(round(diagonal / 1200.0)))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for match in matches:
        face = match.get("face")
        bbox = getattr(face, "bbox", None) if face is not None else match.get("bbox")
        if bbox is None:
            continue
        x1, y1, x2, y2 = np.asarray(bbox).astype(int)

        status = match.get("status")
        if status is None:
            status = "present" if match.get("matched") else "unknown"
        color = STATUS_COLORS.get(status, STATUS_COLORS["unmatched"])

        if status == "present":
            label = f"{match.get('student_name', '?')} ({float(match.get('confidence') or 0):.2f})"
        elif status == "needs_review":
            label = f"? {match.get('student_name') or ''} ({float(match.get('confidence') or 0):.2f})"
        else:
            label = "Unknown" if label_unknown else None

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        if label is None:
            continue
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
