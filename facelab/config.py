"""Configuration and the detector/recognizer registry for FaceLab.

All settings can be overridden with ``FACELAB_``-prefixed environment variables,
for example ``FACELAB_ENV=prod`` on a server.

Storage modes:

* ``FACELAB_ENV=dev`` (the default) keeps the roster, reference photos, frames,
  datasets, caches, and results in ``./.data`` so they survive restarts.
* Any other ``env`` value stores them in an ephemeral directory that is wiped on
  every start, so uploaded photos and generated results are never persisted.
  Set ``FACELAB_EPHEMERAL_DATA=false`` to opt back into persistence.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
# Detectors and recognizers are selected independently. ``pack`` links a weight
# file to the InsightFace model-zoo archive that ships it; ``uniface_weights``
# names the UniFace weights enum member. ``file_candidates`` are searched in
# order inside the models directory.
#
# Every entry carries a license so the UI and benchmark table can show which
# models are safe for commercial deployment.
KNOWN_DETECTORS: dict[str, dict] = {
    "insightface:scrfd_10g_bnkps": {
        "id": "insightface:scrfd_10g_bnkps",
        "label": "SCRFD-10G (5pt)",
        "library": "insightface",
        "kind": "detector",
        "pack": "buffalo_l",
        "file_candidates": ["det_10g.onnx", "scrfd_10g_bnkps.onnx"],
        "landmarks_5pt": True,
        "wider_hard": 83.05,
        "license": "Non-Commercial",
        "role": "Balanced classroom baseline",
    },
    "insightface:scrfd_2.5g_bnkps": {
        "id": "insightface:scrfd_2.5g_bnkps",
        "label": "SCRFD-2.5G (5pt)",
        "library": "insightface",
        "kind": "detector",
        "pack": "buffalo_m",
        "file_candidates": ["det_2.5g.onnx", "det_2.5g_bnkps.onnx"],
        "landmarks_5pt": True,
        "wider_hard": 77.13,
        "license": "Non-Commercial",
        "role": "Fast mobile / CPU baseline",
    },
    "insightface:scrfd_34g": {
        "id": "insightface:scrfd_34g",
        "label": "SCRFD-34G (no kps)",
        "library": "insightface",
        "kind": "detector",
        "file_candidates": ["det_34g.onnx", "scrfd_34g.onnx"],
        "landmarks_5pt": False,
        "wider_hard": 87.0,
        "license": "Non-Commercial",
        "role": "High-recall ablation; drop scrfd_34g.onnx in models/ (needs a landmark predictor)",
    },
    "uniface:retinaface_r50": {
        "id": "uniface:retinaface_r50",
        "label": "RetinaFace R50",
        "library": "uniface",
        "kind": "detector",
        "uniface_weights": "retinaface_r50",
        "landmarks_5pt": True,
        "wider_hard": 84.8,
        "license": "MIT wrapper / Non-Commercial weights",
        "role": "Strong small-face recall in back rows",
    },
    "uniface:yolov8n_face": {
        "id": "uniface:yolov8n_face",
        "label": "YOLOv8n-Face",
        "library": "uniface",
        "kind": "detector",
        "uniface_weights": "yolov8n_face",
        "landmarks_5pt": True,
        "wider_hard": 79.6,
        "license": "AGPL-3.0",
        "role": "Steep pitch / desk-occlusion robustness",
    },
    "opencv:yunet_2023mar": {
        "id": "opencv:yunet_2023mar",
        "label": "YuNet (2023mar)",
        "library": "opencv",
        "kind": "detector",
        "file_candidates": [
            "opencv/face_detection_yunet_2023mar.onnx",
            "face_detection_yunet_2023mar.onnx",
            "yunet_2023mar.onnx",
        ],
        "landmarks_5pt": True,
        "wider_hard": 75.03,
        "license": "Apache-2.0",
        "role": "Commercial-safe lightweight baseline",
    },
}

KNOWN_RECOGNIZERS: dict[str, dict] = {
    "insightface:w600k_r50": {
        "id": "insightface:w600k_r50",
        "label": "ArcFace R50 / WebFace600K",
        "library": "insightface",
        "kind": "recognizer",
        "pack": "buffalo_l",
        "file_candidates": ["w600k_r50.onnx"],
        "license": "Non-Commercial",
        "embedding_dim": 512,
        "role": "Current baseline recognizer",
    },
    "insightface:glintr100": {
        "id": "insightface:glintr100",
        "label": "ArcFace R100 / Glint360K",
        "library": "insightface",
        "kind": "recognizer",
        "pack": "antelopev2",
        "file_candidates": ["glintr100.onnx"],
        "license": "Non-Commercial",
        "embedding_dim": 512,
        "role": "High-capacity ArcFace upgrade",
    },
    "insightface:w600k_mbf": {
        "id": "insightface:w600k_mbf",
        "label": "MobileFaceNet / WebFace600K",
        "library": "insightface",
        "kind": "recognizer",
        "pack": "buffalo_s",
        "file_candidates": ["w600k_mbf.onnx"],
        "license": "Non-Commercial",
        "embedding_dim": 512,
        "role": "Speed-floor benchmark",
    },
    "insightface:raccoon_l": {
        "id": "insightface:raccoon_l",
        "label": "Raccoon-L",
        "library": "insightface",
        "kind": "recognizer",
        "pack": "raccoon_l",
        "file_candidates": ["raccoon_l.onnx"],
        "license": "Non-Commercial",
        "embedding_dim": 512,
        "role": "Latest InsightFace 2.x benchmark pack",
    },
    "uniface:adaface_ir101": {
        "id": "uniface:adaface_ir101",
        "label": "AdaFace IR-101 / WebFace12M",
        "library": "uniface",
        "kind": "recognizer",
        "uniface_weights": "adaface_ir_101",
        "license": "MIT wrapper / Non-Commercial weights",
        "embedding_dim": 512,
        "role": "Primary SOTA for tiny back-row faces",
    },
    "uniface:edgeface_s_gamma_05": {
        "id": "uniface:edgeface_s_gamma_05",
        "label": "EdgeFace-S gamma 0.5",
        "library": "uniface",
        "kind": "recognizer",
        "uniface_weights": "edgeface_s_gamma_05",
        "license": "MIT wrapper / Non-Commercial weights",
        "embedding_dim": 512,
        "role": "Edge / mobile SOTA (1.77M params)",
    },
    "uniface:sphereface": {
        "id": "uniface:sphereface",
        "label": "SphereFace-20",
        "library": "uniface",
        "kind": "recognizer",
        "uniface_weights": "sphere20",
        "license": "MIT wrapper / Non-Commercial weights",
        "embedding_dim": 512,
        "role": "Angular-margin ablation",
    },
    "custom_onnx:facelivtv2_s": {
        "id": "custom_onnx:facelivtv2_s",
        "label": "FaceLiVTv2-Small",
        "library": "custom_onnx",
        "kind": "recognizer",
        "file_candidates": [
            "custom/facelivtv2_s.onnx",
            "facelivtv2_s.onnx",
            "facelivtv2/facelivtv2_s.onnx",
        ],
        "license": "MIT / Research",
        "embedding_dim": 512,
        "role": "Ultra-fast linear ViT mobile benchmark",
    },
}

# InsightFace model-zoo archives and the files FaceLab needs from each. The
# fetch script uses these definitions; extra files in an archive are harmless.
KNOWN_PACKS: dict[str, dict] = {
    "buffalo_l": {
        "required": ["det_10g.onnx", "w600k_r50.onnx"],
        "license": "Non-Commercial",
    },
    "buffalo_m": {
        "required": ["det_2.5g.onnx", "w600k_r50.onnx"],
        "license": "Non-Commercial",
    },
    "buffalo_s": {
        "required": ["det_500m.onnx", "w600k_mbf.onnx"],
        "license": "Non-Commercial",
    },
    "buffalo_sc": {
        "required": ["det_500m.onnx", "w600k_mbf.onnx"],
        "license": "Non-Commercial",
    },
    "antelopev2": {
        "required": ["scrfd_10g_bnkps.onnx", "glintr100.onnx"],
        "license": "Non-Commercial",
    },
    "raccoon_l": {
        "required": ["raccoon_l.onnx"],
        "license": "Non-Commercial",
    },
}


def _registry(kind: str) -> dict[str, dict]:
    source = KNOWN_DETECTORS if kind == "detector" else KNOWN_RECOGNIZERS
    return {key: dict(value) for key, value in source.items()}


def detector_registry() -> dict[str, dict]:
    return _registry("detector")


def recognizer_registry() -> dict[str, dict]:
    return _registry("recognizer")


def pack_for_detector(detector_id: str) -> str | None:
    return KNOWN_DETECTORS.get(detector_id, {}).get("pack")


def pack_for_recognizer(recognizer_id: str) -> str | None:
    return KNOWN_RECOGNIZERS.get(recognizer_id, {}).get("pack")


def default_detector_for_pack(pack: str) -> str:
    """Map a legacy InsightFace pack name to its bundled detector."""
    for detector_id, entry in KNOWN_DETECTORS.items():
        if entry.get("pack") == pack and entry.get("library") == "insightface":
            return detector_id
    return DEFAULT_DETECTOR


def default_recognizer_for_pack(pack: str) -> str:
    """Map a legacy InsightFace pack name to its bundled recognizer."""
    for recognizer_id, entry in KNOWN_RECOGNIZERS.items():
        if entry.get("pack") == pack and entry.get("library") == "insightface":
            return recognizer_id
    return DEFAULT_RECOGNIZER


# Kept for compatibility with the pre-registry API where "model" meant pack.
DEFAULT_DETECTOR = "insightface:scrfd_10g_bnkps"
DEFAULT_RECOGNIZER = "insightface:w600k_r50"
BUILTIN_PACK_MODELS = {"insightface:w600k_r50": "buffalo_l"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        env_prefix="FACELAB_",
        extra="ignore",
    )

    app_name: str = "FaceLab"
    env: str = "dev"

    # Where model packs (``models/<name>/*.onnx``) and runtime data live.
    models_dir: Path = BASE_DIR / "models"
    data_dir: Path = BASE_DIR / ".data"

    # Ephemeral storage. ``None`` means "on whenever env != dev". When on,
    # ``data_dir`` is replaced by ``ephemeral_dir`` and that directory is wiped
    # at startup, so reference photos, frames, and results never persist.
    ephemeral_data: bool | None = None
    ephemeral_dir: Path = Path(tempfile.gettempdir()) / "facelab"

    # Legacy InsightFace pack selection. Retained so existing UI/API calls and
    # the fetch script keep working; the registry below is the new default.
    default_model: str = "buffalo_l"
    default_detector: str = DEFAULT_DETECTOR
    default_recognizer: str = DEFAULT_RECOGNIZER

    # Loading only detection + recognition keeps memory and startup time lower.
    # Set to an empty string to load every model in the pack.
    allowed_modules: str = "detection,recognition"

    # ONNX Runtime execution providers, comma separated. Empty means auto:
    # CUDA -> CoreML -> CPU (first one the installed onnxruntime supports).
    onnx_providers: str = ""

    # Enrollment uses a smaller detector; classroom scans can use a larger one.
    face_det_size: int = 640
    face_scan_det_size: int = 1280
    face_scan_max_frames: int = 12
    # Legacy single-threshold default. New callers use the calibrated pair below.
    face_similarity_threshold: float = 0.42
    accept_threshold: float = 0.42
    review_threshold: float = 0.34
    min_margin: float = 0.04

    # Defaults for scans and benchmark sweeps.
    assignment_mode: str = "hungarian"
    template_strategy: str = "top_k_mean"
    template_top_k: int = 2
    tiling_mode: str = "none"
    multi_frame_fusion: str = "k_of_n_voting"
    fusion_k: int = 2
    min_face_quality: float = 0.0

    # Optional auxiliary predictors (require the uniface package).
    quality_predictor: str = "uniface:ediffiqa_s"
    head_pose_predictor: str = "uniface:headpose_6d"

    # Benchmark grid defaults.
    benchmark_det_sizes: str = "640,1024,1280"
    benchmark_max_rows: int = 32
    benchmark_max_frames: int = 60

    max_photos_per_upload: int = 8
    max_photos_per_person: int = 12

    # Saved scan-result JSON files and frame images to retain.
    results_to_keep: int = 50
    scans_to_keep: int = 50
    datasets_to_keep: int = 25
    # Inference-cache JSON artifacts (and their aligned crop JPEGs) to retain.
    cache_entries_to_keep: int = 2000

    # Useful when the parent Attendyrc app calls this API from another origin.
    cors_origins: str = "http://127.0.0.1:8010,http://localhost:8010"

    # When set, every mutating endpoint requires ``Authorization: Bearer <token>``
    # or ``X-API-Token: <token>``.
    api_token: str = ""

    @model_validator(mode="after")
    def _resolve_paths(self) -> "Settings":
        for field in ("models_dir", "data_dir", "ephemeral_dir"):
            path = getattr(self, field)
            if not path.is_absolute():
                setattr(self, field, (BASE_DIR / path).resolve())
        if self.ephemeral:
            self.data_dir = self.ephemeral_dir
        return self

    @property
    def ephemeral(self) -> bool:
        """Whether uploads/results are discarded on every start."""
        if self.ephemeral_data is not None:
            return self.ephemeral_data
        return self.env.strip().lower() != "dev"

    @property
    def allowed_module_list(self) -> list[str]:
        return [item.strip() for item in self.allowed_modules.split(",") if item.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def benchmark_det_size_list(self) -> list[int]:
        values = []
        for item in self.benchmark_det_sizes.split(","):
            try:
                size = int(item.strip())
            except ValueError:
                continue
            if 320 <= size <= 2048:
                values.append(size)
        return values or [640, 1024, 1280]

    @property
    def provider_list(self) -> list[str]:
        """Resolve the ONNX Runtime execution-provider preference list.

        ``FACELAB_PROVIDERS`` wins when set. Otherwise prefer CUDA, then CoreML,
        then CPU, filtered to providers the installed onnxruntime can actually
        create.
        """
        configured = [
            item.strip() for item in self.onnx_providers.split(",") if item.strip()
        ]
        try:
            import onnxruntime as ort

            available = set(ort.get_available_providers())
        except Exception:  # onnxruntime is imported lazily by the engine
            available = {"CPUExecutionProvider"}
        if configured:
            chosen = [item for item in configured if item in available]
            return chosen or ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CoreMLExecutionProvider" in available:
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @property
    def faces_dir(self) -> Path:
        return self.data_dir / "faces"

    @property
    def scans_dir(self) -> Path:
        return self.data_dir / "scans"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def crops_dir(self) -> Path:
        return self.data_dir / "crops"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def benchmarks_dir(self) -> Path:
        return self.data_dir / "benchmarks"

    @property
    def students_file(self) -> Path:
        return self.data_dir / "students.json"

    @property
    def legacy_people_file(self) -> Path:
        return self.data_dir / "people.json"

    @property
    def people_file(self) -> Path:
        """Backwards-compatible alias used by older callers."""
        return self.students_file

    @property
    def models_manifest_file(self) -> Path:
        return self.data_dir / "models_manifest.json"

    def ensure_dirs(self) -> None:
        for folder in (
            self.data_dir,
            self.faces_dir,
            self.scans_dir,
            self.results_dir,
            self.crops_dir,
            self.cache_dir,
            self.datasets_dir,
            self.benchmarks_dir,
            self.models_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)

    def prepare_dirs(self) -> None:
        """Create storage directories, clearing data first in ephemeral mode."""
        if self.ephemeral:
            # Guarantee a clean slate on every process start, but never wipe an
            # obviously wrong target (a typo in FACELAB_EPHEMERAL_DIR).
            target = self.data_dir.resolve()
            unsafe = {
                Path(target.anchor),
                Path.home().resolve(),
                Path(tempfile.gettempdir()).resolve(),
            }
            if target in unsafe or target == BASE_DIR or BASE_DIR.is_relative_to(target):
                raise RuntimeError(
                    f"Refusing to clear ephemeral data dir {target}. "
                    "Set FACELAB_EPHEMERAL_DIR to a dedicated directory."
                )
            shutil.rmtree(target, ignore_errors=True)
        self.ensure_dirs()


settings = Settings()
settings.prepare_dirs()
