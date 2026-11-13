"""Configuration for the standalone FaceLab tool.

All settings can be overridden with ``FACELAB_``-prefixed environment variables,
for example ``FACELAB_DATA_DIR=/var/data`` on Render.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


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

    # The default InsightFace model pack. Any pack present in ``models_dir`` can
    # be selected from the UI or API.
    default_model: str = "buffalo_l"
    # Loading only detection + recognition keeps memory and startup time lower.
    # Set to an empty string to load every model in the pack.
    allowed_modules: str = "detection,recognition"

    # Enrollment uses a smaller detector; classroom scans can use a larger one.
    face_det_size: int = 640
    face_scan_det_size: int = 960
    face_scan_max_frames: int = 12
    face_similarity_threshold: float = 0.5

    max_photos_per_upload: int = 8
    max_photos_per_person: int = 12

    # Saved scan-result JSON files and frame images to retain.
    results_to_keep: int = 50
    scans_to_keep: int = 50

    # Useful when the parent Attendyrc app calls this API from another origin.
    cors_origins: str = "*"

    @model_validator(mode="after")
    def _resolve_paths(self) -> "Settings":
        for field in ("models_dir", "data_dir"):
            path = getattr(self, field)
            if not path.is_absolute():
                setattr(self, field, (BASE_DIR / path).resolve())
        return self

    @property
    def allowed_module_list(self) -> list[str]:
        return [item.strip() for item in self.allowed_modules.split(",") if item.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

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
    def people_file(self) -> Path:
        return self.data_dir / "people.json"

    def ensure_dirs(self) -> None:
        for folder in (
            self.data_dir,
            self.faces_dir,
            self.scans_dir,
            self.results_dir,
            self.models_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()