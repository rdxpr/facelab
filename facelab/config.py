"""Configuration for the standalone FaceLab tool.

All settings can be overridden with ``FACELAB_``-prefixed environment variables,
for example ``FACELAB_ENV=prod`` on a server.

Storage modes:

* ``FACELAB_ENV=dev`` (the default) keeps the roster, reference photos, frames,
  and results in ``./.data`` so they survive restarts.
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
