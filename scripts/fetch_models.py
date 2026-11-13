#!/usr/bin/env python3
"""Fetch model weights for FaceLab and record their SHA-256 hashes.

Two families are supported:

* **InsightFace model packs** (``buffalo_l``, ``buffalo_m``, ``buffalo_s``,
  ``antelopev2``, ``raccoon_l``) downloaded from the official model zoo and
  extracted into ``models/<pack>/``.
* **UniFace weights** (RetinaFace, YOLOv8-Face, AdaFace, EdgeFace, SphereFace,
  eDifFIQA, head pose) downloaded into the UniFace cache by the ``uniface``
  package itself.

Every ``.onnx`` file under ``models/`` is hashed and recorded in
``.data/models_manifest.json`` so benchmark runs can be tied to exact weights.

Examples:

    python scripts/fetch_models.py                       # default pack
    python scripts/fetch_models.py --pack buffalo_l --pack antelopev2
    python scripts/fetch_models.py --uniface adaface_ir_101 --uniface retinaface_r50
    python scripts/fetch_models.py --all                 # every pack + YuNet + FaceLab's UniFace weights
    python scripts/fetch_models.py --force               # re-download packs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://github.com/deepinsight/insightface/releases/download/model-zoo"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

sys.path.insert(0, str(REPO_ROOT))

from facelab.config import (  # noqa: E402
    KNOWN_PACKS,
    KNOWN_RECOGNIZERS,
    KNOWN_DETECTORS,
    Settings,
)

# Backwards-compatible alias: older docs call packs "models".
KNOWN_MODELS = {name: spec["required"] for name, spec in KNOWN_PACKS.items()}

UNIFACE_WEIGHTS: dict[str, str] = {}
for entry in list(KNOWN_DETECTORS.values()) + list(KNOWN_RECOGNIZERS.values()):
    value = entry.get("uniface_weights")
    if value:
        UNIFACE_WEIGHTS[value] = entry["id"]


def required_uniface_weights() -> list[str]:
    """UniFace weights the registry can actually use, plus the predictors.

    UniFace ships dozens of models (gaze, matting, parsing, ...) that FaceLab
    never loads, so ``--all`` targets this set instead of every enum member.
    """
    values = set(UNIFACE_WEIGHTS)
    try:
        settings = Settings()
        predictors = (settings.quality_predictor, settings.head_pose_predictor)
    except Exception:
        predictors = ("uniface:ediffiqa_s", "uniface:headpose_6d")
    for predictor_id in predictors:
        if not predictor_id or not predictor_id.startswith("uniface:"):
            continue
        value = predictor_id.split(":", 1)[-1]
        if value in ("6d", "headpose_6d"):
            value = "headpose_mobilenetv2"
        values.add(value)
    return sorted(values)


def default_models_dir() -> Path:
    env = os.environ.get("FACELAB_MODELS_DIR")
    if env:
        path = Path(env).expanduser()
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()
    return REPO_ROOT / "models"


def default_data_dir() -> Path:
    env = os.environ.get("FACELAB_DATA_DIR")
    if env:
        path = Path(env).expanduser()
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()
    return REPO_ROOT / ".data"


def needs_download(target: Path, required: list[str], force: bool) -> bool:
    if force:
        return True
    if not target.is_dir():
        return True
    if not list(target.glob("*.onnx")):
        return True
    if required and not all((target / name).exists() for name in required):
        return True
    return False


def download(url: str, destination: Path, retries: int = 3) -> None:
    last_error: Exception | None = None
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            print(f"[fetch_models] downloading {url} (attempt {attempt}/{retries})")
            with urllib.request.urlopen(url, timeout=180) as response:
                header = response.headers.get("Content-Length")
                expected = int(header) if header and header.isdigit() else None
                with partial.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            written = partial.stat().st_size
            if not written:
                raise OSError("empty response")
            if expected is not None and written != expected:
                raise OSError(f"truncated download ({written}/{expected} bytes)")
            os.replace(partial, destination)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            print(f"[fetch_models]   retry needed: {exc}", file=sys.stderr)
    partial.unlink(missing_ok=True)
    raise SystemExit(f"[fetch_models] could not download {url}: {last_error}")


def _extract_flat(zf: zipfile.ZipFile, target: Path) -> None:
    """Extract a model-zoo archive, flattening an optional single top folder."""
    names = zf.namelist()
    roots = {name.split("/", 1)[0] for name in names if "/" in name.strip("/")}
    prefix = ""
    if len(roots) == 1:
        candidate = roots.pop()
        if all(
            name == candidate or name.startswith(candidate + "/")
            for name in names
            if name.strip("/")
        ):
            prefix = candidate + "/"
    for member in zf.infolist():
        if member.is_dir():
            continue
        relative = member.filename
        if prefix and relative.startswith(prefix):
            relative = relative[len(prefix) :]
        relative = relative.lstrip("/")
        if not relative or ".." in Path(relative).parts:
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as source, destination.open("wb") as sink:
            shutil.copyfileobj(source, sink)


def fetch_pack(
    name: str, models_dir: Path, force: bool, base_url: str, attempts: int = 3
) -> bool:
    target = models_dir / name
    required = KNOWN_MODELS.get(name, [])
    if not needs_download(target, required, force):
        print(f"[fetch_models] {name}: already present at {target}")
        return False

    url = f"{base_url.rstrip('/')}/{name}.zip"
    models_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / f"{name}.zip"
            download(url, archive)
            try:
                if force and target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(archive) as zf:
                    _extract_flat(zf, target)
            except zipfile.BadZipFile as exc:
                last_error = exc
                shutil.rmtree(target, ignore_errors=True)
                print(
                    f"[fetch_models] {name}: archive unreadable "
                    f"(attempt {attempt}/{attempts}); re-downloading",
                    file=sys.stderr,
                )
                continue
        break
    else:
        shutil.rmtree(target, ignore_errors=True)
        raise SystemExit(
            f"[fetch_models] {name}: no valid archive after {attempts} attempts ({last_error})"
        )

    onnx_files = sorted(target.glob("*.onnx"))
    if not onnx_files:
        raise SystemExit(f"[fetch_models] {name}: archive had no .onnx files; check {url}")
    missing = [item for item in required if not (target / item).exists()]
    if missing:
        print(
            f"[fetch_models] {name}: warning, expected files missing: {', '.join(missing)}",
            file=sys.stderr,
        )
    size_mb = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) / 1e6
    print(f"[fetch_models] {name}: ready at {target} ({len(onnx_files)} models, {size_mb:.0f} MB)")
    return True


def fetch_yunet(models_dir: Path, force: bool) -> bool:
    """Download the OpenCV Zoo YuNet face detector (Apache-2.0, ~230 KB)."""
    target = models_dir / "opencv" / "face_detection_yunet_2023mar.onnx"
    if target.is_file() and not force:
        print(f"[fetch_models] yunet: already present at {target}")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    download(YUNET_URL, target)
    print(f"[fetch_models] yunet: ready at {target} ({target.stat().st_size / 1024:.0f} KB)")
    return True


def fetch_uniface(names: list[str], force: bool) -> bool:
    try:
        from uniface.model_store import download_models
        from uniface import constants
        import enum as _enum
    except Exception as exc:
        print(
            f"[fetch_models] uniface is not installed ({exc}); skipping UniFace weights. "
            'Install with `uv pip install "uniface[cpu]"`.',
            file=sys.stderr,
        )
        return False

    enums = []
    for name in names:
        if name == "all":
            for enum_cls in vars(constants).values():
                if isinstance(enum_cls, type) and issubclass(enum_cls, _enum.Enum):
                    enums.extend(list(enum_cls))
            continue
        for enum_cls in vars(constants).values():
            if not (isinstance(enum_cls, type) and issubclass(enum_cls, _enum.Enum)):
                continue
            # The bare `Enum` base and memberless classes raise TypeError when
            # called with a value, so skip them before probing membership.
            if enum_cls is _enum.Enum or not list(enum_cls):
                continue
            try:
                enums.append(enum_cls(name))
                break
            except ValueError:
                continue
        else:
            print(f"[fetch_models] unknown uniface weights '{name}'", file=sys.stderr)

    if not enums:
        return False
    seen = set()
    unique = []
    for item in enums:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    print(f"[fetch_models] fetching {len(unique)} UniFace weight files...")
    results = download_models(unique)
    for item, path in results.items():
        print(f"[fetch_models]   {item.value}: {path}")
    return bool(results)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(models_dir: Path, data_dir: Path) -> Path:
    license_by_file: dict[str, str] = {}
    for entry in list(KNOWN_DETECTORS.values()) + list(KNOWN_RECOGNIZERS.values()):
        for name in entry.get("file_candidates", []):
            license_by_file.setdefault(Path(name).name, entry.get("license", "unknown"))

    models = []
    for path in sorted(models_dir.rglob("*.onnx")):
        models.append(
            {
                "file": str(path.relative_to(models_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "license": license_by_file.get(path.name, "unknown"),
            }
        )
    manifest = {"models_dir": str(models_dir), "models": models}
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "models_manifest.json"
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[fetch_models] wrote manifest for {len(models)} files to {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch face models for FaceLab")
    parser.add_argument(
        "--model", "--pack", action="append", dest="packs",
        help="InsightFace pack to fetch (repeatable). Defaults to buffalo_l.",
    )
    parser.add_argument(
        "--uniface", action="append", dest="uniface", default=[],
        help="UniFace weights value to prefetch, or 'all' (repeatable).",
    )
    parser.add_argument(
        "--models-dir", default=None,
        help="Where to place packs (default: $FACELAB_MODELS_DIR or ./models).",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Where to write models_manifest.json (default: $FACELAB_DATA_DIR or ./.data).",
    )
    parser.add_argument("--base-url", default=BASE_URL, help="Model zoo release URL.")
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    parser.add_argument("--list", action="store_true", help="List known packs and exit.")
    parser.add_argument(
        "--all", action="store_true", dest="all_models",
        help="Fetch every known pack, YuNet, and the UniFace weights FaceLab uses.",
    )
    parser.add_argument(
        "--yunet", action="store_true",
        help="Fetch the OpenCV Zoo YuNet detector (Apache-2.0, ~230 KB).",
    )
    args = parser.parse_args()

    if args.list:
        for name, spec in sorted(KNOWN_PACKS.items()):
            print(f"  {name:12s} needs: {', '.join(spec['required'])}")
        print("  yunet        needs: face_detection_yunet_2023mar.onnx (--yunet)")
        print("  uniface weights:", ", ".join(required_uniface_weights()))
        return

    if args.all_models:
        packs = sorted(KNOWN_PACKS)
        uniface_names = required_uniface_weights()
    else:
        packs = args.packs or [os.environ.get("FACELAB_DEFAULT_MODEL") or "buffalo_l"]
        uniface_names = args.uniface

    models_dir = (
        Path(args.models_dir).expanduser().resolve()
        if args.models_dir
        else default_models_dir()
    )
    data_dir = (
        Path(args.data_dir).expanduser().resolve() if args.data_dir else default_data_dir()
    )

    print(f"[fetch_models] models dir: {models_dir}")
    fetched = False
    for name in packs:
        fetched = fetch_pack(name, models_dir, args.force, args.base_url) or fetched
    if args.yunet or args.all_models:
        fetched = fetch_yunet(models_dir, args.force) or fetched
    if uniface_names:
        fetched = fetch_uniface(uniface_names, args.force) or fetched

    write_manifest(models_dir, data_dir)
    print("[fetch_models] done" + ("" if fetched else " (nothing to download)"))


if __name__ == "__main__":
    main()
