#!/usr/bin/env python3
"""Download InsightFace model packs into the FaceLab ``models/`` directory.

The ONNX weights are far too large for git, so they are kept out of the
repository and fetched on demand. This runs automatically at container start
(see ``scripts/start.sh``) and can be run by hand locally:

    python scripts/fetch_models.py                 # default pack
    python scripts/fetch_models.py --model buffalo_s --model buffalo_sc
    python scripts/fetch_models.py --force          # re-download

Packs come from the official InsightFace model zoo release:

    https://github.com/deepinsight/insightface/releases/download/model-zoo/<name>.zip
"""

from __future__ import annotations

import argparse
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

# Known packs and the files we need for detection + recognition. Only the listed
# files are required for FaceLab; extra files in a pack are harmless.
KNOWN_MODELS: dict[str, list[str]] = {
    "buffalo_l": ["det_10g.onnx", "w600k_r50.onnx"],
    "buffalo_s": ["det_500m.onnx", "w600k_mbf.onnx"],
    "buffalo_sc": ["det_500m.onnx", "w600k_mbf.onnx"],
    "antelopev2": ["scrfd_10g_bnkps.onnx", "glintr100.onnx"],
}


def default_models_dir() -> Path:
    env = os.environ.get("FACELAB_MODELS_DIR")
    if env:
        path = Path(env).expanduser()
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()
    return REPO_ROOT / "models"


def needs_download(target: Path, required: list[str], force: bool) -> bool:
    if force:
        return True
    if not target.is_dir():
        return True
    onnx_files = list(target.glob("*.onnx"))
    if not onnx_files:
        return True
    # If the pack is a known one, make sure the files we actually need are here.
    if required and not all((target / name).exists() for name in required):
        return True
    return False


def download(url: str, destination: Path, retries: int = 3) -> None:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"[fetch_models] downloading {url} (attempt {attempt}/{retries})")
            with urllib.request.urlopen(url, timeout=120) as response, destination.open(
                "wb"
            ) as handle:
                shutil.copyfileobj(response, handle)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            print(f"[fetch_models]   retry needed: {exc}", file=sys.stderr)
    raise SystemExit(f"[fetch_models] could not download {url}: {last_error}")


def fetch_one(name: str, models_dir: Path, force: bool, base_url: str) -> bool:
    target = models_dir / name
    required = KNOWN_MODELS.get(name, [])
    if not needs_download(target, required, force):
        print(f"[fetch_models] {name}: already present at {target}")
        return False

    url = f"{base_url.rstrip('/')}/{name}.zip"
    models_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"{name}.zip"
        download(url, archive)
        if force and target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            _extract_flat(zf, target)

    onnx_files = sorted(target.glob("*.onnx"))
    if not onnx_files:
        raise SystemExit(
            f"[fetch_models] {name}: archive had no .onnx files; check {url}"
        )
    missing = [item for item in required if not (target / item).exists()]
    if missing:
        print(
            f"[fetch_models] {name}: warning, expected files missing: {', '.join(missing)}",
            file=sys.stderr,
        )
    size_mb = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) / 1e6
    print(
        f"[fetch_models] {name}: ready at {target} ({len(onnx_files)} models, {size_mb:.0f} MB)"
    )
    return True


def _extract_flat(zf: zipfile.ZipFile, target: Path) -> None:
    """Extract a model-zoo archive, flattening an optional single top folder."""
    names = zf.namelist()
    roots = {name.split("/", 1)[0] for name in names if "/" in name.strip("/")}
    prefix = ""
    if len(roots) == 1:
        candidate = roots.pop()
        # Only treat it as a wrapper dir if every entry starts with it.
        if all(name == candidate or name.startswith(candidate + "/") for name in names if name.strip("/")):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch InsightFace model packs for FaceLab")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model pack to fetch (repeatable). Defaults to buffalo_l.",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Where to place packs (default: $FACELAB_MODELS_DIR or ./models).",
    )
    parser.add_argument("--base-url", default=BASE_URL, help="Model zoo release URL.")
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    parser.add_argument("--list", action="store_true", help="List known packs and exit.")
    args = parser.parse_args()

    if args.list:
        for name, files in sorted(KNOWN_MODELS.items()):
            print(f"  {name:12s} needs: {', '.join(files)}")
        return

    models_dir = (
        Path(args.models_dir).expanduser().resolve()
        if args.models_dir
        else default_models_dir()
    )
    names = args.models or [os.environ.get("FACELAB_DEFAULT_MODEL") or "buffalo_l"]

    print(f"[fetch_models] models dir: {models_dir}")
    fetched = False
    for name in names:
        fetched = fetch_one(name, models_dir, args.force, args.base_url) or fetched
    print("[fetch_models] done" + ("" if fetched else " (nothing to download)"))


if __name__ == "__main__":
    main()
