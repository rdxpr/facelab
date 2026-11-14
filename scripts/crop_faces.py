#!/usr/bin/env python3
"""Batch-extract padded face crops from a folder of reference photos.

This is a standalone helper: it never touches the server, the roster, or the
inference cache. Point it at a directory of photos and it writes a padded,
natural-looking crop around every detected face (with margin, so the enrolment
engine can re-detect the face), plus an optional 112×112 aligned crop, a JSON or
CSV manifest, and a labeled contact sheet.

Typical use:

    python scripts/crop_faces.py
    python scripts/crop_faces.py --input scripts/incoming --padding 0.4
    python scripts/crop_faces.py --recursive --tiling sahi_upper_band

Output layout (default):

    scripts/face-crops/
        <source-stem>/
            <source-stem>__f01.jpg
            <source-stem>__f01_aligned.jpg   # only with --aligned
        manifest.json
        contact-sheet.jpg

Both scripts/incoming/ and scripts/face-crops/ are git-ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


def _reexec_into_venv() -> None:
    """Re-exec into ./.venv so `./scripts/crop_faces.py` works from a bare shell."""
    venv = REPO_ROOT / ".venv" / "bin" / "python"
    if (
        venv.is_file()
        and os.environ.get("FACELAB_SKIP_REEXEC") != "1"
        and Path(sys.executable).resolve() != venv.resolve()
    ):
        os.execv(str(venv), [str(venv), str(Path(__file__).resolve()), *sys.argv[1:]])


# Only re-exec when executed as a program, so tests can import this module.
if __name__ == "__main__":
    _reexec_into_venv()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from facelab import face as engine  # noqa: E402
from facelab import matching  # noqa: E402
from facelab.config import settings  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CONTACT_SHEET_MAX = 240


def sanitize_stem(value: str) -> str:
    """Filesystem-safe name derived from a path fragment."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return stem or "image"


def base_name(path: Path, root: Path, recursive: bool) -> str:
    """Folder/file prefix for one source image, unique across subfolders."""
    if recursive:
        try:
            relative = path.relative_to(root).with_suffix("")
            return sanitize_stem("__".join(relative.parts))
        except ValueError:
            pass
    return sanitize_stem(path.stem)


def padded_box(
    bbox: list[float], width: int, height: int, padding: float = 0.35, square: bool = True
) -> list[int]:
    """Expand a face box by ``padding`` of its larger side and clamp to the image.

    With ``square`` the result is a square around the face centre (a natural
    portrait crop); otherwise each side is padded independently.
    """
    x1, y1, x2, y2 = (float(value) for value in bbox[:4])
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0:
        return [
            max(0, int(round(x1))),
            max(0, int(round(y1))),
            min(width, int(round(x2))),
            min(height, int(round(y2))),
        ]

    side = max(box_w, box_h)
    if square:
        half = side / 2.0 + side * padding
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        px1, py1, px2, py2 = (
            center_x - half,
            center_y - half,
            center_x + half,
            center_y + half,
        )
    else:
        grow = side * padding
        px1, py1, px2, py2 = x1 - grow, y1 - grow, x2 + grow, y2 + grow

    px1 = max(0, int(round(px1)))
    py1 = max(0, int(round(py1)))
    px2 = min(width, int(round(px2)))
    py2 = min(height, int(round(py2)))
    if px2 <= px1:
        px2 = min(width, px1 + 1)
    if py2 <= py1:
        py2 = min(height, py1 + 1)
    return [px1, py1, px2, py2]


def iter_images(root: Path, recursive: bool, exclude: Path | None = None) -> list[Path]:
    candidates = root.rglob("*") if recursive else root.iterdir()
    found: list[Path] = []
    for path in candidates:
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if exclude is not None:
            try:
                if path.resolve().is_relative_to(exclude):
                    continue
            except OSError:
                pass
        found.append(path)
    return sorted(found)


def _fit(image: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_w / max(1, width), max_h / max(1, height))
    return cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def build_contact_sheet(entries: list[dict], path: Path, thumb: int = 160, cols: int = 6) -> None:
    """Tile labelled thumbnails into one browsable index image."""
    if not entries:
        return
    entries = entries[:CONTACT_SHEET_MAX]
    label_h = 18
    margin = 8
    cols = max(1, min(cols, len(entries)))
    rows = math.ceil(len(entries) / cols)
    cell_w, cell_h = thumb, thumb + label_h
    sheet = np.full(
        (rows * cell_h + margin * 2, cols * cell_w + margin * 2, 3),
        250,
        dtype=np.uint8,
    )
    for index, entry in enumerate(entries):
        row, col = divmod(index, cols)
        x = margin + col * cell_w
        y = margin + row * cell_h
        tile = _fit(entry["image"], thumb, thumb)
        tile_h, tile_w = tile.shape[:2]
        ox = x + (thumb - tile_w) // 2
        oy = y + (thumb - tile_h) // 2
        sheet[oy : oy + tile_h, ox : ox + tile_w] = tile
        label = f'{entry["label"]}  {entry["score"]:.2f}'
        cv2.putText(
            sheet,
            label[:30],
            (x + 2, y + thumb + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
    ok, buffer = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        path.write_bytes(buffer.tobytes())


def write_manifest(
    path: Path, fmt: str, payload: dict, rows: list[dict]
) -> None:
    if fmt == "csv":
        fields = [
            "source", "sourceWidth", "sourceHeight", "faceIndex", "bbox",
            "paddedBbox", "score", "faceHeightPx", "sizeBucket", "crop", "aligned",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fields})
        return
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract padded face crops from reference photos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=str(SCRIPTS_DIR / "incoming"),
                        help="Folder of reference photos.")
    parser.add_argument("--output", default=str(SCRIPTS_DIR / "face-crops"),
                        help="Folder to write crops and the manifest into.")
    parser.add_argument("--recursive", action="store_true", help="Walk subfolders.")
    parser.add_argument("--detector", default=None,
                        help="Detector id (default: FACELAB_DEFAULT_DETECTOR).")
    parser.add_argument("--det-size", type=int, default=None,
                        help="Detector input size (default: FACELAB_FACE_DET_SIZE).")
    parser.add_argument("--det-thresh", type=float, default=0.5,
                        help="Detector confidence threshold.")
    parser.add_argument("--nms-thresh", type=float, default=0.45, help="NMS IoU.")
    parser.add_argument("--tiling", choices=["none", "sahi_upper_band", "sahi_full"],
                        default="none", help="SAHI tiling mode (group photos benefit).")
    parser.add_argument("--padding", type=float, default=0.35,
                        help="Margin as a fraction of the face's larger side.")
    parser.add_argument("--square", action=argparse.BooleanOptionalAction, default=True,
                        help="Produce a square crop around the face centre.")
    parser.add_argument("--min-face", type=int, default=48,
                        help="Skip faces shorter than this many pixels.")
    parser.add_argument("--max-faces", type=int, default=None,
                        help="Keep at most this many faces per image (largest first).")
    parser.add_argument("--dominant", action="store_true",
                        help="Keep only the largest face per image.")
    parser.add_argument("--aligned", action="store_true",
                        help="Also write the 112x112 aligned crop.")
    parser.add_argument("--contact-sheet", action=argparse.BooleanOptionalAction, default=True,
                        help="Write a labelled thumbnail contact sheet.")
    parser.add_argument("--manifest", choices=["json", "csv", "none"], default="json",
                        help="Manifest format.")
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing files.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be written without writing.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(
            f"[crop_faces] input folder not found: {input_dir}\n"
            f"  Put photos in {SCRIPTS_DIR / 'incoming'} or pass --input."
        )

    detector_id = args.detector or settings.default_detector
    available, reason = engine.detector_availability(detector_id)
    if not available:
        raise SystemExit(
            f"[crop_faces] detector '{detector_id}' is unavailable: {reason}\n"
            "  Run scripts/fetch_models.py first."
        )
    det_size = args.det_size or settings.face_det_size
    try:
        detector = engine.get_detector(detector_id, det_size, args.det_thresh, args.nms_thresh)
    except engine.FaceEngineUnavailable as exc:
        raise SystemExit(f"[crop_faces] {exc}") from exc

    images = iter_images(input_dir, args.recursive, exclude=output_dir)
    if not images:
        raise SystemExit(f"[crop_faces] no images found under {input_dir}")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    manifest_images: list[dict] = []
    flat_rows: list[dict] = []
    counts = {"images": 0, "faces": 0, "skipped": 0}

    for position, path in enumerate(images, 1):
        try:
            image_bgr = engine.image_to_bgr(path.read_bytes())
        except Exception as exc:  # unreadable file: warn, keep going
            counts["skipped"] += 1
            print(f"[crop_faces] skip {path.name}: {exc}", file=sys.stderr)
            continue
        height, width = image_bgr.shape[:2]
        detections = engine.detect_with_sahi(
            image_bgr,
            lambda image: detector.detect(image),
            tile_size=det_size,
            iou_threshold=args.nms_thresh,
            tiling_mode=args.tiling,
        )
        faces = [det for det in detections if det.height >= args.min_face]
        faces.sort(
            key=lambda det: float((det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1])),
            reverse=True,
        )
        if args.dominant:
            faces = faces[:1]
        if args.max_faces:
            faces = faces[: args.max_faces]
        if not faces:
            counts["skipped"] += 1
            print(f"[crop_faces] {path.name}: no usable faces")
            continue

        counts["images"] += 1
        name = base_name(path, input_dir, args.recursive)
        image_dir = output_dir / name
        if not args.dry_run:
            image_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict] = []

        for face_index, det in enumerate(faces, 1):
            bbox = [float(value) for value in det.bbox[:4]]
            box = padded_box(bbox, width, height, args.padding, args.square)
            crop = image_bgr[box[1] : box[3], box[0] : box[2]]
            if crop.size == 0:
                continue

            crop_name = f"{name}__f{face_index:02d}.jpg"
            crop_path = image_dir / crop_name
            crop_rel = crop_path.relative_to(output_dir).as_posix()
            if args.dry_run:
                print(f"[crop_faces] would write {crop_rel}")
            elif args.overwrite or not crop_path.is_file():
                cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, args.quality])

            aligned_rel = None
            if args.aligned and det.kps is not None:
                aligned = engine.align_face(image_bgr, det.kps.astype(np.float32), 112)
                aligned_name = f"{name}__f{face_index:02d}_aligned.jpg"
                aligned_path = image_dir / aligned_name
                aligned_rel = aligned_path.relative_to(output_dir).as_posix()
                if args.dry_run:
                    print(f"[crop_faces] would write {aligned_rel}")
                elif args.overwrite or not aligned_path.is_file():
                    cv2.imwrite(
                        str(aligned_path), aligned, [cv2.IMWRITE_JPEG_QUALITY, args.quality]
                    )

            score = round(float(det.score), 4)
            record = {
                "faceIndex": face_index,
                "bbox": [round(value, 1) for value in bbox],
                "paddedBbox": box,
                "score": score,
                "faceHeightPx": round(float(det.height), 1),
                "sizeBucket": matching.size_bucket(det.height),
                "crop": crop_rel,
                "aligned": aligned_rel,
            }
            records.append(record)
            flat_rows.append({"source": path.name, "sourceWidth": width,
                              "sourceHeight": height, **record})
            entries.append({"label": crop_name, "image": crop, "score": score})

        counts["faces"] += len(records)
        manifest_images.append(
            {
                "source": path.relative_to(input_dir).as_posix(),
                "width": width,
                "height": height,
                "faces": records,
            }
        )
        if not args.dry_run:
            print(
                f"[crop_faces] {path.name}: {len(records)} face(s) -> "
                f"{image_dir.relative_to(output_dir).as_posix()}/"
            )

    if not args.dry_run:
        if args.manifest != "none":
            suffix = ".csv" if args.manifest == "csv" else ".json"
            manifest_path = output_dir / f"manifest{suffix}"
            payload = {
                "generatedAt": datetime.now(UTC).isoformat(timespec="seconds"),
                "input": str(input_dir),
                "output": str(output_dir),
                "detector": detector_id,
                "params": {
                    "detSize": det_size,
                    "detThresh": args.det_thresh,
                    "nmsThresh": args.nms_thresh,
                    "tilingMode": args.tiling,
                    "padding": args.padding,
                    "square": args.square,
                    "minFace": args.min_face,
                    "aligned": args.aligned,
                },
                "counts": counts,
                "images": manifest_images,
            }
            write_manifest(manifest_path, args.manifest, payload, flat_rows)
            print(f"[crop_faces] wrote {manifest_path.name}")

        if args.contact_sheet and entries:
            sheet_path = output_dir / "contact-sheet.jpg"
            build_contact_sheet(entries, sheet_path)
            print(f"[crop_faces] wrote {sheet_path.name} ({len(entries)} faces)")

    print(
        f"[crop_faces] done: {counts['images']} image(s), {counts['faces']} face(s),"
        f" {counts['skipped']} skipped"
        + (" (dry run)" if args.dry_run else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())