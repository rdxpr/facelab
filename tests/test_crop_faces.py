"""Unit tests for the standalone face-crop helper's pure geometry/name helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location("crop_faces", ROOT / "scripts" / "crop_faces.py")
crop_faces = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(crop_faces)


def test_padded_box_square_centered():
    box = crop_faces.padded_box([100, 100, 200, 200], 1000, 1000, 0.35, True)
    assert box == [65, 65, 235, 235]


def test_padded_box_clamps_to_image_edges():
    box = crop_faces.padded_box([0, 0, 100, 100], 100, 100, 0.35, True)
    assert box == [0, 0, 100, 100]


def test_padded_box_non_square():
    box = crop_faces.padded_box([100, 200, 200, 300], 1000, 1000, 0.35, False)
    assert box == [65, 165, 235, 335]


def test_padded_box_uses_larger_side_for_margin():
    # w=200, h=100 -> margin = 0.35 * 200 = 70 on every side.
    box = crop_faces.padded_box([100, 100, 300, 200], 1000, 1000, 0.35, False)
    assert box == [30, 30, 370, 270]


def test_padded_box_zero_padding_is_a_tight_crop():
    box = crop_faces.padded_box([10, 20, 110, 120], 500, 500, 0.0, True)
    assert box == [10, 20, 110, 120]


def test_sanitize_stem():
    assert crop_faces.sanitize_stem("Room 204 / morning.JPG") == "Room-204-morning.JPG"
    assert crop_faces.sanitize_stem("___") == "image"


def test_base_name_keeps_subfolder_when_recursive():
    root = Path("/photos")
    path = Path("/photos/class-a/saquib 1.jpg")
    assert crop_faces.base_name(path, root, True) == "class-a__saquib-1"
    assert crop_faces.base_name(path, root, False) == "saquib-1"