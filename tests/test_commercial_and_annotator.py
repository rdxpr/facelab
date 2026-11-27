"""Unit tests for score normalization, commercial registry flags, and dataset box updates."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from facelab import datasets, matching
from facelab.config import KNOWN_DETECTORS, KNOWN_RECOGNIZERS, is_commercial_safe, settings


def _make_jpeg(width: int = 1280, height: int = 960) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(120, 130, 140)).save(buf, format="JPEG")
    return buf.getvalue()


def test_impostor_center_suppresses_hub_face_and_keeps_genuine():
    # Row 0: genuine match to student 0 (0.42), low background (0.04, 0.02, 0.06 -> mean 0.04).
    # Row 1: noisy/blurry impostor scoring ~0.30 against everyone (mean non-top1 = 0.28).
    sim = np.array(
        [
            [0.42, 0.04, 0.02, 0.06],
            [0.32, 0.29, 0.28, 0.27],
        ],
        dtype=np.float32,
    )
    normed = matching.apply_score_norm(sim, "impostor_center")
    # Genuine stays well above 0.35; hub impostor drops to ~0.04.
    assert normed[0, 0] > 0.37
    assert normed[1, 0] < 0.06
    assert np.allclose(matching.apply_score_norm(sim, "none"), sim)


def test_cluster_tracks_respects_custom_cos_thresh():
    frames = [
        {
            "width": 1000,
            "height": 1000,
            "detections": [{"bbox": [100, 100, 160, 160], "embedding": [1.0, 0.0, 0.0]}],
        },
        {
            "width": 1000,
            "height": 1000,
            "detections": [{"bbox": [102, 102, 162, 162], "embedding": [0.45, 0.893, 0.0]}],
        },
    ]
    strict = matching._cluster_tracks(frames, cos_thresh=0.55)
    relaxed = matching._cluster_tracks(frames, cos_thresh=0.40)
    assert len(strict) == 2
    assert len(relaxed) == 1


def test_commercial_safe_registry_flags():
    assert is_commercial_safe(KNOWN_DETECTORS["opencv:yunet_2023mar"]) is True
    assert is_commercial_safe(KNOWN_DETECTORS["uniface:centerface"]) is True
    assert is_commercial_safe(KNOWN_DETECTORS["nvidia:scrfd_10g_fld"]) is True
    assert is_commercial_safe(KNOWN_DETECTORS["insightface:scrfd_10g_bnkps"]) is False
    assert is_commercial_safe(KNOWN_DETECTORS["uniface:yolov8n_face"]) is False
    assert is_commercial_safe(KNOWN_RECOGNIZERS["custom_onnx:auraface_v1"]) is True
    assert is_commercial_safe(KNOWN_RECOGNIZERS["opencv:sface_2021dec"]) is True
    assert is_commercial_safe(KNOWN_RECOGNIZERS["uniface:adaface_ir101"]) is False


def test_update_dataset_normalizes_boxes_and_updates_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    settings.ensure_dirs()
    manifest = datasets.create_dataset(
        "Room 204",
        [(_make_jpeg(1280, 960), "frame1.jpg")],
        present_student_ids=[1],
        annotated_boxes=[],
    )
    ds_id = manifest["dataset_id"]

    updated = datasets.update_dataset(
        ds_id,
        present_student_ids=[1, 2],
        annotated_boxes=[
            {"frame_id": "frame_01", "bbox": [120, 400, 180, 470], "student_id": 1},  # 70px -> large
            {"frame_id": "frame_01", "bbox": [520, 410, 500, 390], "student_id": 2},  # reversed -> 20px -> small
        ],
    )
    assert updated is not None
    gt = updated["ground_truth"]
    assert gt["present_student_ids"] == [1, 2]
    assert len(gt["annotated_boxes"]) == 2
    assert gt["annotated_boxes"][0]["row_bucket"] == "large"
    assert gt["annotated_boxes"][1]["row_bucket"] == "small"
    assert gt["annotated_boxes"][1]["bbox"] == [500.0, 390.0, 520.0, 410.0]


def test_threshold_curve_lowers_floor_for_impostor_center():
    from facelab import benchmark

    # 3 students so impostor_center activates; student 1 matches at raw 0.34 while
    # non-top-1 impostors average 0.16 -> shifted genuine score is 0.18 (below 0.25!).
    def _u(v):
        a = np.asarray(v, dtype=np.float32)
        return a / np.linalg.norm(a)

    gallery = {
        1: _u([1.0, 0.0, 0.0, 0.0])[None, :],
        2: _u([0.0, 1.0, 0.0, 0.0])[None, :],
        3: _u([0.0, 0.0, 1.0, 0.0])[None, :],
    }
    probe = _u([0.34, 0.16, 0.16, 0.91])
    frames = [{"frame_id": "f1", "width": 1000, "height": 1000, "detections": [{"bbox": [10, 10, 60, 60], "embedding": probe}]}]
    curve = benchmark.threshold_curve_datasets(
        [(frames, [1])], gallery, variant="raw", k=1, score_norm="impostor_center"
    )
    assert curve[0]["threshold"] == 0.05
    best = benchmark.pick_operating_points(curve)["best_f1"]
    assert best is not None and best["f1"] == 1.0 and best["threshold"] < 0.25
