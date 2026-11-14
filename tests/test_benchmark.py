"""Unit tests for the calibration / metrics engine (pure NumPy, no ONNX)."""

from __future__ import annotations

import numpy as np
import pytest

from facelab import benchmark, matching


def unit(*values):
    vec = np.asarray(values, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def make_student(student_id, raw, flip_avg=None):
    embeddings = {"insightface:w600k_r50": {"raw": list(raw)}}
    if flip_avg is not None:
        embeddings["insightface:w600k_r50"]["flip_avg"] = list(flip_avg)
    return {
        "id": student_id,
        "photos": [{"id": f"ph{student_id}", "embeddings": embeddings}],
    }


def frame(frame_id, detections):
    return {
        "frame_id": frame_id,
        "width": 1000,
        "height": 1000,
        "detections": detections,
    }


def det(embedding, bbox=(100, 100, 180, 180), quality=0.9, flip=None):
    entry = {"bbox": list(bbox), "embedding": embedding, "quality_score": quality}
    if flip is not None:
        entry["embedding_flip_avg"] = flip
    return entry


def test_build_gallery_picks_variant_and_normalizes():
    students = {
        1: make_student(1, [3.0, 0.0, 0.0], flip_avg=[2.0, 1.0, 0.0]),
        2: make_student(2, [0.0, 5.0, 0.0]),
    }
    gallery = benchmark.build_gallery(students, "insightface:w600k_r50", "flip_avg")
    assert set(gallery) == {1, 2}
    assert np.isclose(np.linalg.norm(gallery[1][0]), 1.0, atol=1e-6)
    # student 2 has no flip_avg, so it falls back to raw.
    assert np.allclose(gallery[2][0], unit(0, 1, 0))


def test_build_gallery_skips_students_without_embeddings():
    students = {1: {"id": 1, "photos": [{"embeddings": {}}]}}
    assert benchmark.build_gallery(students, "insightface:w600k_r50") == {}


def test_select_embedding_variant_fallback():
    detection = {"embedding": unit(1, 0, 0)}
    assert np.allclose(benchmark.select_embedding(detection, "flip_avg"), unit(1, 0, 0))
    detection["embedding_flip_avg"] = unit(0, 1, 0)
    assert np.allclose(benchmark.select_embedding(detection, "flip_avg"), unit(0, 1, 0))


def test_evaluate_frames_perfect_attendance():
    students = {1: make_student(1, [1.0, 0.0, 0.0]), 2: make_student(2, [0.0, 1.0, 0.0])}
    gallery = benchmark.build_gallery(students, "insightface:w600k_r50", "raw")
    frames = [
        frame("frame_01", [det(unit(1, 0, 0)), det(unit(0, 1, 0), bbox=(600, 600, 680, 680))]),
        frame("frame_02", [det(unit(1, 0, 0)), det(unit(0, 1, 0), bbox=(600, 600, 680, 680))]),
    ]
    result = benchmark.evaluate_frames(
        frames, gallery, [1, 2], accept_threshold=0.5, review_threshold=0.4, k=1
    )
    assert result["attendance"]["f1"] == 1.0
    assert set(result["predicted_student_ids"]) == {1, 2}


def test_calibration_finds_low_far_threshold_for_good_model():
    students = {1: make_student(1, [1.0, 0.0, 0.0])}
    gallery = benchmark.build_gallery(students, "insightface:w600k_r50", "raw")
    # Face 1 clearly the enrolled student, plus an impostor that never passes.
    frames = [
        frame("frame_01", [det(unit(1.0, 0.05, 0.0)), det(unit(0.0, 0.7, 0.7), bbox=(700, 700, 760, 760))]),
        frame("frame_02", [det(unit(1.0, 0.04, 0.0))]),
    ]
    curve = benchmark.threshold_curve(
        frames, gallery, [1], variant="raw", low=0.2, high=0.7, step=0.02, k=1
    )
    points = benchmark.pick_operating_points(curve)
    assert points["best_f1"]["f1"] == 1.0
    assert points["far0"] is not None
    assert points["far0"]["fp"] == 0


def test_stratified_metrics_buckets_by_row():
    frames = [
        frame(
            "frame_01",
            [
                det(unit(1, 0, 0), bbox=(100, 100, 180, 180)),  # large (80px)
                det(unit(0, 1, 0), bbox=(500, 500, 540, 540)),  # medium (40px)
            ],
        )
    ]
    decisions = [
        [
            {"status": "present", "matched_student_id": 1},
            {"status": "unmatched", "matched_student_id": None},
        ]
    ]
    boxes = [
        {"frame_id": "frame_01", "bbox": [100, 100, 180, 180], "student_id": 1, "row_bucket": "large"},
        {"frame_id": "frame_01", "bbox": [500, 500, 540, 540], "student_id": 2, "row_bucket": "medium"},
        {"frame_id": "frame_01", "bbox": [800, 800, 812, 812], "student_id": 3, "row_bucket": "small"},
    ]
    metrics = benchmark.stratified_metrics(frames, decisions, boxes)
    assert metrics["gt_faces"] == 3
    assert metrics["detected_faces"] == 2
    assert metrics["identified_faces"] == 1
    assert metrics["buckets"]["large"]["identification_recall"] == 1.0
    assert metrics["buckets"]["small"]["detection_recall"] == 0.0


def test_aggregate_attendance_sums_confusion():
    results = [
        {"attendance": {"tp": 2, "fp": 1, "fn": 0, "precision": 0, "recall": 0, "f1": 0}},
        {"attendance": {"tp": 1, "fp": 0, "fn": 1, "precision": 0, "recall": 0, "f1": 0}},
    ]
    aggregate = benchmark.aggregate_attendance(results)
    assert aggregate["tp"] == 3
    assert aggregate["fp"] == 1
    assert aggregate["fn"] == 1
    assert aggregate["f1"] == pytest.approx(0.75)


def test_latency_percentiles():
    timings = [{"detect_ms": 10, "align_embed_ms": 20, "match_ms": 1} for _ in range(10)]
    stats = benchmark.latency_percentiles(timings)
    assert stats["detect_ms_p50"] == 10
    assert stats["total_inference_ms_p50"] == 31
    assert stats["passes"] == 10


def test_select_embedding_lowres_aug_prefers_clean_flip():
    detection = {
        "embedding": unit(1, 0, 0),
        "embedding_flip_avg": unit(0, 1, 0),
        "embedding_lowres_aug": unit(0, 0, 1),
    }
    # Classroom probes are already low-res, so lowres_aug must reuse the clean
    # flip-averaged vector rather than degrading them again.
    assert np.allclose(benchmark.select_embedding(detection, "lowres_aug"), unit(0, 1, 0))


def test_threshold_curve_datasets_aggregates_across_datasets():
    students = {1: make_student(1, [1.0, 0.0, 0.0])}
    gallery = benchmark.build_gallery(students, "insightface:w600k_r50", "raw")
    # Both datasets reuse the same frame id on purpose: they must be scored
    # separately, not concatenated.
    good = [
        frame("frame_01", [det(unit(1, 0, 0))]),
        frame("frame_02", [det(unit(1, 0, 0))]),
    ]
    bad = [frame("frame_01", [det(unit(0.0, 1.0, 0.0), bbox=(700, 700, 760, 760))])]
    curve = benchmark.threshold_curve_datasets(
        [(good, [1]), (bad, [1])],
        gallery,
        variant="raw",
        low=0.2,
        high=0.7,
        step=0.1,
        k=1,
    )
    row = next(item for item in curve if item["threshold"] == pytest.approx(0.2))
    assert row["tp"] == 1  # dataset one recognised
    assert row["fn"] == 1  # dataset two missed
    assert row["fp"] == 0


def test_curve_row_matches_evaluate_frames_with_fixed_review():
    students = {1: make_student(1, [1.0, 0.0, 0.0])}
    gallery = benchmark.build_gallery(students, "insightface:w600k_r50", "raw")
    frames = [frame("frame_01", [det(unit(1.0, 0.05, 0.0))])]
    curve = benchmark.threshold_curve_datasets(
        [(frames, [1])], gallery, variant="raw",
        low=0.5, high=0.5, step=0.1, review_threshold=0.34, k=1,
    )
    direct = benchmark.evaluate_frames(
        frames, gallery, [1], variant="raw",
        accept_threshold=0.5, review_threshold=0.34, k=1,
    )["attendance"]
    assert {key: curve[0][key] for key in direct} == direct


def test_dataset_id_traversal_rejected():
    from facelab import datasets
    assert datasets.delete_dataset("..") is False
    assert datasets.load_manifest("../x") is None


def _row(threshold, tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def test_operating_points_prefer_top_of_plateau():
    curve = [
        _row(0.25, 8, 0, 0), _row(0.30, 8, 0, 0), _row(0.45, 8, 0, 0),
        _row(0.50, 6, 0, 2), _row(0.75, 1, 0, 7),
    ]
    points = benchmark.pick_operating_points(curve)
    assert points["best_f1"]["threshold"] == 0.45
    assert points["far0"]["threshold"] == 0.45  # not 0.75, where tp collapsed to 1
    assert points["prec98"]["threshold"] == 0.45


def test_operating_points_empty_curve():
    assert benchmark.pick_operating_points([]) == {
        "best_f1": None, "far0": None, "prec98": None
    }


def test_curve_default_low_is_025():
    rows = benchmark.threshold_curve_datasets([], {}, high=0.27)
    assert rows[0]["threshold"] == 0.25


def test_aggregate_attendance_similarity_stats():
    att = {"tp": 1, "fp": 0, "fn": 0}
    out = benchmark.aggregate_attendance([
        {"attendance": att, "present_sim_sum": 1.2, "present_sim_n": 2, "max_impostor_sim": 0.3},
        {"attendance": att, "present_sim_sum": 0.6, "present_sim_n": 1, "max_impostor_sim": 0.5},
    ])
    assert out["tp"] == 2
    assert out["mean_present_sim"] == 0.6
    assert out["max_impostor_sim"] == 0.5


def test_evaluate_frames_reports_similarity_stats_and_detected_buckets():
    students = {1: make_student(1, [1.0, 0.0, 0.0])}
    gallery = benchmark.build_gallery(students, "insightface:w600k_r50", "raw")
    frames = [frame("frame_01", [det(unit(1.0, 0.0, 0.0))])]
    result = benchmark.evaluate_frames(
        frames, gallery, [1], variant="raw", accept_threshold=0.5, review_threshold=0.4, k=1
    )
    assert result["mean_present_sim"] > 0.9
    assert result["max_impostor_sim"] == 0.0
    # Detection buckets are counted even without annotated ground-truth boxes.
    assert result["detection"]["detected_buckets"]["large"] == 1


def test_annotate_unknown_label_toggle():
    from facelab import face as engine

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    matches = [
        {
            "bbox": [40, 40, 160, 170],
            "status": "unmatched",
            "student_name": "Unknown",
            "confidence": 0.0,
            "matched": False,
        }
    ]
    without = engine.annotate_classroom_photo(image, matches)
    with_labels = engine.annotate_classroom_photo(image, matches, label_unknown=True)
    assert int(np.count_nonzero(without)) > 0  # the red box is still drawn
    assert int(np.count_nonzero(with_labels)) > int(np.count_nonzero(without))


def test_aggregate_detection_sums_detected_buckets():
    merged = benchmark.aggregate_detection([
        {"buckets": {}, "detected_buckets": {"large": 2, "small": 1}},
        {"buckets": {}, "detected_buckets": {"large": 1, "medium": 4}},
    ])
    assert merged["detected_buckets"] == {"large": 3, "medium": 4, "small": 1}


def test_aggregate_detection_sums_buckets():
    results = [
        {"has_annotations": True, "buckets": {"large": {"gt": 2, "detected": 2, "identified": 1}}},
        {"has_annotations": True, "buckets": {"large": {"gt": 1, "detected": 1, "identified": 1}}},
    ]
    merged = benchmark.aggregate_detection(results)
    assert merged["gt_faces"] == 3
    assert merged["detected_faces"] == 3
    assert merged["identified_faces"] == 2
    assert merged["buckets"]["large"]["detection_recall"] == 1.0
