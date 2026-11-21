"""Unit tests for the pure NumPy/SciPy matching engine."""

from __future__ import annotations

import numpy as np
import pytest

from facelab import matching


def unit(vec):
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def test_size_bucket_boundaries():
    assert matching.size_bucket(120) == "large"
    assert matching.size_bucket(55) == "large"
    assert matching.size_bucket(54.9) == "medium"
    assert matching.size_bucket(30) == "medium"
    assert matching.size_bucket(29.9) == "small"
    assert matching.size_bucket(0) == "small"


def test_flip_average_renormalizes_and_is_symmetric():
    left = unit([1.0, 0.0, 0.0])
    right = unit([0.8, 0.6, 0.0])
    fused = matching.flip_averaged(left, right)
    assert np.isclose(np.linalg.norm(fused), 1.0, atol=1e-6)
    assert np.allclose(fused, matching.flip_averaged(right, left))


def test_compute_student_similarity_strategies():
    face = unit([1.0, 0.0, 0.0])
    templates = np.stack([unit([1.0, 0.0, 0.0]), unit([0.0, 1.0, 0.0])])
    assert matching.compute_student_similarity(face, templates, "max") == pytest.approx(1.0)
    assert matching.compute_student_similarity(face, templates, "mean") == pytest.approx(0.5)
    assert matching.compute_student_similarity(face, templates, "top_k_mean", top_k=1) == pytest.approx(1.0)
    proto = matching.compute_student_similarity(face, templates, "prototype")
    assert 0.5 < proto < 1.0
    with pytest.raises(ValueError):
        matching.compute_student_similarity(face, templates, "nope")


def test_build_similarity_matrix_shape_and_order():
    gallery = {
        "a": unit([1.0, 0.0, 0.0])[None, :],
        "b": unit([0.0, 1.0, 0.0])[None, :],
    }
    faces = [unit([1.0, 0.0, 0.0]), unit([0.0, 1.0, 0.0])]
    matrix, ids = matching.build_similarity_matrix(faces, gallery, "max")
    assert ids == ["a", "b"]
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[1, 1] == pytest.approx(1.0)
    assert matrix[0, 1] == pytest.approx(0.0)


def test_assign_single_frame_present_review_and_unmatched():
    # face 0 -> student a (clear), face 1 -> borderline b, face 2 -> nobody.
    matrix = np.array(
        [
            [0.80, 0.10],
            [0.05, 0.44],
            [0.20, 0.10],
        ],
        dtype=np.float32,
    )
    decisions = matching.assign_single_frame(
        matrix, ["a", "b"], accept_threshold=0.5, review_threshold=0.4, min_margin=0.05
    )
    assert decisions[0]["status"] == "present"
    assert decisions[0]["matched_student_id"] == "a"
    # face 1 lands in the review band (below accept, above review).
    assert decisions[1]["status"] == "needs_review"
    assert decisions[1]["matched_student_id"] == "b"
    assert decisions[2]["status"] == "unmatched"
    assert decisions[0]["top_candidates"][0]["student_id"] == "a"
    assert decisions[0]["margin"] == pytest.approx(0.7, abs=1e-3)


def test_assign_single_frame_margin_gate_rejects_ambiguous():
    matrix = np.array([[0.60, 0.58]], dtype=np.float32)
    decisions = matching.assign_single_frame(
        matrix, ["a", "b"], accept_threshold=0.5, review_threshold=0.4, min_margin=0.05
    )
    assert decisions[0]["status"] == "needs_review"
    loose = matching.assign_single_frame(
        matrix, ["a", "b"], accept_threshold=0.5, review_threshold=0.4, min_margin=0.01
    )
    assert loose[0]["status"] == "present"


def test_hungarian_resolves_conflict_greedily_differently():
    # Two faces both prefer student a; the second face has a strong student b.
    matrix = np.array(
        [
            [0.90, 0.10],
            [0.80, 0.75],
        ],
        dtype=np.float32,
    )
    hungarian = matching.assign_single_frame(
        matrix, ["a", "b"], assignment_mode="hungarian"
    )
    assert {row["matched_student_id"] for row in hungarian} == {"a", "b"}

    greedy = matching.assign_single_frame(matrix, ["a", "b"], assignment_mode="greedy")
    assert greedy[0]["matched_student_id"] == "a"
    # Greedy already used student a, so face 1 must fall back to b.
    assert greedy[1]["matched_student_id"] == "b"

    independent = matching.assign_single_frame(matrix, ["a", "b"], assignment_mode="margin")
    assert independent[0]["matched_student_id"] == "a"
    assert independent[1]["matched_student_id"] == "a"


def test_assign_single_frame_empty_inputs():
    assert matching.assign_single_frame(np.zeros((0, 2)), ["a", "b"]) == []
    # An empty gallery must still return one decision per detected face so the
    # scan UI can show the detection instead of silently dropping it.
    decisions = matching.assign_single_frame(np.zeros((3, 0)), [])
    assert len(decisions) == 3
    assert all(row["status"] == "unmatched" for row in decisions)
    assert all(row["matched_student_id"] is None for row in decisions)


def _frame(detections, width=1000, height=1000):
    return {"detections": detections, "width": width, "height": height}


def _det(embedding, bbox=(100, 100, 200, 200), quality=0.9):
    return {"embedding": unit(embedding), "bbox": list(bbox), "quality": quality}


def test_k_of_n_voting_suppresses_single_frame_impostor():
    gallery = {
        "real": unit([1.0, 0.0, 0.0])[None, :],
        "impostor": unit([0.0, 1.0, 0.0])[None, :],
    }
    # A face that is genuinely 'real' every frame, plus a one-off random blip
    # that happens to resemble the impostor in a single frame.
    frames = [
        _frame([_det([1.0, 0.0, 0.0])]),
        _frame([_det([1.0, 0.0, 0.0]), _det([0.0, 1.0, 0.0], bbox=(700, 700, 760, 760))]),
        _frame([_det([1.0, 0.0, 0.0])]),
    ]
    fused = matching.fuse_multi_frame(
        frames,
        gallery,
        mode="k_of_n_voting",
        accept_threshold=0.5,
        review_threshold=0.4,
        k=2,
    )
    assert fused["students"]["real"]["status"] == "present"
    # The impostor only cleared review once, so it must not be marked present.
    assert fused["students"]["impostor"]["status"] != "present"


def test_k_of_n_requires_one_frame_above_accept():
    gallery = {"s": unit([1.0, 0.0, 0.0])[None, :]}
    # Two frames clear the review band but never the accept band.
    frames = [
        _frame([_det([0.45, 0.893, 0.0])]),
        _frame([_det([0.45, 0.893, 0.0])]),
    ]
    fused = matching.fuse_multi_frame(
        frames,
        gallery,
        mode="k_of_n_voting",
        accept_threshold=0.6,
        review_threshold=0.4,
        k=2,
    )
    assert fused["students"]["s"]["status"] == "needs_review"


def test_max_with_margin_gate_requires_quality():
    gallery = {"s": unit([1.0, 0.0, 0.0])[None, :]}
    frames = [_frame([_det([1.0, 0.0, 0.0], quality=0.1)])]
    fused = matching.fuse_multi_frame(
        frames,
        gallery,
        mode="max_with_margin_gate",
        accept_threshold=0.5,
        min_quality=0.5,
    )
    assert fused["students"]["s"]["status"] == "absent"

    good = matching.fuse_multi_frame(
        frames,
        gallery,
        mode="max_with_margin_gate",
        accept_threshold=0.5,
        min_quality=0.05,
    )
    assert good["students"]["s"]["status"] == "present"


def test_quality_weighted_cluster_merges_tracks_across_frames():
    gallery = {"s": unit([1.0, 0.0, 0.0])[None, :]}
    frames = [
        _frame([_det([1.0, 0.05, 0.0], bbox=(100, 100, 180, 180), quality=0.9)]),
        _frame([_det([0.98, 0.1, 0.0], bbox=(104, 102, 182, 181), quality=0.8)]),
    ]
    fused = matching.fuse_multi_frame(
        frames,
        gallery,
        mode="quality_weighted_embedding_cluster",
        accept_threshold=0.5,
        review_threshold=0.4,
    )
    assert fused["students"]["s"]["status"] == "present"
    # The two detections were clustered into one track, but both keep a status.
    assert all(row["status"] == "present" for row in fused["per_frame"][0])
    assert all(row["status"] == "present" for row in fused["per_frame"][1])


def test_attendance_metrics_confusion_and_f1():
    metrics = matching.attendance_metrics(["a", "b", "c"], ["a", "b", "d"])
    assert metrics == {
        "tp": 2,
        "fp": 1,
        "fn": 1,
        "precision": 0.6667,
        "recall": 0.6667,
        "f1": 0.6667,
    }
    perfect = matching.attendance_metrics(["a"], ["a"])
    assert perfect["f1"] == 1.0
    empty = matching.attendance_metrics([], [])
    assert empty["f1"] == 0.0


def test_k_of_n_voting_respects_hungarian_assignment():
    # Each frame's single face is closest to 'a' but also clears review for 'b'.
    # The one-to-one assignment must give the face to 'a' only; 'b' must not vote.
    gallery = {
        "a": unit([1.0, 0.0])[None, :],
        "b": unit([0.85, 0.5268])[None, :],  # cosine(face, b) ~= 0.85
    }
    frames = [_frame([_det([1.0, 0.0])]), _frame([_det([1.0, 0.0])])]
    fused = matching.fuse_multi_frame(
        frames,
        gallery,
        mode="k_of_n_voting",
        accept_threshold=0.9,
        review_threshold=0.4,
        min_margin=0.05,
        k=1,
    )
    assert fused["students"]["a"]["status"] == "present"
    # 'b' never wins the assignment, so it must be absent even though its raw
    # max similarity clears the review band.
    assert fused["students"]["b"]["status"] == "absent"


def test_k_of_n_voting_honours_min_margin():
    # The face clears accept for 'a', but 'b' is almost as close, so the margin
    # gate must demote it to review instead of calling it present.
    gallery = {
        "a": unit([1.0, 0.0, 0.0])[None, :],
        "b": unit([0.99, 0.14, 0.0])[None, :],
    }
    frames = [_frame([_det([1.0, 0.0, 0.0])]), _frame([_det([1.0, 0.0, 0.0])])]
    fused = matching.fuse_multi_frame(
        frames,
        gallery,
        mode="k_of_n_voting",
        accept_threshold=0.5,
        review_threshold=0.4,
        min_margin=0.05,
        k=1,
    )
    assert fused["students"]["a"]["status"] == "needs_review"
    assert fused["students"]["b"]["status"] == "absent"


def test_single_frame_voting_honours_min_quality():
    gallery = {"s": unit([1.0, 0.0, 0.0])[None, :]}
    frames = [_frame([_det([1.0, 0.0, 0.0], quality=0.1)])]
    fused = matching.fuse_multi_frame(
        frames, gallery, accept_threshold=0.5, review_threshold=0.4, min_quality=0.5
    )
    assert fused["students"]["s"]["status"] == "absent"


def test_k_is_clamped_to_frame_count():
    gallery = {"s": unit([1.0, 0.0, 0.0])[None, :]}
    frames = [_frame([_det([1.0, 0.0, 0.0])]), _frame([_det([1.0, 0.0, 0.0])])]
    fused = matching.fuse_multi_frame(
        frames, gallery, accept_threshold=0.5, review_threshold=0.4, k=5
    )
    assert fused["students"]["s"]["status"] == "present"


def test_max_gate_reports_needs_review():
    gallery = {"s": unit([1.0, 0.0, 0.0])[None, :]}
    frames = [_frame([_det([0.45, 0.893, 0.0])])]
    fused = matching.fuse_multi_frame(
        frames, gallery, mode="max_with_margin_gate",
        accept_threshold=0.6, review_threshold=0.4,
    )
    assert fused["students"]["s"]["status"] == "needs_review"


def test_select_dominant_face_prefers_largest_then_score():
    from facelab import face as face_module

    make = face_module.Detection
    subject = make(bbox=np.array([0, 0, 100, 100], dtype=np.float32), score=0.5)
    background = make(bbox=np.array([0, 0, 30, 30], dtype=np.float32), score=0.99)
    # A small, high-confidence background face must not beat the subject.
    assert face_module.select_dominant_face([background, subject]) is subject

    # Equal areas fall back to the higher detector score.
    twin = make(bbox=np.array([10, 10, 110, 110], dtype=np.float32), score=0.9)
    assert face_module.select_dominant_face([subject, twin]) is twin


def test_annotate_classroom_photo_draws_bbox_only_matches():
    from facelab import face as face_module

    image = np.zeros((120, 160, 3), dtype=np.uint8)
    assert int(face_module.annotate_classroom_photo(image, []).sum()) == 0

    # The scan passes match dicts with a "bbox" but no "face" object; those must
    # still be drawn instead of skipped.
    matches = [
        {
            "bbox": [20, 20, 80, 90],
            "status": "present",
            "student_name": "squid",
            "confidence": 0.9,
            "matched": True,
        }
    ]
    annotated = face_module.annotate_classroom_photo(image, matches)
    assert annotated.shape == image.shape
    assert int(annotated.sum()) > 0


def test_k_of_n_voting_counts_unique_frames_in_margin_mode():
    gallery = {"s": unit([1.0, 0.0, 0.0])[None, :]}
    # Two faces in a single frame both match 's' under independent margin mode;
    # they must count as 1 frame hit, not 2.
    frames = [
        _frame([_det([1.0, 0.0, 0.0]), _det([0.99, 0.1, 0.0], bbox=(500, 500, 600, 600))]),
        _frame([]),
    ]
    fused = matching.fuse_multi_frame(
        frames,
        gallery,
        mode="k_of_n_voting",
        assignment_mode="margin",
        accept_threshold=0.5,
        review_threshold=0.4,
        k=2,
    )
    assert fused["students"]["s"]["frames_hit"] == 1
    assert fused["students"]["s"]["status"] == "needs_review"


def test_max_gate_populates_score_and_frames_for_needs_review():
    gallery = {"s": unit([1.0, 0.0, 0.0])[None, :]}
    frames = [_frame([_det([0.45, 0.893, 0.0], quality=0.8)])]
    fused = matching.fuse_multi_frame(
        frames,
        gallery,
        mode="max_with_margin_gate",
        accept_threshold=0.6,
        review_threshold=0.4,
    )
    assert fused["students"]["s"]["status"] == "needs_review"
    assert fused["students"]["s"]["score"] == pytest.approx(0.45, abs=1e-2)
    assert fused["students"]["s"]["frames_hit"] == 1


def test_cluster_mode_respects_min_quality_and_updates_candidates():
    gallery = {
        "a": unit([1.0, 0.0, 0.0])[None, :],
        "b": unit([0.0, 1.0, 0.0])[None, :],
    }
    frames = [_frame([_det([1.0, 0.0, 0.0], quality=0.1)])]
    filtered = matching.fuse_multi_frame(
        frames,
        gallery,
        mode="quality_weighted_embedding_cluster",
        accept_threshold=0.5,
        min_quality=0.5,
    )
    assert filtered["students"]["a"]["status"] == "absent"
    assert filtered["per_frame"][0][0]["status"] == "unmatched"
    assert filtered["per_frame"][0][0]["top_candidates"] == []


def test_max_gate_zero_accept_threshold_does_not_mark_absent_students_present():
    gallery = {"absent": unit([1.0, 0.0, 0.0])[None, :]}
    frames = [_frame([])]
    fused = matching.fuse_multi_frame(
        frames,
        gallery,
        mode="max_with_margin_gate",
        accept_threshold=0.0,
        review_threshold=0.0,
    )
    assert fused["students"]["absent"]["status"] == "absent"
