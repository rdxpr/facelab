"""Pure NumPy/SciPy matching engine for FaceLab.

This module has **no I/O and no ONNX dependencies**, so every matching rule can
be unit-tested with ``pytest`` and re-evaluated against cached embeddings in a
few milliseconds while the UI sliders move.

Concepts:

* A ``template`` is one L2-normalized embedding vector for one enrolled photo.
* The ``gallery`` maps ``student_id -> np.ndarray`` with shape
  ``(num_templates, embedding_dim)``.
* ``build_similarity_matrix`` compares every detected face against every student.
* ``assign_single_frame`` turns a similarity matrix into per-face decisions with
  ``present`` / ``needs_review`` / ``unmatched`` bands and a top1-top2 margin gate.
* ``fuse_multi_frame`` merges a burst of frames into a single attendance list.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

# Face-size buckets keyed on bounding-box height in pixels.
SIZE_BUCKET_LARGE_MIN = 55.0
SIZE_BUCKET_MEDIUM_MIN = 30.0


def size_bucket(face_height_px: float) -> str:
    """Bucket a detected face by its pixel height (front to back row)."""
    if face_height_px >= SIZE_BUCKET_LARGE_MIN:
        return "large"
    if face_height_px >= SIZE_BUCKET_MEDIUM_MIN:
        return "medium"
    return "small"


def l2_normalize(matrix: np.ndarray, axis: int = -1) -> np.ndarray:
    """Row-wise (or axis-wise) unit-normalize, leaving zero vectors alone."""
    array = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    return array / np.where(norms > 0, norms, 1.0)


def flip_averaged(original: np.ndarray, mirrored: np.ndarray) -> np.ndarray:
    """Combine an embedding and its horizontally flipped twin.

    ``e_flip = normalize(normalize(e_orig) + normalize(e_mirror))`` per the spec.
    """
    left = l2_normalize(np.asarray(original, dtype=np.float32))
    right = l2_normalize(np.asarray(mirrored, dtype=np.float32))
    return l2_normalize(left + right)


def compute_student_similarity(
    face_vec: np.ndarray,
    student_templates: np.ndarray,
    strategy: str = "top_k_mean",
    top_k: int = 2,
) -> float:
    """Similarity between one face vector and one student's template set."""
    templates = np.asarray(student_templates, dtype=np.float32)
    if templates.ndim == 1:
        templates = templates[None, :]
    if templates.shape[0] == 0:
        return 0.0

    face = np.asarray(face_vec, dtype=np.float32).reshape(-1)
    scores = templates @ face

    if strategy == "max":
        return float(np.max(scores))
    if strategy == "mean":
        return float(np.mean(scores))
    if strategy == "top_k_mean":
        k = max(1, min(int(top_k), len(scores)))
        return float(np.mean(np.sort(scores)[-k:]))
    if strategy == "prototype":
        proto = np.mean(templates, axis=0)
        norm = float(np.linalg.norm(proto))
        if norm <= 0:
            return 0.0
        return float(np.dot(proto / norm, face))
    raise ValueError(f"Unknown template strategy: {strategy}")


def build_similarity_matrix(
    face_embeddings: Sequence[np.ndarray] | np.ndarray,
    student_gallery: dict[Any, np.ndarray],
    template_strategy: str = "top_k_mean",
    top_k: int = 2,
) -> tuple[np.ndarray, list[Any]]:
    """Compare every face against every student.

    Returns ``(sim_matrix, student_ids)`` where ``sim_matrix[i, j]`` is the
    similarity of face ``i`` to student ``student_ids[j]``.

    Implementation note: every student's templates are stacked into one
    ``(M, D)`` block and scored with a single matrix multiply, so a sweep over
    the whole gallery costs one BLAS call instead of a Python double loop.
    """
    if template_strategy not in ("max", "mean", "top_k_mean", "prototype"):
        raise ValueError(f"Unknown template strategy: {template_strategy}")
    faces = [np.asarray(vec, dtype=np.float32).reshape(-1) for vec in face_embeddings]
    student_ids = list(student_gallery.keys())
    sim_matrix = np.zeros((len(faces), len(student_ids)), dtype=np.float32)
    if not faces or not student_ids:
        return sim_matrix, student_ids

    face_matrix = np.vstack(faces)
    dim = face_matrix.shape[1]
    blocks: list[np.ndarray] = []
    slices: list[tuple[int, int, int]] = []
    offset = 0
    for j, student_id in enumerate(student_ids):
        templates = np.asarray(student_gallery[student_id], dtype=np.float32)
        if templates.ndim == 1:
            templates = templates[None, :]
        if templates.shape[0] == 0 or templates.shape[1] != dim:
            continue
        if template_strategy == "prototype":
            proto = np.mean(templates, axis=0)
            norm = float(np.linalg.norm(proto))
            if norm <= 0:
                continue
            templates = (proto / norm)[None, :]
        blocks.append(templates)
        end = offset + templates.shape[0]
        slices.append((j, offset, end))
        offset = end

    if not blocks:
        return sim_matrix, student_ids

    all_scores = face_matrix @ np.vstack(blocks).T
    for j, start, end in slices:
        sub = all_scores[:, start:end]
        if template_strategy in ("max", "prototype"):
            sim_matrix[:, j] = np.max(sub, axis=1)
        elif template_strategy == "mean":
            sim_matrix[:, j] = np.mean(sub, axis=1)
        elif template_strategy == "top_k_mean":
            k = max(1, min(int(top_k), end - start))
            sim_matrix[:, j] = np.mean(np.sort(sub, axis=1)[:, -k:], axis=1)
    return sim_matrix, student_ids


def _top_candidates(
    sim_row: np.ndarray, student_ids: list[Any], limit: int = 3
) -> list[dict]:
    if sim_row.size == 0:
        return []
    order = np.argsort(-sim_row)[: min(limit, sim_row.size)]
    return [
        {"student_id": student_ids[int(idx)], "score": round(float(sim_row[idx]), 4)}
        for idx in order
    ]


def assign_single_frame(
    sim_matrix: np.ndarray,
    student_ids: list[Any],
    accept_threshold: float = 0.42,
    review_threshold: float = 0.34,
    min_margin: float = 0.04,
    assignment_mode: str = "hungarian",
) -> list[dict]:
    """Decide which student each detected face belongs to.

    ``assignment_mode`` is one of ``hungarian`` (globally optimal one-to-one),
    ``greedy`` (best-first one-to-one), or ``margin`` (each face takes its top
    candidate independently). Every face gets a ``status`` of ``present``,
    ``needs_review``, or ``unmatched`` plus its top-3 candidates and margin.
    """
    matrix = np.asarray(sim_matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    num_faces, num_students = matrix.shape
    results: list[dict] = []
    if num_faces == 0:
        return results
    if num_students == 0:
        # A scan without an enrolled gallery still has to surface every face;
        # returning no decisions made the caller drop real detections entirely.
        return [
            {
                "det_index": index,
                "matched_student_id": None,
                "status": "unmatched",
                "score": 0.0,
                "margin": 0.0,
                "top_candidates": [],
            }
            for index in range(num_faces)
        ]

    top_k_indices = np.argsort(-matrix, axis=1)[:, : min(3, num_students)]

    # Rank *accepted* pairs above review-band pairs before similarity. Plain
    # sum-of-similarity can trade one present match for two review matches
    # (e.g. 0.75/0.70 vs 0.75/0.40), which silently loses attendance.
    accepted = matrix >= accept_threshold
    weights = np.where(matrix >= review_threshold, matrix, -1.0) + np.where(
        accepted, 2.0, 0.0
    )

    assigned_map: dict[int, int] = {}
    if assignment_mode == "hungarian":
        row_ind, col_ind = linear_sum_assignment(-weights)
        for row, col in zip(row_ind, col_ind):
            if matrix[row, col] >= review_threshold:
                assigned_map[int(row)] = int(col)
    elif assignment_mode == "greedy":
        order = np.dstack(np.unravel_index(np.argsort(-weights.ravel()), matrix.shape))[0]
        used: set[int] = set()
        for row, col in order:
            row_i, col_i = int(row), int(col)
            if row_i in assigned_map or col_i in used:
                continue
            if matrix[row_i, col_i] >= review_threshold:
                assigned_map[row_i] = col_i
                used.add(col_i)
    elif assignment_mode == "margin":
        for row in range(num_faces):
            col = int(top_k_indices[row, 0])
            if matrix[row, col] >= review_threshold:
                assigned_map[row] = col
    else:
        raise ValueError(f"Unknown assignment mode: {assignment_mode}")

    for i in range(num_faces):
        candidates = _top_candidates(matrix[i], student_ids)
        best_score = candidates[0]["score"] if candidates else 0.0
        second_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
        margin = round(best_score - second_score, 4)

        assigned_col = assigned_map.get(i)
        assigned_score = float(matrix[i, assigned_col]) if assigned_col is not None else 0.0

        status = "unmatched"
        matched_student_id = None
        if assigned_col is not None and assigned_score >= review_threshold:
            is_top_choice = assigned_col == int(top_k_indices[i, 0])
            effective_margin = margin if is_top_choice else 0.0
            matched_student_id = student_ids[assigned_col]
            if assigned_score >= accept_threshold and effective_margin >= min_margin:
                status = "present"
            else:
                status = "needs_review"

        results.append(
            {
                "det_index": i,
                "matched_student_id": matched_student_id,
                "status": status,
                "score": round(assigned_score, 4),
                "margin": margin,
                "top_candidates": candidates,
            }
        )
    return results


# --------------------------------------------------------------------------- #
# Multi-frame burst fusion
# --------------------------------------------------------------------------- #
def _normalized_box(bbox: np.ndarray, width: float, height: float) -> np.ndarray:
    x1, y1, x2, y2 = (float(value) for value in bbox[:4])
    width = float(width) or 1.0
    height = float(height) or 1.0
    return np.array([x1 / width, y1 / height, x2 / width, y2 / height], dtype=np.float32)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def fuse_multi_frame(
    frames: list[dict],
    student_gallery: dict[Any, np.ndarray],
    *,
    mode: str = "k_of_n_voting",
    accept_threshold: float = 0.42,
    review_threshold: float = 0.34,
    min_margin: float = 0.04,
    assignment_mode: str = "hungarian",
    template_strategy: str = "top_k_mean",
    top_k: int = 2,
    k: int = 2,
    min_quality: float = 0.0,
    precomputed_sims: Sequence[np.ndarray] | None = None,
    precomputed_tracks: list[dict] | None = None,
    precomputed_track_sim: np.ndarray | None = None,
) -> dict:
    """Fuse a burst of frames into one attendance decision.

    Modes:

    * ``k_of_n_voting``: present when the student clears ``review_threshold`` in
      at least ``k`` frames and reaches ``accept_threshold`` in at least one.
    * ``max_with_margin_gate``: keep the best frame per student only when its
      top1-top2 margin and face quality pass their gates.
    * ``quality_weighted_embedding_cluster``: group detections across frames into
      tracks, quality-weight each track's centroid, then match once.

    ``precomputed_sims`` / ``precomputed_tracks`` / ``precomputed_track_sim``
    let a threshold sweep reuse similarities that do not depend on the
    thresholds being swept (see :func:`facelab.benchmark.threshold_curve_datasets`).
    """
    student_ids = list(student_gallery.keys())
    per_frame: list[list[dict]] = []
    sims: list[np.ndarray] = []

    student_index_map = {sid: idx for idx, sid in enumerate(student_ids)}

    if mode != "quality_weighted_embedding_cluster":
        for frame_index, frame in enumerate(frames):
            detections = frame.get("detections", [])
            if precomputed_sims is not None:
                matrix = precomputed_sims[frame_index]
            else:
                matrix, student_ids = build_similarity_matrix(
                    [det["embedding"] for det in detections],
                    student_gallery,
                    template_strategy,
                    top_k,
                )
            sims.append(matrix)
            per_frame.append(
                assign_single_frame(
                    matrix,
                    student_ids,
                    accept_threshold=accept_threshold,
                    review_threshold=review_threshold,
                    min_margin=min_margin,
                    assignment_mode=assignment_mode,
                )
            )

    num_frames = len(frames)
    student_status: dict[Any, dict] = {
        sid: {"status": "absent", "score": 0.0, "frames_hit": 0, "quality": 0.0}
        for sid in student_ids
    }

    if mode == "k_of_n_voting":
        # Vote on the *assigned* decision per frame, not the raw max similarity:
        # otherwise the Hungarian one-to-one assignment, the margin gate and the
        # min_quality filter are all bypassed here.
        required = max(1, min(int(k), num_frames or 1))
        # Count a frame at most once per student. Under ``assignment_mode="margin"``
        # two faces in the *same* frame can independently match one student, and a
        # naïve counter would treat that as two frames of evidence.
        hit_frames: dict[Any, set[int]] = {sid: set() for sid in student_ids}
        present_frames: dict[Any, set[int]] = {sid: set() for sid in student_ids}
        for frame_index, frame in enumerate(frames):
            detections = frame.get("detections", [])
            for det_index, decision in enumerate(per_frame[frame_index]):
                sid = decision["matched_student_id"]
                if sid is None or sid not in student_status:
                    continue
                # ``None`` means quality was not measured (no quality model).
                # Treat it as unknown rather than as 0.0, which used to reject
                # every valid face as soon as min_quality was non-zero.
                quality = detections[det_index].get("quality")
                if quality is not None and float(quality) < min_quality:
                    continue
                score = float(decision["score"])
                if score >= review_threshold:
                    hit_frames[sid].add(frame_index)
                    if decision["status"] == "present":
                        present_frames[sid].add(frame_index)
                record = student_status[sid]
                if score > record["score"]:
                    record["score"] = round(score, 4)
                    record["quality"] = round(float(quality or 0.0), 4)
        for sid, record in student_status.items():
            frame_hits = len(hit_frames[sid])
            record["frames_hit"] = frame_hits
            if frame_hits >= required and len(present_frames[sid]) >= 1:
                record["status"] = "present"
            elif frame_hits >= 1:
                record["status"] = "needs_review"
    elif mode == "max_with_margin_gate":
        accepted_frames: dict[Any, set[int]] = {sid: set() for sid in student_ids}
        review_frames: dict[Any, set[int]] = {sid: set() for sid in student_ids}
        review_best: dict[Any, tuple[float, float]] = {}
        for frame_index, frame in enumerate(frames):
            detections = frame.get("detections", [])
            assignments = per_frame[frame_index]
            for det_index, decision in enumerate(assignments):
                sid = decision["matched_student_id"]
                if sid is None or sid not in student_status:
                    continue
                quality = detections[det_index].get("quality")
                if quality is not None and float(quality) < min_quality:
                    continue
                score = float(sims[frame_index][det_index, student_index_map[sid]])
                if decision["status"] != "present" or decision["margin"] < min_margin:
                    # Keep the best review-band score so a student who never
                    # reaches ``present`` is still reported with real numbers
                    # instead of ``score: 0.0``.
                    review_frames[sid].add(frame_index)
                    prev_score, _ = review_best.get(sid, (-1.0, 0.0))
                    if score > prev_score:
                        review_best[sid] = (score, float(quality or 0.0))
                    continue
                record = student_status[sid]
                if score > record["score"]:
                    record["score"] = round(score, 4)
                    record["quality"] = round(float(quality or 0.0), 4)
                accepted_frames[sid].add(frame_index)
        for sid, record in student_status.items():
            record["frames_hit"] = len(accepted_frames[sid])
            if record["score"] >= accept_threshold:
                record["status"] = "present"
            elif record["frames_hit"] or review_frames[sid]:
                record["status"] = "needs_review"
                if not record["frames_hit"] and sid in review_best:
                    r_score, r_quality = review_best[sid]
                    record["score"] = round(r_score, 4)
                    record["quality"] = round(r_quality, 4)
                    record["frames_hit"] = len(review_frames[sid])
    elif mode == "quality_weighted_embedding_cluster":
        # Cluster mode decides per *track*, not per detection, so every detection
        # starts unmatched and inherits its track's decision below. Detections
        # filtered out by ``min_quality`` (or left unclustered) stay unmatched.
        per_frame = [
            [
                {
                    "det_index": det_index,
                    "matched_student_id": None,
                    "status": "unmatched",
                    "score": 0.0,
                    "margin": 0.0,
                    "top_candidates": [],
                }
                for det_index in range(len(frame.get("detections", [])))
            ]
            for frame in frames
        ]
        tracks = (
            precomputed_tracks
            if precomputed_tracks is not None
            else _cluster_tracks(frames, min_quality=min_quality)
        )
        if precomputed_track_sim is not None:
            matrix = precomputed_track_sim
        else:
            track_embeddings = [track["embedding"] for track in tracks]
            matrix, _ = build_similarity_matrix(
                track_embeddings, student_gallery, template_strategy, top_k
            )
        decisions = assign_single_frame(
            matrix,
            student_ids,
            accept_threshold=accept_threshold,
            review_threshold=review_threshold,
            min_margin=min_margin,
            assignment_mode=assignment_mode,
        )
        for track_index, decision in enumerate(decisions):
            sid = decision["matched_student_id"]
            if sid is None or sid not in student_status:
                continue
            record = student_status[sid]
            if decision["score"] > record["score"]:
                record["score"] = decision["score"]
                record["quality"] = round(float(tracks[track_index]["quality"] or 0.0), 4)
            record["frames_hit"] = max(record["frames_hit"], tracks[track_index]["count"])
            if decision["status"] == "present":
                record["status"] = "present"
            elif decision["status"] == "needs_review" and record["status"] == "absent":
                record["status"] = "needs_review"
        # Rewrite per-frame decisions so every detection inherits its track's
        # final status (annotation and UI colours). ``margin`` and
        # ``top_candidates`` are copied too, otherwise the UI inspector and
        # ``evaluate_frames`` read a stale candidate list.
        for track_index, track in enumerate(tracks):
            if track_index >= len(decisions):
                continue
            track_decision = decisions[track_index]
            for member in track["members"]:
                frame_index, det_index = member
                frame_decisions = per_frame[frame_index]
                if det_index < len(frame_decisions):
                    for field in (
                        "status",
                        "matched_student_id",
                        "score",
                        "margin",
                        "top_candidates",
                    ):
                        frame_decisions[det_index][field] = track_decision[field]
    else:
        raise ValueError(f"Unknown multi-frame fusion mode: {mode}")

    return {
        "mode": mode,
        "students": student_status,
        "per_frame": per_frame,
    }


def _cluster_tracks(frames: list[dict], *, min_quality: float = 0.0) -> list[dict]:
    """Greedy cross-frame track clustering (spatial IoU + cosine >= 0.55).

    Quality weights each detection's contribution to the track centroid.
    Unmeasured quality (``None``) is kept distinct and weighted at 1.0, and
    detections below ``min_quality`` are dropped before clustering. The centroid
    is maintained as an *unnormalized* weighted sum so the reduced magnitude of
    non-collinear sums is not discarded (which used to over-weight earlier
    frames and make clustering order-dependent).
    """
    tracks: list[dict] = []
    for frame_index, frame in enumerate(frames):
        detections = frame.get("detections", [])
        width = float(frame.get("width") or 0)
        height = float(frame.get("height") or 0)
        for det_index, det in enumerate(detections):
            raw_quality = det.get("quality")
            if raw_quality is not None and float(raw_quality) < min_quality:
                continue
            quality = float(raw_quality) if raw_quality is not None else None
            weight = max(quality, 1e-3) if quality is not None else 1.0
            box = _normalized_box(np.asarray(det["bbox"], dtype=np.float32), width, height)
            embedding = l2_normalize(np.asarray(det["embedding"], dtype=np.float32).reshape(-1))

            best_track: dict | None = None
            best_score = -1.0
            for track in tracks:
                if frame_index in track["frame_indices"]:
                    # One face per frame per track.
                    continue
                cos = float(np.dot(track["centroid"], embedding))
                if cos < 0.55:
                    continue
                if box_iou(track["box"], box) < 0.3:
                    continue
                if cos > best_score:
                    best_score = cos
                    best_track = track

            if best_track is None:
                tracks.append(
                    {
                        "weighted_sum": embedding * weight,
                        "centroid": embedding.copy(),
                        "box_sum": box.copy(),
                        "box": box.copy(),
                        "weight": weight,
                        "quality": quality,
                        "count": 1,
                        "members": [(frame_index, det_index)],
                        "frame_indices": {frame_index},
                    }
                )
            else:
                best_track["weight"] += weight
                best_track["weighted_sum"] = best_track["weighted_sum"] + embedding * weight
                best_track["centroid"] = l2_normalize(best_track["weighted_sum"])
                if quality is not None:
                    best_track["quality"] = (
                        quality
                        if best_track["quality"] is None
                        else max(best_track["quality"], quality)
                    )
                best_track["count"] += 1
                best_track["members"].append((frame_index, det_index))
                best_track["frame_indices"].add(frame_index)
                best_track["box_sum"] = best_track["box_sum"] + box
                best_track["box"] = best_track["box_sum"] / best_track["count"]

    for track in tracks:
        track["embedding"] = track["centroid"]
    return tracks


# --------------------------------------------------------------------------- #
# Attendance metrics
# --------------------------------------------------------------------------- #
def attendance_metrics(
    predicted_student_ids: Sequence[Any], ground_truth_student_ids: Sequence[Any]
) -> dict:
    """Confusion counts and precision/recall/F1 for one attendance decision."""
    predicted = set(predicted_student_ids)
    truth = set(ground_truth_student_ids)
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
