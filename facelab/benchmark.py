"""Automated calibration and metrics engine.

Similarity distributions differ wildly across backbones, so comparing every
model at a fixed ``0.5`` threshold is meaningless. This module calibrates each
model on a ground-truth dataset and reports:

* the threshold that maximises attendance F1 (``tau_f1``);
* the lowest threshold with zero false attendances (``tau_far0``);
* the lowest threshold with precision >= 0.98 (``tau_prec98``);
* detection and identification recall stratified by face-size bucket;
* latency percentiles from the cached inference timings.

Everything here runs on cached embeddings (pure NumPy/SciPy) and never loads an
ONNX model itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from facelab import cache, datasets, matching

# --------------------------------------------------------------------------- #
# Galleries
# --------------------------------------------------------------------------- #
def build_gallery(
    students: dict[int, dict],
    recognizer_id: str,
    variant: str = "flip_avg",
    ensemble: bool = False,
) -> dict[int, np.ndarray]:
    """Collect L2-normalized templates per student for one recognizer.

    ``variant`` selects which stored embedding to use (``raw``, ``flip_avg``,
    ``lowres_aug``). ``ensemble`` stacks every available variant as extra
    templates for the same student.
    """
    gallery: dict[int, np.ndarray] = {}
    for student_id, person in students.items():
        vectors: list[np.ndarray] = []
        for photo in person.get("photos", []):
            entry = (photo.get("embeddings") or {}).get(recognizer_id)
            if not entry:
                continue
            if ensemble:
                candidates = [
                    entry.get(name)
                    for name in ("raw", "flip_avg", "lowres_aug")
                    if entry.get(name) is not None
                ]
            else:
                candidate = entry.get(variant)
                if candidate is None:
                    candidate = entry.get("raw")
                candidates = [candidate]
            for vector in candidates:
                if vector is None:
                    continue
                array = np.asarray(vector, dtype=np.float32).reshape(-1)
                if array.size:
                    vectors.append(matching.l2_normalize(array))
        if vectors:
            gallery[int(student_id)] = np.vstack(vectors)
    return gallery


_EMBEDDING_FIELDS = {
    "raw": "embedding",
    "flip_avg": "embedding_flip_avg",
    "lowres_aug": "embedding_lowres_aug",
}


def select_embedding(detection: dict, variant: str = "flip_avg") -> np.ndarray:
    """Pick the detection embedding to score with, falling back to raw."""
    if variant == "lowres_aug":
        # Classroom probes are already low-resolution; don't degrade them a
        # second time with the low-res augmentation. Use the clean/flip vector.
        keys = ("embedding_flip_avg", "embedding")
    else:
        keys = (_EMBEDDING_FIELDS.get(variant, "embedding"), "embedding")
    for key in keys:
        value = detection.get(key)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size:
            return matching.l2_normalize(array)
    return np.zeros(512, dtype=np.float32)


def _fusion_frames(frames: Sequence[dict], variant: str) -> list[dict]:
    output = []
    for frame in frames:
        detections = []
        for det in frame.get("detections", []):
            detections.append(
                {
                    "embedding": select_embedding(det, variant),
                    "bbox": det.get("bbox"),
                    "quality": det.get("quality_score") or 0.0,
                }
            )
        output.append(
            {
                "frame_id": frame.get("frame_id"),
                "width": frame.get("width") or frame.get("image_width") or 1,
                "height": frame.get("height") or frame.get("image_height") or 1,
                "detections": detections,
            }
        )
    return output


# --------------------------------------------------------------------------- #
# Detection / identification recall
# --------------------------------------------------------------------------- #
def _iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a[:4])
    bx1, by1, bx2, by2 = (float(v) for v in box_b[:4])
    xx1, yy1 = max(ax1, bx1), max(ay1, by1)
    xx2, yy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def stratified_metrics(
    frames: Sequence[dict],
    decisions: Sequence[Sequence[dict]],
    annotated_boxes: Sequence[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """Detection and identification recall split into size buckets."""
    buckets = {
        name: {"gt": 0, "detected": 0, "identified": 0}
        for name in ("large", "medium", "small")
    }
    boxes_by_frame: dict[str, list[dict]] = {}
    for box in annotated_boxes:
        boxes_by_frame.setdefault(box.get("frame_id"), []).append(box)

    has_boxes = bool(annotated_boxes)
    for frame_index, frame in enumerate(frames):
        frame_id = frame.get("frame_id")
        gt_boxes = boxes_by_frame.get(frame_id, [])
        if not gt_boxes:
            continue
        detections = frame.get("detections", [])
        frame_decisions = decisions[frame_index] if frame_index < len(decisions) else []
        used = set()
        for gt in gt_boxes:
            bucket = gt.get("row_bucket") or "medium"
            if bucket not in buckets:
                bucket = "medium"
            buckets[bucket]["gt"] += 1
            best_index, best_iou = None, 0.0
            for index, det in enumerate(detections):
                if index in used:
                    continue
                score = _iou(gt["bbox"], det.get("bbox"))
                if score > best_iou:
                    best_iou, best_index = score, index
            if best_index is None or best_iou < iou_threshold:
                continue
            used.add(best_index)
            buckets[bucket]["detected"] += 1
            decision = frame_decisions[best_index] if best_index < len(frame_decisions) else None
            matched = decision.get("matched_student_id") if decision else None
            if (
                decision
                and decision.get("status") == "present"
                and matched is not None
                and gt.get("student_id") is not None
                and str(matched) == str(gt.get("student_id"))
            ):
                buckets[bucket]["identified"] += 1

    for value in buckets.values():
        value["detection_recall"] = round(
            value["detected"] / value["gt"], 4
        ) if value["gt"] else 0.0
        value["identification_recall"] = round(
            value["identified"] / value["gt"], 4
        ) if value["gt"] else 0.0

    total_gt = sum(value["gt"] for value in buckets.values())
    total_detected = sum(value["detected"] for value in buckets.values())
    total_identified = sum(value["identified"] for value in buckets.values())
    return {
        "has_annotations": has_boxes,
        "gt_faces": total_gt,
        "detected_faces": total_detected,
        "identified_faces": total_identified,
        "detection_recall": round(total_detected / total_gt, 4) if total_gt else 0.0,
        "identification_recall": round(total_identified / total_gt, 4) if total_gt else 0.0,
        "buckets": buckets,
    }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_frames(
    frames: Sequence[dict],
    gallery: dict[int, np.ndarray],
    present_student_ids: Sequence[int],
    *,
    variant: str = "flip_avg",
    accept_threshold: float = 0.42,
    review_threshold: float = 0.34,
    min_margin: float = 0.04,
    assignment_mode: str = "hungarian",
    template_strategy: str = "top_k_mean",
    top_k: int = 2,
    fusion_mode: str = "k_of_n_voting",
    k: int = 2,
    min_quality: float = 0.0,
    annotated_boxes: Sequence[dict] | None = None,
) -> dict:
    """Score one dataset configuration end to end (pure NumPy, no inference)."""
    fusion_frames = _fusion_frames(frames, variant)
    fused = matching.fuse_multi_frame(
        fusion_frames,
        gallery,
        mode=fusion_mode,
        accept_threshold=accept_threshold,
        review_threshold=review_threshold,
        min_margin=min_margin,
        assignment_mode=assignment_mode,
        template_strategy=template_strategy,
        top_k=top_k,
        k=k,
        min_quality=min_quality,
    )
    predicted = [
        student_id
        for student_id, record in fused["students"].items()
        if record["status"] == "present"
    ]
    attendance = matching.attendance_metrics(predicted, present_student_ids)
    detection = stratified_metrics(frames, fused["per_frame"], annotated_boxes or [])
    return {
        "attendance": attendance,
        "predicted_student_ids": sorted(int(value) for value in predicted),
        "students": fused["students"],
        "detection": detection,
        "per_frame": fused["per_frame"],
        "faces_detected": int(sum(len(frame.get("detections", [])) for frame in frames)),
    }


def threshold_curve(
    frames: Sequence[dict],
    gallery: dict[int, np.ndarray],
    present_student_ids: Sequence[int],
    **kwargs,
) -> list[dict]:
    """Sweep ``accept_threshold`` and record attendance confusion per step."""
    return threshold_curve_datasets(
        [(frames, present_student_ids)], gallery, **kwargs
    )


def threshold_curve_datasets(
    datasets: Sequence[tuple[Sequence[dict], Sequence[int]]],
    gallery: dict[int, np.ndarray],
    *,
    variant: str = "flip_avg",
    low: float = 0.15,
    high: float = 0.75,
    step: float = 0.01,
    review_ratio: float = 0.8,
    review_threshold: float | None = None,
    min_quality: float = 0.0,
    min_margin: float = 0.04,
    assignment_mode: str = "hungarian",
    template_strategy: str = "top_k_mean",
    top_k: int = 2,
    fusion_mode: str = "k_of_n_voting",
    k: int = 2,
) -> list[dict]:
    """Sweep ``accept_threshold`` across several datasets, aggregating per step.

    Evaluating datasets separately avoids the frame-id collisions and mixed
    present-rosters you get from concatenating every dataset's frames into one
    call.
    """
    rows = []
    threshold = low
    while threshold <= high + 1e-9:
        review = (
            min(review_threshold, threshold)
            if review_threshold is not None
            else round(threshold * review_ratio, 4)
        )
        results = [
            evaluate_frames(
                frames,
                gallery,
                present_ids,
                variant=variant,
                accept_threshold=threshold,
                review_threshold=review,
                min_margin=min_margin,
                assignment_mode=assignment_mode,
                template_strategy=template_strategy,
                top_k=top_k,
                fusion_mode=fusion_mode,
                k=k,
                min_quality=min_quality,
            )
            for frames, present_ids in datasets
        ]
        rows.append({"threshold": round(threshold, 4), **aggregate_attendance(results)})
        threshold = round(threshold + step, 4)
    return rows


def pick_operating_points(curve: list[dict]) -> dict:
    """Reduce a threshold curve to the three operating points we report."""
    if not curve:
        return {"best_f1": None, "far0": None, "prec98": None}

    best_f1 = max(curve, key=lambda row: (row["f1"], row["recall"], -row["threshold"]))

    far0_candidates = [row for row in curve if row["fp"] == 0 and row["tp"] > 0]
    far0 = min(far0_candidates, key=lambda row: row["threshold"]) if far0_candidates else None

    prec_candidates = [row for row in curve if row["precision"] >= 0.98 and row["tp"] > 0]
    prec98 = (
        min(prec_candidates, key=lambda row: row["threshold"]) if prec_candidates else None
    )
    return {"best_f1": best_f1, "far0": far0, "prec98": prec98}


def aggregate_attendance(results: Sequence[dict]) -> dict:
    """Sum confusion counts across datasets and recompute precision/recall/F1."""
    tp = sum(item["attendance"]["tp"] for item in results)
    fp = sum(item["attendance"]["fp"] for item in results)
    fn = sum(item["attendance"]["fn"] for item in results)
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


def aggregate_detection(results: Sequence[dict]) -> dict:
    """Merge per-dataset stratified detection counts into one report."""
    buckets = {
        name: {"gt": 0, "detected": 0, "identified": 0}
        for name in ("large", "medium", "small")
    }
    for result in results:
        for name, value in (result.get("buckets") or {}).items():
            bucket = buckets.setdefault(name, {"gt": 0, "detected": 0, "identified": 0})
            bucket["gt"] += int(value.get("gt") or 0)
            bucket["detected"] += int(value.get("detected") or 0)
            bucket["identified"] += int(value.get("identified") or 0)

    for value in buckets.values():
        value["detection_recall"] = (
            round(value["detected"] / value["gt"], 4) if value["gt"] else 0.0
        )
        value["identification_recall"] = (
            round(value["identified"] / value["gt"], 4) if value["gt"] else 0.0
        )
    total_gt = sum(value["gt"] for value in buckets.values())
    total_detected = sum(value["detected"] for value in buckets.values())
    total_identified = sum(value["identified"] for value in buckets.values())
    return {
        "has_annotations": any(result.get("has_annotations") for result in results),
        "gt_faces": total_gt,
        "detected_faces": total_detected,
        "identified_faces": total_identified,
        "detection_recall": round(total_detected / total_gt, 4) if total_gt else 0.0,
        "identification_recall": round(total_identified / total_gt, 4) if total_gt else 0.0,
        "buckets": buckets,
    }


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #
def latency_percentiles(timings: Sequence[dict]) -> dict:
    """p50/p95 for each stage plus total inference time."""
    if not timings:
        return {}

    def percentile(values: Sequence[float], q: float) -> float:
        if not values:
            return 0.0
        return float(np.percentile(np.asarray(values, dtype=np.float64), q))

    stages = ("decode_ms", "detect_ms", "align_embed_ms", "quality_pose_ms", "match_ms",
              "annotate_ms")
    output: dict[str, Any] = {}
    totals = [
        sum(float(item.get(stage) or 0.0) for stage in ("detect_ms", "align_embed_ms", "match_ms"))
        for item in timings
    ]
    for stage in stages:
        values = [float(item.get(stage) or 0.0) for item in timings]
        if any(value > 0 for value in values):
            output[f"{stage}_p50"] = round(percentile(values, 50), 3)
            output[f"{stage}_p95"] = round(percentile(values, 95), 3)
    output["total_inference_ms_p50"] = round(percentile(totals, 50), 3)
    output["total_inference_ms_p95"] = round(percentile(totals, 95), 3)
    output["passes"] = len(timings)
    return output


# --------------------------------------------------------------------------- #
# Cached frame loading
# --------------------------------------------------------------------------- #
def load_cached_frames(
    dataset_id: str,
    *,
    detector_id: str,
    recognizer_id: str,
    det_params: dict,
    flip_aug: bool = True,
) -> tuple[list[dict], int]:
    """Load cached analysis artifacts for a dataset.

    Returns ``(frames, missing)`` where ``missing`` counts frames with no cache
    artifact for this exact configuration.
    """
    manifest = datasets.load_manifest(dataset_id)
    if manifest is None:
        raise FileNotFoundError(f"Dataset '{dataset_id}' not found")
    frames: list[dict] = []
    missing = 0
    for frame in manifest["frames"]:
        key = cache.cache_key(
            frame["sha256"], detector_id, det_params, recognizer_id, flip_aug
        )
        payload = cache.load_analysis(key)
        if payload is None:
            missing += 1
            continue
        frames.append(
            {
                "frame_id": frame["frame_id"],
                "path": str(datasets.dataset_dir(dataset_id) / frame["filename"]),
                "width": frame["width"],
                "height": frame["height"],
                "detections": payload["detections"],
                "timings": payload["timings_ms"],
                "cached": True,
            }
        )
    return frames, missing
