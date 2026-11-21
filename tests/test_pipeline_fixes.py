"""Regression tests for detection, alignment, embedding and quality fixes.

No ONNX weights are required: detector/recognizer collaborators are stubbed.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from facelab import benchmark, face, matching


class FakeRecognizer:
    """Embedding that changes with image content so blur/flip are observable."""

    id = "fake:rec"
    embedding_dim = 4

    def __init__(self):
        self.batch_sizes: list[int] = []

    def embed(self, crops):
        self.batch_sizes.append(len(crops))
        vectors = []
        for crop in crops:
            array = crop.astype(np.float32)
            vectors.append(
                np.array(
                    [
                        array.mean(),
                        array.std(),
                        array[0].mean(),
                        array[-1].mean(),
                    ],
                    dtype=np.float32,
                )
                + 1.0
            )
        return np.vstack(vectors)


@pytest.fixture()
def crop():
    ramp = np.arange(112 * 112, dtype=np.uint8).reshape(112, 112)
    return np.dstack([ramp, ramp[::-1], np.roll(ramp, 17, axis=0)])


def test_embed_variants_lowres_template_is_degraded_and_flip_averaged(crop):
    recognizer = FakeRecognizer()
    variants = face.embed_variants(recognizer, crop)

    expected_lowres = matching.flip_averaged(
        recognizer.embed([face.lowres_augment(crop)])[0],
        recognizer.embed([face.lowres_augment(cv2.flip(crop, 1))])[0],
    )
    assert np.allclose(variants["lowres_aug"], expected_lowres)

    # The old enrolment blend mixed the clean vector into the template; the
    # stored variant is now purely the degraded (gallery) template.
    clean = variants["flip_avg"]
    blended = matching.flip_averaged(clean, recognizer.embed([face.lowres_augment(crop)])[0])
    assert not np.allclose(blended, variants["lowres_aug"])


def test_analyze_frame_never_embeds_a_lowres_probe(monkeypatch, crop):
    detections = [
        face.Detection(
            bbox=np.array([20, 20, 180, 180], dtype=np.float32),
            score=0.9,
            kps=np.array(
                [[60, 70], [140, 70], [100, 110], [70, 150], [130, 150]],
                dtype=np.float32,
            ),
        )
    ]

    class StubDetector:
        def detect(self, image):
            return detections

    recognizer = FakeRecognizer()
    monkeypatch.setattr(face, "get_detector", lambda *a, **k: StubDetector())
    monkeypatch.setattr(face, "get_recognizer", lambda *a, **k: recognizer)

    payload = face.analyze_frame(
        np.zeros((240, 240, 3), dtype=np.uint8),
        detector_id="stub",
        recognizer_id="stub",
        lowres_aug=True,
        with_quality=False,
        with_pose=False,
    )

    # Only the clean and mirrored passes run; lowres_aug used to cost a third
    # recognition pass whose result was never scored.
    assert recognizer.batch_sizes == [1, 1]
    assert payload["detections"][0]["embedding_lowres_aug"] is None
    assert payload["detections"][0]["quality_score"] is None
    assert payload["detections"][0]["quality_source"] == "unavailable"


def test_quality_score_has_no_recognizer_dependent_fallback():
    score, source = face._quality_score(None, np.zeros((112, 112, 3), np.uint8))
    assert score is None
    assert source == "unavailable"


def test_align_face_antialiases_large_downscales():
    from insightface.utils.face_align import arcface_dst

    image = np.zeros((900, 900, 3), dtype=np.uint8)
    image[::4, :, :] = 255
    kps = arcface_dst * 4

    aligned = face.align_face(image, kps)
    aliased = face.align_face(image, kps, antialias=False)

    # Averaged stripes keep the source brightness; point-sampled ones saturate.
    assert abs(float(aligned.mean()) - float(image.mean())) < 15
    assert float(aliased.mean()) > 200


class _RecoveryModel:
    def __init__(self, primary, recovery):
        self.primary = primary
        self.recovery = recovery
        self.sizes: list[tuple[int, int]] = []

    def detect(self, image, input_size=None):
        self.sizes.append(tuple(input_size))
        if max(input_size) <= 128:
            boxes = (
                np.array([[10, 10, 210, 210, 0.9]], dtype=np.float32)
                if self.recovery
                else np.empty((0, 5), dtype=np.float32)
            )
        else:
            boxes = (
                np.array([[10, 10, 110, 210, 0.9]], dtype=np.float32)
                if self.primary
                else np.empty((0, 5), dtype=np.float32)
            )
        kps = np.ones((len(boxes), 5, 2), dtype=np.float32)
        return boxes, kps


def _bare_scrfd(model, det_size=(1280, 1280)):
    detector = object.__new__(face.InsightFaceDetector)
    detector.model = model
    detector.det_size = det_size
    return detector


def test_scrfd_retries_closeups_at_a_small_input_size():
    model = _RecoveryModel(primary=False, recovery=True)
    detections = _bare_scrfd(model).detect(np.zeros((300, 300, 3), dtype=np.uint8))

    assert model.sizes == [(1280, 1280), (128, 128)]
    assert len(detections) == 1
    assert detections[0].height == pytest.approx(200)


def test_scrfd_skips_recovery_for_normal_frames():
    model = _RecoveryModel(primary=True, recovery=True)
    detections = _bare_scrfd(model).detect(np.zeros((1000, 1000, 3), dtype=np.uint8))

    # Detected face is 200 px of a 1000 px frame: no second pass needed.
    assert model.sizes == [(1280, 1280)]
    assert len(detections) == 1


def test_merge_detections_recovers_a_clipped_primary_box():
    clipped = face.Detection(
        bbox=np.array([10, 10, 110, 210], dtype=np.float32), score=0.9
    )
    recovered = face.Detection(
        bbox=np.array([10, 10, 210, 210], dtype=np.float32), score=0.8
    )
    merged = face.merge_detections([clipped], [recovered])
    assert len(merged) == 1
    assert merged[0].bbox[2] == pytest.approx(210)


class _FakeYuNet:
    def __init__(self, row):
        self.row = row
        self.input_size = None

    def setInputSize(self, size):
        self.input_size = size

    def detect(self, image):
        return 1, self.row


def test_yunet_honours_the_requested_detector_size():
    row = np.array(
        [[100, 50, 80, 80, 110, 70, 170, 70, 140, 100, 120, 150, 160, 150, 0.9]],
        dtype=np.float32,
    )
    detector = object.__new__(face.OpenCVYuNetDetector)
    detector.model = _FakeYuNet(row)
    detector.det_size = (640, 640)
    detector.id = "opencv:yunet_2023mar"

    detections = detector.detect(np.zeros((1000, 2000, 3), dtype=np.uint8))

    scale = 640 / 2000
    assert detector.model.input_size == (640, 320)
    assert detections[0].bbox == pytest.approx(
        np.array([100, 50, 180, 130], dtype=np.float32) / scale
    )
    assert detections[0].kps[0, 0] == pytest.approx(110 / scale)


def test_tile_merge_drops_a_clipped_fragment_below_the_iou_threshold():
    boxes = np.array(
        [
            [480, 60, 700, 280],
            [480, 60, 560, 280],
            [480, 60, 645, 280],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.85, 0.99, 0.97], dtype=np.float32)
    landmarks = np.ones((3, 5, 2), dtype=np.float32)
    truncated = np.array([False, True, True])

    kept_boxes, kept_scores, _ = face._merge_tiled_detections(
        boxes, scores, landmarks, truncated, iou_threshold=0.45
    )
    assert len(kept_boxes) == 1
    assert kept_boxes[0][2] == pytest.approx(700)
    assert kept_scores[0] == pytest.approx(0.85)


def test_tile_merge_orders_clipped_fragments_by_score():
    boxes = np.array(
        [[480, 60, 645, 280], [480, 60, 620, 280]], dtype=np.float32
    )
    scores = np.array([0.8, 0.9], dtype=np.float32)
    landmarks = np.ones((2, 5, 2), dtype=np.float32)
    truncated = np.array([True, True])

    kept_boxes, kept_scores, _ = face._merge_tiled_detections(
        boxes, scores, landmarks, truncated, iou_threshold=0.45
    )
    assert len(kept_boxes) == 1
    assert kept_scores[0] == pytest.approx(0.9)
    assert kept_boxes[0][2] == pytest.approx(620)


def test_hungarian_prefers_one_accepted_match_over_two_review_matches():
    matrix = np.array([[0.75, 0.69], [0.70, 0.40]], dtype=np.float32)
    decisions = matching.assign_single_frame(
        matrix,
        ["A", "B"],
        accept_threshold=0.72,
        review_threshold=0.34,
        min_margin=0.03,
        assignment_mode="hungarian",
    )
    assert decisions[0]["status"] == "present"
    assert decisions[0]["matched_student_id"] == "A"
    assert decisions[1]["matched_student_id"] == "B"
    assert decisions[1]["status"] == "needs_review"


def test_fusion_treats_unmeasured_quality_as_unknown_not_zero():
    frames = [
        {
            "detections": [
                {
                    "embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    "bbox": [100, 100, 180, 180],
                    "quality": None,
                }
            ]
        }
    ]
    fused = matching.fuse_multi_frame(
        frames,
        {1: np.array([[1.0, 0.0, 0.0]], dtype=np.float32)},
        mode="k_of_n_voting",
        accept_threshold=0.5,
        review_threshold=0.4,
        k=1,
        min_quality=0.5,
    )
    assert fused["students"][1]["status"] == "present"


def test_stratified_recall_does_not_depend_on_annotation_order():
    detections = [
        {"bbox": [1, 0, 11, 10], "embedding": np.array([1.0, 0.0])},
        {"bbox": [-3, 0, 7, 10], "embedding": np.array([1.0, 0.0])},
    ]
    frames = [
        {"frame_id": "f", "width": 1000, "height": 1000, "detections": detections}
    ]
    first = {"frame_id": "f", "bbox": [0, 0, 10, 10]}
    second = {"frame_id": "f", "bbox": [4, 0, 14, 10]}

    def detected(boxes):
        return benchmark.stratified_metrics(frames, [], boxes)["detected_faces"]

    assert detected([first, second]) == 2
    assert detected([second, first]) == 2

def test_detect_with_sahi_preserves_missing_keypoints():
    # 400x400 is smaller than the tile size, so the only tile is the full frame
    # and the base detection wins the merge with its keypoints intact.
    image = np.zeros((400, 400, 3), dtype=np.uint8)

    def fake_detect(_img):
        return [face.Detection(bbox=np.array([10, 10, 90, 90], dtype=np.float32), score=0.9, kps=None)]

    detections = face.detect_with_sahi(
        image, fake_detect, tile_size=640, tiling_mode="sahi_upper_band"
    )
    assert len(detections) == 1
    assert detections[0].kps is None


def test_mid_band_tiles_stay_in_band_and_preserve_missing_kps():
    height, width = 960, 1280
    # Tight tile size for a 1280x960 classroom frame.
    assert face._mid_band_tile_size(height, width, 640) == 403
    band_lo, band_hi = int(height * 0.20), int(height * 0.80)
    tiles = face.generate_tiles(height, width, tile_size=403, band=(0.20, 0.80))
    assert tiles
    for x0, y0, x1, y1 in tiles:
        assert 0 <= x0 < x1 <= width
        assert 0 <= y0 < y1 <= height
        # Every tile overlaps the seated-student band.
        assert y1 > band_lo and y0 < band_hi

    image = np.zeros((height, width, 3), dtype=np.uint8)

    def fake_detect(_img):
        return [face.Detection(bbox=np.array([100, 300, 180, 380], dtype=np.float32), score=0.9, kps=None)]

    detections = face.detect_with_sahi(image, fake_detect, tiling_mode="sahi_mid_band")
    assert detections
    assert all(det.kps is None for det in detections)
