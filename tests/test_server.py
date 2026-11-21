"""FastAPI TestClient & persistence integration tests (stubbed ONNX engine)."""

from __future__ import annotations

import base64
import io
import json
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import server
from facelab import cache as cache_store
from facelab import face as engine
from facelab.config import settings


def _jpeg_bytes(
    width: int = 160,
    height: int = 160,
    color=(120, 140, 160),
    orientation: int | None = None,
) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    if orientation is not None:
        exif = Image.Exif()
        exif[274] = orientation
        img.save(buf, format="JPEG", exif=exif)
    else:
        img.save(buf, format="JPEG")
    return buf.getvalue()


class _StubDetector:
    def __init__(self, detector_id: str, det_size: tuple[int, int]):
        self.id = detector_id
        self.label = detector_id
        self.det_size = det_size
        self.active_providers = ["CPUExecutionProvider"]
        self.calls = 0

    def detect(self, image_bgr: np.ndarray):
        self.calls += 1
        h, w = image_bgr.shape[:2]
        # Blank warm-up frames (constant 127) have no faces.
        if int(image_bgr.min()) == int(image_bgr.max()) == 127:
            return []
        kps = np.array(
            [
                [w * 0.35, h * 0.35],
                [w * 0.65, h * 0.35],
                [w * 0.50, h * 0.50],
                [w * 0.38, h * 0.68],
                [w * 0.62, h * 0.68],
            ],
            dtype=np.float32,
        )
        return [
            engine.Detection(
                bbox=np.array([w * 0.15, h * 0.15, w * 0.85, h * 0.85], dtype=np.float32),
                score=0.95,
                kps=kps,
            )
        ]

    def warm_up(self):
        return None


class _StubRecognizer:
    embedding_dim = 8

    def __init__(self, recognizer_id: str):
        self.id = recognizer_id
        self.label = recognizer_id
        self.active_providers = ["CPUExecutionProvider"]

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        vectors = []
        for crop in crops:
            mean_bgr = crop.mean(axis=(0, 1)).astype(np.float32) / 255.0
            vec = np.zeros(self.embedding_dim, dtype=np.float32)
            vec[:3] = mean_bgr + 0.05
            vec[3] = 1.0
            vectors.append(vec)
        return np.vstack(vectors)

    def warm_up(self):
        return None


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Isolated runtime directory, empty roster, and stubbed detectors/recognizers."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "api_token", "")
    settings.ensure_dirs()

    monkeypatch.setattr(server, "_people", {})
    monkeypatch.setattr(server, "_next_number", 1)
    monkeypatch.setattr(server, "_MANIFEST_CACHE", (None, {}))

    detectors: dict[tuple, _StubDetector] = {}
    recognizers: dict[str, _StubRecognizer] = {}
    available_recs = {"insightface:w600k_r50", "insightface:glintr100"}
    available_dets = {"insightface:scrfd_10g_bnkps", "uniface:yolov8n_face"}

    def det_avail(detector_id: str):
        ok = detector_id in available_dets
        return (ok, "" if ok else "stub unavailable")

    def rec_avail(recognizer_id: str):
        ok = recognizer_id in available_recs
        return (ok, "" if ok else "stub unavailable")

    def get_det(detector_id=None, det_size=None, det_thresh=0.35, nms_thresh=0.45):
        did = detector_id or settings.default_detector
        size = (det_size, det_size) if isinstance(det_size, int) else (det_size or (640, 640))
        key = (did, size)
        detectors.setdefault(key, _StubDetector(did, size))
        return detectors[key]

    def get_rec(recognizer_id=None):
        rid = recognizer_id or settings.default_recognizer
        recognizers.setdefault(rid, _StubRecognizer(rid))
        return recognizers[rid]

    monkeypatch.setattr(engine, "detector_availability", det_avail)
    monkeypatch.setattr(engine, "recognizer_availability", rec_avail)
    monkeypatch.setattr(engine, "get_detector", get_det)
    monkeypatch.setattr(engine, "get_recognizer", get_rec)
    monkeypatch.setattr(engine, "get_quality_predictor", lambda: None)
    monkeypatch.setattr(engine, "get_head_pose_predictor", lambda: None)
    monkeypatch.setattr(engine, "unload_other_models", lambda *a, **k: {"total": 0})
    monkeypatch.setattr(
        engine,
        "align_face",
        lambda image_bgr, kps, size=112, antialias=True: np.resize(image_bgr, (size, size, 3)),
    )
    monkeypatch.setattr(engine, "_project_landmarks_to_crop", lambda kps, size=112: None)

    return {
        "client": TestClient(server.app),
        "detectors": detectors,
        "recognizers": recognizers,
        "tmp_path": tmp_path,
    }


def test_enroll_scan_and_live_sweep_roundtrip(env):
    client: TestClient = env["client"]
    alice_jpg = _jpeg_bytes(180, 180, color=(200, 40, 40))

    res = client.post(
        "/api/people",
        data={"name": "Alice", "recognizer": "insightface:w600k_r50"},
        files=[("images", ("alice.jpg", alice_jpg, "image/jpeg"))],
    )
    assert res.status_code == 200, res.text
    person = res.json()["person"]
    assert person["name"] == "Alice"
    assert settings.students_file.is_file()

    # A different (but same-face) frame: similar enough for the first scan, but
    # not above the high threshold used for the live sweep.
    class_jpg = _jpeg_bytes(180, 180, color=(40, 40, 200))
    scan_res = client.post(
        "/api/scan",
        data={
            "detector": "insightface:scrfd_10g_bnkps",
            "recognizer": "insightface:w600k_r50",
            "detSize": "640",
            "acceptThreshold": "0.5",
            "reviewThreshold": "0.3",
            "minQuality": "0.8",
            "fusionK": "1",
        },
        files=[("images", ("class.jpg", class_jpg, "image/jpeg"))],
    )
    assert scan_res.status_code == 200, scan_res.text
    scan = scan_res.json()
    assert scan["summary"]["matchedCount"] == 1
    assert scan["people"][0]["status"] == "present"

    det = env["detectors"][("insightface:scrfd_10g_bnkps", (640, 640))]
    calls_before_sweep = det.calls

    sweep_res = client.post(
        f"/api/results/{scan['scanId']}/sweep",
        json={
            "detector": "insightface:scrfd_10g_bnkps",
            "recognizer": "insightface:w600k_r50",
            "det_size": 640,
            "accept_threshold": 0.9,
            "review_threshold": 0.9,
            "min_quality": 0.8,
            "k": 1,
        },
    )
    assert sweep_res.status_code == 200, sweep_res.text
    assert sweep_res.json()["matchedCount"] == 0
    # Live sweep must reuse the inference cache without running the detector.
    assert det.calls == calls_before_sweep


def test_reembed_releases_state_lock_during_inference(env, monkeypatch):
    client: TestClient = env["client"]
    res = client.post(
        "/api/people",
        data={"name": "Bob", "recognizer": "insightface:w600k_r50"},
        files=[("images", ("bob.jpg", _jpeg_bytes(), "image/jpeg"))],
    )
    assert res.status_code == 200, res.text

    real_embed_ref = engine.embed_reference
    observed = {"state_lock_free": False, "model_lock_held": False}

    def spy_embed_reference(*args, **kwargs):
        observed["model_lock_held"] = server._model_lock.locked()
        acquired = [False]

        def probe():
            if server._state_lock.acquire(blocking=False):
                acquired[0] = True
                server._state_lock.release()

        t = threading.Thread(target=probe)
        t.start()
        t.join()
        observed["state_lock_free"] = acquired[0]
        return real_embed_ref(*args, **kwargs)

    monkeypatch.setattr(engine, "embed_reference", spy_embed_reference)

    re_res = client.post(
        "/api/students/re-embed",
        json={"recognizer_ids": ["insightface:glintr100"]},
    )
    assert re_res.status_code == 200, re_res.text
    assert re_res.json()["results"]["insightface:glintr100"]["updated"] == 1
    assert observed == {"state_lock_free": True, "model_lock_held": True}


def test_api_token_guards_mutations_and_model_endpoint(env, monkeypatch):
    client: TestClient = env["client"]
    monkeypatch.setattr(settings, "api_token", "top-secret")

    assert client.get("/api/health").status_code == 200
    assert (
        client.get(
            "/api/model?detector=insightface:scrfd_10g_bnkps&recognizer=insightface:w600k_r50"
        ).status_code
        == 401
    )
    assert client.post("/api/reset").status_code == 401
    assert client.post("/api/reset", headers={"X-API-Token": "wrong"}).status_code == 401

    ok_model = client.get(
        "/api/model?detector=insightface:scrfd_10g_bnkps&recognizer=insightface:w600k_r50",
        headers={"Authorization": "Bearer top-secret"},
    )
    assert ok_model.status_code == 200, ok_model.text
    assert ok_model.json()["ready"] is True
    assert ok_model.json()["providers"] == ["CPUExecutionProvider"]
    assert client.post("/api/reset", headers={"X-API-Token": "top-secret"}).status_code == 200


def test_model_warmup_clamps_fixed_input_size_detector(env, monkeypatch):
    client: TestClient = env["client"]
    warmed_sizes: list[int] = []
    real_warm_up = engine.warm_up

    def spy_warm_up(detector_id=None, recognizer_id=None, det_size=None, *args, **kwargs):
        warmed_sizes.append(det_size)
        return real_warm_up(detector_id, recognizer_id, det_size, *args, **kwargs)

    monkeypatch.setattr(engine, "warm_up", spy_warm_up)
    res = client.get("/api/model?detector=uniface:yolov8n_face&recognizer=insightface:w600k_r50")
    assert res.status_code == 200, res.text
    assert res.json()["ready"] is True
    assert warmed_sizes == [640]


def test_dataset_exif_orientation_and_cache_first_benchmark(env):
    client: TestClient = env["client"]
    person_jpg = _jpeg_bytes(160, 160, color=(40, 200, 40))
    enroll = client.post(
        "/api/people",
        data={"name": "Cara", "recognizer": "insightface:w600k_r50"},
        files=[("images", ("cara.jpg", person_jpg, "image/jpeg"))],
    )
    assert enroll.status_code == 200, enroll.text
    sid = enroll.json()["person"]["id"]

    # 200x100 with EXIF orientation 6 (90° CW) -> stored/displayed size is 100x200.
    rotated = _jpeg_bytes(200, 100, color=(40, 200, 40), orientation=6)
    ds_res = client.post(
        "/api/datasets",
        data={"name": "Rotated Room", "presentIds": json.dumps([sid]), "boxes": "[]"},
        files=[("images", ("rot.jpg", rotated, "image/jpeg"))],
    )
    assert ds_res.status_code == 200, ds_res.text
    ds = ds_res.json()["dataset"]
    assert ds["frames"][0]["width"] == 100
    assert ds["frames"][0]["height"] == 200

    req = {
        "dataset_ids": [ds["datasetId"]],
        "detectors": ["insightface:scrfd_10g_bnkps"],
        "recognizers": ["insightface:w600k_r50"],
        "det_sizes": [640],
        "tiling_modes": ["none"],
        "k": 1,
    }
    first = client.post("/api/benchmark/run", json=req)
    assert first.status_code == 200, first.text
    row1 = first.json()["rows"][0]
    assert row1["cachedFrames"] == 0
    assert row1["providers"] == ["CPUExecutionProvider"]
    assert (settings.benchmarks_dir / f"{first.json()['runId']}.summary.json").is_file()

    det = env["detectors"][("insightface:scrfd_10g_bnkps", (640, 640))]
    calls_after_first = det.calls

    second = client.post("/api/benchmark/run", json=req)
    assert second.status_code == 200, second.text
    assert second.json()["rows"][0]["cachedFrames"] == 1
    # Only the warm-up blank pass runs (1 call); the cached frame is never re-analyzed.
    assert det.calls == calls_after_first + 1

    listing = client.get("/api/benchmarks")
    assert listing.status_code == 200
    assert len(listing.json()["benchmarks"]) == 2


def test_upload_size_cap_and_enum_validation(env, monkeypatch):
    client: TestClient = env["client"]
    bad_bench = client.post(
        "/api/benchmark/run",
        json={"dataset_ids": ["missing"], "fusion_mode": "not_a_mode"},
    )
    assert bad_bench.status_code == 422

    bad_scan = client.post(
        "/api/scan",
        data={"assignmentMode": "not_a_mode"},
        files=[("images", ("f.jpg", _jpeg_bytes(), "image/jpeg"))],
    )
    assert bad_scan.status_code == 422

    monkeypatch.setattr(server, "MAX_IMAGE_BYTES", 32)
    too_big = client.post(
        "/api/scan",
        files=[("images", ("huge.jpg", b"x" * 64, "image/jpeg"))],
    )
    assert too_big.status_code == 413


def test_cache_lru_pruning_and_legacy_migration(env):
    for idx in range(3):
        key = f"k{idx}"
        cache_store.save_analysis({"cache_key": key, "detections": [], "timings_ms": {}})
        (settings.crops_dir / f"{key}_det0.jpg").write_bytes(b"jpg")
        time.sleep(0.02)

    # Touch k0 so k1 becomes the least-recently-used entry.
    assert cache_store.load_analysis("k0") is not None
    removed = cache_store.prune_cache(keep=2)
    assert removed == 1
    assert cache_store.has_analysis("k0")
    assert not cache_store.has_analysis("k1")
    assert not (settings.crops_dir / "k1_det0.jpg").exists()
    assert cache_store.has_analysis("k2")

    # Legacy people.json migration -> students.json with None quality_score.
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    legacy = {
        "version": 2,
        "nextNumber": 2,
        "people": [
            {
                "id": 1,
                "number": 1,
                "name": "Legacy",
                "photos": [
                    {
                        "id": "p1",
                        "filename": "faces/p1/p1.jpg",
                        "model": "buffalo_l",
                        "embedding": base64.b64encode(vec.tobytes()).decode("ascii"),
                    }
                ],
            }
        ],
    }
    settings.legacy_people_file.write_text(json.dumps(legacy), encoding="utf-8")
    server._load_state()
    assert 1 in server._people
    photo = server._people[1]["photos"][0]
    assert photo["quality_score"] is None
    assert np.allclose(photo["embeddings"]["insightface:w600k_r50"]["raw"], vec)