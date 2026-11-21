"""Regression tests for ONNX execution-provider initialization fallbacks."""

from __future__ import annotations

import pytest

from facelab import face


def test_gpu_initialization_failure_retries_with_cpu(monkeypatch):
    attempts: list[list[str]] = []
    monkeypatch.setattr(
        face, "_evict_idle_models", lambda keep_ids: {"total": 0}
    )

    def load(providers):
        attempts.append(providers)
        if providers != ["CPUExecutionProvider"]:
            raise RuntimeError("CUBLAS failure 3: the resource allocation failed")
        return "cpu-session"

    session = face._with_cpu_fallback(
        load,
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "test-recognizer",
    )

    assert session == "cpu-session"
    assert attempts == [
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]


def test_gpu_retry_after_evicting_idle_models(monkeypatch):
    attempts: list[list[str]] = []
    evicted: list[set[str]] = []

    def evict(keep_ids):
        evicted.append(keep_ids)
        return {"total": 3}

    monkeypatch.setattr(face, "_evict_idle_models", evict)

    def load(providers):
        attempts.append(providers)
        if len(attempts) == 1:
            raise RuntimeError("CUBLAS failure 3: the resource allocation failed")
        return "cuda-session"

    session = face._with_cpu_fallback(
        load,
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "test-recognizer",
    )

    # Eviction made room, so the model stays on the GPU instead of dropping to CPU.
    assert session == "cuda-session"
    assert attempts == [
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
    ]
    assert evicted == [{"test-recognizer"}]


def test_explicit_gpu_only_failure_is_not_hidden_by_cpu_fallback(monkeypatch):
    attempts = 0
    monkeypatch.setattr(
        face, "_evict_idle_models", lambda keep_ids: {"total": 0}
    )

    def load(_providers):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("CUDA initialization failed")

    with pytest.raises(RuntimeError, match="CUDA initialization failed"):
        face._with_cpu_fallback(load, ["CUDAExecutionProvider"], "test-recognizer")

    assert attempts == 1


def test_unload_other_models_releases_idle_caches(monkeypatch):
    from collections import OrderedDict

    monkeypatch.setattr(
        face,
        "_DETECTORS",
        OrderedDict(
            {(name, (640, 640), 0.35, 0.45): object() for name in ("a", "b")}
        ),
    )
    monkeypatch.setattr(
        face, "_RECOGNIZERS", {(name, ("CPU",)): object() for name in ("a", "b")}
    )
    monkeypatch.setattr(face, "_QUALITY", {"q": object()})
    monkeypatch.setattr(face, "_POSE", {"p": object()})

    result = face.unload_other_models("a", "a")

    assert list(face._DETECTORS) == [("a", (640, 640), 0.35, 0.45)]
    assert list(face._RECOGNIZERS) == [("a", ("CPU",))]
    assert face._QUALITY == {} and face._POSE == {}
    assert result["detectors"] == 1
    assert result["recognizers"] == 1
    assert result["total"] == 4


def test_uniface_recognizer_retries_session_on_cpu(monkeypatch):
    from uniface import recognition

    attempts: list[list[str]] = []

    class FakeSphereFace:
        def __init__(self, *, model_name, providers):
            attempts.append(providers)
            if providers != ["CPUExecutionProvider"]:
                raise RuntimeError("Failed to initialize model session")
            self.providers = providers

    monkeypatch.setattr(recognition, "SphereFace", FakeSphereFace)
    recognizer = face.UnifaceRecognizer(
        "uniface:sphereface",
        {"uniface_weights": "sphere20", "embedding_dim": 512},
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    assert recognizer.model.providers == ["CPUExecutionProvider"]
    assert attempts == [
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]


def test_warm_up_runs_recognizer_even_when_blank_frame_has_no_faces(monkeypatch):
    calls: list[str] = []

    class FakeRecognizer:
        def warm_up(self):
            calls.append("recognizer")

    monkeypatch.setattr(face, "analyze_frame", lambda *args, **kwargs: calls.append("frame"))
    monkeypatch.setattr(face, "get_recognizer", lambda recognizer_id: FakeRecognizer())

    face.warm_up("test-detector", "test-recognizer", 640)

    assert calls == ["frame", "recognizer"]


def test_warm_up_does_not_hide_provider_failures(monkeypatch):
    monkeypatch.setattr(
        face,
        "analyze_frame",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("CUDA initialization failed")),
    )

    with pytest.raises(face.FaceEngineUnavailable, match="Model warm-up failed"):
        face.warm_up("test-detector", "test-recognizer", 640)
