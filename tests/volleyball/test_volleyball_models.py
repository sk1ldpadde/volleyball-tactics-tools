from pathlib import Path
from types import SimpleNamespace

import numpy as np
import supervision as sv

from examples.volleyball.detection import UltralyticsPlayerDetector
from examples.volleyball.models import (
    HardwareInfo,
    choose_auto_model,
    resolve_player_model,
    select_device,
)


class _FakeCuda:
    def __init__(self, available=False):
        self._available = available

    def is_available(self):
        return self._available


class _FakeMps(_FakeCuda):
    pass


def _fake_torch(cuda=False, mps=False):
    return SimpleNamespace(
        __version__="test",
        cuda=_FakeCuda(cuda),
        backends=SimpleNamespace(mps=_FakeMps(mps)),
    )


class _DownloadingYolo:
    calls = []

    def __init__(self, path):
        self.calls.append(path)
        Path(path).write_bytes(b"official test weights")


def test_auto_policy_uses_current_compatible_generation_and_hardware() -> None:
    assert choose_auto_model("8.4.0", "cpu") == "yolo26n.pt"
    assert choose_auto_model("8.4.163", "cuda:0") == "yolo26s.pt"
    assert choose_auto_model("8.4.163", "mps") == "yolo26s.pt"
    assert choose_auto_model("8.3.99", "cpu") == "yolo11n.pt"
    assert select_device("auto", HardwareInfo("x", True, False)) == "cuda:0"
    assert select_device("auto", HardwareInfo("x", False, True)) == "mps"
    assert select_device("auto", HardwareInfo("x", False, False)) == "cpu"


def test_official_model_download_and_cache(tmp_path) -> None:
    _DownloadingYolo.calls.clear()

    first = resolve_player_model(
        "auto", models_dir=tmp_path, torch_module=_fake_torch(),
        yolo_class=_DownloadingYolo, ultralytics_version="8.4.0")
    second = resolve_player_model(
        "auto", models_dir=tmp_path, torch_module=_fake_torch(),
        yolo_class=_DownloadingYolo, ultralytics_version="8.4.0")

    assert first.path == tmp_path.resolve() / "yolo26n.pt"
    assert first.source_type == "official"
    assert second.path == first.path
    assert len(_DownloadingYolo.calls) == 1


class _Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _FakePredictionModel:
    task = "detect"
    names = {0: "person", 1: "bicycle"}

    def predict(self, *_args, **_kwargs):
        self.predict_kwargs = _kwargs
        boxes = SimpleNamespace(
            xyxy=_Tensor([[10.0, 20.0, 30.0, 80.0]]),
            conf=_Tensor([0.9]),
            cls=_Tensor([0]),
            id=None,
        )
        return [SimpleNamespace(boxes=boxes, masks=None, names=self.names)]


def test_detector_adapter_converts_person_result_and_bottom_center() -> None:
    detector = object.__new__(UltralyticsPlayerDetector)
    detector._model = _FakePredictionModel()
    detector._confidence = 0.25
    detector._device = "cpu"
    detector._person_class_id = 0
    detector._image_size = 640

    detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert isinstance(detections, sv.Detections)
    np.testing.assert_allclose(detections.xyxy, [[10, 20, 30, 80]])
    assert detector.task == "detect"
    assert detector.class_names[0] == "person"
    assert detector._model.predict_kwargs["imgsz"] == 640
