"""Unit tests for main.py.

These tests focus on app wiring and control flow, while replacing heavy runtime
dependencies (model loading, external libraries, and camera/runtime side
effects) with lightweight stubs.
"""

import importlib
import sys
import types
from unittest.mock import Mock, patch

import numpy as np
import pytest


def _load_main_with_stubs(monkeypatch: pytest.MonkeyPatch):
    """Import main.py with stubbed third-party modules.

    This keeps the tests deterministic and fast by avoiding model downloads,
    hardware dependencies, and full inference stack initialization.
    """
    class DummyRFDETRNano:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def optimize_for_inference(self) -> None:
            pass

    class DummyAnnotator:
        def annotate(self, scene, detections, labels=None):
            return scene

    rfdetr_module = types.ModuleType("rfdetr")
    rfdetr_module.RFDETRNano = DummyRFDETRNano

    rfdetr_assets_module = types.ModuleType("rfdetr.assets")
    rfdetr_coco_module = types.ModuleType("rfdetr.assets.coco_classes")
    rfdetr_coco_module.COCO_CLASSES = {16: "bird", 1: "person"}

    supervision_module = types.ModuleType("supervision")
    supervision_module.BoxAnnotator = DummyAnnotator
    supervision_module.LabelAnnotator = DummyAnnotator
    supervision_module.Detections = object

    transformers_module = types.ModuleType("transformers")
    transformers_module.pipeline = lambda *args, **kwargs: Mock(
        return_value=[{"label": "TEST", "score": 1.0}]
    )

    monkeypatch.setitem(sys.modules, "rfdetr", rfdetr_module)
    monkeypatch.setitem(sys.modules, "rfdetr.assets", rfdetr_assets_module)
    monkeypatch.setitem(sys.modules, "rfdetr.assets.coco_classes", rfdetr_coco_module)
    monkeypatch.setitem(sys.modules, "supervision", supervision_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    import utils

    monkeypatch.setattr(utils, "get_platform", lambda: "Linux")
    sys.modules.pop("main", None)

    import main

    return importlib.reload(main)


@pytest.fixture
def main_module(monkeypatch: pytest.MonkeyPatch):
    """Provide a reloaded main module that uses local stubs."""
    return _load_main_with_stubs(monkeypatch)


def test_platform_uses_get_platform(main_module) -> None:
    """Ensure PLATFORM is derived from the platform helper at import time."""
    assert main_module.PLATFORM == "Linux"


def test_on_frame_calls_classifier_when_bird_detected(main_module, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify bird detections trigger crop + classification path."""
    detector = Mock()
    detector.predict.return_value = (types.SimpleNamespace(class_id=[16]), np.array([0, 0, 2, 2]))

    classifier = Mock()
    frame_getter = Mock()
    app = main_module.BirdWatcherApp(frame_getter, detector=detector, classifier=classifier)

    monkeypatch.setattr(
        main_module,
        "crop_image",
        lambda frame, xyxy: np.ones((2, 2, 3), dtype=np.uint8),
    )

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    result = app.on_frame(frame)

    classifier.predict.assert_called_once()
    assert result.shape == frame.shape


def test_on_frame_skips_classifier_when_no_bird(main_module) -> None:
    """Verify classification is skipped when no bird bbox is returned."""
    detector = Mock()
    detector.predict.return_value = (types.SimpleNamespace(class_id=[1]), None)

    classifier = Mock()
    frame_getter = Mock()
    app = main_module.BirdWatcherApp(frame_getter, detector=detector, classifier=classifier)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    app.on_frame(frame)

    classifier.predict.assert_not_called()


def test_bird_classifier_predict_returns_birdie(main_module, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify classifier output is mapped to a Birdie domain object."""
    classifier = object.__new__(main_module.BirdClassifier)
    classifier.pipe = Mock(return_value=[{"label": "BLUE JAY", "score": 0.87}])

    monkeypatch.setattr(main_module.cv2, "cvtColor", lambda frame, _: frame)
    monkeypatch.setattr(main_module.Image, "fromarray", lambda arr: arr)

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    bird = main_module.BirdClassifier.predict(classifier, frame)

    assert isinstance(bird, main_module.Birdie)
    assert bird.species == "BLUE JAY"
    assert bird.confidence == pytest.approx(0.87)


def test_run_creates_components_and_starts_app(main_module) -> None:
    """Verify run() wires dependencies and starts the app lifecycle."""
    fake_frame_getter = Mock()
    fake_app = Mock()

    with patch.object(main_module, "FrameGetter", return_value=fake_frame_getter) as frame_getter_cls, patch.object(
        main_module, "BirdClassifier", return_value="classifier"
    ) as classifier_cls, patch.object(main_module, "BirdDetector", return_value="detector") as detector_cls, patch.object(
        main_module,
        "BirdWatcherApp",
        return_value=fake_app,
    ) as app_cls:
        main_module.run()

    frame_getter_cls.assert_called_once_with(video_source=0)
    fake_frame_getter.start.assert_called_once()
    classifier_cls.assert_called_once_with()
    detector_cls.assert_called_once_with()
    app_cls.assert_called_once_with(fake_frame_getter, detector="detector", classifier="classifier")
    fake_app.run.assert_called_once_with()
