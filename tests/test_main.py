"""Unit tests for main.py after refactor.

The tests target behavior and wiring while stubbing heavy dependencies.
"""

import importlib
import sys
import types
from unittest.mock import Mock, patch

import numpy as np
import pytest


def _load_main_with_stubs(monkeypatch: pytest.MonkeyPatch):
    """Import main.py with lightweight stubs for third-party modules."""

    class DummyRFDETRNano:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def optimize_for_inference(self) -> None:
            pass

        def predict(self, frame):
            return frame

    rfdetr_module = types.ModuleType("rfdetr")
    rfdetr_module.RFDETRNano = DummyRFDETRNano

    supervision_module = types.ModuleType("supervision")
    supervision_module.Detections = object

    transformers_module = types.ModuleType("transformers")
    transformers_module.pipeline = lambda *args, **kwargs: Mock(
        return_value=[{"label": "TEST", "score": 1.0}]
    )

    monkeypatch.setitem(sys.modules, "rfdetr", rfdetr_module)
    monkeypatch.setitem(sys.modules, "supervision", supervision_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    import utils

    monkeypatch.setattr(utils, "get_platform", lambda: "Linux")

    sys.modules.pop("main", None)
    import main

    return importlib.reload(main)


@pytest.fixture
def main_module(monkeypatch: pytest.MonkeyPatch):
    return _load_main_with_stubs(monkeypatch)


def test_platform_uses_get_platform(main_module) -> None:
    assert main_module.PLATFORM == "Linux"


def test_find_bird_returns_first_bird_bbox(main_module) -> None:
    detector = object.__new__(main_module.BirdDetector)

    detections = types.SimpleNamespace(
        class_id=[1, 16, 16],
        xyxy=[
            np.array([0, 0, 1, 1]),
            np.array([2, 2, 4, 4]),
            np.array([5, 5, 7, 7]),
        ],
    )

    bbox = main_module.BirdDetector.find_bird(detector, detections)

    assert np.array_equal(bbox, np.array([2, 2, 4, 4]))


def test_find_bird_returns_none_when_missing(main_module) -> None:
    detector = object.__new__(main_module.BirdDetector)

    detections = types.SimpleNamespace(
        class_id=[1, 3],
        xyxy=[np.array([0, 0, 1, 1]), np.array([2, 2, 3, 3])],
    )

    bbox = main_module.BirdDetector.find_bird(detector, detections)

    assert bbox is None


def test_pipeline_process_returns_none_when_no_bird(main_module) -> None:
    detector = Mock()
    detector.predict.return_value = "detections"
    detector.find_bird.return_value = None

    classifier = Mock()
    pipeline = main_module.BirdPipeline(detector=detector, classifier=classifier)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    result = pipeline.process(frame)

    assert result is None
    classifier.predict.assert_not_called()


def test_pipeline_process_crops_and_classifies(main_module, monkeypatch: pytest.MonkeyPatch) -> None:
    detector = Mock()
    detector.predict.return_value = "detections"
    detector.find_bird.return_value = np.array([1, 1, 4, 4])

    expected = main_module.BirdPrediction(species="BLUE JAY", confidence=0.91)
    classifier = Mock()
    classifier.predict.return_value = expected

    pipeline = main_module.BirdPipeline(detector=detector, classifier=classifier)

    cropped = np.ones((3, 3, 3), dtype=np.uint8)
    monkeypatch.setattr(main_module, "crop_image", lambda frame, bbox: cropped)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    result = pipeline.process(frame)

    assert result == expected
    classifier.predict.assert_called_once_with(cropped)


def test_bird_classifier_predict_returns_bird_prediction(main_module, monkeypatch: pytest.MonkeyPatch) -> None:
    classifier = object.__new__(main_module.BirdClassifier)
    classifier.model = Mock(return_value=[{"label": "BLUE JAY", "score": 0.87}])

    monkeypatch.setattr(main_module.cv2, "cvtColor", lambda frame, _: frame)
    monkeypatch.setattr(main_module.Image, "fromarray", lambda arr: arr)

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    bird = main_module.BirdClassifier.predict(classifier, frame)

    assert isinstance(bird, main_module.BirdPrediction)
    assert bird.species == "BLUE JAY"
    assert bird.confidence == pytest.approx(0.87)


def test_process_frame_logs_when_prediction_exists(main_module) -> None:
    camera = Mock()
    led = Mock()
    pipeline = Mock()
    pipeline.process.return_value = main_module.BirdPrediction(
        species="NUTHATCH",
        confidence=0.65,
    )

    app = main_module.BirdWatcherApp(led=led, camera=camera, pipeline=pipeline)
    frame = np.zeros((5, 5, 3), dtype=np.uint8)

    with patch.object(main_module.logger, "info") as logger_info:
        out = app.process_frame(frame)

    assert out is frame
    logger_info.assert_called_once()


def test_main_creates_components_and_runs_app(main_module) -> None:
    fake_led = Mock()
    fake_camera = Mock()
    fake_pipeline = Mock()
    fake_app = Mock()

    with patch.object(main_module, "LedManager", return_value=fake_led) as led_cls, patch.object(
        main_module, "VideoSource", return_value=fake_camera
    ) as camera_cls, patch.object(
        main_module, "BirdDetector", return_value="detector"
    ) as detector_cls, patch.object(main_module, "BirdClassifier", return_value="classifier") as classifier_cls, patch.object(
        main_module, "BirdPipeline", return_value=fake_pipeline
    ) as pipeline_cls, patch.object(main_module, "BirdWatcherApp", return_value=fake_app) as app_cls:
        main_module.main()

    led_cls.assert_called_once_with(main_module.PLATFORM)
    camera_cls.assert_called_once_with(0)
    detector_cls.assert_called_once_with()
    classifier_cls.assert_called_once_with()
    pipeline_cls.assert_called_once_with("detector", "classifier")
    app_cls.assert_called_once_with(fake_led, fake_camera, fake_pipeline)
    fake_app.run.assert_called_once_with()


def test_app_run_starts_and_stops_camera(main_module, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "PLATFORM", "Darwin")

    led = Mock()
    camera = Mock()
    camera.get_latest_frame.return_value = np.zeros((5, 5, 3), dtype=np.uint8)

    pipeline = Mock()
    pipeline.process.return_value = None

    app = main_module.BirdWatcherApp(led=led, camera=camera, pipeline=pipeline)

    monkeypatch.setattr(main_module.cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_module.cv2, "waitKey", lambda _: ord("q"))
    monkeypatch.setattr(main_module.cv2, "destroyAllWindows", lambda: None)

    app.run()

    camera.start.assert_called_once_with()
    camera.stop.assert_called_once_with()
    led.led_off.assert_called_once_with()
    led.led_close.assert_called_once_with()
