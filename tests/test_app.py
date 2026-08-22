"""Unit tests for the Flask web app in app.py."""

import importlib
import sys
import types
from unittest.mock import Mock

import numpy as np
import pytest


class _FakeCamera:
    def __init__(self, frame: np.ndarray | None = None) -> None:
        self._frame = frame if frame is not None else np.zeros((8, 8, 3), dtype=np.uint8)
        self.start = Mock()
        self.stop = Mock()

    def get_latest_frame(self) -> np.ndarray:
        return self._frame.copy()


class _FakePipeline:
    def __init__(self, prediction=None) -> None:
        self._prediction = prediction

    def process(self, frame):
        return self._prediction


def _load_app_with_stubs(monkeypatch: pytest.MonkeyPatch):
    """Import app.py with lightweight stubs for heavy third-party modules."""

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

    sys.modules.pop("app", None)
    import app

    return importlib.reload(app)


@pytest.fixture
def app_module(monkeypatch: pytest.MonkeyPatch):
    return _load_app_with_stubs(monkeypatch)


def test_index_page_contains_stream_and_button(app_module) -> None:
    camera = _FakeCamera()
    pipeline = _FakePipeline()

    web_app = app_module.BirdWatcherWebApp(camera=camera, pipeline=pipeline)
    client = web_app.flask_app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "video_feed" in html
    assert "Analyze current frame" in html


def test_video_feed_returns_mjpeg_response(app_module, monkeypatch: pytest.MonkeyPatch) -> None:
    camera = _FakeCamera(frame=np.zeros((8, 8, 3), dtype=np.uint8))
    pipeline = _FakePipeline()

    web_app = app_module.BirdWatcherWebApp(camera=camera, pipeline=pipeline)
    client = web_app.flask_app.test_client()

    fake_jpeg = np.array([255, 216, 255], dtype=np.uint8)
    monkeypatch.setattr(app_module.cv2, "imencode", lambda ext, frame: (True, fake_jpeg))

    response = client.get("/video_feed", buffered=False)

    assert response.status_code == 200
    assert response.mimetype == "multipart/x-mixed-replace"

    first_chunk = next(response.response)
    assert b"--frame" in first_chunk
    assert b"Content-Type: image/jpeg" in first_chunk


def test_analyze_returns_confidence_without_decimals(app_module, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = np.ones((8, 8, 3), dtype=np.uint8)
    camera = _FakeCamera(frame=frame)

    prediction = app_module.BirdPrediction(species="Blue Jay", confidence=0.87)
    pipeline = _FakePipeline(prediction=prediction)

    web_app = app_module.BirdWatcherWebApp(camera=camera, pipeline=pipeline)
    client = web_app.flask_app.test_client()

    fake_jpeg = np.array([255, 216, 255], dtype=np.uint8)
    monkeypatch.setattr(app_module.cv2, "imencode", lambda ext, stream_frame: (True, fake_jpeg))

    stream_response = client.get("/video_feed", buffered=False)
    _ = next(stream_response.response)

    response = client.post("/analyze")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["species"] == "Blue Jay"
    assert payload["confidence_percent"] == 87
    assert payload["result_text"] == "Blue Jay (87%)"


def test_analyze_returns_409_when_no_frame_available(app_module) -> None:
    camera = _FakeCamera()
    pipeline = _FakePipeline()

    web_app = app_module.BirdWatcherWebApp(camera=camera, pipeline=pipeline)
    client = web_app.flask_app.test_client()

    response = client.post("/analyze")

    assert response.status_code == 409
    payload = response.get_json()
    assert "No frame available" in payload["error"]


def test_create_components_uses_linux_camera_path(app_module, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "get_platform", lambda: "Linux")

    class FakePicamera2:
        def create_preview_configuration(self):
            return "preview"

        def configure(self, config):
            self.config = config

        def start(self):
            pass

        def stop(self):
            pass

        def capture_array(self):
            return np.zeros((4, 4, 4), dtype=np.uint8)

    picamera_module = types.ModuleType("picamera2")
    picamera_module.Picamera2 = FakePicamera2
    monkeypatch.setitem(sys.modules, "picamera2", picamera_module)

    camera, pipeline = app_module.create_components()

    assert camera.platform == "Linux"
    assert isinstance(pipeline, app_module.BirdPipeline)


def test_create_app_does_not_stop_camera_after_single_request(app_module, monkeypatch: pytest.MonkeyPatch) -> None:
    camera = _FakeCamera()
    pipeline = _FakePipeline()

    monkeypatch.setattr(app_module, "create_components", lambda: (camera, pipeline))

    flask_app = app_module.create_app()
    client = flask_app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    camera.start.assert_called_once_with()
    camera.stop.assert_not_called()


def test_create_app_exposes_video_source_for_shutdown(app_module, monkeypatch: pytest.MonkeyPatch) -> None:
    camera = _FakeCamera()
    pipeline = _FakePipeline()

    monkeypatch.setattr(app_module, "create_components", lambda: (camera, pipeline))

    flask_app = app_module.create_app()

    assert flask_app.config["VIDEO_SOURCE"] is camera
