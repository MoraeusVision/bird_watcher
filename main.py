import logging
import threading
from dataclasses import dataclass
import atexit

import cv2
import numpy as np
import supervision as sv
from PIL import Image
from flask import Flask, Response, jsonify, render_template
from rfdetr import RFDETRNano
from transformers import pipeline

from utils import crop_image, get_platform


logging.basicConfig(
	level=logging.INFO,
	format="[%(asctime)s] [%(levelname)s] %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/rf-detr-nano.pth"
BIRD_CLASS_ID = 16


@dataclass
class BirdPrediction:
	species: str
	confidence: float


class VideoSource:
	"""Threaded camera reader with platform-specific camera backend."""

	def __init__(self, source: int | str = 0):
		self.platform = get_platform()

		if self.platform == "Linux":
			from picamera2 import Picamera2

			self.camera = Picamera2()
			self.camera.configure(self.camera.create_preview_configuration())
			self.camera.start()
			self.cap = None
		else:
			self.cap = cv2.VideoCapture(source)
			if not self.cap.isOpened():
				raise RuntimeError(f"Could not open source {source}")
			self.camera = None

		self.frame: np.ndarray | None = None
		self.running = False
		self.lock = threading.Lock()
		self.thread: threading.Thread | None = None
		self.first_frame_ready = threading.Event()

	def start(self) -> None:
		if self.running:
			return

		self.running = True
		self.thread = threading.Thread(target=self._capture_loop, daemon=True)
		self.thread.start()
		self.first_frame_ready.wait()

	def _capture_loop(self) -> None:
		while self.running:
			if self.platform == "Linux":
				frame = self.camera.capture_array()
				frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
			else:
				if self.cap is None:
					break

				ret, frame = self.cap.read()
				if not ret:
					continue

			with self.lock:
				self.frame = frame

			self.first_frame_ready.set()

	def get_latest_frame(self) -> np.ndarray:
		with self.lock:
			if self.frame is None:
				raise RuntimeError("No frame available")
			return self.frame.copy()

	def stop(self) -> None:
		self.running = False

		if self.platform != "Linux" and self.cap is not None:
			self.cap.release()
			self.cap = None

		if self.platform == "Linux" and self.camera is not None:
			self.camera.stop()

		if self.thread is not None:
			self.thread.join(timeout=1.0)
			self.thread = None

		self.first_frame_ready.clear()


class BirdDetector:
	"""RF-DETR bird detector."""

	def __init__(self):
		self.model = RFDETRNano(pretrain_weights=MODEL_PATH)
		self.model.optimize_for_inference()

	def predict(self, frame: np.ndarray) -> sv.Detections:
		return self.model.predict(frame)

	def find_bird(self, detections: sv.Detections) -> np.ndarray | None:
		for class_id, bbox in zip(detections.class_id, detections.xyxy):
			if class_id == BIRD_CLASS_ID:
				return bbox

		return None


class BirdClassifier:
	"""Bird species classifier."""

	def __init__(self):
		self.model = pipeline(
			"image-classification",
			model="dennisjooo/Birds-Classifier-EfficientNetB2",
		)

	def predict(self, image: np.ndarray) -> BirdPrediction:
		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		result = self.model(Image.fromarray(image_rgb))
		best = result[0]
		return BirdPrediction(species=best["label"], confidence=float(best["score"]))


class BirdPipeline:
	"""Coordinates detection and classification."""

	def __init__(self, detector: BirdDetector, classifier: BirdClassifier):
		self.detector = detector
		self.classifier = classifier

	def process(self, frame: np.ndarray) -> BirdPrediction | None:
		detections = self.detector.predict(frame)
		bbox = self.detector.find_bird(detections)
		if bbox is None:
			return None

		bird_image = crop_image(frame, bbox)
		return self.classifier.predict(bird_image)


class BirdWatcherWebApp:
	"""Flask app that streams camera frames and analyzes on button click."""

	def __init__(self, camera: VideoSource, pipeline: BirdPipeline):
		self.camera = camera
		self.pipeline = pipeline
		self.frame_lock = threading.Lock()
		self.latest_frame: np.ndarray | None = None

		self.flask_app = Flask(__name__, template_folder="templates")
		self._register_routes()

	def _register_routes(self) -> None:
		@self.flask_app.get("/")
		def index():
			return render_template("index.html")

		@self.flask_app.get("/video_feed")
		def video_feed():
			return Response(
				self._stream_frames(),
				mimetype="multipart/x-mixed-replace; boundary=frame",
			)

		@self.flask_app.post("/analyze")
		def analyze():
			with self.frame_lock:
				if self.latest_frame is None:
					return jsonify({"error": "No frame available for analysis"}), 409
				frame = self.latest_frame.copy()

			prediction = self.pipeline.process(frame)
			if prediction is None:
				return jsonify(
					{
						"species": None,
						"confidence_percent": None,
						"result_text": "No bird detected",
					}
				)

			confidence_percent = int(round(prediction.confidence * 100))
			result_text = f"{prediction.species} ({confidence_percent}%)"
			logger.info(result_text)

			return jsonify(
				{
					"species": prediction.species,
					"confidence_percent": confidence_percent,
					"result_text": result_text,
				}
			)

	def _stream_frames(self):
		while True:
			frame = self.camera.get_latest_frame()

			with self.frame_lock:
				self.latest_frame = frame.copy()

			ok, jpeg = cv2.imencode(".jpg", frame)
			if not ok:
				continue

			payload = jpeg.tobytes()
			yield (
				b"--frame\r\n"
				b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
			)


def create_components() -> tuple[VideoSource, BirdPipeline]:
	camera = VideoSource(0)
	detector = BirdDetector()
	classifier = BirdClassifier()
	pipeline = BirdPipeline(detector=detector, classifier=classifier)
	return camera, pipeline


def create_app() -> Flask:
	camera, bird_pipeline = create_components()
	camera.start()

	web_app = BirdWatcherWebApp(camera=camera, pipeline=bird_pipeline)
	web_app.flask_app.config["VIDEO_SOURCE"] = camera

	atexit.register(camera.stop)

	return web_app.flask_app


app = create_app()


if __name__ == "__main__":
	video_source = app.config.get("VIDEO_SOURCE")

	try:
		app.run(host="0.0.0.0", port=8080, debug=False)
	finally:
		if video_source is not None:
			video_source.stop()
