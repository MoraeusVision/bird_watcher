import logging
import threading
from dataclasses import dataclass
import atexit
import torch
from datetime import datetime, timedelta
import birder
from birder.inference.classification import infer_image

import cv2
import numpy as np
import supervision as sv
from PIL import Image
from flask import Flask, Response, jsonify, render_template
from rfdetr import RFDETRNano
from transformers import pipeline
from huggingface_hub import hf_hub_download

from utils import crop_image, get_platform


logging.basicConfig(
	level=logging.INFO,
	format="[%(asctime)s] [%(levelname)s] %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BIRD_MODEL_PATH = "models/rf-detr-nano.pth"
BIRD_CLASS_ID = 16

EGG_MODEL_REPO = "moraeusvision/rf-detr-egg-detector"
EGG_MODEL_FILENAME = "checkpoint_best_ema.pth"


@dataclass
class BirdPrediction:
	species: str
	confidence: float

@dataclass
class EggPrediction:
	bbox: list
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


class EggDetector:
    def __init__(self, threshold: float = 0.5):
        # Download model from Hugging Face
        model_path = hf_hub_download(
            repo_id=EGG_MODEL_REPO,
            filename=EGG_MODEL_FILENAME,
        )

        # Load RF-DETR
        self.model = RFDETRNano(
            pretrain_weights=model_path
        )

        # Optimize for inference
        self.model.optimize_for_inference()

        self.threshold = threshold

    def predict(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return self.model.predict(
            frame_rgb,
            threshold=self.threshold
        )

    def _parse_eggs(self, detections):
        eggs = []

        for class_id, confidence, bbox in zip(
            detections.class_id,
            detections.confidence,
            detections.xyxy,
        ):
            if class_id == 0:
                eggs.append(
                    EggPrediction(
                        bbox=bbox.tolist(),
                        confidence=float(confidence),
                    )
                )

        return eggs

    def get_eggs(self, frame):
        detections = self.predict(frame)
        return self._parse_eggs(detections)


class EggMonitor:
	def __init__(
			self,
			detector: EggDetector,
			batch_size: int = 500,
			required_detections: int = 25,
			cooldown_days: int = 90,
			check_frequency_hours: int = 1):
	
		self.detector = detector
		self.batch_size = batch_size
		self.required_detections = required_detections
		self.cooldown_days = cooldown_days
		self.check_frequency_hours = check_frequency_hours
		self.frames = []
		self.egg_detected = False
		self.next_check = datetime.now()

	def update(self, frame):
		if datetime.now() < self.next_check:
			return
		
		self.fill_batch(frame)

		if not self.batch_is_full():
			return

		self.check_batch()

		if not self.egg_detected:
			self.frames.clear()
			self.next_check = datetime.now() + timedelta(hours=self.check_frequency_hours)
			logger.info(f"Checking again at {self.next_check}")
			return

		self.alert_subscribers()
		self.frames.clear()
		self.next_check = datetime.now() + timedelta(days=self.cooldown_days)
		logger.info(f"Cooldown, looking for eggs again at {self.next_check}")

	def alert_subscribers(self):
		logger.info("Eggs are detected!") # Placeholder
		
	def check_batch(self):
		detections = 0

		for frame in self.frames:
			eggs = self.detector.get_eggs(frame)

			if eggs:
				detections += 1

		self.egg_detected = detections >= self.required_detections
		
	def fill_batch(self, frame):
		logger.info(f"frames in batch: {len(self.frames)}")
		self.frames.append(frame)

	def batch_is_full(self):
		logger.info("Batch is full")
		return len(self.frames) >= self.batch_size

	def egg_status(self):
		return self.egg_detected


class BirdDetector:
	"""RF-DETR bird detector."""

	def __init__(self):
		self.model = RFDETRNano(pretrain_weights=BIRD_MODEL_PATH)
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

	def __init__(self, model_name: str = "convnext_v2_tiny_eu-common"):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.net, model_info, self.transform = birder.load_pretrained_model_and_transform(
				model_name,
				inference=True,
				device=self.device,
				progress_bar=False,
			)

		self.class_names = [name for name, _ in sorted(model_info.class_to_idx.items(), key=lambda item: item[1])]

	def predict(self, image: np.ndarray) -> BirdPrediction:
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(rgb)

		logits, _ = infer_image(
			self.net,
			image,
			self.transform,
			device=self.device)
		
		pred_idx = int(np.argmax(logits[0]))
		confidence = float(logits[0, pred_idx])
		label = self.class_names[pred_idx]

		return BirdPrediction(species=label, confidence=confidence)


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

	def __init__(self, camera: VideoSource, pipeline: BirdPipeline, egg_monitor: EggMonitor):
		self.camera = camera
		self.pipeline = pipeline
		self.egg_monitor = egg_monitor
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
			print(prediction)
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

			self.egg_monitor.update(frame)

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
	egg_detector = EggDetector()
	classifier = BirdClassifier()
	egg_monitor = EggMonitor(detector=egg_detector)
	pipeline = BirdPipeline(detector=detector, classifier=classifier)
	return camera, pipeline, egg_monitor


def create_app() -> Flask:
	camera, bird_pipeline, egg_monitor = create_components()
	camera.start()

	web_app = BirdWatcherWebApp(camera=camera, pipeline=bird_pipeline, egg_monitor=egg_monitor)
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
