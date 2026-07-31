import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv
from PIL import Image
from transformers import pipeline
from rfdetr import RFDETRNano

from utils import get_platform, crop_image, LedManager


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PLATFORM = get_platform()
MODEL_PATH = "models/rf-detr-nano.pth"
BIRD_CLASS_ID = 16


@dataclass(slots=True)
class BirdPrediction:
    species: str
    confidence: float


class VideoSource:
    """Threaded camera reader."""

    def __init__(self, source: int | str = 0):
        self.platform = get_platform()

        if self.platform == "Linux":
            from picamera2 import Picamera2

            self.camera = Picamera2()
            self.camera.configure(
                self.camera.create_preview_configuration()
            )
            self.camera.start()

        else:
            self.cap = cv2.VideoCapture(source)

            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Could not open source {source}"
                )

        self.frame: np.ndarray | None = None
        self.running = False
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None

        self.first_frame_ready = threading.Event()


    def start(self):
        self.running = True

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
        )
        self.thread.start()

        self.first_frame_ready.wait()


    def _capture_loop(self):
        while self.running:

            if self.platform == "Linux":
                frame = self.camera.capture_array()

                # Picamera2 ger ofta RGBA
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_RGBA2BGR,
                )

            else:
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


    def stop(self):
        self.running = False

        if self.thread:
            self.thread.join()

        if self.platform == "Linux":
            self.camera.stop()
        else:
            self.cap.release()

        self.first_frame_ready.clear()


class BirdDetector:
    """RF-DETR bird detector."""

    def __init__(self):
        self.model = RFDETRNano(
            pretrain_weights=MODEL_PATH
        )
        self.model.optimize_for_inference()

    def predict(self, frame: np.ndarray) -> sv.Detections:
        return self.model.predict(frame)

    def find_bird(
        self,
        detections: sv.Detections,
    ) -> np.ndarray | None:

        for class_id, bbox in zip(
            detections.class_id,
            detections.xyxy,
        ):
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

    def predict(
        self,
        image: np.ndarray,
    ) -> BirdPrediction:

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = self.model(Image.fromarray(image_rgb))

        best = result[0]

        return BirdPrediction(
            species=best["label"],
            confidence=float(best["score"]),
        )


class BirdPipeline:
    """Coordinates detection and classification."""

    def __init__(
        self,
        detector: BirdDetector,
        classifier: BirdClassifier,
    ):
        self.detector = detector
        self.classifier = classifier

    def process(
        self,
        frame: np.ndarray,
    ) -> BirdPrediction | None:

        detections = self.detector.predict(frame)

        bbox = self.detector.find_bird(detections)

        if bbox is None:
            return None

        bird_image = crop_image(frame, bbox)

        return self.classifier.predict(bird_image)


class BirdWatcherApp:

    def __init__(
        self,
        led: LedManager,
        camera: VideoSource,
        pipeline: BirdPipeline,
    ):
        self.led = led
        self.camera = camera
        self.pipeline = pipeline

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:

        prediction = self.pipeline.process(frame)

        if prediction:
            logger.info(
                "%s %.1f%%",
                prediction.species,
                prediction.confidence * 100,
            )

        return frame

    def run(self):

        self.camera.start()

        try:
            while True:
                frame = self.camera.get_latest_frame()
                
                output = self.process_frame(frame)

                cv2.imshow(
                    "BirdWatcher",
                    output,
                )
                self.led.led_on() # Only for raspberry

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    break

        finally:
            self.camera.stop()
            self.led.led_off()
            self.led.led_close()
            cv2.destroyAllWindows()


def main():
    led = LedManager(PLATFORM)
    camera = VideoSource(0)

    detector = BirdDetector()
    classifier = BirdClassifier()

    pipeline = BirdPipeline(
        detector,
        classifier,
    )

    app = BirdWatcherApp(
        led,
        camera,
        pipeline,
    )

    app.run()


if __name__ == "__main__":
    main()