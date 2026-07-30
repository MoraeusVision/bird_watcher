import logging
from pathlib import Path
import threading
import cv2
import torch
import numpy as np
from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES
import supervision as sv
from PIL import Image
from dataclasses import dataclass

from transformers import pipeline

from utils import create_led, get_device, get_platform, crop_image

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PLATFORM = get_platform()
MODEL_PATH = "models/rf-detr-nano.pth"
BIRD_CLASS_ID = 16  # COCO class ID for bird

@dataclass
class Birdie:
    species: str
    confidence: float

class FrameGetter:
    """Fetches frames from the video source"""
    def __init__(self, video_source: int | str = 0) -> None:
        self.cap: cv2.VideoCapture = cv2.VideoCapture(video_source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open source: {video_source}")
        
        self.frame: np.ndarray | None = None
        self.running: bool = False
        self.lock: threading.Lock = threading.Lock()
        self.thread: threading.Thread | None = None
        
    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self) -> None:
        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                continue

            with self.lock:
                self.frame = frame

    def get_frame(self) -> np.ndarray:
        with self.lock:
            if self.frame is None:
                raise RuntimeError("No frame available yet")

            return self.frame.copy()
        
    def stop(self) -> None:
        self.running = False

        if self.thread:
            self.thread.join()

        self.cap.release()
    

class BirdDetector:
    """RF-DETR bird detector."""

    def __init__(self) -> None:
        self.model = RFDETRNano(pretrain_weights=MODEL_PATH)
        self.model.optimize_for_inference()

    def predict(self, frame: np.ndarray) -> sv.Detections:
        detections = self.model.predict(frame)
        bird_xyxy = None

        for class_id in detections.class_id:
            if class_id == BIRD_CLASS_ID:
                bird_xyxy = detections.xyxy[0]
                
                return detections, bird_xyxy
            
        return detections, bird_xyxy
        
        
class BirdClassifier:
    """Takes an image, crop the bird and classify"""
    def __init__(self) -> None:
        self.pipe = pipeline("image-classification", model="dennisjooo/Birds-Classifier-EfficientNetB2")

    def predict(self, frame: np.ndarray) -> list[dict[str, float | str]]:
        # OpenCV frames are BGR numpy arrays; pipeline expects PIL image/path/url/base64.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        result = self.pipe(pil_image)
        best = result[0]

        bird = Birdie(
            species=best["label"],
            confidence=float(best["score"]))

        logger.info(f"Bird species detected: {bird}")
        
        return bird
    

class BirdWatcherApp:
    """Bird watcher app"""

    def __init__(
        self,
        source: FrameGetter,
        detector: BirdDetector | None = None,
        classifier: BirdClassifier | None = None,
    ) -> None:
        self.frame_getter: FrameGetter = source
        self.bird_detector: BirdDetector | None = detector
        self.bird_classifier: BirdClassifier | None = classifier
        self.box_annotator: sv.BoxAnnotator = sv.BoxAnnotator()
        self.label_annotator: sv.LabelAnnotator = sv.LabelAnnotator()

    def on_frame(self, frame: np.ndarray) -> np.ndarray:
        detections, bird_xyxy = self.bird_detector.predict(frame)

        if bird_xyxy is not None:
            bird_img = crop_image(frame=frame, xyxy=bird_xyxy)
            self.bird_classifier.predict(bird_img)

        labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]
        
        annotated_image = sv.BoxAnnotator().annotate(frame, detections)
        annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

        return annotated_image
        

    def run(self) -> None:
        led = None
        if PLATFORM == "Linux":
            led = create_led(PLATFORM)

        try:
            while True:
                frame = self.frame_getter.get_frame()

                processed_frame = self.on_frame(frame)

                cv2.imshow("BirdWatcher", processed_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
        finally:
            self.frame_getter.stop()
            cv2.destroyAllWindows()
            if led is not None:
                led.close()


def run() -> None:

    # Start the camera
    frame_getter = FrameGetter(video_source=0)
    frame_getter.start()

    # Initiate the models
    bird_classifier = BirdClassifier()
    bird_detector = BirdDetector()

    app = BirdWatcherApp(
        source=frame_getter,
        detector=bird_detector,
        classifier=bird_classifier,
    )
    app.run()


if __name__ == "__main__":
    run()
