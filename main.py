import logging
from pathlib import Path

import cv2
import numpy as np
from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES
import supervision as sv

from utils import create_led, get_device, get_platform

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PLATFORM = get_platform()

class BirdDetector:
    """Simple RF-DETR bird detector."""

    def __init__(self) -> None:
        self.model = RFDETRNano()
        self.model.optimize_for_inference()

    def predict(self, frame: np.ndarray) -> sv.Detections:
        detections = self.model.predict(frame)

        return detections

class BirdWatcherApp:
    """Bird watcher app"""

    def __init__(self, video_source: int | str = 0, detector: BirdDetector | None = None) -> None:
        self.video_source = video_source
        self.bird_detector = detector
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def on_prediction(self, detections: sv.Detections) -> None:
        pass

    def on_frame(self, frame: np.ndarray) -> np.ndarray:
        detections = self.bird_detector.predict(frame)
        labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]
        
        annotated_image = sv.BoxAnnotator().annotate(detections.metadata["source_image"], detections)
        annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

        return annotated_image

    def run(self) -> None:
        led = None
        if PLATFORM == "Linux":
            led = create_led(PLATFORM)

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {self.video_source}")

        logger.info("BirdWatcher started. Press 'q' or Esc to exit.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break

                processed_frame = self.on_frame(frame)

                cv2.imshow("BirdWatcher", processed_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if led is not None:
                led.close()


def run() -> None:
    bird_detector = BirdDetector()
    app = BirdWatcherApp(video_source=0, detector=bird_detector)
    app.run()


if __name__ == "__main__":
    run()
