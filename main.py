import logging

import cv2
import numpy as np

from utils import create_led, get_platform

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PLATFORM = get_platform()

Frame = np.ndarray
Prediction = object


class BirdWatcherApp:
    """Bird watcher app"""

    def __init__(self, video_source: int | str = 0) -> None:
        self.video_source = video_source

    def on_frame(self, frame: Frame) -> Frame:
        return frame

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
    app = BirdWatcherApp(video_source=0)
    app.run()


if __name__ == "__main__":
    run()
