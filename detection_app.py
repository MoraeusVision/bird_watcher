import cv2
import logging
from abc import ABC, abstractmethod
from typing import List, Any, Optional
from inference.core.interfaces.camera.entities import VideoFrame
from inference import InferencePipeline
from rfdetr import RFDETRNano
import supervision as sv
from utils import get_device, parse_video_source
import argparse

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

VIDEO_PATH = "example_media/eggs_1.mp4"
MODEL_PATH = "models/egg_ema.pth"
OUTPUT_VIDEO_PATH = "output/saved_video.mp4"
PREDICT_THRESHOLD = 0.5


class PredictionFrameRenderer:
    def __init__(self, track: bool):
        self.track = track
        self.label_annotator = sv.LabelAnnotator()
        self.box_annotator = sv.BoxAnnotator()

    def get_labels(self, prediction) -> List[str]:
        if self.track and prediction.tracker_id is not None:
            return [
                f"DroEggne - Tracking ID:{int(track_id)}"
                for track_id, conf in zip(
                    prediction.tracker_id,
                    prediction.confidence,
                )
            ]

        return [f"Egg {conf}" for conf in prediction.confidence]

    def render(self, prediction, video_frame):
        labels = self.get_labels(prediction)
        annotated_image = video_frame.image.copy()

        annotated_image = self.box_annotator.annotate(
            scene=annotated_image,
            detections=prediction,
        )

        annotated_image = self.label_annotator.annotate(
            annotated_image,
            detections=prediction,
            labels=labels,
        )

        return annotated_image  # type: ignore


class FrameOutputManager:
    def __init__(self, show: bool, save: bool):
        self.show = show
        self.save = save
        self.paused = False
        self.sink = None
        self.paused_frame = None

    def set_sink(self, sink) -> None:
        self.sink = sink

    def emit(self, frame, pipeline) -> None:
        if self.show and frame is not None:
            self.visualize(frame, pipeline)

        if self.save and self.sink is not None and frame is not None:
            self.sink.write_frame(frame)

    def visualize(self, frame, pipeline) -> None:
        # Pause only affects what is displayed; inference and saving continue.
        display_frame = self.paused_frame if self.paused and self.paused_frame is not None else frame
        cv2.imshow("Predictions", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            pipeline.terminate()
            return

        if key == ord(" "):
            self.paused = not self.paused
            if self.paused:
                self.paused_frame = frame.copy()
            else:
                self.paused_frame = None
            logging.info("Paused" if self.paused else "Resumed")

class BaseDetectionApp(ABC):
    """Abstract base class for detection applications with custom frame processing logic.
    
    Subclasses should implement the `process_predicted_frame` method to add custom
    logic that processes predictions and video frames during the detection pipeline.
    """
    
    def __init__(self, weights_path: str, video_source: str, show: bool, save: bool, output_path: str, track: bool):
        self.show = show
        self.save = save
        self.track = track
        self.output_path = output_path
        self.last_frame = None
        self.video_source = parse_video_source(video_source)

        # Model
        self.model = RFDETRNano(pretrain_weights=weights_path, device=get_device())
        self.model.optimize_for_inference()

        # Supervision tools
        self.tracker = sv.ByteTrack()
        self.renderer = PredictionFrameRenderer(track=self.track)
        self.output_manager = FrameOutputManager(show=self.show, save=self.save)

        # Video info - used for saving
        if self.save:
            self.video_info = sv.VideoInfo.from_video_path(self.video_source)

        # Pipeline
        self.pipeline = InferencePipeline.init_with_custom_logic(
            video_reference=self.video_source,
            on_video_frame=self.infer,
            on_prediction=self.on_prediction,
        )

    @abstractmethod
    def process_predicted_frame(self, prediction: Any, video_frame: VideoFrame) -> Any:
        """Hook method for subclasses to implement custom frame processing logic.
        
        This method is called after object detection and tracking (if enabled).
        Subclasses should override this method to add custom processing logic.
        
        Args:
            prediction: The detection predictions object from the model.
            video_frame: The current video frame with metadata.
            
        Returns:
            The (possibly modified) prediction object.
        """
        pass




##### Detection and tracking functions #####


    def on_prediction(self, prediction: Any, video_frame: VideoFrame) -> None:
        if self.track:
            prediction = self.track_objects(prediction, video_frame)

        self.process_predicted_frame(prediction=prediction, video_frame=video_frame)
        
        if self.show or self.save:
            rendered_frame = self.renderer.render(prediction, video_frame)
            self.handle_rendered_outputs(rendered_frame)

    
    def infer(self, video_frames: List[VideoFrame]) -> List[Any]:
        """Run inference on frames"""
        predictions = self.model.predict(
            [v.image for v in video_frames],
            threshold=PREDICT_THRESHOLD,
        )
        return [predictions]

    def track_objects(self, prediction: Any, video_frame: VideoFrame) -> Any:
        """Track objects with ByteTrack"""
        tracked_detections = self.tracker.update_with_detections(prediction)
        return tracked_detections

    def handle_rendered_outputs(self, frame) -> None:
        """Handles rendered outputs, for saving or visualization"""
        self.last_frame = frame
        self.output_manager.emit(frame, self.pipeline)

    def run(self) -> None:
        if self.save:
            with sv.VideoSink(self.output_path, self.video_info, codec="H264") as sink:
                self.output_manager.set_sink(sink)
                self.pipeline.start()
                self.pipeline.join()
        else:
            self.pipeline.start()
            self.pipeline.join()

        cv2.destroyAllWindows()


class DetectionApp(BaseDetectionApp):
    """Concrete implementation of BaseDetectionApp with default frame processing."""
    
    def process_predicted_frame(self, prediction: Any, video_frame: VideoFrame) -> Any:
        """Default implementation that performs no additional processing."""
        return prediction


def parse_args():
    parser = argparse.ArgumentParser(description="Detection App")

    parser.add_argument("--source", type=str, default=VIDEO_PATH)
    parser.add_argument("--weights", type=str, default=MODEL_PATH)
    parser.add_argument("--output", type=str, default=OUTPUT_VIDEO_PATH)
    
    parser.add_argument("--track", action="store_true", help="Enable tracking")
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    app = DetectionApp(
        weights_path=args.weights,
        video_source=args.source,
        show=args.show,
        save=args.save,
        track=args.track,
        output_path=args.output
    )
    app.run()