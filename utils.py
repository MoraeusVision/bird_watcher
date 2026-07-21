import torch
import logging


def get_device():
    """Return the best available device: cuda, mps, or cpu. Logs the selected device."""
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logging.info(f"Using device: {device}")
    return device

def parse_video_source(video_arg):
    """If user provides "--source 0" """
    if video_arg.isdigit():
        return int(video_arg)
    return video_arg