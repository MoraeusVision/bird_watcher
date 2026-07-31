import platform

import numpy as np

def get_platform() -> str:
    """Return the current platform name."""
    return platform.system()

def create_led(platform_name: str):
    """Create a hardware LED on GPIO pin 17 on Linux only."""
    if platform_name == "Linux":
        from gpiozero import LED

        return LED(17)

    raise RuntimeError(f"Unsupported platform for GPIO LED: {platform_name}")

def crop_image(frame: np.ndarray, xyxy) -> np.ndarray:
    """Crop a frame using an ``xyxy`` bounding box."""
    x1, y1, x2, y2 = (int(round(value)) for value in xyxy)
    height, width = frame.shape[:2]

    left = max(0, min(width, x1))
    top = max(0, min(height, y1))
    right = max(0, min(width, x2))
    bottom = max(0, min(height, y2))

    if right <= left or bottom <= top:
        return frame[0:0, 0:0].copy()

    return frame[top:bottom, left:right].copy()


    