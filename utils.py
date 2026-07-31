import platform

import numpy as np

class LedManager:
    def __init__(self, platform_name):
        self.led = None

        if platform_name == "Linux":
            from gpiozero import LED
            self.led = LED(17)

    def led_on(self):
        if self.led:
            self.led.on()

    def led_off(self):
        if self.led:
            self.led.off()

    def led_close(self):
        if self.led:
            self.led.close()

def get_platform() -> str:
    """Return the current platform name."""
    return platform.system()

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


    