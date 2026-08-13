from picamera2 import Picamera2
import cv2

from ..utils import LedManager, get_platform


picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration()
)
picam2.start()

led = LedManager(get_platform())

print("Kamera startad. Tryck q för att avsluta.")

try:
    while True:
        try:
            frame = picam2.capture_array()
            led.led_on()
        except Exception as e:
            print(f"Kunde inte hämta frame: {e}")
            led.led_off()
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        cv2.imshow("Picamera2 test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    led.led_off()
    led.led_close()
    picam2.stop()
    cv2.destroyAllWindows()