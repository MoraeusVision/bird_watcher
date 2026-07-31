from picamera2 import Picamera2
import cv2


picam2 = Picamera2()

picam2.configure(
    picam2.create_preview_configuration()
)

picam2.start()

print("Kamera startad. Tryck q för att avsluta.")

while True:
    frame = picam2.capture_array()

    # Picamera2 ger ofta RGBA
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    cv2.imshow("Picamera2 test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


picam2.stop()
cv2.destroyAllWindows()