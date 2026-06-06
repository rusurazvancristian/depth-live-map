"""
Color channel diagnostic — shows 4 test rectangles + camera frame.
Run and tell me what colors you SEE for each numbered rectangle.
"""
import time
import cv2
import numpy as np
from picamera2 import Picamera2

cam = Picamera2()
cam.configure(cam.create_preview_configuration(
    main={"size": (640, 480), "format": "BGR888"}
))
cam.start()
time.sleep(1.0)

frame = cam.capture_array()
print(f"shape={frame.shape}  dtype={frame.dtype}")
print(f"center pixel ch0={frame[240,320,0]}  ch1={frame[240,320,1]}  ch2={frame[240,320,2]}")

cv2.namedWindow("test", cv2.WINDOW_NORMAL)

while True:
    frame = cam.capture_array()

    # Bottom strip: 4 known-color blocks drawn by OpenCV (BGR values)
    strip = np.zeros((80, 640, 3), dtype=np.uint8)
    strip[:,   0:160] = (0,   0, 255)  # OpenCV BGR RED  → label "1"
    strip[:, 160:320] = (255, 0,   0)  # OpenCV BGR BLUE → label "2"
    strip[:, 320:480] = (0, 255,   0)  # OpenCV BGR GREEN→ label "3"
    strip[:, 480:640] = (0, 255, 255)  # OpenCV BGR YELLOW→label "4"
    for i, label in enumerate(["1:BGR(0,0,255)", "2:BGR(255,0,0)",
                                "3:BGR(0,255,0)", "4:BGR(0,255,255)"]):
        cv2.putText(strip, label, (i*160+4, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    display = np.vstack([frame, strip])
    cv2.imshow("test", display)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break

cam.stop()
cv2.destroyAllWindows()
