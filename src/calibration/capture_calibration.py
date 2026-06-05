"""Capture calibration images using Camera Module 3.

Usage:
    python3 src/calibration/capture_calibration.py --out calibration_images/

Controls:
    SPACE  — capture frame (saves to out/)
    Q      — quit
    D      — delete last capture
"""

import argparse
import os
import time

import cv2
from picamera2 import Picamera2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="calibration_images")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cam = Picamera2()
    cfg = cam.create_preview_configuration(
        main={"size": (args.width, args.height), "format": "BGR888"},
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(1.0)

    count = len([f for f in os.listdir(args.out) if f.endswith(".jpg")])
    last_path = None

    print(f"Camera started. Images will be saved to: {args.out}/")
    print("SPACE=capture  D=delete last  Q=quit")
    print(f"Already have {count} images.")

    cv2.namedWindow("Calibration Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration Capture", 960, 540)

    while True:
        frame = cam.capture_array()
        vis = frame.copy()

        # Status overlay
        status = f"Captured: {count} | SPACE=capture  D=delete last  Q=quit"
        color = (0, 200, 0) if count >= 10 else (0, 140, 255)
        cv2.putText(vis, status, (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        if count < 10:
            cv2.putText(vis, f"Need {10 - count} more images",
                        (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 140, 255), 1, cv2.LINE_AA)

        cv2.imshow("Calibration Capture", vis)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" "):
            path = os.path.join(args.out, f"calib_{count:03d}.jpg")
            cv2.imwrite(path, frame)
            last_path = path
            count += 1
            print(f"  Saved: {path}")

        elif key == ord("d") and last_path and os.path.exists(last_path):
            os.remove(last_path)
            count -= 1
            print(f"  Deleted: {last_path}")
            last_path = None

    cam.stop()
    cv2.destroyAllWindows()
    print(f"\nTotal images: {count}")
    if count >= 10:
        print(f"\nRuleaza calibrarea cu:")
        print(f"  python3 src/calibration/calibrate_camera.py \\")
        print(f"    --images_dir {args.out} \\")
        print(f"    --pattern 9 6 \\")
        print(f"    --square_mm 25.0 \\")
        print(f"    --out src/calibration/intrinsics.json")
    else:
        print(f"Prea putine imagini ({count}). Mai ai nevoie de {10 - count}.")


if __name__ == "__main__":
    main()
