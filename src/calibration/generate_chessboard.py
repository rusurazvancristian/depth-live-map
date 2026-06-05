"""Generate a printable chessboard PDF/PNG for camera calibration."""

import cv2
import numpy as np
import os

COLS = 9        # inner corners orizontal
ROWS = 6        # inner corners vertical
SQUARE_PX = 80  # pixels per square

def generate(out_path: str = "src/calibration/chessboard_9x6.png") -> None:
    w = (COLS + 1) * SQUARE_PX
    h = (ROWS + 1) * SQUARE_PX
    img = np.ones((h, w), dtype=np.uint8) * 255

    for r in range(ROWS + 1):
        for c in range(COLS + 1):
            if (r + c) % 2 == 0:
                x1, y1 = c * SQUARE_PX, r * SQUARE_PX
                img[y1:y1 + SQUARE_PX, x1:x1 + SQUARE_PX] = 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)
    print(f"Caroiaj salvat: {out_path}")
    print(f"Dimensiune: {w}x{h} px  ({COLS+1}x{ROWS+1} patrate)")
    print(f"Tipareste la 100% scala — masoar un patrat si noteaza marimea in mm.")

if __name__ == "__main__":
    generate()
