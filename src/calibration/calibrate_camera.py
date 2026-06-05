"""Camera calibration via chessboard pattern. [TRACK A]

Usage:
    python3 src/calibration/calibrate_camera.py --images_dir calibration_images/
                                                  --pattern 9 6
                                                  --square_mm 25.0
                                                  --out src/calibration/intrinsics.json

Saves intrinsics.json with camera_matrix (3x3) and dist_coeffs (5,).
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np


def calibrate_camera(
    images_dir: str,
    pattern_size: tuple[int, int] = (9, 6),
    square_size_m: float = 0.025,
) -> tuple[np.ndarray, np.ndarray]:
    """Run OpenCV chessboard calibration on a set of images.

    Args:
        images_dir: Directory containing calibration JPG/PNG images.
        pattern_size: (cols, rows) inner corners on the chessboard.
        square_size_m: Physical size of one chessboard square in metres.

    Returns:
        (camera_matrix K (3,3), dist_coeffs (5,))

    Raises:
        ValueError: If fewer than 5 valid calibration images are found.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cols, rows = pattern_size

    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_m

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    img_shape: tuple[int, int] = (0, 0)

    paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.png"))
    )

    found_count = 0
    for path in paths:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        ok, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if ok:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)
            found_count += 1
            print(f"  [OK] {os.path.basename(path)}")
        else:
            print(f"  [--] {os.path.basename(path)} — pattern not found")

    if found_count < 5:
        raise ValueError(f"Only {found_count} valid images found, need at least 5.")

    print(f"\nCalibrating with {found_count} images...")
    _, K, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, img_shape, None, None,
    )
    print(f"Calibration done. f_x={K[0,0]:.1f} f_y={K[1,1]:.1f}")
    return K, dist.flatten()


def save_intrinsics(K: np.ndarray, dist: np.ndarray, out_path: str) -> None:
    """Save camera matrix and distortion coefficients to JSON.

    Args:
        K: 3x3 camera intrinsic matrix.
        dist: Distortion coefficients array of shape (5,).
        out_path: Output JSON file path.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = {
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.tolist(),
        "focal_length_px": float(K[1, 1]),
        "principal_point": [float(K[0, 2]), float(K[1, 2])],
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved intrinsics to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera calibration via chessboard.")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--pattern", nargs=2, type=int, default=[9, 6], metavar=("COLS", "ROWS"))
    parser.add_argument("--square_mm", type=float, default=25.0)
    parser.add_argument("--out", default="src/calibration/intrinsics.json")
    args = parser.parse_args()

    K, dist = calibrate_camera(
        args.images_dir,
        pattern_size=(args.pattern[0], args.pattern[1]),
        square_size_m=args.square_mm / 1000.0,
    )
    save_intrinsics(K, dist, args.out)
