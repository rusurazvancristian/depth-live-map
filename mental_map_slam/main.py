"""
Mental-map SLAM entry point.

Usage:
    python main.py

Reads all .mp4 files from INPUT_DIR, writes BEV maps + trajectories to OUTPUT_DIR.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("HF_HOME", "D:/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/hf_cache")

INPUT_DIR  = Path(__file__).resolve().parent / "output"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(INPUT_DIR.glob("*.mp4"))
    if not videos:
        print(f"No .mp4 files found in {INPUT_DIR}")
        sys.exit(1)

    print(f"Found {len(videos)} video(s): {[v.name for v in videos]}")

    # Lazy import after env vars are set
    from config import CAMERA, SLAM
    from depth_engine import DepthEngine
    from slam_pipeline import process_video

    depth_engine = DepthEngine(SLAM.depth_model_id, SLAM.depth_fallback_id, device="cuda",
                               metric_scale=SLAM.depth_metric_scale)

    bev_images: list[tuple[str, np.ndarray]] = []

    for video_path in videos:
        print(f"\nProcessing {video_path.name} ...")
        bev = process_video(video_path, OUTPUT_DIR, depth_engine, CAMERA, SLAM)
        bev_images.append((video_path.stem, bev))

    # Composite: all BEVs in a grid
    _save_composite(bev_images, OUTPUT_DIR / "all_maps.png")
    print(f"\nDone. Outputs saved to {OUTPUT_DIR}")


def _save_composite(images: list[tuple[str, np.ndarray]], out_path: Path) -> None:
    """Tile all BEV images into a single overview PNG."""
    if not images:
        return

    # Thumbnail size
    thumb = 600
    cols = min(3, len(images))
    rows = (len(images) + cols - 1) // cols
    canvas = np.full((rows * thumb, cols * thumb, 3), 10, dtype=np.uint8)

    for idx, (name, bev) in enumerate(images):
        r, c = divmod(idx, cols)
        tile = cv2.resize(bev, (thumb, thumb), interpolation=cv2.INTER_AREA)
        canvas[r * thumb:(r + 1) * thumb, c * thumb:(c + 1) * thumb] = tile
        cv2.putText(canvas, name,
                    (c * thumb + 8, r * thumb + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    cv2.imwrite(str(out_path), canvas)
    print(f"Composite saved: {out_path}")


if __name__ == "__main__":
    main()
