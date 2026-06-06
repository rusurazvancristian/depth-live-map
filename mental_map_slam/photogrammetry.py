"""
Generate per-video and merged point clouds for the 3D Space Explorer.

Output:
  output/video_N_cloud.ply   — per-video, up to 150K points (subsample=8)
  output/merged_cloud.ply    — all videos merged, up to 300K points
"""
import os, sys
from pathlib import Path
import numpy as np

os.environ.setdefault("HF_HOME", "D:/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/hf_cache")

INPUT_DIR  = Path(__file__).resolve().parent / "output"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

SUBSAMPLE      = 8        # denser than SLAM demo (16)
VOXEL_SIZE_V   = 0.07     # 7cm voxels per video
VOXEL_SIZE_M   = 0.10     # 10cm voxels merged
MAX_PER_VIDEO  = 150_000
MAX_MERGED     = 300_000


def main() -> None:
    from config import CAMERA, SLAM
    from depth_engine import DepthEngine
    from slam_pipeline import process_video
    from pointcloud_builder import PointCloudBuilder

    depth_engine = DepthEngine(SLAM.depth_model_id, SLAM.depth_fallback_id, "cuda",
                               metric_scale=SLAM.depth_metric_scale)

    all_pts:  list[np.ndarray] = []
    all_cols: list[np.ndarray] = []

    for video_path in sorted(INPUT_DIR.glob("*.mp4")):
        stem = video_path.stem
        out  = OUTPUT_DIR / f"{stem}_cloud.ply"
        print(f"\n{video_path.name}")

        pcd = PointCloudBuilder(CAMERA, subsample=SUBSAMPLE, depth_max=12.0)
        process_video(video_path, OUTPUT_DIR, depth_engine, CAMERA, SLAM, pcd_builder=pcd)

        pts, cols = pcd.get()
        pts[:, 1] *= -1   # camera Y=down → viewer Y=up

        pts, cols = PointCloudBuilder.voxel_downsample(pts, cols, VOXEL_SIZE_V)
        if len(pts) > MAX_PER_VIDEO:
            idx = np.random.choice(len(pts), MAX_PER_VIDEO, replace=False)
            pts, cols = pts[idx], cols[idx]

        PointCloudBuilder.save_ply(out, pts, cols)
        sz = out.stat().st_size
        print(f"  {len(pts):,} pts  ->  {out.name}  ({sz//1024} KB)")

        all_pts.append(pts)
        all_cols.append(cols)

    # Merged
    print("\nBuilding merged cloud…")
    m_pts  = np.concatenate(all_pts)
    m_cols = np.concatenate(all_cols)
    m_pts, m_cols = PointCloudBuilder.voxel_downsample(m_pts, m_cols, VOXEL_SIZE_M)
    if len(m_pts) > MAX_MERGED:
        idx = np.random.choice(len(m_pts), MAX_MERGED, replace=False)
        m_pts, m_cols = m_pts[idx], m_cols[idx]

    out_m = OUTPUT_DIR / "merged_cloud.ply"
    PointCloudBuilder.save_ply(out_m, m_pts, m_cols)
    print(f"  {len(m_pts):,} pts  ->  {out_m.name}  ({out_m.stat().st_size//1024} KB)")
    print("\nDone. Open the Space Explorer in the pitch app.")


if __name__ == "__main__":
    main()
