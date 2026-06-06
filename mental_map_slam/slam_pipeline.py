"""
Monocular visual SLAM pipeline using DepthAnything V2 + ORB + PnP.

Coordinate convention (OpenCV camera standard):
  Camera frame: X=right, Y=down, Z=forward
  World frame: same, initialized at first keyframe

Pose chain:
  T_prev_to_curr  = solvePnP output   (prev_cam → curr_cam)
  T_curr_c2w      = T_prev_c2w  @  inv(T_prev_to_curr)
"""

from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

from config import CameraConfig, SLAMConfig, CAMERA, SLAM
from data_contract import FrameState
from depth_engine import DepthEngine
from visual_odometry import VisualOdometry
from map_builder import MapBuilder


def process_video(
    video_path: str | Path,
    output_dir: str | Path,
    depth_engine: DepthEngine,
    camera: CameraConfig = CAMERA,
    cfg: SLAMConfig = SLAM,
    pcd_builder=None,       # optional PointCloudBuilder — receives (depth, bgr, pose)
) -> np.ndarray:
    """
    Process one video through the SLAM pipeline.
    Returns the rendered BEV image.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    vo = VisualOdometry(camera, cfg)
    mapper = MapBuilder(camera, cfg)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or camera.fps

    pose_c2w = np.eye(4, dtype=np.float64)  # camera-to-world; starts at identity
    prev_state: FrameState | None = None
    frame_id = 0
    keyframe_count = 0

    pbar = tqdm(total=total // cfg.frame_stride, desc=video_path.stem, unit="kf")

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if frame_id % cfg.frame_stride != 0:
            frame_id += 1
            continue

        timestamp = frame_id / fps

        # --- Depth inference ---
        depth = depth_engine.infer(bgr)

        # --- Feature detection ---
        kps, descs = vo.detect(bgr)

        # --- Pose estimation from previous keyframe ---
        if prev_state is not None:
            T_prev_to_curr = vo.estimate_pose(prev_state, kps, descs)
            if T_prev_to_curr is not None:
                # T_curr_c2w = T_prev_c2w @ inv(T_prev_to_curr)
                pose_c2w = prev_state.pose_c2w @ np.linalg.inv(T_prev_to_curr)
            # On failure: keep previous pose (static assumption)

        # --- Accumulate into map ---
        mapper.update(depth, pose_c2w)

        if pcd_builder is not None:
            pcd_builder.update(depth, bgr, pose_c2w)

        state = FrameState(
            frame_id=frame_id,
            timestamp=timestamp,
            image=bgr,
            depth_map=depth,
            keypoints=kps,
            descriptors=descs,
            pose_c2w=pose_c2w.copy(),
        )
        prev_state = state
        frame_id += 1
        keyframe_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    bev = mapper.render_bev()

    stem = video_path.stem
    cv2.imwrite(str(output_dir / f"{stem}_bev.png"), bev)
    np.save(str(output_dir / f"{stem}_trajectory.npy"),
            np.array(mapper.state.trajectory))

    print(f"  [{stem}] {keyframe_count} keyframes | "
          f"trajectory range: "
          f"X={_range(mapper.state.trajectory, 0):.1f}m  "
          f"Z={_range(mapper.state.trajectory, 2):.1f}m")

    return bev


def _range(traj: list, axis: int) -> float:
    if not traj:
        return 0.0
    vals = [p[axis] for p in traj]
    return max(vals) - min(vals)
