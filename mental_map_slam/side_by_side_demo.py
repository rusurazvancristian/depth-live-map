"""
Side-by-side SLAM + detection demo renderer.

  python D:\\mental_map_slam\\side_by_side_demo.py

Output per video:  output/video_N_demo.mp4

Layout (1920 x 400):
  [ RGB + ORB + YOLO boxes + distances ]
  [ Depth map (DepthAnything V2)       ]
  [ SLAM BEV (live) + object labels    ]
  [           info bar                 ]
"""

import os, sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

os.environ.setdefault("HF_HOME", "D:/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/hf_cache")

INPUT_DIR  = Path(__file__).resolve().parent / "output"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

PANEL_W, PANEL_H = 640, 360
INFO_H           = 40
OUT_W            = PANEL_W * 3     # 1920
OUT_H            = PANEL_H + INFO_H  # 400
OUT_FPS          = 5.0
BEV_WIN_W_M      = 32.0            # BEV crop width in metres
BEV_WIN_H_M      = 18.0            # BEV crop height in metres


# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    videos = sorted(INPUT_DIR.glob("*.mp4"))
    if not videos:
        print(f"No videos in {INPUT_DIR}"); sys.exit(1)

    from config import CAMERA, SLAM
    from depth_engine import DepthEngine
    from object_detector import ObjectDetector

    depth_engine = DepthEngine(SLAM.depth_model_id, SLAM.depth_fallback_id, "cuda",
                               metric_scale=SLAM.depth_metric_scale)

    try:
        detector = ObjectDetector("yolov8n.pt", conf_thresh=0.35, device="cuda")
    except Exception as e:
        print(f"[warn] YOLO unavailable ({e}) — running without detection")
        detector = None

    for vp in videos:
        print(f"\nRendering: {vp.name}")
        _render_video(vp, OUTPUT_DIR, depth_engine, detector, CAMERA, SLAM)

    print("\nAll done.", OUTPUT_DIR)


# ---------------------------------------------------------------------------
def _render_video(video_path, output_dir, depth_engine, detector, camera, cfg):
    from visual_odometry import VisualOdometry
    from map_builder import MapBuilder
    from data_contract import FrameState
    from object_detector import ObjectMemory

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Cannot open {video_path}"); return

    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = output_dir / f"{video_path.stem}_demo.mp4"
    writer   = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"avc1"),  # H.264 — browser compatible
        OUT_FPS, (OUT_W, OUT_H),
    )

    vo         = VisualOdometry(camera, cfg)
    mapper     = MapBuilder(camera, cfg)
    obj_memory = ObjectMemory(merge_dist=0.8)

    pose_c2w   = np.eye(4, dtype=np.float64)
    prev_state = None
    frame_id   = 0
    kf_id      = 0

    pbar = tqdm(total=total // cfg.frame_stride, desc=video_path.stem, unit="kf")

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if frame_id % cfg.frame_stride != 0:
            frame_id += 1
            continue

        timestamp = frame_id / camera.fps

        # ── SLAM ────────────────────────────────────────────────────────────
        depth      = depth_engine.infer(bgr)
        kps, descs = vo.detect(bgr)

        if prev_state is not None:
            T = vo.estimate_pose(prev_state, kps, descs)
            if T is not None:
                pose_c2w = prev_state.pose_c2w @ np.linalg.inv(T)

        mapper.update(depth, pose_c2w)

        # ── Object detection + distance + 3-D projection ────────────────────
        if detector is not None:
            dets = detector.detect(bgr)
            detector.measure_distances(dets, depth, camera=camera,
                                       d_min=cfg.depth_min_m,
                                       d_max=cfg.depth_max_m)
            detector.project_to_world(dets, pose_c2w, camera)
            obj_memory.update(dets, frame_id)
        else:
            dets = []

        # ── Compose panels ───────────────────────────────────────────────────
        p_rgb   = _panel_rgb(bgr, kps, dets, depth, cfg.depth_max_m)
        p_depth = _panel_depth(depth, cfg.depth_min_m, cfg.depth_max_m, dets)
        p_bev   = _panel_bev(mapper, obj_memory, pose_c2w, cfg)

        _stamp(p_rgb,   "RGB  |  ORB  |  YOLO + Distance")
        _stamp(p_depth, "Depth — DepthAnything V2")
        _stamp(p_bev,   "SLAM — Bird's Eye View  |  Semantic Map")

        top = np.hstack([p_rgb, p_depth, p_bev])
        cv2.line(top, (PANEL_W,   0), (PANEL_W,   PANEL_H), (55, 55, 55), 2)
        cv2.line(top, (PANEL_W*2, 0), (PANEL_W*2, PANEL_H), (55, 55, 55), 2)

        scale = detector.depth_scale if detector is not None else 1.0
        bar = _info_bar(frame_id, timestamp, len(kps), len(dets), pose_c2w, scale)
        writer.write(np.vstack([top, bar]))

        prev_state = FrameState(
            frame_id=frame_id, timestamp=timestamp,
            image=bgr, depth_map=depth,
            keypoints=kps, descriptors=descs,
            pose_c2w=pose_c2w.copy(),
        )
        frame_id += 1
        kf_id    += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    print(f"  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def _panel_rgb(frame, keypoints, dets, depth_map, depth_max):
    """Original frame + ORB keypoints (depth-coloured) + YOLO boxes with distances."""
    sx = PANEL_W / frame.shape[1]
    sy = PANEL_H / frame.shape[0]
    panel = cv2.resize(frame, (PANEL_W, PANEL_H))

    # ORB features: green=close, red=far
    for kp in keypoints[:350]:
        u0, v0 = int(kp.pt[0]), int(kp.pt[1])
        if 0 <= u0 < frame.shape[1] and 0 <= v0 < frame.shape[0]:
            t = min(float(depth_map[v0, u0]) / depth_max, 1.0)
            col = (0, int(255*(1-t)), int(255*t))
            px, py = int(kp.pt[0]*sx), int(kp.pt[1]*sy)
            cv2.circle(panel, (px, py), 2, (0, 0, 0), -1)
            cv2.circle(panel, (px, py), 1, col, -1)

    # YOLO bounding boxes + distance labels
    from object_detector import ObjectDetector
    for det in dets:
        if det.distance_m is None:
            continue
        x1, y1, x2, y2 = det.bbox_xyxy
        px1, py1 = int(x1*sx), int(y1*sy)
        px2, py2 = int(x2*sx), int(y2*sy)
        color = ObjectDetector.class_color(det.class_name)

        # Pinhole-anchored detections get a thicker box (they set the scale)
        is_anchor = det.method == "pinhole"
        cv2.rectangle(panel, (px1, py1), (px2, py2), color, 3 if is_anchor else 2)

        # Label: "class  X.Xm" + marker for known-height calibration anchor
        marker = " [cal]" if is_anchor else ""
        label = f"{det.class_name}  {det.distance_m:.1f}m{marker}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        ly = max(py1 - 2, th + 4)
        cv2.rectangle(panel, (px1, ly - th - 4), (px1 + tw + 6, ly + 2), color, -1)
        cv2.putText(panel, label, (px1 + 3, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    return panel


def _panel_depth(depth_map, d_min, d_max, dets):
    """Colorised depth map (warm=close, cool=far) with distance cross-hairs."""
    clipped = np.clip(depth_map, d_min, d_max)
    norm    = ((clipped - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    panel   = cv2.applyColorMap(255 - norm, cv2.COLORMAP_TURBO)
    panel   = cv2.resize(panel, (PANEL_W, PANEL_H))

    # Re-draw measurement cross at bbox centre
    sx = PANEL_W / depth_map.shape[1]
    sy = PANEL_H / depth_map.shape[0]
    from object_detector import ObjectDetector
    for det in dets:
        if det.distance_m is None:
            continue
        x1, y1, x2, y2 = det.bbox_xyxy
        cx = int((x1 + x2) * 0.5 * sx)
        cy = int((y1 + y2) * 0.5 * sy)
        color = ObjectDetector.class_color(det.class_name)
        cv2.drawMarker(panel, (cx, cy), color,
                       cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
        cv2.putText(panel, f"{det.distance_m:.1f}m", (cx + 8, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    return panel


def _panel_bev(mapper, obj_memory, pose_c2w, cfg):
    """Live BEV crop with all remembered object positions labelled."""
    panel = mapper.render_crop(pose_c2w, PANEL_W, PANEL_H,
                               BEV_WIN_W_M, BEV_WIN_H_M)

    # Overlay persistent semantic objects
    from object_detector import ObjectDetector
    for obj in obj_memory.objects:
        wx, _, wz = obj["pos"]
        px, pz = _world_to_bev_px(wx, wz, pose_c2w, cfg)
        if not (0 <= px < PANEL_W and 0 <= pz < PANEL_H):
            continue
        color = ObjectDetector.class_color(obj["name"])
        dist_str = f"{obj['dist']:.1f}m" if obj["dist"] else ""
        label    = f"{obj['name']} {dist_str}"

        # Dot
        cv2.circle(panel, (px, pz), 6, (0, 0, 0), -1)
        cv2.circle(panel, (px, pz), 4, color, -1)

        # Label with background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
        lx, ly = px + 7, pz + 4
        cv2.rectangle(panel, (lx - 1, ly - th - 1), (lx + tw + 2, ly + 2),
                      (0, 0, 0), -1)
        cv2.putText(panel, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

    return panel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _world_to_bev_px(wx: float, wz: float, pose_c2w: np.ndarray, cfg) -> tuple[int, int]:
    """Map world (X, Z) → panel pixel, matching render_crop's coordinate transform."""
    sz     = cfg.bev_size_px
    res    = cfg.bev_res_m
    origin = sz // 2

    cam_x = pose_c2w[0, 3] / res + origin
    cam_z = pose_c2w[2, 3] / res + origin
    hw    = BEV_WIN_W_M / 2 / res
    hh    = BEV_WIN_H_M / 2 / res

    x0 = max(0.0, cam_x - hw)
    x1 = min(float(sz), cam_x + hw)
    z0 = max(0.0, cam_z - hh)
    z1 = min(float(sz), cam_z + hh)

    gx = wx / res + origin
    gz = wz / res + origin

    px = int((gx - x0) / max(x1 - x0, 1) * PANEL_W)
    pz = int((gz - z0) / max(z1 - z0, 1) * PANEL_H)
    return px, pz


def _stamp(panel: np.ndarray, text: str) -> None:
    tw = len(text) * 7 + 10
    cv2.rectangle(panel, (0, 0), (tw, 24), (0, 0, 0), -1)
    cv2.putText(panel, text, (5, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 215, 255), 1, cv2.LINE_AA)


def _info_bar(frame_id, timestamp, n_features, n_dets, pose_c2w, depth_scale=1.0):
    bar = np.zeros((INFO_H, OUT_W, 3), dtype=np.uint8)
    cv2.rectangle(bar, (0, 0), (OUT_W, INFO_H), (8, 8, 18), -1)
    x, z = float(pose_c2w[0, 3]), float(pose_c2w[2, 3])
    text = (f"  frame {frame_id:05d}  |  t {timestamp:6.2f}s  |  "
            f"cam  X{x:+6.2f}m  Z{z:+6.2f}m  |  "
            f"ORB {n_features:4d}  |  det {n_dets:2d}  |  "
            f"depth scale x{depth_scale:.2f}")
    cv2.putText(bar, text, (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (120, 190, 120), 1, cv2.LINE_AA)
    return bar


if __name__ == "__main__":
    main()
