"""Visualization utilities — draws the live side-by-side HUD on the monitor."""

import cv2
import numpy as np
import math
from typing import Tuple, Dict
from data_contract import FrameResult

# Colormaps registry
COLORMAPS: Dict[str, int] = {
    "Magma": cv2.COLORMAP_MAGMA,
    "Turbo": cv2.COLORMAP_TURBO,
    "Inferno": cv2.COLORMAP_INFERNO,
    "Jet": cv2.COLORMAP_JET,
    "Hot": cv2.COLORMAP_HOT,
    "Viridis": cv2.COLORMAP_VIRIDIS,
}


def draw_hud(
    result: FrameResult,
    cmap_name: str = "Turbo",
    invert_depth: bool = False,
    display_w: int = 1280,
    display_h: int = 480,
) -> np.ndarray:
    """Create the side-by-side HUD display.

    Left side: Camera frame with tracking bboxes, lock status, and arrival triggers.
    Right side: Colormapped SCDepthV3 depth map with target projection.

    Returns:
        Combined BGR image of shape (display_h, display_w, 3) ready for cv2.imshow.
    """
    frame = result.frame
    orig_h, orig_w = frame.shape[:2]

    # 1. Resize camera frame to half the display width
    half_w = display_w // 2
    cam_vis = cv2.resize(frame, (half_w, display_h), interpolation=cv2.INTER_LINEAR)

    # Calculate scale factor from original camera frame to HUD coordinate space
    scale_x = half_w / orig_w
    scale_y = display_h / orig_h

    # 2. Draw centering region box (±10% of center)
    cx, cy = half_w // 2, display_h // 2
    tol_x = int(half_w * 0.10)
    tol_y = int(display_h * 0.10)
    cv2.rectangle(
        cam_vis,
        (cx - tol_x, cy - tol_y),
        (cx + tol_x, cy + tol_y),
        (100, 100, 100),
        1,
        lineType=cv2.LINE_AA,
    )

    # 3. Draw tracked objects
    for obj in result.tracked_objects:
        x1_orig, y1_orig, x2_orig, y2_orig = obj.bbox
        x1 = int(round(x1_orig * scale_x))
        y1 = int(round(y1_orig * scale_y))
        x2 = int(round(x2_orig * scale_x))
        y2 = int(round(y2_orig * scale_y))

        # Check if this object is the locked target
        is_target = (result.target_id != -1) and (obj.track_id == result.target_id)

        # Style configuration
        if is_target:
            if result.target_is_arrived:
                color = (0, 0, 255)  # Red for arrival
                thickness = 3
            else:
                color = (255, 0, 255)  # Purple for target lock
                thickness = 2
        else:
            color = (0, 255, 0)  # Green for general tracks
            thickness = 1

        # Draw bbox
        cv2.rectangle(cam_vis, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        
        # If target, draw corner brackets for premium look
        if is_target:
            length = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
            # Top-left corner
            cv2.line(cam_vis, (x1, y1), (x1 + length, y1), color, thickness + 1)
            cv2.line(cam_vis, (x1, y1), (x1, y1 + length), color, thickness + 1)
            # Top-right corner
            cv2.line(cam_vis, (x2, y1), (x2 - length, y1), color, thickness + 1)
            cv2.line(cam_vis, (x2, y1), (x2, y1 + length), color, thickness + 1)
            # Bottom-left corner
            cv2.line(cam_vis, (x1, y2), (x1 + length, y2), color, thickness + 1)
            cv2.line(cam_vis, (x1, y2), (x1, y2 - length), color, thickness + 1)
            # Bottom-right corner
            cv2.line(cam_vis, (x2, y2), (x2 - length, y2), color, thickness + 1)
            cv2.line(cam_vis, (x2, y2), (x2, y2 - length), color, thickness + 1)

        # Format label
        label_parts = [f"ID {obj.track_id}", obj.class_name]
        
        if not math.isnan(obj.kalman_distance_m):
            label_parts.append(f"{obj.kalman_distance_m:.2f}m")
            # Show standard deviation (sqrt of variance) if available
            if not math.isnan(obj.kalman_variance):
                std_dev = math.sqrt(obj.kalman_variance)
                label_parts.append(f"\u00b1{std_dev:.2f}")
        elif not math.isnan(obj.d_geometric_m):
            label_parts.append(f"~{obj.d_geometric_m:.1f}m")

        label = " | ".join(label_parts)

        # Draw text background banner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        text_thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        
        # Position label above bbox (or inside top if bbox is too high)
        ty = max(y1 - 6, th + 4)
        cv2.rectangle(
            cam_vis,
            (x1, ty - th - 2),
            (x1 + tw + 4, ty + baseline - 2),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.putText(
            cam_vis,
            label,
            (x1 + 2, ty),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            lineType=cv2.LINE_AA,
        )

    # 4. Translucent status display overlay in top-left
    panel_h = 95
    panel_w = 260
    status_panel = cam_vis[10 : 10 + panel_h, 10 : 10 + panel_w].copy()
    overlay = np.zeros_like(status_panel)
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(status_panel, 0.4, overlay, 0.6, 0, dst=cam_vis[10 : 10 + panel_h, 10 : 10 + panel_w])
    cv2.rectangle(cam_vis, (10, 10), (10 + panel_w, 10 + panel_h), (60, 60, 60), 1, lineType=cv2.LINE_AA)

    # Write status panel text
    cv2.putText(
        cam_vis,
        f"PIPELINE: ACTIVE (Hailo-8)",
        (20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        lineType=cv2.LINE_AA,
    )

    # Select color based on target status
    status = result.target_status
    if status == "LOCKED":
        status_color = (0, 255, 0)
    elif status == "SEARCHING":
        status_color = (0, 255, 255)
    elif status == "LOST":
        status_color = (0, 0, 255)
    else:
        status_color = (180, 180, 180)

    cv2.putText(
        cam_vis,
        f"STATUS: {status}",
        (20, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        status_color,
        2,
        lineType=cv2.LINE_AA,
    )

    if result.target_id != -1:
        cv2.putText(
            cam_vis,
            f"TARGET ID: {result.target_id}",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )
        if not math.isnan(result.target_distance_m):
            cv2.putText(
                cam_vis,
                f"TARGET DISTANCE: {result.target_distance_m:.2f} m",
                (20, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )
    else:
        cv2.putText(
            cam_vis,
            f"TARGET ID: None",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 150, 150),
            1,
            lineType=cv2.LINE_AA,
        )

    # 5. Flashing target arrived overlay
    if result.target_is_arrived:
        # Flash every ~15 frames (0.5s at 30fps)
        flash_on = (int(result.timestamp * 10) % 2) == 0
        if flash_on:
            overlay_banner = cam_vis[display_h - 70 : display_h - 20, cx - 180 : cx + 180].copy()
            banner_black = np.zeros_like(overlay_banner)
            cv2.rectangle(banner_black, (0, 0), (360, 50), (0, 0, 150), cv2.FILLED)
            cv2.addWeighted(overlay_banner, 0.2, banner_black, 0.8, 0, dst=cam_vis[display_h - 70 : display_h - 20, cx - 180 : cx + 180])
            cv2.rectangle(cam_vis, (cx - 180, display_h - 70), (cx + 180, display_h - 20), (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(
                cam_vis,
                "WARNING: TARGET ARRIVED",
                (cx - 150, display_h - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )

    # 6. Right side: Colormapped depth map representation
    if result.depth_map is not None:
        depth_norm = (result.depth_map * 255).astype(np.uint8)
        if invert_depth:
            depth_norm = 255 - depth_norm

        cmap_id = COLORMAPS.get(cmap_name, cv2.COLORMAP_TURBO)
        depth_color = cv2.applyColorMap(depth_norm, cmap_id)
        
        # Resize to HUD half-width
        depth_vis = cv2.resize(depth_color, (half_w, display_h), interpolation=cv2.INTER_LINEAR)

        # Draw target bbox projection on the depth map
        for obj in result.tracked_objects:
            is_target = (result.target_id != -1) and (obj.track_id == result.target_id)
            if is_target:
                x1_orig, y1_orig, x2_orig, y2_orig = obj.bbox
                x1 = int(round(x1_orig * scale_x))
                y1 = int(round(y1_orig * scale_y))
                x2 = int(round(x2_orig * scale_x))
                y2 = int(round(y2_orig * scale_y))

                color = (0, 0, 255) if result.target_is_arrived else (255, 0, 255)
                cv2.rectangle(depth_vis, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
                
                # Draw crosshair inside target on depth map
                tx_c = (x1 + x2) // 2
                ty_c = (y1 + y2) // 2
                cv2.drawMarker(depth_vis, (tx_c, ty_c), color, cv2.MARKER_CROSS, 14, 1)

        # Draw scalebar overlay on depth map
        scalebar_w = 30
        scalebar = np.zeros((display_h, scalebar_w, 3), dtype=np.uint8)
        grad = np.arange(display_h, dtype=np.uint8).reshape(-1, 1)
        # Scale to 0-255
        grad = (grad * 255 / display_h).astype(np.uint8)
        if not invert_depth:
            grad = 255 - grad
        colored_grad = cv2.applyColorMap(grad, cmap_id)
        scalebar[:] = np.repeat(colored_grad, scalebar_w, axis=1)
        
        # Draw "near" / "far" text on scalebar
        cv2.putText(scalebar, "Near", (2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.putText(scalebar, "Far", (2, display_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        # Overlay scalebar on the rightmost edge of depth map
        depth_vis[:, half_w - scalebar_w :] = scalebar
    else:
        # Fallback if depth map is missing
        depth_vis = np.zeros((display_h, half_w, 3), dtype=np.uint8)
        cv2.putText(
            depth_vis,
            "DEPTH MAP UNAVAILABLE",
            (half_w // 2 - 120, display_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            1,
            lineType=cv2.LINE_AA,
        )

    # 7. Stack side-by-side
    combined_hud = np.hstack((cam_vis, depth_vis))
    return combined_hud
