import cv2
import numpy as np
from data_contract import FrameState
from config import CameraConfig, SLAMConfig


class VisualOdometry:
    """ORB feature tracking + PnP-RANSAC pose estimation."""

    def __init__(self, camera: CameraConfig, cfg: SLAMConfig):
        self.cam = camera
        self.cfg = cfg
        self.K = np.array(
            [[camera.fx, 0, camera.cx],
             [0, camera.fy, camera.cy],
             [0, 0, 1]],
            dtype=np.float64,
        )
        self.orb = cv2.ORB_create(
            nfeatures=cfg.max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def detect(self, bgr: np.ndarray) -> tuple[list, np.ndarray]:
        """Return (keypoints, descriptors) for frame."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        kps, descs = self.orb.detectAndCompute(gray, None)
        if kps is None:
            return [], np.zeros((0, 32), dtype=np.uint8)
        if descs is None:
            return list(kps), np.zeros((0, 32), dtype=np.uint8)
        return list(kps), descs

    def estimate_pose(
        self,
        prev: FrameState,
        curr_kps: list,
        curr_descs: np.ndarray,
    ) -> np.ndarray | None:
        """
        Estimate T_prev_to_curr (4×4 SE3) using depth-lifted PnP.

        PnP solves: p_curr_cam = R * p_prev_cam + t
        Returns None when tracking fails.
        """
        if (
            prev.depth_map is None
            or prev.descriptors is None
            or len(prev.descriptors) == 0
            or len(curr_descs) == 0
        ):
            return None

        matches = self.matcher.match(prev.descriptors, curr_descs)
        matches = sorted(matches, key=lambda m: m.distance)[:500]

        pts3d, pts2d = [], []
        for m in matches:
            u1, v1 = prev.keypoints[m.queryIdx].pt
            ui, vi = int(round(u1)), int(round(v1))
            if not (0 <= ui < self.cam.width and 0 <= vi < self.cam.height):
                continue
            z = float(prev.depth_map[vi, ui])
            if not (self.cfg.depth_min_m < z < self.cfg.depth_max_m):
                continue
            pts3d.append([
                (u1 - self.cam.cx) * z / self.cam.fx,
                (v1 - self.cam.cy) * z / self.cam.fy,
                z,
            ])
            pts2d.append(curr_kps[m.trainIdx].pt)

        if len(pts3d) < self.cfg.min_pnp_inliers:
            return None

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.array(pts3d, dtype=np.float64),
            np.array(pts2d, dtype=np.float64),
            self.K,
            None,
            reprojectionError=self.cfg.pnp_reproj_error,
            iterationsCount=1000,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok or inliers is None or len(inliers) < self.cfg.min_pnp_inliers:
            return None

        translation_m = float(np.linalg.norm(tvec))
        if translation_m > self.cfg.max_translation_m:
            return None  # outlier jump

        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T
