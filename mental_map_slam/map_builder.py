import numpy as np
import cv2
from data_contract import MapState
from config import CameraConfig, SLAMConfig


class MapBuilder:
    """Accumulates a bird's-eye-view occupancy + height map."""

    def __init__(self, camera: CameraConfig, cfg: SLAMConfig):
        self.cam = camera
        self.cfg = cfg
        sz = cfg.bev_size_px
        self._origin = sz // 2  # world (0,0) maps to image center
        self.state = MapState(
            bev_count=np.zeros((sz, sz), dtype=np.int32),
            bev_height_acc=np.zeros((sz, sz), dtype=np.float64),
        )

    def update(self, depth_map: np.ndarray, pose_c2w: np.ndarray) -> None:
        """Project depth pixels into world space and accumulate into BEV grid."""
        h, w = depth_map.shape
        s = self.cfg.depth_subsample
        vs, us = np.arange(0, h, s), np.arange(0, w, s)
        uu, vv = np.meshgrid(us, vs)
        uu, vv = uu.ravel(), vv.ravel()

        zz = depth_map[vv, uu].astype(np.float64)
        valid = (zz > self.cfg.depth_min_m) & (zz < self.cfg.depth_max_m)
        uu, vv, zz = uu[valid], vv[valid], zz[valid]

        # Unproject to camera frame  (X=right, Y=down, Z=forward)
        xx = (uu - self.cam.cx) * zz / self.cam.fx
        yy = (vv - self.cam.cy) * zz / self.cam.fy

        pts_cam = np.stack([xx, yy, zz, np.ones_like(zz)], axis=1)  # N×4
        pts_world = (pose_c2w @ pts_cam.T).T                          # N×4

        xw = pts_world[:, 0]
        yw = pts_world[:, 1]
        zw = pts_world[:, 2]

        sz = self.cfg.bev_size_px
        gx = (xw / self.cfg.bev_res_m + self._origin).astype(int)
        gz = (zw / self.cfg.bev_res_m + self._origin).astype(int)

        in_bounds = (gx >= 0) & (gx < sz) & (gz >= 0) & (gz < sz)
        np.add.at(self.state.bev_count,      (gz[in_bounds], gx[in_bounds]), 1)
        np.add.at(self.state.bev_height_acc, (gz[in_bounds], gx[in_bounds]), yw[in_bounds])

        self.state.trajectory.append(pose_c2w[:3, 3].copy())
        self.state.frame_count += 1

    def render_bev(self) -> np.ndarray:
        """Return BGR BEV image with height coloring and trajectory."""
        cnt = self.state.bev_count
        hacc = self.state.bev_height_acc
        sz = self.cfg.bev_size_px

        seen = cnt > 0
        img = np.full((sz, sz, 3), 20, dtype=np.uint8)  # dark background

        if seen.any():
            avg_h = np.zeros((sz, sz), dtype=np.float32)
            avg_h[seen] = (hacc[seen] / cnt[seen]).astype(np.float32)

            h_vals = avg_h[seen]
            lo = float(np.percentile(h_vals, 5))
            hi = float(np.percentile(h_vals, 95))
            norm = np.zeros((sz, sz), dtype=np.float32)
            norm[seen] = np.clip((avg_h[seen] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

            norm_u8 = (norm * 255).astype(np.uint8)
            colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)
            img[seen] = colored[seen]

        # Grid lines every 10m (200 px at 5cm/px)
        step = int(10.0 / self.cfg.bev_res_m)
        for g in range(0, sz, step):
            cv2.line(img, (g, 0), (g, sz - 1), (40, 40, 40), 1)
            cv2.line(img, (0, g), (sz - 1, g), (40, 40, 40), 1)

        # Origin cross
        o = self._origin
        cv2.line(img, (o - 20, o), (o + 20, o), (80, 80, 80), 1)
        cv2.line(img, (o, o - 20), (o, o + 20), (80, 80, 80), 1)

        # Trajectory (yellow)
        for pos in self.state.trajectory:
            gx = int(pos[0] / self.cfg.bev_res_m + self._origin)
            gz = int(pos[2] / self.cfg.bev_res_m + self._origin)
            if 0 <= gx < sz and 0 <= gz < sz:
                cv2.circle(img, (gx, gz), 2, (0, 255, 255), -1)

        # Start (green) / end (red) markers
        traj = self.state.trajectory
        if traj:
            def _gpos(p: np.ndarray) -> tuple[int, int]:
                return (
                    int(p[0] / self.cfg.bev_res_m + self._origin),
                    int(p[2] / self.cfg.bev_res_m + self._origin),
                )

            s_pos = _gpos(traj[0])
            e_pos = _gpos(traj[-1])
            cv2.circle(img, s_pos, 8, (0, 255, 0), 2)
            cv2.circle(img, e_pos, 8, (0, 0, 255), 2)
            cv2.putText(img, "S", (s_pos[0] + 10, s_pos[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(img, "E", (e_pos[0] + 10, e_pos[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Scale bar: 10m
        bar_px = int(10.0 / self.cfg.bev_res_m)
        bx, by = 40, sz - 40
        cv2.line(img, (bx, by), (bx + bar_px, by), (200, 200, 200), 2)
        cv2.putText(img, "10 m", (bx, by - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return img

    def render_crop(
        self,
        pose_c2w: np.ndarray,
        out_w: int,
        out_h: int,
        window_w_m: float = 32.0,
        window_h_m: float = 18.0,
    ) -> np.ndarray:
        """Fast BEV crop centred on current camera — only processes the visible region."""
        sz  = self.cfg.bev_size_px
        res = self.cfg.bev_res_m

        cam_x = int(pose_c2w[0, 3] / res + self._origin)
        cam_z = int(pose_c2w[2, 3] / res + self._origin)

        hw = int(window_w_m / 2 / res)
        hh = int(window_h_m / 2 / res)
        x0, x1 = max(0, cam_x - hw), min(sz, cam_x + hw)
        z0, z1 = max(0, cam_z - hh), min(sz, cam_z + hh)

        if x1 <= x0 or z1 <= z0:
            return np.zeros((out_h, out_w, 3), dtype=np.uint8)

        cnt  = self.state.bev_count[z0:z1, x0:x1]
        hacc = self.state.bev_height_acc[z0:z1, x0:x1]
        seen = cnt > 0

        tile = np.full((z1 - z0, x1 - x0, 3), 20, dtype=np.uint8)
        if seen.any():
            avg_h = np.zeros((z1 - z0, x1 - x0), dtype=np.float32)
            avg_h[seen] = (hacc[seen] / cnt[seen]).astype(np.float32)
            h_vals = avg_h[seen]
            lo = float(np.percentile(h_vals, 5))
            hi = float(np.percentile(h_vals, 95))
            norm = np.zeros_like(avg_h)
            norm[seen] = np.clip((avg_h[seen] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            tile[seen] = colored[seen]

        # Trajectory dots inside the crop window
        for pos in self.state.trajectory:
            tx = int(pos[0] / res + self._origin) - x0
            tz = int(pos[2] / res + self._origin) - z0
            if 0 <= tx < tile.shape[1] and 0 <= tz < tile.shape[0]:
                cv2.circle(tile, (tx, tz), 1, (0, 255, 255), -1)

        # Camera marker (cyan filled + black outline)
        mx = np.clip(cam_x - x0, 0, tile.shape[1] - 1)
        mz = np.clip(cam_z - z0, 0, tile.shape[0] - 1)
        cv2.circle(tile, (mx, mz), 9, (0, 0, 0), -1)
        cv2.circle(tile, (mx, mz), 7, (0, 255, 255), -1)

        return cv2.resize(tile, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    def reset(self) -> None:
        sz = self.cfg.bev_size_px
        self.state = MapState(
            bev_count=np.zeros((sz, sz), dtype=np.int32),
            bev_height_acc=np.zeros((sz, sz), dtype=np.float64),
        )
