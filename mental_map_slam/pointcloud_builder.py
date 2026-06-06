import numpy as np
from pathlib import Path
from config import CameraConfig, CAMERA


class PointCloudBuilder:
    """Accumulates a colored 3D point cloud from depth maps + camera poses."""

    def __init__(
        self,
        camera: CameraConfig = CAMERA,
        subsample: int = 16,
        depth_min: float = 0.3,
        depth_max: float = 12.0,
    ):
        self.camera = camera
        self.subsample = subsample
        self.depth_min = depth_min
        self.depth_max = depth_max
        self._pts: list[np.ndarray] = []
        self._cols: list[np.ndarray] = []

    def update(
        self,
        depth_map: np.ndarray,
        bgr_frame: np.ndarray,
        pose_c2w: np.ndarray,
    ) -> None:
        h, w = depth_map.shape
        s = self.subsample
        uu, vv = np.meshgrid(np.arange(0, w, s), np.arange(0, h, s))
        uu, vv = uu.ravel(), vv.ravel()

        zz = depth_map[vv, uu].astype(np.float64)
        valid = (zz > self.depth_min) & (zz < self.depth_max)
        uu, vv, zz = uu[valid], vv[valid], zz[valid]

        xx = (uu - self.camera.cx) * zz / self.camera.fx
        yy = (vv - self.camera.cy) * zz / self.camera.fy

        pts_cam = np.stack([xx, yy, zz, np.ones_like(zz)], axis=1)
        pts_world = (pose_c2w @ pts_cam.T).T[:, :3].astype(np.float32)
        rgb = bgr_frame[vv, uu, ::-1].astype(np.uint8)   # BGR → RGB

        self._pts.append(pts_world)
        self._cols.append(rgb)

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (N×3 float32 XYZ, N×3 uint8 RGB)."""
        if not self._pts:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
        return np.concatenate(self._pts), np.concatenate(self._cols)

    def reset(self) -> None:
        self._pts = []
        self._cols = []

    @staticmethod
    def voxel_downsample(
        pts: np.ndarray,
        cols: np.ndarray,
        voxel_size: float = 0.12,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Keep one point per voxel using a hash-based approach."""
        if len(pts) == 0:
            return pts, cols
        vox = np.floor(pts / voxel_size).astype(np.int32)
        key = (vox[:, 0].astype(np.int64) * 1_299_721
               + vox[:, 1].astype(np.int64) * 1_000_003
               + vox[:, 2].astype(np.int64))
        _, first = np.unique(key, return_index=True)
        return pts[first], cols[first]

    @staticmethod
    def save_ply(path: Path, pts: np.ndarray, cols: np.ndarray) -> None:
        """Save as binary PLY (float32 xyz + uint8 rgb) — opens in MeshLab / CloudCompare / Open3D."""
        n = len(pts)
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        pts_f32 = np.ascontiguousarray(pts.astype(np.float32))
        cols_u8 = cols.astype(np.uint8)
        vtx = np.empty((n, 15), dtype=np.uint8)
        vtx[:, :12] = pts_f32.view(np.uint8)   # 3×float32 = 12 bytes
        vtx[:, 12:] = cols_u8
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(vtx.tobytes())
