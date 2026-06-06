"""
Demo: interactive 3D point cloud from mental-map videos.

  python D:\\mental_map_slam\\pointcloud_demo.py

Outputs:
  output/merged_pointcloud.ply   — open in MeshLab / CloudCompare / Open3D
  output/demo_3d.html            — self-contained Three.js interactive viewer
"""

import os, sys, base64, json, webbrowser
from pathlib import Path
import numpy as np

os.environ.setdefault("HF_HOME", "D:/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/hf_cache")

INPUT_DIR  = Path(__file__).resolve().parent / "output"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Per-video trajectory colors (hex int for JS)
TRAJ_COLORS = [0xFFFF00, 0x00FFFF, 0xFF44FF, 0xFF8800, 0x88FF00]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    videos = sorted(INPUT_DIR.glob("*.mp4"))
    if not videos:
        print(f"No videos in {INPUT_DIR}"); sys.exit(1)

    from config import CAMERA, SLAM
    from depth_engine import DepthEngine
    from slam_pipeline import process_video
    from pointcloud_builder import PointCloudBuilder

    depth_engine = DepthEngine(SLAM.depth_model_id, SLAM.depth_fallback_id, "cuda",
                               metric_scale=SLAM.depth_metric_scale)

    all_pts:  list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    trajectories: list[np.ndarray] = []

    for video_path in videos:
        print(f"\n3D pass: {video_path.name}")
        pcd = PointCloudBuilder(CAMERA, subsample=16)
        process_video(video_path, OUTPUT_DIR, depth_engine, CAMERA, SLAM, pcd_builder=pcd)
        pts, cols = pcd.get()
        all_pts.append(pts)
        all_cols.append(cols)
        traj_path = OUTPUT_DIR / f"{video_path.stem}_trajectory.npy"
        if traj_path.exists():
            trajectories.append(np.load(str(traj_path)).astype(np.float32))

    merged_pts  = np.concatenate(all_pts)
    merged_cols = np.concatenate(all_cols)

    # Camera convention: Y=down → viewer convention: Y=up
    merged_pts[:, 1] *= -1
    for t in trajectories:
        t[:, 1] *= -1

    print(f"\nRaw point count : {len(merged_pts):,}")

    # Save full-resolution PLY
    ply_path = OUTPUT_DIR / "merged_pointcloud.ply"
    PointCloudBuilder.save_ply(ply_path, merged_pts, merged_cols)
    print(f"Saved PLY        : {ply_path}")

    # Downsample for HTML (target ≤ 250 K points)
    html_pts, html_cols = PointCloudBuilder.voxel_downsample(merged_pts, merged_cols, 0.12)
    if len(html_pts) > 250_000:
        idx = np.random.choice(len(html_pts), 250_000, replace=False)
        html_pts, html_cols = html_pts[idx], html_cols[idx]
    print(f"HTML points      : {len(html_pts):,}")

    # Pre-compute height colormap (viridis-like)
    height_cols = _height_colormap(html_pts[:, 1])

    # Generate HTML
    html_path = OUTPUT_DIR / "demo_3d.html"
    _write_html(html_path, html_pts, html_cols, height_cols, trajectories)
    print(f"Saved HTML       : {html_path}")

    # Open browser unless running in batch/headless mode
    if "--no-open" not in sys.argv:
        webbrowser.open(html_path.as_uri())
        print("Browser opened!")
        _try_open3d(str(ply_path))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _height_colormap(y: np.ndarray) -> np.ndarray:
    """Viridis-like uint8 RGB based on Y height."""
    lo, hi = float(np.percentile(y, 2)), float(np.percentile(y, 98))
    t = np.clip((y - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
    # Viridis approximation
    r = np.clip(t * 2.5 - 0.5, 0, 1)
    g = np.clip(np.sin(t * np.pi) * 1.3, 0, 1)
    b = np.clip(1.5 - t * 2.5, 0, 1)
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)


def _enc(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def _write_html(
    path: Path,
    pts: np.ndarray,
    cols_rgb: np.ndarray,
    cols_height: np.ndarray,
    trajectories: list[np.ndarray],
) -> None:
    centroid = pts.mean(axis=0).tolist()

    traj_data = []
    for i, traj in enumerate(trajectories):
        traj_data.append({
            "b64": _enc(traj.astype(np.float32)),
            "color": TRAJ_COLORS[i % len(TRAJ_COLORS)],
        })

    html = _HTML_TEMPLATE
    html = html.replace("__POS_B64__",    _enc(pts.astype(np.float32)))
    html = html.replace("__COL_RGB_B64__", _enc(cols_rgb))
    html = html.replace("__COL_HGT_B64__", _enc(cols_height))
    html = html.replace("__TRAJ_JSON__",   json.dumps(traj_data))
    html = html.replace("__CENTROID__",    json.dumps(centroid))
    html = html.replace("__N_POINTS__",    f"{len(pts):,}")
    html = html.replace("__N_VIDEOS__",    str(len(trajectories)))

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def _try_open3d(ply_path: str) -> None:
    try:
        import open3d as o3d
        print("Launching Open3D viewer (press Q to quit) …")
        pcd = o3d.io.read_point_cloud(ply_path)
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="Mental Map — 3D Point Cloud",
            width=1600, height=900,
        )
    except ImportError:
        print("Tip: pip install open3d  ->  native 3D viewer")
    except Exception as e:
        print(f"Open3D: {e}")


# ---------------------------------------------------------------------------
# HTML / Three.js template
# All JS braces are literal — placeholders use __MARKER__ syntax.
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Mental Map — 3D Point Cloud</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#05050f;overflow:hidden;font-family:'Courier New',monospace;color:#ccc}
  #c{display:block}
  #hud{
    position:absolute;top:16px;left:16px;
    background:rgba(0,0,12,.75);padding:12px 16px;border-radius:8px;
    border:1px solid #223;font-size:12px;line-height:2;min-width:220px
  }
  #hud b{color:#7cf;font-size:13px}
  #hud .dim{color:#556}
  #panel{
    position:absolute;bottom:16px;left:16px;
    background:rgba(0,0,12,.75);padding:10px 14px;border-radius:8px;
    border:1px solid #223;font-size:12px
  }
  .row{display:flex;align-items:center;gap:8px;margin:4px 0}
  input[type=range]{width:110px;accent-color:#7cf}
  .btn{
    cursor:pointer;background:#0a0a2a;border:1px solid #335;
    color:#acd;padding:3px 10px;border-radius:4px;font-size:11px
  }
  .btn:hover{background:#112244}
  .tog{cursor:pointer;user-select:none;padding:2px 8px;border-radius:3px;border:1px solid #335;font-size:11px}
  .tog.on{background:#113355;color:#7cf;border-color:#336}
  .tog.off{background:#0a0a1a;color:#556}
  #fps{position:absolute;top:16px;right:16px;font-size:11px;color:#334}
</style>
</head>
<body>
<canvas id="c"></canvas>

<div id="hud">
  <b>Mental Map — 3D Point Cloud</b><br>
  Points &nbsp;<span id="npts">__N_POINTS__</span><br>
  Videos &nbsp;<span>__N_VIDEOS__</span><br>
  <span class="dim">Drag: rotate &nbsp;|&nbsp; Right: pan &nbsp;|&nbsp; Scroll: zoom</span>
</div>

<div id="panel">
  <div class="row">
    Point size
    <input type="range" id="sizeSlider" min="0.005" max="0.15" step="0.005" value="0.035">
    <span id="sizeVal">0.035</span>
  </div>
  <div class="row">
    Color &nbsp;
    <span class="tog on"  id="togRGB"    onclick="setColor('rgb')">RGB</span>
    <span class="tog off" id="togHeight" onclick="setColor('height')">Height</span>
  </div>
  <div class="row">
    Rotate &nbsp;
    <span class="tog off" id="togAuto" onclick="toggleAuto()">Auto</span>
    &nbsp;
    <button class="btn" onclick="resetCam()">Reset cam</button>
  </div>
</div>

<div id="fps"></div>

<script type="importmap">
{"imports":{
  "three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
}}
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Decode helpers ──────────────────────────────────────────────────────────
function b64F32(b64){
  const s=atob(b64),n=s.length,b=new Uint8Array(n);
  for(let i=0;i<n;i++)b[i]=s.charCodeAt(i);
  return new Float32Array(b.buffer);
}
function b64U8(b64){
  const s=atob(b64),n=s.length,b=new Uint8Array(n);
  for(let i=0;i<n;i++)b[i]=s.charCodeAt(i);
  return b;
}
function u8toF32(u8){
  const f=new Float32Array(u8.length);
  for(let i=0;i<u8.length;i++)f[i]=u8[i]/255;
  return f;
}

// ── Embedded data ───────────────────────────────────────────────────────────
const POS      = b64F32("__POS_B64__");
const COL_RGB  = u8toF32(b64U8("__COL_RGB_B64__"));
const COL_HGT  = u8toF32(b64U8("__COL_HGT_B64__"));
const TRAJS    = __TRAJ_JSON__;
const CENTROID = __CENTROID__;

// ── Scene setup ─────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({canvas:document.getElementById('c'),antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setSize(innerWidth,innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05050f);
scene.fog = new THREE.FogExp2(0x05050f, 0.012);

const camera = new THREE.PerspectiveCamera(55,innerWidth/innerHeight,0.05,500);
camera.position.set(CENTROID[0], CENTROID[1]+18, CENTROID[2]+35);

const controls = new OrbitControls(camera,renderer.domElement);
controls.target.set(CENTROID[0],CENTROID[1],CENTROID[2]);
controls.enableDamping=true; controls.dampingFactor=0.06;
controls.minDistance=0.5; controls.maxDistance=300;
controls.update();

// ── Point cloud geometry ────────────────────────────────────────────────────
const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(POS,3));
geo.setAttribute('color',    new THREE.BufferAttribute(COL_RGB.slice(),3));  // .slice() = copy

const mat = new THREE.PointsMaterial({vertexColors:true,size:0.035,sizeAttenuation:true});
const cloud = new THREE.Points(geo,mat);
scene.add(cloud);

// ── Trajectory lines ────────────────────────────────────────────────────────
TRAJS.forEach(({b64,color})=>{
  const pts=b64F32(b64);
  if(pts.length<6) return;
  const tg=new THREE.BufferGeometry();
  tg.setAttribute('position',new THREE.BufferAttribute(pts,3));
  scene.add(new THREE.Line(tg, new THREE.LineBasicMaterial({color})));

  // Start sphere (green) and end sphere (red)
  const addSphere=(i,col)=>{
    const m=new THREE.Mesh(
      new THREE.SphereGeometry(0.25,8,8),
      new THREE.MeshBasicMaterial({color:col})
    );
    m.position.set(pts[i],pts[i+1],pts[i+2]);
    scene.add(m);
  };
  addSphere(0, 0x00ff44);
  addSphere(pts.length-3, 0xff3300);
});

// ── Grid ─────────────────────────────────────────────────────────────────────
const grid=new THREE.GridHelper(200,100,0x111133,0x111133);
grid.position.set(CENTROID[0], CENTROID[1]-2, CENTROID[2]);
scene.add(grid);

// ── Axes helper (small) ──────────────────────────────────────────────────────
const ax=new THREE.AxesHelper(3);
ax.position.set(CENTROID[0],CENTROID[1],CENTROID[2]);
scene.add(ax);

// ── UI wiring ────────────────────────────────────────────────────────────────
const slider=document.getElementById('sizeSlider');
const sizeVal=document.getElementById('sizeVal');
slider.addEventListener('input',e=>{
  mat.size=parseFloat(e.target.value);
  sizeVal.textContent=e.target.value;
});

let colorMode='rgb';
window.setColor=mode=>{
  colorMode=mode;
  const arr=mode==='rgb'?COL_RGB:COL_HGT;
  geo.attributes.color.array.set(arr);
  geo.attributes.color.needsUpdate=true;
  document.getElementById('togRGB').className   ='tog '+(mode==='rgb'?'on':'off');
  document.getElementById('togHeight').className='tog '+(mode==='height'?'on':'off');
};

let autoRotate=false;
window.toggleAuto=()=>{
  autoRotate=!autoRotate;
  controls.autoRotate=autoRotate;
  controls.autoRotateSpeed=0.6;
  document.getElementById('togAuto').className='tog '+(autoRotate?'on':'off');
};

window.resetCam=()=>{
  camera.position.set(CENTROID[0],CENTROID[1]+18,CENTROID[2]+35);
  controls.target.set(CENTROID[0],CENTROID[1],CENTROID[2]);
  controls.update();
};

// ── Resize ───────────────────────────────────────────────────────────────────
window.addEventListener('resize',()=>{
  camera.aspect=innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth,innerHeight);
});

// ── Render loop ──────────────────────────────────────────────────────────────
let last=performance.now(), frames=0;
const fpsEl=document.getElementById('fps');

(function animate(){
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene,camera);
  frames++;
  const now=performance.now();
  if(now-last>1000){
    fpsEl.textContent=frames+' fps';
    frames=0; last=now;
  }
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
