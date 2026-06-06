"""Re-encode demo videos from mp4v to H.264 (browser-compatible)."""
import subprocess
from pathlib import Path
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

TARGETS = [
    Path(r"D:\mental_map_slam\output"),
    Path(r"D:\mental_map_slam\dist\MentalMapSLAM_Demo\_internal\output"),
]

for folder in TARGETS:
    if not folder.exists():
        print(f"skip (not found): {folder}")
        continue
    print(f"\n{folder}")
    for i in range(1, 6):
        src = folder / f"video_{i}_demo.mp4"
        if not src.exists():
            continue
        tmp = folder / f"video_{i}_demo_tmp.mp4"
        r = subprocess.run([
            FFMPEG, "-y", "-i", str(src),
            "-c:v", "libx264", "-preset", "fast", "-crf", "21",
            "-movflags", "+faststart",
            "-an",
            str(tmp),
        ], capture_output=True)
        if r.returncode == 0 and tmp.stat().st_size > 10_000:
            src.unlink()
            tmp.rename(src)
            print(f"  OK  {src.name}  ({src.stat().st_size//1024} KB)")
        else:
            if tmp.exists(): tmp.unlink()
            print(f"  FAIL {src.name}: {r.stderr.decode()[-200:]}")

print("\nDone.")
