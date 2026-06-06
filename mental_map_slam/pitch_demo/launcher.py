"""
Mental Map SLAM — Pitch Demo Launcher
  python launcher.py          (dev)
  MentalMapSLAM_Demo.exe      (built)
"""
import os, sys, threading, time, json, atexit
from pathlib import Path
import http.server
import cv2
import numpy as np
import webview

# Ensure pitch_demo directory is in sys.path to resolve imports cleanly
pitch_demo_dir = Path(__file__).resolve().parent
if str(pitch_demo_dir) not in sys.path:
    sys.path.insert(0, str(pitch_demo_dir))

from live_handler import LivePipelineManager

# Global LivePipelineManager instance
pipeline_manager = LivePipelineManager()

# Precompute fallback frames
_fallback_hud_bytes = None
_fallback_depth_bytes = None

def make_fallback_frame(msg):
    try:
        width = getattr(pipeline_manager.config, "display_width", 1280)
        height = getattr(pipeline_manager.config, "display_height", 720)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        color = (100, 100, 100)
        thickness = 2
        text_size = cv2.getTextSize(msg, font, scale, thickness)[0]
        tx = (width - text_size[0]) // 2
        ty = (height + text_size[1]) // 2
        cv2.putText(img, msg, (tx, ty), font, scale, color, thickness, lineType=cv2.LINE_AA)
        _, jpeg = cv2.imencode(".jpg", img)
        return jpeg.tobytes()
    except Exception as e:
        print(f"[fallback] Error generating fallback frame: {e}")
        return b'\xff\xd8\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\x37\xff\xd9'

def get_fallback_hud_bytes():
    global _fallback_hud_bytes
    if _fallback_hud_bytes is None:
        _fallback_hud_bytes = make_fallback_frame("Live Feed Offline")
    return _fallback_hud_bytes

def get_fallback_depth_bytes():
    global _fallback_depth_bytes
    if _fallback_depth_bytes is None:
        _fallback_depth_bytes = make_fallback_frame("Depth Map Offline")
    return _fallback_depth_bytes

# Register global exit handler to stop pipeline manager on program exit
@atexit.register
def cleanup_pipeline():
    print("Exiting program. Stopping pipeline manager...")
    try:
        if pipeline_manager.is_running():
            pipeline_manager.stop()
    except Exception as e:
        print(f"Error stopping pipeline manager: {e}")

# ── Resolve base directory ─────────────────────────────────────────────────
# In dev  → D:\mental_map_slam  (two levels above this file)
# In exe  → sys._MEIPASS  (where PyInstaller extracts bundled data)
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "output"
THUMB_DIR  = BASE_DIR / "pitch_demo" / "thumbs"
PORT       = 8765


# ── Thumbnail extraction ───────────────────────────────────────────────────
def extract_thumbnails() -> None:
    """Grab frame 60 of each demo video and save as JPEG thumbnail."""
    try:
        import cv2
        THUMB_DIR.mkdir(parents=True, exist_ok=True)
        for i in range(1, 6):
            out = THUMB_DIR / f"video_{i}_thumb.jpg"
            if out.exists():
                continue
            mp4 = OUTPUT_DIR / f"video_{i}_demo.mp4"
            if not mp4.exists():
                continue
            cap = cv2.VideoCapture(str(mp4))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            target = min(60, max(5, total // 3))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            cap.release()
    except Exception as e:
        print(f"[thumb] {e}")


# ── Local HTTP server ──────────────────────────────────────────────────────
class _SilentHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def log_message(self, *_): pass
    def log_error(self, *_):   pass

    def write_json_response(self, data, status_code=200):
        try:
            response_bytes = json.dumps(data).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response_bytes)
        except Exception as e:
            print(f"[API Error] Failed to write JSON response: {e}")

    def read_json_body(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                return {}
            body = self.rfile.read(content_length)
            return json.loads(body.decode("utf-8"))
        except Exception as e:
            print(f"[API Error] Failed to parse JSON body: {e}")
            return {}

    def do_GET(self):
        if self.path == "/live_feed":
            self.handle_live_feed()
        elif self.path == "/depth_feed":
            self.handle_depth_feed()
        elif self.path == "/api/pipeline/status":
            self.handle_pipeline_status()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/pipeline/start":
            self.handle_pipeline_start()
        elif self.path == "/api/pipeline/stop":
            self.handle_pipeline_stop()
        elif self.path == "/api/pipeline/lock":
            self.handle_pipeline_lock()
        elif self.path == "/api/pipeline/unlock":
            self.handle_pipeline_unlock()
        elif self.path == "/api/pipeline/config":
            self.handle_pipeline_config()
        elif self.path == "/api/calibrate/start":
            self.handle_calibrate_start()
        elif self.path == "/api/calibrate/click":
            self.handle_calibrate_click()
        elif self.path == "/api/calibrate/save":
            self.handle_calibrate_save()
        elif self.path == "/api/calibrate/verify":
            self.handle_calibrate_verify()
        else:
            self.send_error(404, "Endpoint not found")

    # ── GET stream handlers ──
    def handle_live_feed(self):
        try:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            
            last_frame_time = time.time()
            frame_interval = 1.0 / 30.0
            
            while True:
                now = time.time()
                elapsed = now - last_frame_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_frame_time = time.time()
                
                jpeg_bytes = pipeline_manager.get_latest_hud_jpeg()
                if jpeg_bytes is None:
                    jpeg_bytes = get_fallback_hud_bytes()
                
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(f"Content-Type: image/jpeg\r\nContent-Length: {len(jpeg_bytes)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(jpeg_bytes)
                self.wfile.write(b"\r\n")
        except Exception as e:
            # Clean exit on connection drop (BrokenPipeError / ConnectionAbortedError)
            pass

    def handle_depth_feed(self):
        try:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            
            last_frame_time = time.time()
            frame_interval = 1.0 / 30.0
            
            while True:
                now = time.time()
                elapsed = now - last_frame_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_frame_time = time.time()
                
                jpeg_bytes = pipeline_manager.get_latest_depth_jpeg()
                if jpeg_bytes is None:
                    jpeg_bytes = get_fallback_depth_bytes()
                
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(f"Content-Type: image/jpeg\r\nContent-Length: {len(jpeg_bytes)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(jpeg_bytes)
                self.wfile.write(b"\r\n")
        except Exception as e:
            # Clean exit on connection drop
            pass

    # ── GET api handlers ──
    def handle_pipeline_status(self):
        try:
            status = pipeline_manager.get_status()
            self.write_json_response(status)
        except Exception as e:
            print(f"[API Error] Exception in /api/pipeline/status: {e}")
            self.write_json_response({"success": False, "error": str(e)}, status_code=500)

    # ── POST api handlers ──
    def handle_pipeline_start(self):
        try:
            pipeline_manager.start()
            self.write_json_response({"success": True})
        except Exception as e:
            print(f"[API Error] Exception in /api/pipeline/start: {e}")
            self.write_json_response({"success": False, "error": str(e)}, status_code=500)

    def handle_pipeline_stop(self):
        try:
            pipeline_manager.stop()
            self.write_json_response({"success": True})
        except Exception as e:
            print(f"[API Error] Exception in /api/pipeline/stop: {e}")
            self.write_json_response({"success": False, "error": str(e)}, status_code=500)

    def handle_pipeline_lock(self):
        try:
            body = self.read_json_body()
            track_id = body.get("track_id")
            if track_id is not None:
                track_id = int(track_id)
            pipeline_manager.lock_target(track_id)
            self.write_json_response({"success": True})
        except Exception as e:
            print(f"[API Error] Exception in /api/pipeline/lock: {e}")
            self.write_json_response({"success": False, "error": str(e)}, status_code=500)

    def handle_pipeline_unlock(self):
        try:
            pipeline_manager.unlock_target()
            self.write_json_response({"success": True})
        except Exception as e:
            print(f"[API Error] Exception in /api/pipeline/unlock: {e}")
            self.write_json_response({"success": False, "error": str(e)}, status_code=500)

    def handle_pipeline_config(self):
        try:
            config_data = self.read_json_body()
            pipeline_manager.update_config(**config_data)
            self.write_json_response({"success": True})
        except Exception as e:
            print(f"[API Error] Exception in /api/pipeline/config: {e}")
            self.write_json_response({"success": False, "error": str(e)}, status_code=500)

    def handle_calibrate_start(self):
        try:
            pipeline_manager.start_calibration()
            self.write_json_response({"success": True})
        except Exception as e:
            print(f"[API Error] Exception in /api/calibrate/start: {e}")
            self.write_json_response({"success": False, "error": str(e)}, status_code=500)

    def handle_calibrate_click(self):
        try:
            body = self.read_json_body()
            x1 = int(body.get("x1", 0))
            y1 = int(body.get("y1", 0))
            x2 = int(body.get("x2", 0))
            y2 = int(body.get("y2", 0))
            pipeline_manager.set_calibration_rect(x1, y1, x2, y2)
            self.write_json_response({"success": True})
        except Exception as e:
            print(f"[API Error] Exception in /api/calibrate/click: {e}")
            self.write_json_response({"success": False, "error": str(e)}, status_code=500)

    def handle_calibrate_save(self):
        try:
            focal_length_px = pipeline_manager.calculate_and_save_focal_length()
            self.write_json_response({"success": True, "focal_length_px": focal_length_px})
        except Exception as e:
            print(f"[API Error] Exception in /api/calibrate/save: {e}")
            self.write_json_response({"success": False, "error": str(e)}, status_code=500)

    def handle_calibrate_verify(self):
        try:
            body = self.read_json_body()
            enable = bool(body.get("enable", False))
            pipeline_manager.verify_calibration(enable)
            self.write_json_response({"success": True})
        except Exception as e:
            print(f"[API Error] Exception in /api/calibrate/verify: {e}")
            self.write_json_response({"success": False, "error": str(e)}, status_code=500)


def _start_server() -> None:
    srv = http.server.HTTPServer(("127.0.0.1", PORT), _SilentHandler)
    srv.serve_forever()


# ── Entry point ────────────────────────────────────────────────────────────
def main() -> None:
    extract_thumbnails()

    threading.Thread(target=_start_server, daemon=True).start()
    time.sleep(0.4)                  # let the server bind before opening window

    webview.create_window(
        title="Mental Map SLAM — Pitch Demo",
        url=f"http://127.0.0.1:{PORT}/pitch_demo/ui.html",
        width=1460,
        height=920,
        min_size=(1200, 720),
        background_color="#07071a",
        text_select=False,
        zoomable=False,
    )
    webview.start(debug=False)

    # Stop the pipeline manager after the pywebview window is closed
    print("Window closed. Stopping pipeline manager...")
    try:
        if pipeline_manager.is_running():
            pipeline_manager.stop()
    except Exception as e:
        print(f"Error stopping pipeline manager: {e}")


if __name__ == "__main__":
    main()
