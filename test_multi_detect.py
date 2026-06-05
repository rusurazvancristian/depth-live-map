#!/usr/bin/env python3
"""
Live multi-class YOLO detection demo — toate clasele simultan.
Foloseste yolov8s_h8.hef pe Hailo-8.
Q = quit, +/- = threshold, M = schimba model
"""

import time
import cv2
import numpy as np
from picamera2 import Picamera2
from hailo_platform import (
    VDevice, HEF, ConfigureParams, InferVStreams,
    InputVStreamParams, OutputVStreamParams, FormatType, HailoStreamInterface,
)

MODELS = [
    "/usr/share/hailo-models/yolov8s_h8.hef",
    "/home/martir/Downloads/yolov8m_h8.hef",
    "/home/martir/Downloads/yolov8l_h8.hef",
]
MODEL_NAMES = ["yolov8s", "yolov8m", "yolov8l"]

COCO = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]

rng = np.random.default_rng(42)
COLORS = [tuple(int(x) for x in rng.integers(80, 240, 3)) for _ in range(80)]


def load_model(path):
    hef    = HEF(path)
    device = VDevice()
    params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    ng     = device.configure(hef, params)[0]
    in_p   = InputVStreamParams.make_from_network_group(ng, quantized=False, format_type=FormatType.UINT8)
    out_p  = OutputVStreamParams.make_from_network_group(ng, quantized=False, format_type=FormatType.FLOAT32)
    in_name  = ng.get_input_vstream_infos()[0].name
    out_name = ng.get_output_vstream_infos()[0].name
    return device, ng, in_p, out_p, in_name, out_name


def draw_detections(frame, raw, conf_thr, orig_h, orig_w):
    count = 0
    for cid, dets in enumerate(raw[0]):
        if dets is None or len(dets) == 0:
            continue
        for det in np.asarray(dets):
            score = float(det[4])
            if score < conf_thr:
                continue
            # Hailo NMS output order: [y1n, x1n, y2n, x2n, score]
            y1 = int(np.clip(det[0] * orig_h, 0, orig_h - 1))
            x1 = int(np.clip(det[1] * orig_w, 0, orig_w - 1))
            y2 = int(np.clip(det[2] * orig_h, 0, orig_h - 1))
            x2 = int(np.clip(det[3] * orig_w, 0, orig_w - 1))
            col = COLORS[cid]
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            label = f"{COCO[cid]} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), col, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            count += 1
    return count


def main():
    model_idx = 0
    conf_thr  = 0.45

    cam = Picamera2()
    cfg = cam.create_preview_configuration(
        main={"size": (640, 480), "format": "BGR888"},
        controls={"AfMode": 0, "LensPosition": 0.0},
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(1.5)
    print(f"Camera OK | Model: {MODEL_NAMES[model_idx]} | Conf: {conf_thr:.2f}")
    print("Q=quit  +/-=threshold  M=next model")

    device, ng, in_p, out_p, in_name, out_name = load_model(MODELS[model_idx])

    WINDOW = "Multi-class YOLO"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 540)

    fps = 0.0
    t_prev = time.perf_counter()

    with InferVStreams(ng, in_p, out_p) as pipeline:
        with ng.activate(ng.create_params()):
            while True:
                frame = cam.capture_array()
                h, w = frame.shape[:2]

                resized = cv2.resize(frame, (640, 640))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                batch = np.expand_dims(rgb, 0)

                result = pipeline.infer({in_name: batch})
                raw = result[out_name]

                vis = frame.copy()
                n = draw_detections(vis, raw, conf_thr, h, w)

                t_now = time.perf_counter()
                fps = 0.9 * fps + 0.1 / max(t_now - t_prev, 1e-6)
                t_prev = t_now

                info = f"{MODEL_NAMES[model_idx]}  conf={conf_thr:.2f}  det={n}  {fps:.1f}fps  +/-=thr  M=model  Q=quit"
                cv2.rectangle(vis, (0, h - 22), (w, h), (20, 20, 20), -1)
                cv2.putText(vis, info, (6, h - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

                cv2.imshow(WINDOW, cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('+') or key == ord('='):
                    conf_thr = min(0.95, conf_thr + 0.05)
                elif key == ord('-'):
                    conf_thr = max(0.10, conf_thr - 0.05)
                elif key == ord('m'):
                    break  # reload with next model

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
