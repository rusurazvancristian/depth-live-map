### 📋 PROJECT OUTLINE FOR CLAUDE

**1. Context & Objective**
*   **Event:** Hack a Ton 2026 (Monsson Track).
*   **Challenge:** "How Far?" - Estimate metric distance to an object from a single cropped image and class label.
*   **Target:** Mean Absolute Error (MAE) < 15%.
*   **Bonus Goals:** Calibrated confidence intervals, graceful degradation, methodological rigor.
*   **Hardware Constraint:** Raspberry Pi + Hailo-8L (26 TOPS NPU) + Standard RGB Camera.
*   **Tutor Constraint:** Must "train a model ourselves" (cannot just use off-the-shelf APIs).

**2. Team & GitHub Collision Strategy**
*   To prevent merge conflicts, the codebase is strictly divided into two parallel tracks with a defined "Data Contract" (`FrameResult` with nested `TrackedObject` list).
*   **Track A (Edge, Detection & Tracking):** Focuses on `src/hailo_inference/`, `src/engines/yolo_engine.py`, `src/tracking/`, and the main orchestration script `main.py`.
*   **Track B (Depth & Fusion):** Focuses on `src/engines/depth_engine.py`, `src/engines/reid_engine.py`, and `src/engines/kalman_depth_engine.py`.

**3. The 5-Stage Architecture Pipeline**
*   **Stage 1: YOLO26 (Perception):** Detects objects, extracts bounding boxes, classes, and confidence. *Deployment: Pre-compiled `.hef` running on Hailo NPU.*
*   **Stage 2: ByteTrack (Tracking):** Associates detections across frames to assign unique temporal IDs. *Deployment: Raspberry Pi CPU.*
*   **Stage 3: Geometric Math (Pinhole Baseline):** Computes pixel-to-metric distance using calibrated focal length and class height priors. *Deployment: Raspberry Pi CPU.*
*   **Stage 4: SCDepthV3 (Relative Depth Cue):** Predicts a high-fidelity relative depth map, extracting median depth inside object ROIs. *Deployment: Pre-compiled `.hef` running on Hailo NPU.*
*   **Stage 5: Kalman Depth Fusion (Estimation & Smoothing):** Fuses geometric distance and relative depth using a Kalman filter. It dynamically aligns relative depth to metric space using an EMA scale factor. *Deployment: Raspberry Pi CPU.*

**4. Coding Standards & Constraints (From `computer-vision-expert` Skill)**
*   **Paradigms:** Use Object-Oriented Programming (OOP) for model architectures/engines. Use Functional Programming for image processing pipelines and data transformations.
*   **Style:** Strict PEP 8 compliance, comprehensive type hinting (`typing` module), and descriptive variable names reflecting CV operations.
*   **Hardware:** Proper GPU/NPU utilization handling (device placement, fallback logic).
*   **Documentation:** Clear docstrings for all classes and methods.

**5. Instructions for Claude**
*   Using the context, architecture, and coding standards above, generate a comprehensive `.md` system prompt.
*   This `.md` prompt will be used by the development team to instruct their coding AI (Cursor/Copilot) on how to write the actual Python code for this project.
*   The generated `.md` prompt must explicitly forbid the AI from writing monolithic scripts, ignoring the Hailo/CPU split, or failing to maintain the strict inter-stage data contract boundaries.