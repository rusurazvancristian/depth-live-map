### 📋 PROJECT OUTLINE FOR CLAUDE

**1. Context & Objective**
*   **Event:** Hack a Ton 2026 (Monsson Track).
*   **Challenge:** "How Far?" - Estimate metric distance to an object from a single cropped image and class label.
*   **Target:** Mean Absolute Error (MAE) < 15%.
*   **Bonus Goals:** Calibrated confidence intervals, graceful degradation, methodological rigor.
*   **Hardware Constraint:** Raspberry Pi + Hailo-8L (26 TOPS NPU) + Standard RGB Camera.
*   **Tutor Constraint:** Must "train a model ourselves" (cannot just use off-the-shelf APIs).

**2. Team & GitHub Collision Strategy**
*   To prevent merge conflicts, the codebase is strictly divided into two parallel tracks with a defined "Data Contract" (Inputs: image, class, bbox. Outputs: distance_m, confidence).
*   **Track A (Edge & Geometry):** Focuses on `src/hailo_inference/`, `src/engines/geometry_engine.py`, and the main orchestration script.
*   **Track B (Neural & Colab):** Focuses on `src/engines/depth_engine.py`, `src/mlp_training/` (Colab notebooks), and the fusion logic.

**3. The 4-Part Architecture Pipeline**
*   **Engine 1: YOLOv8 (Perception):** Detects object, extracts bounding box (pixel height), class, and confidence. *Deployment: Pre-compiled `.hef` running on Hailo NPU via HailoRT API.*
*   **Engine 2: Geometric Math (The Baseline):** Uses Pinhole Camera Model (`Distance = (Real_Height * Focal_Length) / Pixel_Height`). *Deployment: Raspberry Pi CPU (NumPy).*
*   **Engine 3: Depth Anything V2 (Visual Cue):** Processes the YOLO crop to output a relative depth map and variance. *Deployment: Pre-compiled `.hef` running on Hailo NPU via HailoRT API.*
*   **Engine 4: Custom Fusion MLP (The Brain & Tutor Loophole):** A lightweight PyTorch Multi-Layer Perceptron trained on Colab. Takes `[Geometric_Distance, Relative_Depth_Score, Class_ID]` and outputs `[Final_Metric_Distance, Log_Variance]`. Solves the unit-mismatch between meters and relative depth, satisfies the "train it yourself" requirement, and provides the required confidence intervals. *Deployment: Exported to ONNX, runs on Raspberry Pi CPU via ONNX Runtime.*

**4. Coding Standards & Constraints (From `computer-vision-expert` Skill)**
*   **Paradigms:** Use Object-Oriented Programming (OOP) for model architectures/engines. Use Functional Programming for image processing pipelines and data transformations.
*   **Style:** Strict PEP 8 compliance, comprehensive type hinting (`typing` module), and descriptive variable names reflecting CV operations.
*   **Hardware:** Proper GPU/NPU utilization handling (device placement, fallback logic).
*   **Documentation:** Clear docstrings for all classes and methods.

**5. Instructions for Claude**
*   Using the context, architecture, and coding standards above, generate a comprehensive `.md` system prompt.
*   This `.md` prompt will be used by the development team to instruct their coding AI (Cursor/Copilot) on how to write the actual Python code for this project.
*   The generated `.md` prompt must explicitly forbid the AI from writing monolithic scripts, training heavy backbones from scratch, or ignoring the Hailo/CPU split.