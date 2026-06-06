"""HEF model multiplexer — manages VDevice and multiple loaded networks.

Allows concurrent inference across YOLO, Depth, and ReID models on Hailo-8.
"""

import logging
from typing import Dict, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)

try:
    from hailo_platform import (
        VDevice,
        HEF,
        ConfigureParams,
        InputVStreamParams,
        OutputVStreamParams,
        FormatType,
        HailoStreamInterface,
        InferVStreams,
        HailoSchedulingAlgorithm,
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    logger.warning("hailo_platform not found. HailoMultiplexer will run in mock mode.")


class HailoMultiplexer:
    """Manages a single Hailo VDevice loaded with multiple HEF models.

    Uses ROUND_ROBIN scheduling to run inference on multiple networks.
    """

    def __init__(self, model_registry: Dict[str, str]) -> None:
        """Initialize the multiplexer.

        Args:
            model_registry: Dict mapping model name -> absolute path to .hef file.
        """
        self._model_registry = model_registry
        self._vdevice: Optional[Any] = None
        self._configured_g: Dict[str, Any] = {}
        self._input_names: Dict[str, str] = {}
        self._output_names: Dict[str, str] = {}
        self._all_output_names: Dict[str, list] = {}
        self._input_shapes: Dict[str, Tuple[int, ...]] = {}
        self._output_shapes: Dict[str, Tuple[int, ...]] = {}
        self._pipelines: Dict[str, Any] = {}
        self._pipeline_contexts: Dict[str, Any] = {}

        if not HAILO_AVAILABLE:
            logger.info("Initializing mock HailoMultiplexer shapes.")
            # Set default mock shapes
            self._input_shapes = {
                "yolo": (1, 640, 640, 3),
                "depth": (1, 256, 320, 3),
                "reid": (1, 256, 128, 3),
            }
            self._output_shapes = {
                "yolo": (1, 100, 6),
                "depth": (1, 256, 320, 1),
                "reid": (1, 512),
            }

    def __enter__(self) -> "HailoMultiplexer":
        if not HAILO_AVAILABLE:
            return self

        try:
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            self._vdevice = VDevice(params)

            for name, hef_path in self._model_registry.items():
                logger.info("Loading HEF: %s from %s", name, hef_path)
                hef = HEF(hef_path)
                configure_params = ConfigureParams.create_from_hef(
                    hef, interface=HailoStreamInterface.PCIe
                )
                
                # Configure network groups on the shared device
                network_groups = self._vdevice.configure(hef, configure_params)
                if not network_groups:
                    raise RuntimeError(f"Failed to configure network group for {name}")
                
                ng = network_groups[0]
                self._configured_g[name] = ng

                # Format inputs as UINT8 and outputs as FLOAT32 (normalised/unquantised output)
                in_params = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
                out_params = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)

                # Store metadata
                in_infos = ng.get_input_vstream_infos()
                out_infos = ng.get_output_vstream_infos()
                
                self._input_names[name] = in_infos[0].name
                self._output_names[name] = out_infos[0].name
                self._all_output_names[name] = [o.name for o in out_infos]

                self._input_shapes[name] = tuple(in_infos[0].shape)
                self._output_shapes[name] = tuple(out_infos[0].shape)

                # Create and activate InferVStreams pipeline context
                pipeline_ctx = InferVStreams(ng, in_params, out_params)
                self._pipeline_contexts[name] = pipeline_ctx
                self._pipelines[name] = pipeline_ctx.__enter__()
                
                logger.info(
                    "Model '%s' configured: input='%s' %s, output='%s' %s",
                    name, self._input_names[name], self._input_shapes[name],
                    self._output_names[name], self._output_shapes[name]
                )

        except Exception as exc:
            logger.error("Failed to initialize Hailo VDevice or configure models: %s", exc, exc_info=True)
            self.__exit__(None, None, None)
            raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not HAILO_AVAILABLE:
            return

        # Close pipelines
        for name, pipeline_ctx in list(self._pipeline_contexts.items()):
            try:
                pipeline_ctx.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.warning("Error exiting pipeline context for %s: %s", name, e)
        
        self._pipelines.clear()
        self._pipeline_contexts.clear()
        self._configured_g.clear()

        # Release VDevice
        if self._vdevice is not None:
            try:
                del self._vdevice
            except Exception as e:
                logger.warning("Error releasing VDevice: %s", e)
            self._vdevice = None

        logger.info("HailoMultiplexer shutdown complete.")

    def infer(self, model_name: str, input_data: np.ndarray) -> np.ndarray:
        """Run synchronous inference on the named model.

        Args:
            model_name: The name of the model registered (e.g. 'yolo', 'depth', 'reid').
            input_data: NHWC numpy array matching the model's input shape.

        Returns:
            Output numpy array of the shape expected from the NPU.
        """
        if not HAILO_AVAILABLE:
            return self._generate_mock_output(model_name)

        if model_name not in self._pipelines:
            raise ValueError(f"Model '{model_name}' not loaded or initialized.")

        pipeline = self._pipelines[model_name]
        input_name = self._input_names[model_name]
        output_name = self._output_names[model_name]

        # Perform inference
        result_dict = pipeline.infer({input_name: input_data})
        return result_dict[output_name]

    def infer_all(self, model_name: str, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference and return the full output dict (all tensor names).

        Needed for multi-output models like yolo26m (6 raw tensors).

        Args:
            model_name: Registered model key.
            input_data: NHWC uint8 array.

        Returns:
            Dict mapping output tensor name -> np.ndarray.
        """
        if not HAILO_AVAILABLE:
            return {self._output_names[model_name]: self._generate_mock_output(model_name)}

        if model_name not in self._pipelines:
            raise ValueError(f"Model '{model_name}' not loaded or initialized.")

        pipeline = self._pipelines[model_name]
        input_name = self._input_names[model_name]
        return pipeline.infer({input_name: input_data})

    def get_output_count(self, model_name: str) -> int:
        """Return the number of output tensors for the named model."""
        names = self._all_output_names.get(model_name)
        return len(names) if names else 1

    def get_all_output_names(self, model_name: str) -> list:
        """Return all output tensor names for the named model."""
        return self._all_output_names.get(model_name, [self._output_names[model_name]])

    def get_input_shape(self, model_name: str) -> Tuple[int, ...]:
        """Get the model's expected input shape (usually NHWC)."""
        if model_name not in self._input_shapes:
            raise ValueError(f"Unknown model: {model_name}")
        return self._input_shapes[model_name]

    def get_output_shape(self, model_name: str) -> Tuple[int, ...]:
        """Get the model's expected output shape."""
        if model_name not in self._output_shapes:
            raise ValueError(f"Unknown model: {model_name}")
        return self._output_shapes[model_name]

    def _generate_mock_output(self, model_name: str) -> np.ndarray:
        """Generate mock predictions if Hailo hardware is unavailable."""
        if model_name == "yolo":
            # (1, 100, 6) NMS-free output: x1n, y1n, x2n, y2n, confidence, class_id
            res = np.zeros((1, 100, 6), dtype=np.float32)
            # Add one person in center
            res[0, 0] = [0.25, 0.20, 0.75, 0.85, 0.88, 0.0]
            # Add a bicycle class
            res[0, 1] = [0.35, 0.50, 0.65, 0.90, 0.75, 1.0]
            return res

        elif model_name == "depth":
            # (1, 256, 320, 1) vertical depth gradient: 0.0 at top, 1.0 at bottom
            grad = np.linspace(0.1, 0.9, 256, dtype=np.float32).reshape(1, 256, 1, 1)
            res = np.repeat(grad, 320, axis=2)
            return res

        elif model_name == "reid":
            # (1, 512) L2-normalized embedding
            emb = np.random.randn(1, 512).astype(np.float32)
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            return emb / (norm + 1e-8)

        else:
            raise ValueError(f"No mock generator for model: {model_name}")
