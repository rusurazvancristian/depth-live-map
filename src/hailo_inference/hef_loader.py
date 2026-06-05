"""HEF model loader — wraps VDevice + network group setup. [TRACK A]"""

import logging
from contextlib import contextmanager

import numpy as np

try:
    from hailo_platform import (
        VDevice, HEF, ConfigureParams,
        InputVStreamParams, OutputVStreamParams,
        FormatType, HailoStreamInterface, InferVStreams,
        HailoSchedulingAlgorithm,
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    VDevice = None
    logger = logging.getLogger(__name__)
    logger.warning("hailo_platform not found. HEFModel will run in fallback/mock mode.")

logger = logging.getLogger(__name__)


class HEFModel:
    """Encapsulates a single HEF model loaded on the Hailo-8 NPU.

    Args:
        hef_path: Path to the compiled .hef model file.
        device: Shared VDevice instance. If None, creates its own.
        quantized_input: Whether the input tensor is pre-quantized.

    Usage:
        model = HEFModel("models/yolov8s_h8.hef", device=shared_vdevice)
        with model.session() as infer:
            output = infer({model.input_name: batch_nhwc})
    """

    def __init__(
        self,
        hef_path: str,
        device: 'VDevice' = None,
        quantized_input: bool = False,
    ) -> None:
        self.hef_path = hef_path
        self._owns_device = False

        if not HAILO_AVAILABLE:
            self.input_name = "input"
            self.output_name = "output"
            self.input_shape = (480, 480, 3)
            logger.info(
                "HEFModel loaded (MOCK): %s | input=%s %s",
                hef_path, self.input_name, self.input_shape,
            )
            return

        self._hef = HEF(hef_path)
        self._owns_device = device is None
        
        if device is not None:
            self._device = device
        else:
            # Create a VDevice with round-robin scheduling for multi-network concurrency
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            self._device = VDevice(params)

        params = ConfigureParams.create_from_hef(
            self._hef, interface=HailoStreamInterface.PCIe,
        )
        self._network_groups = self._device.configure(self._hef, params)
        self._ng = self._network_groups[0]
        self._ng_params = self._ng.create_params()

        self._in_params = InputVStreamParams.make_from_network_group(
            self._ng, quantized=quantized_input, format_type=FormatType.UINT8,
        )
        self._out_params = OutputVStreamParams.make_from_network_group(
            self._ng, quantized=False, format_type=FormatType.FLOAT32,
        )

        self.input_name = self._ng.get_input_vstream_infos()[0].name
        self.output_name = self._ng.get_output_vstream_infos()[0].name
        self.input_shape = tuple(self._ng.get_input_vstream_infos()[0].shape)

        logger.info(
            "HEFModel loaded: %s | input=%s %s | output=%s",
            hef_path, self.input_name, self.input_shape, self.output_name,
        )

    @contextmanager
    def session(self):
        """Context manager: yields an InferVStreams callable inside an active network group."""
        if not HAILO_AVAILABLE:
            yield None
            return

        with InferVStreams(self._ng, self._in_params, self._out_params) as pipeline:
            with self._ng.activate(self._ng_params):
                yield pipeline

    def infer(self, pipeline, batch: np.ndarray) -> list:
        """Run a single inference call.

        Args:
            pipeline: The InferVStreams object from session().
            batch: NHWC uint8 array, e.g. shape (1, H, W, 3).

        Returns:
            Raw output: list[batch_idx] -> list[class_idx] -> np.ndarray
        """
        if not HAILO_AVAILABLE:
            # Mock YOLO-like output: 80 classes, class 0 (person) has 1 detection
            # coordinates: y1n, x1n, y2n, x2n, score
            mock_dets = [np.array([[0.2, 0.2, 0.8, 0.8, 0.85]], dtype=np.float32)] + [
                np.empty((0, 5), dtype=np.float32) for _ in range(79)
            ]
            return [mock_dets]

        result = pipeline.infer({self.input_name: batch})
        return result[self.output_name]

    def __del__(self) -> None:
        if HAILO_AVAILABLE and self._owns_device and hasattr(self, "_device") and self._device is not None:
            try:
                del self._device
            except Exception:
                pass
