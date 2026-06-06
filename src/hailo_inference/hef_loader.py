"""HEF model loader — wraps VDevice + network group setup. [TRACK A]"""

import logging
from contextlib import contextmanager

import numpy as np
from hailo_platform import (
    VDevice, HEF, ConfigureParams,
    InputVStreamParams, OutputVStreamParams,
    FormatType, HailoStreamInterface, InferVStreams,
)

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
        device: VDevice | None = None,
        quantized_input: bool = False,
        use_scheduler: bool = False,
    ) -> None:
        self._hef = HEF(hef_path)
        self._owns_device = device is None
        self._device = device if device is not None else VDevice()
        self._use_scheduler = use_scheduler

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

        self.input_name: str = self._ng.get_input_vstream_infos()[0].name
        self.output_name: str = self._ng.get_output_vstream_infos()[0].name
        self.output_names: list[str] = [i.name for i in self._ng.get_output_vstream_infos()]
        self.input_shape: tuple = tuple(self._ng.get_input_vstream_infos()[0].shape)

        logger.info(
            "HEFModel loaded: %s | input=%s %s | output=%s",
            hef_path, self.input_name, self.input_shape, self.output_name,
        )

    @contextmanager
    def session(self):
        """Context manager: yields an InferVStreams pipeline ready for inference.

        With scheduler (use_scheduler=True): activation is managed by the scheduler.
        Without scheduler: explicit activation is required.
        """
        with InferVStreams(self._ng, self._in_params, self._out_params) as pipeline:
            if self._use_scheduler:
                yield pipeline
            else:
                with self._ng.activate(self._ng_params):
                    yield pipeline

    def infer(self, pipeline, batch: np.ndarray) -> list:
        """Run a single inference call.

        Args:
            pipeline: The InferVStreams object from session().
            batch: NHWC uint8 array, e.g. shape (1, 640, 640, 3).

        Returns:
            Raw output: list[batch_idx] -> list[class_idx] -> np.ndarray
        """
        result = pipeline.infer({self.input_name: batch})
        return result[self.output_name]

    def infer_all(self, pipeline, batch: np.ndarray) -> dict:
        """Run inference and return full output dict (all tensor names).

        Args:
            pipeline: The InferVStreams object from session().
            batch: NHWC uint8 array, e.g. shape (1, 640, 640, 3).

        Returns:
            Dict mapping output tensor name -> np.ndarray.
        """
        return pipeline.infer({self.input_name: batch})

    def __del__(self) -> None:
        if self._owns_device and hasattr(self, "_device"):
            try:
                del self._device
            except Exception:
                pass
