"""Abstract base class for all pipeline engines. [SHARED]"""

from abc import ABC, abstractmethod
from data_contract import FrameResult


class BaseEngine(ABC):
    """Abstract base for all pipeline engines.

    Contract:
        - process() takes a FrameResult, modifies ONLY its own fields, returns it.
        - Engines are stateless per-frame (no cross-frame memory unless documented).
        - Engines must handle graceful degradation: bad input -> NaN output, never crash.
        - start() and stop() handle lifecycle setup and teardown (e.g. NPU activations).
    """

    @abstractmethod
    def process(self, result: FrameResult) -> FrameResult:
        """Process one frame through this engine.
        
        Args:
            result: The current FrameResult object.
            
        Returns:
            The modified FrameResult object.
        """
        ...

    def start(self) -> None:
        """Initialize pipeline context, open hardware sessions, or activate NPU networks."""
        pass

    def stop(self) -> None:
        """Terminate pipeline context, close hardware sessions, or release NPU networks."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
