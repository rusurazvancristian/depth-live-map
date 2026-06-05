from abc import ABC, abstractmethod
from data_contract import FrameResult

class BaseEngine(ABC):
    """Abstract base for all pipeline engines.
    
    Contract:
        - process() takes a FrameResult, modifies ONLY its own fields, returns it.
        - Engines are stateless per-frame (no cross-frame memory unless explicitly documented).
        - Engines must handle graceful degradation (bad input -> NaN output, never crash).
    """

    @abstractmethod
    def process(self, result: FrameResult) -> FrameResult:
        """Process one frame through this engine.
        
        Args:
            result: The current FrameResult object being passed through the pipeline.
            
        Returns:
            The modified FrameResult object.
        """
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
