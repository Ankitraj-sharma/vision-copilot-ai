"""
Abstract Base Interface for Vision Copilot Domain Analyzers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseVisionEngine(ABC):
    """
    Standard interface for all specialized multimodal vision engines.
    """

    def __init__(self, name: str, domain_icon: str) -> None:
        self.name = name
        self.domain_icon = domain_icon

    @abstractmethod
    def analyze(self, img_bgr: np.ndarray, user_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Processes frame pixels and returns domain-specific insights,
        narration speech cues, confidence score, and visual tags.
        """
        pass
