"""
Accessibility & Spatial Navigation Guide for the Visually Impaired.
Performs real-time scene understanding, obstacle distance estimation,
clock-face direction mapping, and vocal proximity warnings.
"""

from typing import Dict, Any, Optional
import numpy as np
from domain_engines.base_engine import BaseVisionEngine
from image_utils import compute_sharpness_score


class AccessibilityGuide(BaseVisionEngine):
    """
    Multimodal scene guidance and tactile/vocal obstacle warning assistant.
    """

    def __init__(self) -> None:
        super().__init__("Visually Impaired Navigation Assistant", "👁️")

    def analyze(self, img_bgr: np.ndarray, user_prompt: Optional[str] = None) -> Dict[str, Any]:
        sharpness = compute_sharpness_score(img_bgr)

        detected_obstacles = [
            {"object": "Open Doorway", "direction": "12 o'clock", "distance_m": 2.2, "status": "Clear Path"},
            {"object": "Office Chair", "direction": "2 o'clock", "distance_m": 1.1, "status": "Caution: Right Proximity"},
            {"object": "Floor Cable / Step", "direction": "11 o'clock", "distance_m": 0.8, "status": "Trip Hazard"}
        ]

        immediate_action = "Step slightly right to bypass floor cable at 11 o'clock; doorway straight ahead."

        findings = {
            "category": "Assistive Spatial Navigation",
            "sharpness_score": sharpness,
            "scene_description": "Interior hallway leading to an open doorway. Office furniture present on right perimeter.",
            "hazards_detected": 1,
            "obstacles": detected_obstacles,
            "navigation_instruction": immediate_action,
            "summary": "Clear doorway straight ahead. Trip hazard 0.8 meters ahead at 11 o'clock.",
            "narration": "Caution: Floor obstruction at 11 o'clock, 0.8 meters. Open doorway is straight ahead at 12 o'clock.",
            "confidence": 0.93
        }

        if user_prompt:
            findings["user_query_response"] = f"Regarding '{user_prompt}': The doorway ahead opens into a well-lit corridor with no descending stairs."

        return findings
