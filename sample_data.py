"""
Sample Dataset & Synthetic Test Image Generator for Vision Copilot.
Generates test frames for automated testing without requiring a physical camera.
"""

from typing import Tuple
import numpy as np


def generate_synthetic_medicine_box() -> np.ndarray:
    """Creates a synthetic 480x640 medicine blister frame with contrast text."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 230
    # Add simulated blister pack foil pattern
    img[100:380, 150:490] = [180, 190, 200]
    return img


def generate_synthetic_meal_plate() -> np.ndarray:
    """Creates a synthetic 480x640 food plate frame with colorful ingredients."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 40
    # Add warm plate colors
    img[120:360, 160:480] = [30, 140, 60]  # Green vegetable tone
    img[200:320, 220:420] = [40, 80, 180]  # Cooked protein tone
    return img


def generate_synthetic_pcb_board() -> np.ndarray:
    """Creates a synthetic 480x640 circuit board frame with green solder mask."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 20
    # Green PCB substrate
    img[80:400, 120:520] = [40, 110, 35]
    # Gold test pads
    img[120:160, 160:200] = [60, 215, 255]
    return img
