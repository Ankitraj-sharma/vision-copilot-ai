"""
Asynchronous WebRTC Video Stream Processor for Vision Copilot.
Streams live webcam frames, extracts visual features, renders a futuristic
Sci-Fi targeting reticle HUD, and streams telemetry to Streamlit in real time.
"""

import threading
import time
from typing import Dict, Any, Optional
import av
import cv2
import numpy as np
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase

from copilot_engine import MultimodalCopilotEngine
from image_utils import compute_sharpness_score

# Public STUN servers for NAT traversal on Streamlit Cloud
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
        ]
    }
)


class LiveVisionCopilotProcessor(VideoProcessorBase):
    """
    Asynchronous WebRTC processor executing real-time augmented vision scanning.
    """

    def __init__(self) -> None:
        self.engine = MultimodalCopilotEngine()
        self.active_domain: str = "Medicine & Prescriptions"

        # Thread synchronization
        self._lock = threading.Lock()
        self.fps: float = 0.0
        self.sharpness: float = 120.0
        self.summary: str = "Scanning active camera view..."
        self.narration: str = ""
        self.confidence: float = 0.90
        self.last_analysis_time: float = 0.0

        # Animation state
        self._scan_line_y: int = 0
        self._frame_count = 0
        self._last_time = time.time()

    def set_domain(self, domain_name: str) -> None:
        """Switch active inspection domain from sidebar."""
        with self._lock:
            self.active_domain = domain_name
            self.engine.set_domain(domain_name)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process video frame and render cybernetic vision reticle HUD."""
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        now = time.time()
        self._frame_count += 1
        elapsed = now - self._last_time
        if elapsed >= 1.0:
            self.fps = round(self._frame_count / elapsed, 1)
            self._frame_count = 0
            self._last_time = now

        # Measure sharpness
        sharp = compute_sharpness_score(img)

        # Trigger periodic domain analysis (every 1.5 seconds)
        if (now - self.last_analysis_time) >= 1.5:
            self.last_analysis_time = now
            res = self.engine.analyze_frame(img)
            with self._lock:
                self.sharpness = sharp
                self.summary = res.get("summary", "Analysis updated.")
                self.narration = res.get("narration", "")
                self.confidence = res.get("confidence", 0.85)

        # ── Draw Sci-Fi HUD ───────────────────────────────────────────
        # Top Telemetry Header
        cv2.rectangle(img, (0, 0), (w, 50), (10, 15, 26), -1)
        cv2.line(img, (0, 50), (w, 50), (0, 245, 212), 2)

        cv2.putText(img, f"VISION COPILOT: {self.active_domain.upper()}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 245, 212), 2)
        cv2.putText(img, f"SHARPNESS: {sharp:.0f}", (w - 240, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(img, f"{self.fps} FPS", (w - 80, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 245, 212), 1)

        # Central Target Reticle Box
        rw, rh = int(w * 0.55), int(h * 0.55)
        rx = (w - rw) // 2
        ry = (h - rh) // 2

        # Draw 4 corner brackets
        bracket_len = 28
        color_reticle = (0, 245, 212)
        # Top-Left
        cv2.line(img, (rx, ry), (rx + bracket_len, ry), color_reticle, 2)
        cv2.line(img, (rx, ry), (rx, ry + bracket_len), color_reticle, 2)
        # Top-Right
        cv2.line(img, (rx + rw, ry), (rx + rw - bracket_len, ry), color_reticle, 2)
        cv2.line(img, (rx + rw, ry), (rx + rw, ry + bracket_len), color_reticle, 2)
        # Bottom-Left
        cv2.line(img, (rx, ry + rh), (rx + bracket_len, ry + rh), color_reticle, 2)
        cv2.line(img, (rx, ry + rh), (rx, ry + rh - bracket_len), color_reticle, 2)
        # Bottom-Right
        cv2.line(img, (rx + rw, ry + rh), (rx + rw - bracket_len, ry + rh), color_reticle, 2)
        cv2.line(img, (rx + rw, ry + rh), (rx + rw, ry + rh - bracket_len), color_reticle, 2)

        # Center Crosshair
        cx, cy = w // 2, h // 2
        cv2.line(img, (cx - 10, cy), (cx + 10, cy), (0, 245, 212), 1)
        cv2.line(img, (cx, cy - 10), (cx, cy + 10), (0, 245, 212), 1)

        # Animated Sci-Fi Scan Line
        self._scan_line_y = (self._scan_line_y + 8) % rh
        scan_actual_y = ry + self._scan_line_y
        cv2.line(img, (rx, scan_actual_y), (rx + rw, scan_actual_y), (120, 255, 235), 1)

        # Bottom Information Banner
        cv2.rectangle(img, (0, h - 45), (w, h), (10, 15, 26), -1)
        cv2.line(img, (0, h - 45), (w, h - 45), (0, 245, 212), 1)
        display_summary = self.summary if len(self.summary) < 70 else (self.summary[:67] + "...")
        cv2.putText(img, f"COPILOT: {display_summary}", (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 220, 240), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_telemetry(self) -> Dict[str, Any]:
        """Provides thread-safe status readout for Streamlit frontend."""
        with self._lock:
            return {
                "fps": self.fps,
                "sharpness": self.sharpness,
                "active_domain": self.active_domain,
                "summary": self.summary,
                "narration": self.narration,
                "confidence": self.confidence,
            }
