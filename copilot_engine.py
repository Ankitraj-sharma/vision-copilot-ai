"""
Multimodal Vision Copilot Core Engine.
Coordinates domain-specific computer vision analyzers with optional Gemini Multimodal Vision API,
providing dual Edge-Rule / Cloud-LLM reasoning and interactive Q&A.
"""

import os
from typing import Dict, Any, Optional
import numpy as np

from domain_engines import (
    BaseVisionEngine,
    MedicineScanner,
    NutritionTracker,
    CircuitInspector,
    AccessibilityGuide,
    DocumentTranslator,
)
from image_utils import ndarray_to_base64


class MultimodalCopilotEngine:
    """
    Central orchestrator for multimodal vision queries and interactive frame dialogue.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.domain_engines: Dict[str, BaseVisionEngine] = {
            "Medicine & Prescriptions": MedicineScanner(),
            "Food Nutrition & Calories": NutritionTracker(),
            "PCB Circuits & Hardware": CircuitInspector(),
            "Accessibility & Navigation": AccessibilityGuide(),
            "Document OCR & Translation": DocumentTranslator(),
        }
        self.active_domain: str = "Medicine & Prescriptions"

    def set_domain(self, domain_name: str) -> None:
        """Switches active domain analyzer."""
        if domain_name in self.domain_engines:
            self.active_domain = domain_name

    def analyze_frame(
        self,
        img_bgr: np.ndarray,
        user_question: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes domain analysis. If Gemini API key is configured, invokes multimodal LLM
        for deep qualitative reasoning; otherwise uses high-speed edge heuristics.
        """
        engine = self.domain_engines.get(self.active_domain, self.domain_engines["Medicine & Prescriptions"])

        # Run primary domain engine
        results = engine.analyze(img_bgr, user_prompt=user_question)

        # Cloud Multimodal Fallback / Augmentation
        results["cloud_model_used"] = False
        if self.api_key and len(self.api_key) > 10:
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key)
                b64_img = ndarray_to_base64(img_bgr)
                
                prompt_text = (
                    f"You are an expert AI Vision Copilot in domain: '{self.active_domain}'. "
                    f"Analyze this image carefully. Specific user inquiry: '{user_question or 'Provide comprehensive analysis'}'."
                )
                
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        genai.types.Part.from_bytes(data=b64_img.encode(), mime_type="image/jpeg"),
                        prompt_text
                    ]
                )
                if response and response.text:
                    results["llm_insights"] = response.text
                    results["cloud_model_used"] = True
                    results["summary"] = response.text[:220] + "..."
            except Exception as e:
                results["llm_error"] = str(e)

        return results
