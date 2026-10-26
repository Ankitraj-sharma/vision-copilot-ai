"""
Domain Vision Engines Registry.
"""

from domain_engines.base_engine import BaseVisionEngine
from domain_engines.medicine_scanner import MedicineScanner
from domain_engines.nutrition_tracker import NutritionTracker
from domain_engines.circuit_inspector import CircuitInspector
from domain_engines.accessibility_guide import AccessibilityGuide
from domain_engines.doc_translator import DocumentTranslator

__all__ = [
    "BaseVisionEngine",
    "MedicineScanner",
    "NutritionTracker",
    "CircuitInspector",
    "AccessibilityGuide",
    "DocumentTranslator",
]
