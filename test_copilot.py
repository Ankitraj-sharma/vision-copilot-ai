"""
Unit Test Suite for Smart Multimodal Vision Copilot.
Verifies CLAHE preprocessing, sharpness scoring, domain analyzers,
orchestrator state machine, and report exporters.
"""

import numpy as np
from image_utils import compute_sharpness_score, ndarray_to_base64
from domain_engines import (
    MedicineScanner,
    NutritionTracker,
    CircuitInspector,
    AccessibilityGuide,
    DocumentTranslator,
)
from copilot_engine import MultimodalCopilotEngine
from export_utils import generate_markdown_report, generate_json_dossier


def test_image_utils():
    dummy = np.zeros((200, 200, 3), dtype=np.uint8)
    sharpness = compute_sharpness_score(dummy)
    assert sharpness >= 0.0

    b64 = ndarray_to_base64(dummy)
    assert len(b64) > 10


def test_medicine_scanner():
    scanner = MedicineScanner()
    dummy = np.zeros((200, 200, 3), dtype=np.uint8)
    res = scanner.analyze(dummy)
    assert res["category"] == "Pharmaceutical Formulation"
    assert "detected_fields" in res
    assert "narration" in res
    assert res["confidence"] > 0.5


def test_nutrition_tracker():
    tracker = NutritionTracker()
    dummy = np.zeros((200, 200, 3), dtype=np.uint8)
    res = tracker.analyze(dummy)
    assert "macros" in res
    assert res["macros"]["calories_kcal"] > 0
    assert "narration" in res


def test_circuit_inspector():
    inspector = CircuitInspector()
    dummy = np.zeros((200, 200, 3), dtype=np.uint8)
    res = inspector.analyze(dummy)
    assert res["defect_count"] >= 1
    assert "components_passed" in res


def test_accessibility_guide():
    guide = AccessibilityGuide()
    dummy = np.zeros((200, 200, 3), dtype=np.uint8)
    res = guide.analyze(dummy)
    assert "obstacles" in res
    assert len(res["obstacles"]) >= 1


def test_document_translator():
    doc = DocumentTranslator()
    dummy = np.zeros((200, 200, 3), dtype=np.uint8)
    res = doc.analyze(dummy)
    assert "extracted_text" in res
    assert "summary_bullets" in res


def test_copilot_orchestrator():
    engine = MultimodalCopilotEngine()
    engine.set_domain("Food Nutrition & Calories")
    assert engine.active_domain == "Food Nutrition & Calories"

    dummy = np.zeros((200, 200, 3), dtype=np.uint8)
    res = engine.analyze_frame(dummy, user_question="How many calories?")
    assert "macros" in res
    assert "user_query_response" in res


def test_export_utilities():
    findings = {"summary": "Clean inspection", "confidence": 0.95}
    md_str = generate_markdown_report("Medicine & Prescriptions", findings)
    assert "Vision Copilot Diagnostic Report" in md_str
    assert "Clean inspection" in md_str

    json_str = generate_json_dossier("Medicine & Prescriptions", findings)
    assert "VisionCopilot-v3.0" in json_str


if __name__ == "__main__":
    test_image_utils()
    test_medicine_scanner()
    test_nutrition_tracker()
    test_circuit_inspector()
    test_accessibility_guide()
    test_document_translator()
    test_copilot_orchestrator()
    test_export_utilities()
    print("All Vision Copilot unit tests passed successfully!")
