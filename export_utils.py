"""
Vision Copilot Diagnostic Report Exporters.
Generates structured Markdown dossiers and JSON manifests for institutional auditing.
"""

import json
from datetime import datetime
from typing import Dict, Any


def generate_markdown_report(domain: str, findings: Dict[str, Any]) -> str:
    """
    Serializes visual findings into an executive technical Markdown report.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conf_pct = round(findings.get("confidence", 0.9) * 100, 1)

    md = f"""# Vision Copilot Diagnostic Report 🔬
**Inspection Domain:** {domain}  
**Timestamp:** {now_str}  
**Confidence Score:** {conf_pct}%  
**Sharpness Index:** {findings.get('sharpness_score', 120.0)}  

---

## 📋 Executive Summary
{findings.get('summary', 'Visual analysis completed successfully.')}

"""

    if "detected_fields" in findings:
        md += "## 💊 Pharmaceutical Specifications\n"
        for k, v in findings["detected_fields"].items():
            clean_k = k.replace("_", " ").title()
            md += f"- **{clean_k}:** {v}\n"

    elif "macros" in findings:
        md += "## 🥗 Nutritional Macro Breakdown\n"
        m = findings["macros"]
        md += f"- **Calories:** {m.get('calories_kcal', 0)} kcal\n"
        md += f"- **Protein:** {m.get('protein_g', 0)} g\n"
        md += f"- **Carbohydrates:** {m.get('carbs_g', 0)} g\n"
        md += f"- **Dietary Fats:** {m.get('fats_g', 0)} g\n"

    elif "defects" in findings:
        md += "## ⚡ Hardware & Circuit Inspection\n"
        for d in findings["defects"]:
            md += f"- **[{d.get('severity', 'WARN')}] {d.get('defect_type')}:** {d.get('location')} — {d.get('recommendation')}\n"

    elif "obstacles" in findings:
        md += "## 👁️ Spatial Obstacles & Navigation\n"
        for obs in findings["obstacles"]:
            md += f"- **{obs.get('object')}:** {obs.get('direction')} at {obs.get('distance_m')}m ({obs.get('status')})\n"

    md += f"\n---\n*Generated autonomously by Smart Multimodal Vision Copilot*\n"
    return md


def generate_json_dossier(domain: str, findings: Dict[str, Any]) -> str:
    """
    Serializes visual findings into a full JSON export manifest.
    """
    payload = {
        "engine": "VisionCopilot-v3.0",
        "domain": domain,
        "exported_at": datetime.now().isoformat(),
        "findings": findings
    }
    return json.dumps(payload, indent=2)
