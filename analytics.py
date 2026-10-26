"""
Multimodal Vision Telemetry & Inspection Analytics.
Renders Plotly interactive charts for recognition confidence,
nutritional macro breakdowns, and optical sharpness tracking.
"""

from typing import List, Dict, Any
import plotly.graph_objects as go


def confidence_gauge(confidence: float) -> go.Figure:
    """
    Renders semicircular confidence gauge (0-100%).
    """
    pct = max(0.0, min(100.0, confidence * 100.0 if confidence <= 1.0 else confidence))
    color = "#00F5D4" if pct >= 80 else ("#FEE440" if pct >= 60 else "#FF0055")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            title={"text": "Multimodal Model Confidence", "font": {"color": "#00F5D4", "size": 17}},
            number={"suffix": "%", "font": {"color": "#FFFFFF", "size": 34}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8B949E"},
                "bar": {"color": color},
                "bgcolor": "#101626",
                "steps": [
                    {"range": [0, 50], "color": "rgba(255, 0, 85, 0.25)"},
                    {"range": [50, 75], "color": "rgba(254, 228, 64, 0.25)"},
                    {"range": [75, 100], "color": "rgba(0, 245, 212, 0.25)"},
                ],
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="#0A0F1D",
        font=dict(color="#9FB3C8"),
        height=240,
        margin=dict(l=30, r=30, t=40, b=20),
    )
    return fig


def macronutrient_donut(macros: Dict[str, float]) -> go.Figure:
    """
    Visualizes protein, carbs, and fats breakdown for nutrition mode.
    """
    labels = ["Protein (g)", "Carbohydrates (g)", "Fats (g)"]
    values = [
        macros.get("protein_g", 30.0),
        macros.get("carbs_g", 40.0),
        macros.get("fats_g", 15.0),
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                marker=dict(colors=["#00F5D4", "#00BBF9", "#FEE440"]),
                textinfo="label+percent",
            )
        ]
    )

    fig.update_layout(
        title="Meal Macronutrient Ratio",
        title_font=dict(size=17, color="#00F5D4"),
        paper_bgcolor="#0A0F1D",
        plot_bgcolor="#070B14",
        font=dict(color="#FFFFFF"),
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def sharpness_history_chart(sharpness_data: List[float]) -> go.Figure:
    """
    Plots rolling optical sharpness scores to verify steady camera focus.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=sharpness_data,
            mode="lines",
            name="Sharpness (Laplacian Var)",
            line=dict(color="#00F5D4", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0, 245, 212, 0.12)",
        )
    )

    fig.add_hline(
        y=50,
        line_dash="dot",
        line_color="#FEE440",
        annotation_text="Minimum Focus Baseline (50)",
        annotation_position="bottom right",
        annotation_font_color="#FEE440",
    )

    fig.update_layout(
        title="Optical Focus & Sharpness Timeline",
        title_font=dict(size=17, color="#00F5D4"),
        xaxis_title="Captured Frames",
        yaxis_title="Sharpness Score",
        paper_bgcolor="#0A0F1D",
        plot_bgcolor="#070B14",
        font=dict(color="#9FB3C8"),
        margin=dict(l=40, r=40, t=45, b=35),
        height=260,
    )
    return fig
