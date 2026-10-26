# Smart Multimodal Vision Copilot 👁️✨

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org)
[![Gemini](https://img.shields.io/badge/Gemini%202.5%20Flash-Multimodal%20LLM-blueviolet.svg)](https://deepmind.google/technologies/gemini/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](Dockerfile)

> **Real-time, zero-upload Multimodal Computer Vision AI Copilot for automated pharmaceutical medicine scanning, food nutrition decomposition, printed circuit board (PCB) QA, multilingual document OCR, and hands-free spatial accessibility guidance.** Runs continuously on live webcam frames with instant client-side voice narration!

---

## 🎯 Specialized Multimodal Domains

1. **💊 Medicine & Prescription Scanner:**
   - Detects tablet blister packs, prescription boxes, and syrups.
   - Extracts active chemical formulations, recommended dosages, and expiration dates.
2. **🥗 Food Nutrition & Calorie Estimator:**
   - Localizes meals and calculates caloric content (kcal) and macronutrient ratios (Protein, Carbs, Healthy Fats).
3. **⚡ PCB Circuit & Hardware Defect Inspector:**
   - Automated Optical Inspection (AOI) identifying dry/cold solder joints, bridged traces, and capacitor venting.
4. **👁️ Assistive Spatial Navigation (for Visually Impaired):**
   - Real-time obstacle distance estimation and clock-face direction alerts (e.g., *"Trip hazard at 11 o'clock, 0.8 meters"*).
5. **📄 Multilingual Document OCR & Summarizer:**
   - Extracts printed and handwritten text and compiles a 3-bullet executive summary.

---

## 🔬 Mathematical Formulations

### 1. Optical Focus Sharpness Index (Laplacian Variance)
Image focus quality and motion blur are quantified using the variance of the 2D discrete Laplacian operator:

$$\Delta f(x, y) = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$$

$$\text{Sharpness} = \sigma^2(\Delta f) = \frac{1}{N} \sum_{x, y} \left( \Delta f(x, y) - \mu_{\Delta f} \right)^2$$

- $\text{Sharpness} \ge 100 \implies$ Crisp, optimal focus.
- $\text{Sharpness} < 50 \implies$ Out-of-focus warning triggered.

### 2. CLAHE Adaptive Contrast Enhancement
Contrast Limited Adaptive Histogram Equalization prevents over-amplification of packaging noise by clipping the local contextual histogram at slope $\beta$:

$$P_{\text{clipped}}(k) = \min(P(k), \beta)$$

The redistributive factor is then uniformly spread across all gray levels to ensure ultra-sharp character boundaries.

---

## 🛡️ Feature Matrix

| Feature | Edge Mode (Offline) | Cloud Multimodal Boost (Gemini) |
| :--- | :--- | :--- |
| **Video Latency** | $< 2\text{ ms / frame}$ (60+ FPS) | Hybrid Periodic Polling (~1.5s) |
| **API Key Required** | ❌ No API key needed | Optional Gemini API key |
| **Voice Narration** | Client-Side Web Speech API | Synchronized spoken findings |
| **Camera Feed** | WebRTC Zero-Upload Stream | Full-resolution snapshot frames |
| **Privacy / Cloud** | 100% Local In-Browser | Zero-data-retention API calls |

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Ankitraj-sharma/vision-copilot-ai.git
cd vision-copilot-ai
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Streamlit Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🐳 Docker Deployment

Run seamlessly in a containerized environment with hardware camera passthrough:

```bash
docker-compose up --build
```

---

## 🧪 Running Unit Tests & Benchmarks

```bash
# Execute unit test suite
python test_copilot.py

# Execute FPS throughput benchmark
python benchmark.py
```

---

## 📜 License
Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
