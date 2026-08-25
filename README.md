<div align="center">

# 🌿 AgriScan — Autonomous Crop Monitoring & Disease Detection System

### AI-Powered Crop Disease Detection & Precision Agriculture Intelligence

*Real-time plant disease classification across 57 disease classes and 14 crop types with Next.js frontend and Flask inference backend — built for farmers, agronomists, and researchers.*

[![Python](https://img.shields.io/badge/Python-3.11%2F3.12-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-black.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-MobileNetV2-orange.svg)](https://www.tensorflow.org/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v4-38bdf8.svg)](https://tailwindcss.com/)

</div>

---

## 🔗 Live Demo & Links

- **➡️ Live Deployment**: [AgriScan Web App](https://agriscan-1-3aza.onrender.com)
- **🎥 Video Demo**: [AgriScan Video Walkthrough](https://drive.google.com/file/d/1vPA6HuWiGDg_IwlNVWht3NKPhUqPC-j1/view?usp=drivesdk)

---

## 📖 Overview

AgriScan is an end-to-end autonomous crop health monitoring and plant disease detection system. Users can photograph a crop leaf and receive an instant AI diagnosis — complete with primary signals, confidence scores, top-5 probability breakdown, severity assessment, and actionable treatment recommendations.

The system features real-time environmental tracking (temperature, humidity, wind, rainfall), disease risk forecasting per crop, and bilingual interface support (English / हिन्दी).

---

## ✨ Features

- 🔍 **Real-Time Leaf Disease Scanning**: Instant diagnosis with primary signals, confidence score gauges, and top-5 probability breakdowns.
- 🎯 **57 Disease Classes Across 14 Crops**: Apple, Tomato, Potato, Corn, Rice, Wheat, Grape, Peach, Pepper, Blueberry, Cherry, Raspberry, Soybean, and Squash.
- 💡 **Actionable Field Recommendations**: Step-by-step treatment and organic remediation notes tailored to detected diseases.
- 🌐 **Bilingual UI (English & हिन्दी)**: Complete language toggle for localized field notes and disease advice.
- 🌤 **Live Weather & Disease Risk Engine**: Real-time environmental metrics (Temperature, Humidity, Wind speed, Precipitation) + disease trigger forecasting and spray suitability advisories.
- 🗂 **Scan History**: Local history archiving for past leaf inspections with instant recall and cleanup.
- 📄 **Export Reports**: Clean print and PDF report export for offline farm records.
- ⚡ **Sub-100ms Inference**: MobileNetV2 transfer learning architecture optimized for rapid CPU and edge inference.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│              Next.js 16 + React 19 UI                    │
│   (Dark Field-Intelligence Interface · Lucide · Tailwind) │
│                                                          │
│  • Single Leaf Scan & Dropzone   • Top-5 Confidence Bar  │
│  • Actionable Recommendations    • Weather Risk Engine   │
│  • Scan History Archive          • Bilingual (EN / HI)   │
└────────────────────────────┬─────────────────────────────┘
                             │ HTTP (REST API)
┌────────────────────────────▼─────────────────────────────┐
│                   Flask Backend (app.py)                 │
│                                                          │
│  ┌──────────────────────────┐ ┌────────────────────────┐ │
│  │ MobileNetV2 Inference    │ │ Weather & Disease Risk │ │
│  │ (57 Disease Classes)     │ │ Engine Rules (14 Crops)│ │
│  └──────────────────────────┘ └────────────────────────┘ │
│  ┌──────────────────────────┐ ┌────────────────────────┐ │
│  │ Field History Database   │ │ SPA Static File Server │ │
│  │ (data/fields.json)       │ │ (static/ build)        │ │
│  └──────────────────────────┘ └────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

Frontend and backend are served from a **single origin** — no CORS issues, no separate hosting needed, one seamless deploy.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Model** | MobileNetV2 (Transfer Learning) across 57 classes |
| **Backend** | Python 3.12, Flask 3.0, Gunicorn, Pillow, NumPy |
| **Deep Learning** | TensorFlow 2.16 |
| **Frontend** | Next.js 16, React 19, Tailwind CSS 4, Lucide Icons, TypeScript |
| **Weather API** | Open-Meteo & OpenWeatherMap |
| **Deployment** | Single Web Service (Render / Gunicorn / Colab Tunnel) |

---

## 🧪 Model Details

| Metric | Value |
|---|---|
| **Architecture** | MobileNetV2 (Transfer Learning) |
| **Disease Classes** | 57 classes |
| **Crop Types** | 14 crops |
| **Validation Accuracy** | 84.6% – 85.3% |
| **Training Dataset** | PlantVillage + PlantDoc + Rice + Wheat (79,898 images) |
| **Input Resolution** | 224 × 224 px |

---

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the web application |
| `/health` | `GET` | Health check, returns loaded disease class count |
| `/predict` | `POST` | Upload a leaf image (`image` multipart form), returns diagnosis & top-5 |
| `/weather` | `GET` | Fetches live weather, spray advisories & disease risks (`lat`, `lon`, `crop`) |
| `/fields` | `GET` | List all tracked fields |
| `/fields/add` | `POST` | Register a new agricultural field |
| `/fields/<id>` | `GET` | Get details for a specific field |
| `/fields/<id>/update` | `POST` | Update field details |
| `/fields/<id>/delete` | `POST` | Remove a field |
| `/fields/<id>/history` | `GET` | Retrieve scan history for a field |
| `/summary` | `GET` | High-level summary stats across all fields |

---

## 📁 Repository Structure

```
AgriScan/
├── app.py                     # Flask inference API & static asset server
├── Procfile                   # Gunicorn process file
├── requirements.txt           # Python dependencies
├── runtime.txt                # Python runtime definition
├── .gitignore                 # Git ignore configuration
├── data/
│   └── fields.json            # Field tracking database
├── models/
│   ├── class_mapping.json     # 57 class label mappings
│   └── plant_disease_model.keras # Trained MobileNetV2 model
├── frontend/                  # Next.js / React source code
│   ├── app/                   # App router pages & layouts
│   ├── components/            # UI components
│   ├── lib/                   # Utility helpers
│   ├── public/                # Static assets & icons
│   └── package.json           # Frontend dependencies & scripts
└── static/                    # Production static build served by Flask
```

---

## 🚀 Getting Started

### 1. Run Backend Server (Flask)
```bash
# Clone the repository
git clone https://github.com/manasvikhare19/AgriScan.git
cd AgriScan

# Install Python dependencies
pip install -r requirements.txt

# Start the Flask app (serves UI and API at http://localhost:5000)
python app.py
```

### 2. Frontend Development (Optional)
If you want to modify or develop the Next.js frontend:
```bash
cd frontend

# Install Node dependencies
npm install

# Run frontend dev server (http://localhost:3000)
npm run dev

# Build production static export to update static/ folder
npm run build
```

---

## 🎓 Project
VIT Bhopal University — EPICS (Engineering Projects in Community Service)
