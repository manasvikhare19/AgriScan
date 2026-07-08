<div align="center">

# 🌾 AgriScan

### AI-Powered Crop Disease Detection System

*Real-time plant disease classification across 57 disease classes and 14 crop types — built for farmers, agronomists, and researchers.*

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-black.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow%20Lite-MobileNetV2-orange.svg)](https://www.tensorflow.org/lite)

</div>

---

## 🔗 Live Demo

**➡️ [DEPLOYED-LINK](https://agriscan-1-3aza.onrender.com)**

> Hosted on Render's free tier — the first request after a period of inactivity may take 30–50 seconds to wake up. Subsequent requests are near-instant.

---
## Demo
[https://drive.google.com/file/d/1NW45AYpmR5CGjyLKQ-R3N_6P89Ytvrz9/view?usp=drivesdk](https://drive.google.com/file/d/1hAsFsuKLKhiHjpezOX_llZju7HJg-Pg4/view?usp=drivesdk)

## 📖 Overview

AgriScan is an end-to-end plant disease detection system that lets a user photograph a crop leaf and receive an instant AI diagnosis — complete with confidence scores, severity assessment, and actionable treatment recommendations. It's built to be genuinely usable in the field, not just a model demo: batch scanning for multiple leaves at once, per-field scan history tracking, and weather-based risk scoring for proactive disease management.

The system is trained across four combined datasets (PlantVillage, PlantDoc, and dedicated Rice and Wheat disease sets), giving it broader real-world generalization than single-dataset models typically achieve.

## ✨ Features

- 🔍 **Single & Batch Scanning** — Diagnose one leaf or an entire folder of images in one pass
- 🎯 **57 Disease Classes, 14 Crops** — Apple, tomato, potato, corn, rice, wheat, grape, and more
- 📊 **Top-5 Confidence Breakdown** — Transparent predictions, not just a single black-box label
- 🌡️ **Weather-Aware Risk Scoring** — Cross-references live weather data to flag high-risk disease conditions per crop
- 🗂️ **Field Management & Scan History** — Track disease trends across multiple fields over time
- 💊 **Actionable Recommendations** — Each diagnosis pairs with practical treatment guidance
- 🌐 **Bilingual UI** — English and Hindi language support
- ⚡ **Sub-100ms Inference** — Optimized TFLite model runs fast even on constrained CPU environments

## 🏗️ Architecture

```
┌─────────────────┐         ┌──────────────────────────────┐
│   Browser UI     │  HTTP   │        Flask Backend          │
│  (static/index)  │◄───────►│                                │
│                  │         │  ┌──────────────────────────┐  │
│  • Upload image  │         │  │   TFLite Interpreter      │  │
│  • View results  │         │  │   (MobileNetV2, 57 cls)   │  │
│  • Field mgmt    │         │  └──────────────────────────┘  │
│  • Weather panel │         │  ┌──────────────────────────┐  │
└─────────────────┘         │  │  Field & Scan History     │  │
                             │  │  (JSON-backed storage)     │  │
                             │  └──────────────────────────┘  │
                             │  ┌──────────────────────────┐  │
                             │  │  OpenWeatherMap Client     │  │
                             │  └──────────────────────────┘  │
                             └──────────────────────────────┘
```

Frontend and backend are served from a **single origin** — no CORS, no separate hosting, one deploy.

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Model** | MobileNetV2 (transfer learning), converted to TensorFlow Lite |
| **Backend** | Flask, Gunicorn |
| **Inference** | TensorFlow Lite Interpreter |
| **Frontend** | Vanilla JS, HTML5, CSS3 |
| **External API** | OpenWeatherMap (risk scoring) |
| **Deployment** | Render (Web Service) |
| **Data Storage** | JSON-based field & scan history |

## 🧪 Model Details

| Metric | Value |
|---|---|
| Architecture | MobileNetV2 (transfer learning) |
| Disease classes | 57 |
| Crop types covered | 14 |
| Validation accuracy | 85.3% |
| Test accuracy | 85.0% |
| Training images | 79,898 |
| Datasets | PlantVillage, PlantDoc, Rice, Wheat |
| Inference format | TensorFlow Lite (quantized) |
| Inference time | ~25ms per image |

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the web application |
| `/health` | `GET` | Health check, returns model class count |
| `/predict` | `POST` | Upload an image, returns top-5 disease predictions |
| `/weather` | `GET` | Fetches current weather for risk scoring |
| `/fields` | `GET` | List all tracked fields |
| `/fields/add` | `POST` | Register a new field |
| `/fields/<id>` | `GET` | Get details for a specific field |
| `/fields/<id>/update` | `POST` | Update field details |
| `/fields/<id>/delete` | `POST` | Remove a field |
| `/fields/<id>/history` | `GET` | Retrieve scan history for a field |
| `/summary` | `GET` | Aggregate stats across all fields |

## 💻 Local Setup

```bash
# Clone the repo
git clone https://github.com/manasvikhare19/AgriScan.git
cd AgriScan

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

Visit `http://localhost:5000` in your browser.

## 📁 Project Structure

```
AgriScan/
├── app.py                          # Flask backend + inference logic
├── requirements.txt                 # Python dependencies
├── Procfile                          # Production start command
├── static/
│   └── index.html                   # Frontend (single-page app)
├── models/
│   ├── plant_disease_model.tflite    # Trained model (TFLite)
│   └── class_mapping.json            # Class index → disease label
└── data/
    └── fields.json                   # Field & scan history storage
```

## 🚀 Deployment

Deployed as a single web service on **Render**, serving both the API and frontend from one origin. See [Render documentation](https://render.com/docs) for deployment details specific to Flask + TensorFlow Lite applications.



