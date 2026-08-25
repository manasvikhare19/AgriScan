# -*- coding: utf-8 -*-
"""
AgriScan — production inference server.

Serves the trained plant-disease model over HTTP and hosts the
frontend from the same origin, so there's no separate tunnel URL
to paste in each time.
"""

import os
import io
import json
import datetime
import numpy as np
import tensorflow as tf
import requests
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "plant_disease_model.tflite")
MAPPING_PATH = os.path.join(BASE_DIR, "models", "class_mapping.json")
FIELDS_PATH  = os.path.join(BASE_DIR, "data", "fields.json")
FRONTEND_PATH = os.path.join(BASE_DIR, "static", "index.html")

os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

print("Loading model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
_input_details = interpreter.get_input_details()
_output_details = interpreter.get_output_details()
print("Model loaded (TFLite)")

with open(MAPPING_PATH) as f:
    ID_TO_LABEL = {int(k): v for k, v in json.load(f).items()}
print(f"Class mapping loaded — {len(ID_TO_LABEL)} classes")


def preprocess(pil_image):
    img = pil_image.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = arr / 255.0
    arr = (arr - 0.5) * 2.0
    return tf.constant(arr[np.newaxis], dtype=tf.float32)


RECS = {
    # ── APPLE ──────────────────────────────────────────────────────────────────
    "Apple___Apple_scab": [
        "Spray captan or myclobutanil fungicide every 7–10 days from early spring",
        "Pick up and burn all fallen leaves — they spread the disease",
        "Trim branches so air can move freely through the tree",
        "Use drip watering instead of sprinklers to keep leaves dry",
        "Next planting: choose scab-resistant apple varieties like Liberty or GoldRush",
    ],
    "Apple___Black_rot": [
        "Cut off infected branches 20–25 cm below the black area during dry weather",
        "Remove all shrivelled, rotten fruits left on the tree — they carry the disease",
        "Spray captan fungicide after flowers fall and again 2 weeks later",
        "Keep the tree well-fed with balanced fertilizer — weak trees get infected faster",
        "Dip pruning tools in diluted bleach (1 part bleach, 9 parts water) between cuts",
    ],
    "Apple___Cedar_apple_rust": [
        "Spray myclobutanil or propiconazole fungicide every 7–14 days from pink bud stage",
        "Start spraying early — do not wait until you see orange spots on leaves",
        "If possible, remove cedar or juniper trees growing within 300 m of your apple orchard",
        "In March–April, check cedar trees for orange jelly-like growths and remove them",
        "For future planting: choose rust-resistant varieties like Redfree or Williams Pride",
    ],
    "Apple___healthy": [
        "Tree looks healthy — keep up your current care routine",
        "Do a copper spray once in the dormant season to prevent fungal buildup",
        "Check trees weekly during rainy periods for early signs of scab or rust",
    ],
    "Blueberry___healthy": [
        "Plants look healthy — no treatment needed right now",
        "Keep soil slightly acidic (pH 4.5–5.5); add sulfur if soil is too alkaline",
        "Check plants weekly during flowering for any signs of disease",
    ],
    "Cherry_(including_sour)___Powdery_mildew": [
        "Spray sulfur or neem oil every 10 days when you see white powdery patches",
        "Cut off and throw away any branches with curled, white-coated leaves",
        "Do not over-fertilize with nitrogen — too much makes leaves soft and disease-prone",
        "Trim the tree so air flows well inside the canopy",
    ],
    "Cherry_(including_sour)___healthy": [
        "Tree looks healthy — no treatment needed",
        "Apply a copper spray before winter to protect against bacterial diseases",
        "Thin out crowded fruit clusters to improve airflow",
    ],
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": [
        "Spray azoxystrobin or propiconazole fungicide when the crop is tasseling",
        "Only spray if more than half the plants show spots — small infections often heal on their own",
        "Buy corn seed varieties that are rated resistant to gray leaf spot",
        "Rotate with soybean or wheat next season to reduce leftover disease in the soil",
        "After harvest, plow the field to bury infected crop remains",
    ],
    "Corn_(maize)___Common_rust_": [
        "Spray propiconazole fungicide if rust covers more than half the leaves before tasseling",
        "In warm climates, rust rarely causes serious damage — monitor before spending on sprays",
        "Choose rust-resistant hybrid seeds for next planting season",
        "Remove any stray corn plants growing around the field — they carry disease over seasons",
    ],
    "Corn_(maize)___Northern_Leaf_Blight": [
        "Spray propiconazole or azoxystrobin as soon as you see long grey-green spots on leaves",
        "Choose resistant corn varieties for next season — ask your seed dealer",
        "Rotate with soybean or sunflower for 1–2 years to clean up the field",
        "Plow infected crop remains deep into the soil after harvest",
        "Use drip irrigation instead of sprinklers — wet leaves spread the disease faster",
    ],
    "Corn_(maize)___healthy": [
        "Crop looks healthy — no treatment needed",
        "Keep checking weekly for rust or blight spots, especially during humid weather",
    ],
    "Grape___Black_rot": [
        "Spray mancozeb fungicide every 10–14 days starting when buds open in spring",
        "Remove all shrivelled black berries from vines and the ground before spring",
        "Prune vines so sunlight and air can reach all parts of the plant",
        "Rake and burn all fallen leaves under the vines",
    ],
    "Grape___Esca_(Black_Measles)": [
        "Only prune on dry days — avoid pruning when it is raining or just after rain",
        "Seal every pruning cut immediately with wound-sealing paint",
        "Remove and burn vines that are severely affected inside the wood",
        "There is no spray cure — prevention through careful pruning is the only option",
    ],
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": [
        "Spray copper fungicide or mancozeb every 10–14 days from early shoot growth",
        "Pick off and destroy leaves with brown spots",
        "Remove some leaves near the grape clusters to improve airflow",
        "Avoid watering from above — water at the base of the plant",
    ],
    "Grape___healthy": [
        "Vines look healthy — no treatment needed",
        "Scout weekly during the growing season for early signs of mildew or blight",
    ],
    "Orange___Haunglongbing_(Citrus_greening)": [
        "⚠ URGENT: Tell your local agriculture office immediately — this disease is very serious",
        "Cut down and destroy the infected tree completely, including roots",
        "Spray imidacloprid insecticide to kill the tiny psyllid insect that spreads this disease",
        "Put up yellow sticky traps around your grove to catch and track psyllids",
        "There is no cure — the only option is removing infected trees and killing the insect",
    ],
    "Peach___Bacterial_spot": [
        "Spray copper fungicide every 10–14 days from when fruit starts forming",
        "Water at the base of the tree — never spray water on the leaves",
        "Choose resistant peach varieties like Contender or Redhaven for new plantings",
        "Avoid cutting or bruising leaves and fruit during daily work in the orchard",
    ],
    "Peach___healthy": [
        "Tree looks healthy — no treatment needed",
        "Apply a copper spray before buds open in spring as a preventive step",
    ],
    "Pepper,_bell___Bacterial_spot": [
        "Spray copper fungicide every 7 days as soon as you see spots on leaves",
        "Only use certified, disease-free seeds or seedlings from a trusted nursery",
        "Do not grow pepper or tomato in the same field for at least 2 years",
        "Use drip irrigation and avoid walking in the field when plants are wet",
        "Pull out and destroy badly infected plants right away",
    ],
    "Pepper,_bell___healthy": [
        "Plants look healthy — no treatment needed",
        "Check weekly during warm and wet weather for small water-soaked spots on leaves",
    ],
    "Potato___Early_blight": [
        "Spray chlorothalonil or mancozeb fungicide every 7–10 days when spots appear",
        "Make sure plants get enough potassium fertilizer — low potassium makes plants more vulnerable",
        "Do not grow potatoes or tomatoes in the same field more than once every 3 years",
        "Remove and burn any stray potato plants near the field",
    ],
    "Potato___Late_blight": [
        "⚠ URGENT: Spray mancozeb fungicide today — do not wait even one more day",
        "Cut off and bury all infected leaves and stems — do not leave them on the ground",
        "Check your neighbors' fields too — this disease can travel long distances in the wind",
        "Harvest as early as you can once the plant tops die back",
        "Use only certified, healthy seed potatoes for planting",
    ],
    "Potato___healthy": [
        "Crop looks healthy — no treatment needed",
        "Always use certified seed potatoes from a trusted source",
        "Keep checking during cool, wet weather for first signs of blight",
    ],
    "Raspberry___healthy": [
        "Plants look healthy — no treatment needed",
        "After harvest, cut out all the old canes that have fruited",
        "Check dormant canes in winter for any signs of cane blight",
    ],
    "Rice___Bacterialblight": [
        "⚠ URGENT: Spray copper oxychloride (3 g per litre of water) as soon as you see yellow edges on leaves",
        "Drain the flooded field immediately if water came from an infected area",
        "Do not apply extra urea — too much nitrogen makes rice plants weaker and more prone to disease",
        "Use resistant varieties like IR64 or Swarna-Sub1 for your next crop",
        "Remove and burn badly infected tillers — do not let them float in the irrigation water",
    ],
    "Rice___Blast": [
        "Spray tricyclazole (0.6 g per litre) at the flowering stage and again 10 days later",
        "Do not apply all your urea at once — split it into smaller doses",
        "Use blast-resistant varieties like CO 51 or Improved Sambha Mahsuri",
        "Avoid irrigating in the evening — wet leaves overnight greatly increases blast risk",
        "Check the neck of the spike carefully at flowering — neck blast can destroy the entire yield",
    ],
    "Rice___Brownspot": [
        "Spray mancozeb or propiconazole at the tillering stage and again at booting",
        "Check soil nutrients — brown spot usually means the crop is stressed or not getting enough food",
        "Keep 5 cm of standing water in the field during tillering to reduce plant stress",
        "Soak seeds in carbendazim solution (1 g per litre) for 24 hours before sowing",
    ],
    "Rice___Tungro": [
        "⚠ URGENT: Pull out and destroy infected plants right away — the disease is spread by a hopper insect",
        "Spray imidacloprid or thiamethoxam to kill the green leafhopper that carries the virus",
        "Use Tungro-resistant varieties like IR36, IR64, or locally recommended lines",
        "Try to plant your crop at the same time as your neighbors — staggered sowing helps the disease spread",
        "Clear weed grasses on field bunds — they shelter the insects that spread Tungro",
    ],
    "Soybean___healthy": [
        "Crop looks healthy — no treatment needed",
        "Check weekly from flowering to pod fill for any signs of rust or mold",
    ],
    "Squash___Powdery_mildew": [
        "Spray potassium bicarbonate or diluted neem oil when you first see white patches on leaves",
        "Give more space between plants — at least 90 cm apart — so air can flow freely",
        "Do not over-fertilize with nitrogen; it makes leaves soft and easy to infect",
        "Choose resistant squash varieties for your next planting",
    ],
    "Strawberry___Leaf_scorch": [
        "Spray captan fungicide every 10–14 days from the start of the growing season",
        "Remove all leaves with purple spots and brown scorched edges",
        "Plant strawberries at least 30 cm apart so air can move between plants",
        "Water at the base using drip lines — avoid wetting the leaves",
    ],
    "Strawberry___healthy": [
        "Plants look healthy — no treatment needed",
        "After the last harvest, mow the plants, thin the rows, and apply fertilizer",
    ],
    "Tomato___Bacterial_spot": [
        "Spray copper fungicide every 5–7 days as a preventive measure",
        "Use only certified, disease-free seedlings or seeds",
        "Do not grow tomatoes or peppers in the same bed for 2 years",
        "Never work in the field when plants are wet — you will spread the bacteria",
        "Remove and bury infected plant debris at the end of the season",
    ],
    "Tomato___Early_blight": [
        "Spray chlorothalonil or azoxystrobin every 7 days when dark spots appear on lower leaves",
        "Put mulch (straw or plastic sheet) on the soil to stop mud from splashing onto leaves",
        "Remove and throw away infected lower leaves — do not compost them",
        "Make sure plants get enough potassium and calcium in their fertilizer",
    ],
    "Tomato___Late_blight": [
        "⚠ URGENT: Spray mancozeb fungicide within 24 hours of seeing any brown, water-soaked patches",
        "Remove all infected leaves and stems and put them in bags — do not leave on the ground",
        "Improve drainage in the field — blight spreads fast in wet, waterlogged conditions",
        "Check nearby tomato fields too — this disease travels in wind and rain",
    ],
    "Tomato___Leaf_Mold": [
        "Spray copper fungicide when you see pale yellow patches on tops of leaves",
        "Open vents or windows in the greenhouse to bring humidity down",
        "Stake plants and remove extra side shoots to let more air reach the leaves",
        "Remove leaves that show olive-brown fuzzy patches on the underside",
    ],
    "Tomato___Septoria_leaf_spot": [
        "Spray mancozeb or chlorothalonil every 7–10 days when you see small circular spots with dark edges",
        "Put mulch on the soil to stop spores splashing from the ground onto leaves",
        "Remove spotted leaves and throw them away — do not leave in the field",
        "Use drip irrigation instead of sprinklers to keep leaves dry",
    ],
    "Tomato___Spider_mites Two-spotted_spider_mite": [
        "Spray abamectin miticide on the undersides of leaves where mites hide",
        "Water the plants more regularly — mites thrive when it is hot and dry",
        "Blast leaves with a strong jet of water to knock mites off",
        "Avoid using broad-spectrum insecticides that kill helpful insects which eat mites",
    ],
    "Tomato___Target_Spot": [
        "Spray azoxystrobin or tebuconazole when you see round brown spots with rings like a target",
        "Remove extra side shoots and lower leaves to improve airflow through the plant",
        "Remove and discard infected leaves — do not leave them on the soil",
        "Increase spacing between plants — at least 60 cm to allow air circulation",
    ],
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": [
        "⚠ URGENT: Pull out and bag all plants with curled, yellowing leaves right away",
        "Spray imidacloprid to kill the whitefly insect that carries this virus",
        "Cover nursery seedlings with fine insect-proof nets to keep whiteflies out",
        "Place silver reflective mulch on the soil — it confuses and repels whiteflies",
        "For next planting: buy TYLCV-resistant tomato varieties",
    ],
    "Tomato___Tomato_mosaic_virus": [
        "Remove and destroy all plants showing patchy, mottled, or distorted leaves immediately",
        "After touching any plant, wash hands and dip tools in diluted bleach before touching others",
        "Spray imidacloprid to control aphids that carry and spread the virus",
        "Do not smoke or handle tobacco near tomato plants — this virus can transfer from tobacco",
        "Use certified, virus-tested seedlings from a reliable nursery",
    ],
    "Tomato___healthy": [
        "Plants look healthy — no treatment needed",
        "Check weekly during warm, humid weather for early signs of blight or whiteflies",
    ],
    "Wheat___aphid_test": [
        "Spray dimethoate or thiamethoxam insecticide when you see more than 5 aphids per plant",
        "Do not over-apply nitrogen fertilizer — too much makes soft leaves that attract aphids",
        "Check carefully during the grain filling stage — this is when aphids cause the most damage",
        "Use resistant wheat varieties like WH-1105 if aphids are a regular problem in your area",
    ],
    "Wheat___black_rust_test": [
        "⚠ URGENT: Spray tebuconazole or propiconazole immediately — black rust spreads very fast",
        "Treat the whole field — do not treat only a few plants",
        "Plant resistant varieties like PBW-550 or HD-2967 for next season",
        "Remove any barberry bushes near the field — they help the rust disease survive",
        "Report unusual rust outbreaks to your local agriculture office right away",
    ],
    "Wheat___blast_test": [
        "⚠ URGENT: Spray tricyclazole or tebuconazole within 48 hours of seeing white, empty spikes",
        "Do not sow late — crops sown at the right time avoid the worst blast season",
        "Drain any waterlogged low areas of the field — blast is worst in wet spots",
        "Plant blast-tolerant varieties recommended in your area",
    ],
    "Wheat___brown_rust_test": [
        "Spray propiconazole or mancozeb at the flag leaf stage before rust spreads to upper leaves",
        "Scout from the tillering stage — brown rust is the most common rust in wheat",
        "Plant resistant varieties like GW-496 or HD-2781 for next season",
        "Spray again after 14 days if humidity stays high after the first spray",
    ],
    "Wheat___common_root_rot_test": [
        "Treat seeds with carboxin + thiram fungicide before sowing",
        "Keep soil moist but not waterlogged after sowing — dry-wet cycles make root rot worse",
        "Rotate with mustard, chickpea, or other non-wheat crops for 3 years",
        "Apply phosphorus fertilizer at sowing to help roots grow strong",
        "After harvest, plow deeply to bury infected crop leftovers",
    ],
    "Wheat___fusarium_head_blight_test": [
        "⚠ URGENT: Spray tebuconazole when exactly half the crop is in flower — timing is very important",
        "Do not irrigate during flowering — wet conditions at flowering trigger infection",
        "Harvest on time — leaving the crop too long increases harmful toxin levels in the grain",
        "Plant moderately resistant varieties; ask your local seed dealer for recommendations",
        "Do not feed visibly infected grain to pigs or poultry — it contains harmful toxins",
    ],
    "Wheat___healthy_test": [
        "Crop looks healthy — no treatment needed",
        "Keep checking weekly from tillering to grain fill for rust or blight signs",
    ],
    "Wheat___leaf_blight_test": [
        "Spray propiconazole or mancozeb when you see tan or brown spots on leaf blades",
        "Make sure the crop is getting balanced fertilizer — nutrient shortage makes blight worse",
        "After harvest, plow the field deeply to bury infected straw",
        "Use disease-tolerant varieties recommended by your local agriculture department",
    ],
    "Wheat___mildew_test": [
        "Spray triadimefon or myclobutanil when you see white powdery patches on lower leaves",
        "Do not sow too densely — crowded plants have poor airflow and get mildew faster",
        "Cut back on nitrogen top-dressing — too much nitrogen makes leaves soft and prone to mildew",
        "Use resistant varieties like WH-711 or K-307 for next season",
    ],
    "Wheat___mite_test": [
        "Spray dicofol or abamectin on the undersides of leaves — mites hide there",
        "Irrigate the field if it is very dry — mites thrive in hot, dusty conditions",
        "Do not rely only on top sprays — mites on leaf undersides will not be reached",
        "Avoid dust roads near the field — road dust helps mite populations grow faster",
    ],
    "Wheat___septoria_test": [
        "Spray azoxystrobin at the flag leaf stage before the disease moves to the top leaves",
        "Start scouting from the first node stage — septoria blotch moves slowly upward",
        "Rotate with chickpea, mustard, or other non-wheat crops for at least 2 years",
        "Use certified seed treated with fludioxonil to reduce seed-carried infection",
    ],
    "Wheat___smut_test": [
        "Treat seeds with tebuconazole or carboxin + thiram before sowing — this is the most important step",
        "Only use certified, clean seed — never use seed saved from a smutted field",
        "Before harvest, bag and burn any smutted spikes to stop spores from spreading",
        "Clean the thresher thoroughly between fields to avoid spreading smut spores",
    ],
    "Wheat___stem_fly_test": [
        "Treat seeds with imidacloprid or chlorpyrifos before sowing to protect young seedlings",
        "Sow on time — early sown crops escape the worst stem fly attacks",
        "Look for 'dead heart' — the central leaf of a young tiller turns yellow and pulls out easily",
        "Spray thiamethoxam on young plants at the 2–3 leaf stage if stem fly is active",
        "Pull out and destroy affected tillers to stop the insect from completing its life cycle",
    ],
    "Wheat___tan_spot_test": [
        "Spray propiconazole when you see oval tan or brown spots with a yellow ring around them",
        "After harvest, plow deeply to bury infected wheat straw in the soil",
        "Rotate with chickpea, canola, or other non-grass crops for 2 years",
        "Use certified treated seed — tan spot can carry over from one season to the next through seeds",
    ],
    "Wheat___yellow_rust_test": [
        "⚠ URGENT: Spray tebuconazole or propiconazole immediately — yellow rust spreads very fast in cool weather",
        "Start spraying as soon as you see yellow stripes on leaves — do not wait",
        "Plant resistant varieties like PBW-677 or HD-3059 for next season",
        "Check neighboring farms too — yellow rust can travel many kilometres through the air",
        "Spray again 14 days later if the weather stays cool and cloudy",
    ],
}
DEFAULT_RECS = [
    "Walk the field and inspect plants closely with a crop expert",
    "Contact your local agriculture extension officer (KVK/ATMA) for a confirmed diagnosis",
    "Retake the photo in good daylight with a clear, close-up view of the affected leaf",
]

SEVERITY = {
    "Apple___Black_rot": "high", "Apple___Apple_scab": "med", "Apple___Cedar_apple_rust": "med",
    "Cherry_(including_sour)___Powdery_mildew": "med",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "med", "Corn_(maize)___Common_rust_": "med",
    "Corn_(maize)___Northern_Leaf_Blight": "high",
    "Grape___Black_rot": "high", "Grape___Esca_(Black_Measles)": "high", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "med",
    "Orange___Haunglongbing_(Citrus_greening)": "high",
    "Peach___Bacterial_spot": "med", "Pepper,_bell___Bacterial_spot": "med",
    "Potato___Early_blight": "med", "Potato___Late_blight": "high",
    "Rice___Bacterialblight": "high", "Rice___Blast": "high", "Rice___Brownspot": "med", "Rice___Tungro": "high",
    "Squash___Powdery_mildew": "med", "Strawberry___Leaf_scorch": "med",
    "Tomato___Bacterial_spot": "med", "Tomato___Early_blight": "med", "Tomato___Late_blight": "high",
    "Tomato___Leaf_Mold": "med", "Tomato___Septoria_leaf_spot": "med",
    "Tomato___Spider_mites Two-spotted_spider_mite": "med", "Tomato___Target_Spot": "med",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "high", "Tomato___Tomato_mosaic_virus": "high",
    "Wheat___black_rust_test": "high", "Wheat___blast_test": "high", "Wheat___brown_rust_test": "med",
    "Wheat___common_root_rot_test": "med", "Wheat___fusarium_head_blight_test": "high",
    "Wheat___leaf_blight_test": "med", "Wheat___mildew_test": "med", "Wheat___mite_test": "med",
    "Wheat___septoria_test": "med", "Wheat___smut_test": "high", "Wheat___stem_fly_test": "med",
    "Wheat___tan_spot_test": "med", "Wheat___yellow_rust_test": "high", "Wheat___aphid_test": "med",
}
def get_severity(label):
    if label in SEVERITY:
        return SEVERITY[label]
    if "healthy" in label.lower(): return "low"
    if any(k in label.lower() for k in ["late_blight","tungro","blast","haunglongbing",
        "mosaic_virus","yellow_leaf_curl","black_rot","esca","smut","fusarium","black_rust","yellow_rust"]):
        return "high"
    return "med"

# ── 4. NEW: Disease Risk Engine ──────────────────────────────
# Maps each crop to its key disease triggers based on temperature & humidity.
DISEASE_RISK_RULES = {
    "Rice": [
        {"disease": "Rice Blast",            "temp_min": 20, "temp_max": 30, "humidity_min": 80, "risk": "High"},
        {"disease": "Bacterial Blight",      "temp_min": 25, "temp_max": 35, "humidity_min": 70, "risk": "High"},
        {"disease": "Brown Spot",            "temp_min": 25, "temp_max": 35, "humidity_min": 65, "risk": "Medium"},
    ],
    "Wheat": [
        {"disease": "Yellow Rust",           "temp_min": 8,  "temp_max": 15, "humidity_min": 70, "risk": "High"},
        {"disease": "Brown Rust",            "temp_min": 15, "temp_max": 22, "humidity_min": 70, "risk": "High"},
        {"disease": "Powdery Mildew",        "temp_min": 15, "temp_max": 20, "humidity_min": 60, "risk": "Medium"},
        {"disease": "Fusarium Head Blight",  "temp_min": 20, "temp_max": 30, "humidity_min": 80, "risk": "High"},
    ],
    "Tomato": [
        {"disease": "Late Blight",           "temp_min": 10, "temp_max": 25, "humidity_min": 85, "risk": "High"},
        {"disease": "Early Blight",          "temp_min": 24, "temp_max": 35, "humidity_min": 70, "risk": "Medium"},
        {"disease": "Septoria Leaf Spot",    "temp_min": 20, "temp_max": 25, "humidity_min": 75, "risk": "Medium"},
        {"disease": "Leaf Mold",             "temp_min": 20, "temp_max": 30, "humidity_min": 85, "risk": "High"},
    ],
    "Potato": [
        {"disease": "Late Blight",           "temp_min": 10, "temp_max": 25, "humidity_min": 85, "risk": "High"},
        {"disease": "Early Blight",          "temp_min": 24, "temp_max": 32, "humidity_min": 70, "risk": "Medium"},
    ],
    "Corn": [
        {"disease": "Gray Leaf Spot",        "temp_min": 22, "temp_max": 32, "humidity_min": 80, "risk": "High"},
        {"disease": "Northern Leaf Blight",  "temp_min": 18, "temp_max": 27, "humidity_min": 75, "risk": "Medium"},
        {"disease": "Common Rust",           "temp_min": 15, "temp_max": 25, "humidity_min": 70, "risk": "Medium"},
    ],
    "Grape": [
        {"disease": "Black Rot",             "temp_min": 20, "temp_max": 30, "humidity_min": 75, "risk": "High"},
        {"disease": "Powdery Mildew",        "temp_min": 20, "temp_max": 30, "humidity_min": 50, "risk": "Medium"},
    ],
    "Apple": [
        {"disease": "Apple Scab",            "temp_min": 10, "temp_max": 24, "humidity_min": 75, "risk": "High"},
        {"disease": "Cedar Apple Rust",      "temp_min": 15, "temp_max": 25, "humidity_min": 70, "risk": "Medium"},
    ],
}

def compute_disease_risk(crop, temp_c, humidity_pct):
    """Return list of disease risk warnings for current weather conditions."""
    rules = DISEASE_RISK_RULES.get(crop, [])
    warnings = []
    for rule in rules:
        temp_ok     = rule["temp_min"] <= temp_c <= rule["temp_max"]
        humidity_ok = humidity_pct >= rule["humidity_min"]
        if temp_ok and humidity_ok:
            warnings.append({
                "disease":  rule["disease"],
                "risk":     rule["risk"],
                "reason":   f"Temp {temp_c}°C + Humidity {humidity_pct}% favour this disease"
            })
    if not warnings:
        warnings.append({
            "disease": "No major threat",
            "risk":    "Low",
            "reason":  "Current weather conditions are not ideal for common diseases"
        })
    return warnings

# ── 5. NEW: Field Database helpers ───────────────────────────
def load_fields():
    if os.path.exists(FIELDS_PATH):
        with open(FIELDS_PATH) as f:
            return json.load(f)
    return {}

def save_fields(fields):
    with open(FIELDS_PATH, "w") as f:
        json.dump(fields, f, indent=2)

# ── 6. Flask app ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "classes": len(ID_TO_LABEL)})

# ── /predict — inference and optional field history logging ──
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400
    try:
        field_id = request.form.get("field_id", None)   # optional field tag

        img    = Image.open(io.BytesIO(request.files["image"].read()))
        tensor = preprocess(img)
        interpreter.set_tensor(_input_details[0]["index"], tensor.numpy())
        interpreter.invoke()
        probs  = interpreter.get_tensor(_output_details[0]["index"])[0]
        top5   = np.argsort(probs)[::-1][:5]

        preds = [{"class_id": int(i), "label": ID_TO_LABEL.get(int(i), f"class_{i}"),
                  "confidence": float(probs[i]),
                  "confidence_pct": f"{probs[i]*100:.1f}%"} for i in top5]

        top_label = preds[0]["label"]
        display_name = top_label.replace("___", " — ").replace("_", " ")
        recs = RECS.get(top_label, DEFAULT_RECS)
        top5_formatted = [
            {
                "class_id": p["class_id"],
                "label": p["label"],
                "disease": p["label"].replace("___", " — ").replace("_", " "),
                "display_name": p["label"].replace("___", " — ").replace("_", " "),
                "confidence": p["confidence"],
                "confidence_pct": p["confidence_pct"],
            }
            for p in preds
        ]

        result = {
            "success": True,
            "disease": display_name,
            "prediction": display_name,
            "confidence": preds[0]["confidence"],
            "confidence_pct": preds[0]["confidence_pct"],
            "severity": get_severity(top_label),
            "is_healthy": "healthy" in top_label.lower(),
            "recommendations": recs,
            "top_prediction": {
                "label":          top_label,
                "display_name":   display_name,
                "confidence":     preds[0]["confidence"],
                "confidence_pct": preds[0]["confidence_pct"],
                "severity":       get_severity(top_label),
                "is_healthy":     "healthy" in top_label.lower(),
                "recommendations": recs,
            },
            "top5": top5_formatted
        }

        # Log scan into field history if field_id provided
        if field_id:
            fields = load_fields()
            if field_id in fields:
                scan_entry = {
                    "timestamp":   datetime.datetime.now().isoformat(),
                    "label":       top_label,
                    "display":     display_name,
                    "confidence":  preds[0]["confidence_pct"],
                    "severity":    get_severity(top_label),
                    "is_healthy":  "healthy" in top_label.lower(),
                }
                fields[field_id].setdefault("scan_history", []).insert(0, scan_entry)
                fields[field_id]["scan_history"] = fields[field_id]["scan_history"][:50]  # keep last 50
                fields[field_id]["last_scanned"] = scan_entry["timestamp"]
                save_fields(fields)
                result["field_logged"] = True

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── /weather — fetch live weather + disease risk ─────────────
@app.route("/weather")
def weather():
    lat  = request.args.get("lat")
    lon  = request.args.get("lon")
    crop = request.args.get("crop", "")
    api_key = request.args.get("api_key", "")

    if not lat or not lon:
        return jsonify({"error": "lat and lon are required"}), 400

    try:
        if api_key:
            url = (f"https://api.openweathermap.org/data/2.5/weather"
                   f"?lat={lat}&lon={lon}&appid={api_key}&units=metric")
            resp = requests.get(url, timeout=8)
            if resp.status_code != 200:
                return jsonify({"error": f"Weather API error: {resp.status_code}"}), 502

            w = resp.json()
            temp      = w["main"]["temp"]
            humidity  = w["main"]["humidity"]
            feels     = w["main"]["feels_like"]
            wind_kph  = round(w["wind"]["speed"] * 3.6, 1)
            condition = w["weather"][0]["description"].capitalize()
            city      = w.get("name", "Unknown location")
            rain_1h   = w.get("rain", {}).get("1h", 0)
        else:
            # Fallback to free Open-Meteo service
            url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                   f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,wind_speed_10m,weather_code"
                   f"&wind_speed_unit=kmh&timezone=auto")
            resp = requests.get(url, timeout=8)
            if resp.status_code != 200:
                return jsonify({"error": f"Open-Meteo error: {resp.status_code}"}), 502

            w = resp.json()
            c = w.get("current", {})
            temp      = c.get("temperature_2m", 25.0)
            humidity  = c.get("relative_humidity_2m", 60)
            feels     = c.get("apparent_temperature", temp)
            wind_kph  = c.get("wind_speed_10m", 5.0)
            rain_1h   = c.get("precipitation", 0)
            condition = "Clear"
            city      = f"Field Region ({lat}, {lon})"

        # Disease risk for the selected crop
        risk_warnings = []
        if crop:
            risk_warnings = compute_disease_risk(crop, temp, humidity)

        # General spray advisory
        spray_ok = wind_kph < 20 and rain_1h == 0
        spray_advice = (
            "Good conditions for spraying — wind is calm and no rain" if spray_ok
            else "Avoid spraying now — wind too strong or rain expected"
        )

        return jsonify({
            "city":          city,
            "temperature":   temp,
            "feels_like":    feels,
            "humidity":      humidity,
            "wind_kph":      wind_kph,
            "condition":     condition,
            "rain_1h_mm":    rain_1h,
            "spray_ok":      spray_ok,
            "spray_advice":  spray_advice,
            "disease_risk":  risk_warnings,
        })
    except requests.exceptions.Timeout:
        return jsonify({"error": "Weather service timed out — check your internet connection"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Static File / SPA Route ──────────────────────────────────
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    static_folder = os.path.join(BASE_DIR, "static")
    if path != "" and os.path.exists(os.path.join(static_folder, path)):
        return send_from_directory(static_folder, path)
    return send_from_directory(static_folder, "index.html")

# ── NEW: /fields — list all fields ───────────────────────────
@app.route("/fields", methods=["GET"])
def list_fields():
    return jsonify(load_fields())

# ── NEW: /fields/add — create a new field ───────────────────
@app.route("/fields/add", methods=["POST"])
def add_field():
    try:
        data = request.get_json()
        required = ["id", "name", "crop", "location"]
        for key in required:
            if key not in data:
                return jsonify({"error": f"Missing field: {key}"}), 400

        fields = load_fields()
        if data["id"] in fields:
            return jsonify({"error": "A field with this ID already exists"}), 409

        fields[data["id"]] = {
            "id":           data["id"],
            "name":         data["name"],
            "crop":         data["crop"],
            "location":     data["location"],         # plain text e.g. "Village Rampur, MP"
            "area_acres":   data.get("area_acres", ""),
            "soil_type":    data.get("soil_type", ""),
            "notes":        data.get("notes", ""),
            "created":      datetime.datetime.now().isoformat(),
            "last_scanned": None,
            "scan_history": [],
        }
        save_fields(fields)
        return jsonify({"success": True, "field": fields[data["id"]]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── NEW: /fields/<id> — get one field with full history ─────
@app.route("/fields/<field_id>", methods=["GET"])
def get_field(field_id):
    fields = load_fields()
    if field_id not in fields:
        return jsonify({"error": "Field not found"}), 404
    return jsonify(fields[field_id])

# ── NEW: /fields/<id>/update — edit field details ────────────
@app.route("/fields/<field_id>/update", methods=["POST"])
def update_field(field_id):
    try:
        fields = load_fields()
        if field_id not in fields:
            return jsonify({"error": "Field not found"}), 404
        data = request.get_json()
        allowed = ["name", "crop", "location", "area_acres", "soil_type", "notes"]
        for key in allowed:
            if key in data:
                fields[field_id][key] = data[key]
        save_fields(fields)
        return jsonify({"success": True, "field": fields[field_id]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── NEW: /fields/<id>/delete — remove a field ────────────────
@app.route("/fields/<field_id>/delete", methods=["POST"])
def delete_field(field_id):
    fields = load_fields()
    if field_id not in fields:
        return jsonify({"error": "Field not found"}), 404
    removed = fields.pop(field_id)
    save_fields(fields)
    return jsonify({"success": True, "removed": removed["name"]})

# ── NEW: /fields/<id>/history — just the scan log ────────────
@app.route("/fields/<field_id>/history", methods=["GET"])
def field_history(field_id):
    fields = load_fields()
    if field_id not in fields:
        return jsonify({"error": "Field not found"}), 404
    history = fields[field_id].get("scan_history", [])
    # Summary stats
    total   = len(history)
    healthy = sum(1 for s in history if s.get("is_healthy"))
    disease = total - healthy
    high    = sum(1 for s in history if s.get("severity") == "high")
    return jsonify({
        "field_id":     field_id,
        "field_name":   fields[field_id]["name"],
        "crop":         fields[field_id]["crop"],
        "total_scans":  total,
        "healthy":      healthy,
        "diseased":     disease,
        "high_severity":high,
        "history":      history,
    })

# ── NEW: /summary — officer dashboard overview ───────────────
@app.route("/summary", methods=["GET"])
def summary():
    """High-level stats across all fields — for agriculture officers."""
    fields = load_fields()
    total_fields  = len(fields)
    total_scans   = 0
    disease_count = {}
    high_alert_fields = []

    for fid, field in fields.items():
        history = field.get("scan_history", [])
        total_scans += len(history)
        for scan in history:
            if not scan.get("is_healthy"):
                label = scan.get("label", "Unknown")
                disease_count[label] = disease_count.get(label, 0) + 1
            if scan.get("severity") == "high":
                if fid not in [f["id"] for f in high_alert_fields]:
                    high_alert_fields.append({
                        "id":   fid,
                        "name": field["name"],
                        "crop": field["crop"],
                        "last_disease": scan.get("display", ""),
                    })

    top_diseases = sorted(disease_count.items(), key=lambda x: x[1], reverse=True)[:5]

    return jsonify({
        "total_fields":     total_fields,
        "total_scans":      total_scans,
        "top_diseases":     [{"label": d, "count": c} for d, c in top_diseases],
        "high_alert_fields": high_alert_fields,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
