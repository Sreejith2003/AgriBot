from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import joblib
import pandas as pd
import numpy as np
import bcrypt
import time
import google.generativeai as genai
import logging
import random
from gtts import gTTS
import sklearn

# Import all_states from data.schemes
from data.schemes import all_states

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Initialize Flask app
app = Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = os.urandom(24).hex()
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5000", "http://localhost:3000"]}})

# --- Model Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOIL_MODEL_PATH = os.path.join(BASE_DIR, "model", "efficientnet_soil (1).pth")
CROP_MODEL_PATH = os.path.join(BASE_DIR, "model", "crop_model (2) (1).pkl")
IRRIGATION_MODEL_PATH = os.path.join(BASE_DIR, "model", "irrigation_model (1).pkl")

# Verify model files exist
for path in [SOIL_MODEL_PATH, CROP_MODEL_PATH, IRRIGATION_MODEL_PATH]:
    if not os.path.exists(path):
        logging.warning(f"Model file not found: {path}")

# Print working directory and paths
logging.info(f"Current working directory: {os.getcwd()}")
logging.info(f"Absolute path to crop model: {os.path.abspath(CROP_MODEL_PATH)}")
logging.info(f"Absolute path to irrigation model: {os.path.abspath(IRRIGATION_MODEL_PATH)}")
logging.info(f"Scikit-learn version: {sklearn.__version__}")

# --- Supported Languages ---
SUPPORTED_LANGUAGES = {
    'en': {'gtts': 'en', 'name': 'English'},
    'ta': {'gtts': 'ta', 'name': 'Tamil'},
    'hi': {'gtts': 'hi', 'name': 'Hindi'},
    'te': {'gtts': 'te', 'name': 'Telugu'},
    'kn': {'gtts': 'kn', 'name': 'Kannada'},
    'ml': {'gtts': 'ml', 'name': 'Malayalam'},
    'mr': {'gtts': 'mr', 'name': 'Marathi'},
    'bn': {'gtts': 'bn', 'name': 'Bengali'},
    'gu': {'gtts': 'gu', 'name': 'Gujarati'},
    'pa': {'gtts': 'pa', 'name': 'Punjabi'}
}

# --- Define CustomEfficientNet ---
class CustomEfficientNet(torch.nn.Module):
    def __init__(self):
        super(CustomEfficientNet, self).__init__()
        base_model = efficientnet_b0(weights=None)
        self.features = base_model.features
        self.pooling = base_model.avgpool
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1280, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

soil_classes = ["Alluvial Soil", "Black Soil", "Clay Soil", "Red Soil"]
irrigation_classes = ["Very low", "Low", "Moderate", "High", "Very high"]

# Expected feature names for crop model
EXPECTED_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# --- Crop Labels ---
crop_labels = {
    "rice": "rice", "wheat": "wheat", "maize": "maize", "sugarcane": "sugarcane",
    "cotton": "cotton", "groundnut": "groundnut", "barley": "barley",
    "millet": "millet", "sorghum": "sorghum", "soybean": "soybean",
    "sunflower": "sunflower", "lentil": "lentil", "chickpea": "chickpea",
    "pea": "pea", "mustard": "mustard", "safflower": "safflower",
    "sesame": "sesame", "jute": "jute", "tobacco": "tobacco", "unknown": "unknown"
}

# --- Heuristic Crop Recommendation ---
def heuristic_crop_recommendation(features, soil_type):
    logging.info("Running heuristic crop recommendation")
    nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall = features
    soil_type = soil_type.lower()

    crop_scores = {
        "rice": 0, "wheat": 0, "maize": 0, "sugarcane": 0, "cotton": 0,
        "groundnut": 0, "barley": 0, "millet": 0, "sorghum": 0, "soybean": 0
    }

    soil_prefs = {
        "alluvial": ["rice", "wheat", "sugarcane"],
        "black": ["cotton", "soybean", "groundnut"],
        "clay": ["rice", "sugarcane"],
        "red": ["groundnut", "millet", "sorghum"]
    }
    for crop in soil_prefs.get(soil_type, []):
        crop_scores[crop] += 2

    if nitrogen > 80:
        for crop in ["rice", "maize", "sugarcane"]:
            crop_scores[crop] += 1
    if phosphorus > 40:
        for crop in ["wheat", "soybean"]:
            crop_scores[crop] += 1
    if potassium > 40:
        for crop in ["cotton", "groundnut"]:
            crop_scores[crop] += 1

    if rainfall > 1000:
        for crop in ["rice", "sugarcane"]:
            crop_scores[crop] += 2
    elif rainfall < 500:
        for crop in ["millet", "sorghum", "barley"]:
            crop_scores[crop] += 2
    if temperature > 25:
        for crop in ["cotton", "maize", "sorghum"]:
            crop_scores[crop] += 1
    if 6.0 <= ph <= 7.5:
        for crop in ["wheat", "soybean", "barley"]:
            crop_scores[crop] += 1

    top_crops = [
        {"crop": crop, "probability": min(0.9, 0.3 + score * 0.1)}
        for crop, score in sorted(crop_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    ]
    logging.info(f"Heuristic top crops: {top_crops}")
    return top_crops

# --- Irrigation Check (Heuristic) ---
def check_irrigation_heuristic(crop, soil_type, features):
    logging.info("Running heuristic irrigation check")
    crop = str(crop).lower()
    soil_type = soil_type.lower()
    nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall = features

    crop_irrigation = {
        "rice": "high", "sugarcane": "high", "maize": "moderate", "wheat": "moderate",
        "cotton": "moderate", "groundnut": "low", "millet": "low", "barley": "low",
        "sorghum": "low", "soybean": "moderate", "sunflower": "moderate", "lentil": "low",
        "chickpea": "low", "pea": "low", "mustard": "low", "safflower": "low",
        "sesame": "low", "jute": "high", "tobacco": "moderate"
    }
    soil_retention = {
        "clay": "high", "alluvial": "moderate", "black": "high", "red": "low"
    }
    crop_need = crop_irrigation.get(crop, "moderate")
    soil_hold = soil_retention.get(soil_type, "moderate")

    irrigation_score = 0
    if rainfall < 50:
        irrigation_score += 2
    elif rainfall < 200:
        irrigation_score += 1
    if humidity < 60:
        irrigation_score += 1
    if temperature > 30:
        irrigation_score += 1
    if crop_need == "high":
        irrigation_score += 2
    elif crop_need == "moderate":
        irrigation_score += 1
    if soil_hold == "low":
        irrigation_score += 1
    elif soil_hold == "high":
        irrigation_score -= 1

    irrigation_levels = {
        4: "Very high irrigation required",
        3: "High irrigation required",
        2: "Moderate irrigation required",
        1: "Low irrigation required",
        0: "Very low irrigation required"
    }
    result = irrigation_levels.get(min(irrigation_score, 4), "Moderate irrigation required")
    logging.info(f"Heuristic irrigation result: {result}")
    return result

# --- Irrigation Check (ML Model) ---
def check_irrigation(crop, soil_type, features):
    try:
        # Load irrigation model on-demand
        logging.info(f"Loading irrigation model from {IRRIGATION_MODEL_PATH}")
        irrigation_model = joblib.load(IRRIGATION_MODEL_PATH)
        logging.info("Irrigation model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading irrigation model: {str(e)}")
        return check_irrigation_heuristic(crop, soil_type, features)

    try:
        soil_type_map = {"alluvial": 0, "black": 1, "clay": 2, "red": 3}
        crop_map = {v: k for k, v in crop_labels.items()}
        soil_type_encoded = soil_type_map.get(soil_type.lower(), 0)
        crop_encoded = crop_map.get(crop.lower(), "unknown")
        input_data = features + [soil_type_encoded, crop_encoded]
        input_df = pd.DataFrame([input_data], columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "soil_type", "crop"])
        prediction = irrigation_model.predict(input_df)[0]
        irrigation_level = irrigation_classes[prediction]
        logging.info(f"ML irrigation prediction: {irrigation_level}")
        return irrigation_level
    except Exception as e:
        logging.error(f"Error using irrigation model: {str(e)}")
        return check_irrigation_heuristic(crop, soil_type, features)

# --- Yield Estimation ---
def estimate_yield(crop, features):
    logging.info("Estimating yield")
    rainfall = features[6]
    base_yield = {
        "rice": 3.5, "wheat": 2.8, "maize": 2.2, "sugarcane": 6.5, "cotton": 1.5,
        "groundnut": 1.2, "barley": 2.0, "millet": 1.8, "sorghum": 2.0, "soybean": 2.5,
        "sunflower": 1.8, "lentil": 1.5, "chickpea": 1.6, "pea": 1.7, "mustard": 1.4,
        "safflower": 1.3, "sesame": 1.2, "jute": 2.0, "tobacco": 2.2
    }
    crop = str(crop).lower()
    yield_value = base_yield.get(crop, 2.0)
    if rainfall > 200:
        yield_value *= 1.2
    elif rainfall < 50:
        yield_value *= 0.8
    result = float(round(yield_value, 2))
    logging.info(f"Estimated yield: {result} tons/ha")
    return result

# --- API Key Handling ---
API_KEY_FILE = os.path.join(BASE_DIR, "api", "API_KEY.py")
API_KEY_ROOT = os.path.join(BASE_DIR, "API_KEY.py")
key = os.getenv("GOOGLE_API_KEY")

if not key:
    try:
        api_key_file = API_KEY_FILE if os.path.exists(API_KEY_FILE) else API_KEY_ROOT
        if not os.path.exists(api_key_file):
            raise FileNotFoundError(f"API key file not found at '{api_key_file}' or '{API_KEY_ROOT}'")
        import importlib.util
        spec = importlib.util.spec_from_file_location("API_KEY", api_key_file)
        api_key_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api_key_module)
        key = api_key_module.key
        if not key or not isinstance(key, str) or "YOUR_API_KEY" in key:
            raise ValueError("Invalid API key format")
        logging.info(f"API key loaded successfully from {api_key_file}")
    except Exception as e:
        logging.error(f"Error loading API key: {str(e)}")
        key = None

if not key:
    logging.warning("No valid API key found")

# --- Initialize Gemini ---
gemini_model = None
gemini_chat = None
model_name_to_use = 'gemini-1.5-flash'

if key:
    try:
        genai.configure(api_key=key)
        gemini_model = genai.GenerativeModel(model_name_to_use)
        gemini_chat = gemini_model.start_chat(history=[])
        logging.info("Gemini API initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing Gemini: {str(e)}")
        gemini_model = None
        gemini_chat = None
else:
    logging.warning("Gemini API not initialized due to missing API key")

# --- Helper Functions ---
def clean_text(text):
    return text.replace('*', '').strip() if text else text

def translate_text(text, dest_lang='en'):
    if not text or dest_lang == 'en':
        return text
    if dest_lang not in SUPPORTED_LANGUAGES:
        logging.warning(f"Unsupported language for translation: {dest_lang}")
        return text
    if not gemini_model:
        logging.error("Gemini model not available for translation")
        return text
    try:
        prompt = f"Translate the following text to {SUPPORTED_LANGUAGES[dest_lang]['name']}: {text}"
        response = gemini_model.generate_content(prompt)
        if response.parts:
            translated_text = clean_text("".join(part.text for part in response.parts))
            logging.debug(f"Translated to {dest_lang}: {translated_text}")
            return translated_text
        else:
            logging.warning(f"No translation parts returned for {dest_lang}")
            return text
    except Exception as e:
        logging.error(f"Translation error ({dest_lang}): {str(e)}")
        return text

def translate_response(data, lang='en'):
    if lang == 'en':
        return data
    try:
        if isinstance(data, dict):
            return {k: translate_response(v, lang) for k, v in data.items()}
        elif isinstance(data, list):
            return [translate_response(i, lang) for i in data]
        elif isinstance(data, str):
            return translate_text(data, lang)
        else:
            return str(data)
    except Exception as e:
        logging.error(f"Translation error in response ({lang}): {str(e)}")
        return data

def translate_to_english(text, source_lang_code):
    if source_lang_code == 'en' or not text:
        return text
    if source_lang_code not in SUPPORTED_LANGUAGES:
        logging.warning(f"Unsupported source language: {source_lang_code}")
        return text
    if not gemini_model:
        logging.error("Gemini model not available for translation")
        return text
    try:
        prompt = f"Translate the following text from {SUPPORTED_LANGUAGES[source_lang_code]['name']} to English: {text}"
        response = gemini_model.generate_content(prompt)
        if response.parts:
            translated_text = clean_text("".join(part.text for part in response.parts))
            logging.debug(f"Translated to English from {source_lang_code}: {translated_text}")
            return translated_text
        else:
            logging.warning(f"No translation parts returned from {source_lang_code}")
            return text
    except Exception as e:
        logging.error(f"Translation error from {source_lang_code}: {e}")
        return text

def get_gemini_response(query_en):
    if not query_en or len(query_en.strip()) < 2:
        return "Please provide a more detailed question."
    
    if gemini_chat is None:
        return "Chat service is currently unavailable due to missing API key."

    full_prompt = (
        "You are an AI assistant. If the user asks about agriculture, farming, crops, soil, weather for farming, "
        "or related topics, especially concerning Tamil Nadu, India, act as an agricultural expert providing detailed, practical advice. "
        "For all other general knowledge queries, provide accurate and concise answers. "
        "Format your response clearly and concisely. If applicable, use a numbered or bulleted list with roughly 4 main points or steps. "
        "Avoid using asterisks (*) for formatting."
        f"User query: {query_en}"
    )

    try:
        response = gemini_chat.send_message(full_prompt, stream=False)
        if not hasattr(response, 'parts') or not response.parts:
            return "Sorry, I couldn't process that request."
        return clean_text("".join(part.text for part in response.parts if hasattr(part, 'text')).strip())
    except Exception as e:
        logging.error(f"Gemini API error: {str(e)}")
        return "Sorry, I encountered an error processing your request."

# --- Routes ---
@app.route('/')
def home():
    return render_template('login.html', languages=SUPPORTED_LANGUAGES)

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/chat')
def chat_page():
    try:
        return send_from_directory(app.static_folder, 'chatindex.html')
    except FileNotFoundError:
        logging.error("chatindex.html not found in static folder")
        return jsonify({"error": "Chat page not found"}), 404

@app.route('/register', methods=['POST'])
def register():
    logging.info("Register request received")
    data = request.get_json()
    full_name = data.get('full_name')
    email = data.get('email')
    password = data.get('password')

    if not full_name or not email or not password:
        return jsonify({'error': 'All fields are required'}), 400

    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    return jsonify({'success': True})

@app.route('/login', methods=['POST'])
def login():
    logging.info("Login request received")
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    try:
        return jsonify({'success': True, 'user': {'full_name': 'Test User', 'email': email}})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_soil', methods=['POST'])
def predict_soil():
    logging.info("Received /predict_soil request")
    if 'image' not in request.files:
        logging.error("No image provided")
        return jsonify(translate_response({"error": "No image provided"}, "en")), 400

    lang = request.form.get("language", "en")
    if lang not in SUPPORTED_LANGUAGES:
        lang = "en"

    try:
        # Load soil model on-demand
        logging.info(f"Loading soil model from {SOIL_MODEL_PATH}")
        soil_model = CustomEfficientNet()
        soil_model.load_state_dict(torch.load(SOIL_MODEL_PATH, map_location=torch.device('cpu')))
        soil_model.eval()
        logging.info("Soil model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading soil model: {str(e)}")
        return jsonify(translate_response({"error": "Soil model not loaded"}, lang)), 500

    try:
        image_file = request.files['image']
        logging.info(f"Processing image: {image_file.filename}")
        img = Image.open(image_file).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = soil_model(img_tensor)
            predicted = torch.argmax(output, 1).item()

        soil_type = soil_classes[predicted]
        pest = random.choice(["None", "Locusts", "Aphids", "Armyworm"])
        pest_detection = [] if pest == "None" else [pest]
        logging.info(f"Prediction: soil_type={soil_type}, pest_detection={pest_detection}")

        response = {
            "success": True,
            "data": {
                "soil_type": soil_type,
                "pest_detection": pest_detection
            }
        }
        return jsonify(translate_response(response, lang))

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify(translate_response({"error": f"Image processing failed: {str(e)}"}, lang)), 400

@app.route('/recommend_crop', methods=['POST'])
def recommend_crop():
    logging.info("Received /recommend_crop request")
    logging.debug(f"Raw request data: {request.data}")
    data = request.get_json()
    logging.debug(f"Parsed JSON data: {data}")
    lang = data.get("lang", "en") if data else "en"
    if lang not in SUPPORTED_LANGUAGES:
        lang = "en"

    if not data:
        logging.error("No data provided in request")
        return jsonify(translate_response({"error": "No data provided"}, lang)), 400

    try:
        # Load crop model on-demand
        logging.info(f"Loading crop model from {CROP_MODEL_PATH}")
        crop_model = joblib.load(CROP_MODEL_PATH)
        if not hasattr(crop_model, 'predict_proba'):
            logging.error("Crop model does not support predict_proba")
            return jsonify(translate_response({"error": "Crop model does not support probability prediction"}, lang)), 500
        if hasattr(crop_model, 'feature_names_in_'):
            logging.info(f"Crop model expected features: {crop_model.feature_names_in_}")
            if not all(f in crop_model.feature_names_in_ for f in EXPECTED_FEATURES):
                logging.error(f"Crop model features mismatch. Expected: {EXPECTED_FEATURES}")
                return jsonify(translate_response({"error": f"Crop model features mismatch. Expected: {EXPECTED_FEATURES}"}, lang)), 500
        if hasattr(crop_model, 'classes_'):
            logging.info(f"Crop model classes: {crop_model.classes_}")
        logging.info("Crop model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading crop model: {str(e)}")
        # Fallback to heuristic if model loading fails
        pass

    try:
        logging.info("Validating input fields")
        required_fields = ["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall", "soil_type"]
        missing_fields = [field for field in required_fields if field not in data or data[field] is None]
        if missing_fields:
            logging.error(f"Missing required fields: {missing_fields}")
            return jsonify(translate_response({"error": f"Missing required fields: {missing_fields}"}, lang)), 400

        features = [
            float(data.get("nitrogen", 0)),
            float(data.get("phosphorus", 0)),
            float(data.get("potassium", 0)),
            float(data.get("temperature", 20)),
            float(data.get("humidity", 50)),
            float(data.get("ph", 6.5)),
            float(data.get("rainfall", 100))
        ]
        
        logging.debug(f"Parsed features: {features}")
        if any(f < 0 for f in features):
            logging.error("Features cannot be negative")
            return jsonify(translate_response({"error": "Features cannot be negative"}, lang)), 400
        if not (0 <= features[5] <= 14):
            logging.error("pH must be between 0 and 14")
            return jsonify(translate_response({"error": "pH must be between 0 and 14"}, lang)), 400
        if features[4] > 100:
            logging.error("Humidity cannot exceed 100%")
            return jsonify(translate_response({"error": "Humidity cannot exceed 100%"}, lang)), 400
        if features[6] > 5000:
            logging.error("Rainfall cannot exceed 5000 mm")
            return jsonify(translate_response({"error": "Rainfall cannot exceed 5000 mm"}, lang)), 400

        soil_type = str(data.get("soil_type", "Alluvial")).capitalize()
        logging.debug(f"Soil type: {soil_type}")
        valid_soils = ["Alluvial", "Black", "Clay", "Red"]
        if soil_type not in valid_soils:
            logging.error(f"Invalid soil type: {soil_type}")
            return jsonify(translate_response({"error": "Invalid soil type. Must be Alluvial, Black, Clay, or Red."}, lang)), 400

        logging.info("Generating heuristic recommendation")
        top_crops = heuristic_crop_recommendation(features, soil_type)
        top_crop = top_crops[0]["crop"] if top_crops else "unknown"
        irrigation_status = check_irrigation(top_crop, soil_type, features)
        estimated_yield = estimate_yield(top_crop, features)
        response = {
            "crop": top_crop,
            "crops": top_crops,
            "irrigation": irrigation_status,
            "estimated_yield": f"{estimated_yield} tons/ha",
            "note": "Using heuristic recommendation"
        }
        logging.info(f"Response: {response}")
        return jsonify(translate_response(response, lang))

    except Exception as e:
        logging.error(f"Error processing crop recommendation: {str(e)}")
        return jsonify(translate_response({"error": f"Error processing crop recommendation: {str(e)}"}, lang)), 500

@app.route('/government_aids', methods=['POST'])
def government_aids():
    logging.info("Received /government_aids request")
    data = request.get_json()
    logging.debug(f"Request data: {data}")
    lang = data.get("lang", "en") if data else "en"
    if lang not in SUPPORTED_LANGUAGES:
        lang = "en"

    if not data:
        logging.error("No data provided in request")
        return jsonify(translate_response({"error": "No data provided"}, lang)), 400

    try:
        required_fields = ["state", "land_size"]
        missing_fields = [field for field in required_fields if field not in data or data[field] is None]
        if missing_fields:
            logging.error(f"Missing required fields: {missing_fields}")
            return jsonify(translate_response({"error": f"Missing required fields: {missing_fields}"}, lang)), 400

        state = str(data.get("state", "")).strip().lower()
        land_size = float(data.get("land_size", 0))
        logging.debug(f"State: {state}, Land Size: {land_size}")

        if state not in all_states:
            logging.error("Invalid state provided")
            return jsonify(translate_response({"error": "Invalid state provided"}, lang)), 400
        if land_size < 0:
            logging.error("Land size cannot be negative")
            return jsonify(translate_response({"error": "Land size cannot be negative"}, lang)), 400

        for s in all_states:
            if s["state"].lower() == state:
                eligibility_msg = f"{s['eligibility']} You have {land_size} acres of land."
                response = {
                    "success": True,
                    "data": {
                        "state": state.title(),
                        "land_size": float(land_size),
                        "available_schemes": s["available_schemes"],
                        "eligibility": eligibility_msg,
                        "contact": s["contact"]
                    }
                }
                logging.info(f"Government aids response: {response}")
                return jsonify(translate_response(response, lang))

        response = {
            "success": True,
            "data": {
                "state": state.title(),
                "land_size": float(land_size),
                "available_schemes": ["No official schemes found for your state"],
                "eligibility": "Unknown",
                "contact": "Visit your local agriculture office for accurate details"
            }
        }
        logging.info(f"Government aids response: {response}")
        return jsonify(translate_response(response, lang))

    except Exception as e:
        logging.error(f"Error processing government aids: {str(e)}")
        return jsonify(translate_response({"error": str(e)}, lang)), 500

@app.route('/api/grok', methods=['POST'])
def chat():
    logging.info("Received /api/grok request")
    data = request.json
    if not data:
        logging.error("No data provided in request")
        return jsonify(translate_response({'error': 'No data provided'}, "en")), 400

    message = data.get('message', '').strip()
    lang = data.get('language', 'en')
    
    if not message:
        logging.error("Empty message received")
        return jsonify(translate_response({'error': 'Empty message'}, lang)), 400
    
    if lang not in SUPPORTED_LANGUAGES:
        logging.warning(f"Unsupported language: {lang}, defaulting to 'en'")
        lang = 'en'
    
    query_en = translate_to_english(message, lang)
    response_en = get_gemini_response(query_en)
    response = translate_text(response_en, lang)
    
    logging.info(f"Chat response: {response}")
    return jsonify({
        'response': response,
        'language': lang
    })

@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    logging.info("Received /api/tts request")
    data = request.json
    text = data.get('text', '').strip()
    lang = data.get('language', 'en')
    
    if not text:
        logging.error("Empty text received")
        return jsonify(translate_response({'error': 'Empty text'}, "en")), 400
    
    if lang not in SUPPORTED_LANGUAGES:
        logging.warning(f"Unsupported language: {lang}, defaulting to 'en'")
        lang = 'en'
    
    gtts_lang = SUPPORTED_LANGUAGES[lang]['gtts']
    
    audio_dir = os.path.join(app.root_path, 'audio_output')
    os.makedirs(audio_dir, exist_ok=True)
    
    timestamp = int(time.time())
    filename = f"tts_{lang}_{timestamp}.mp3"
    filepath = os.path.join(audio_dir, filename)
    
    try:
        logging.info(f"Generating TTS with text: {text}, language: {gtts_lang}")
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        tts.save(filepath)
        logging.info(f"Audio saved to: {filepath}")
        return jsonify({
            'audio_url': f'/audio/{filename}',
            'filename': filename
        })
    except Exception as e:
        logging.error(f"TTS error: {str(e)}")
        return jsonify(translate_response({'error': f'Failed to generate speech: {str(e)}'}, "en")), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    audio_dir = os.path.join(app.root_path, 'audio_output')
    try:
        return send_from_directory(audio_dir, filename)
    except FileNotFoundError:
        logging.error(f"Audio file not found: {filename}")
        return jsonify(translate_response({'error': 'Audio file not found'}, "en")), 404

@app.route('/translations.json')
def serve_translations():
    try:
        return send_from_directory(app.static_folder, 'translations.json')
    except Exception as e:
        logging.error(f"Error serving translations: {str(e)}")
        return jsonify(translate_response({'error': 'Translations not found'}, "en")), 404

@app.route('/health', methods=['GET'])
def health():
    status = {
        'api': 'running',
        'soil_model': os.path.exists(SOIL_MODEL_PATH),
        'crop_model': os.path.exists(CROP_MODEL_PATH),
        'irrigation_model': os.path.exists(IRRIGATION_MODEL_PATH),
        'gemini': gemini_chat is not None,
        'api_key': key is not None
    }
    return jsonify(status)

@app.route('/model_info', methods=['GET'])
def model_info():
    crop_model_info = {
        'loaded': False,
        'path': CROP_MODEL_PATH,
        'exists': os.path.exists(CROP_MODEL_PATH),
        'predict_proba': False,
        'features': [],
        'classes': []
    }
    
    try:
        crop_model = joblib.load(CROP_MODEL_PATH)
        crop_model_info.update({
            'loaded': True,
            'predict_proba': hasattr(crop_model, 'predict_proba'),
            'features': getattr(crop_model, 'feature_names_in_', []),  # Use empty list if attribute not present
            'classes': getattr(crop_model, 'classes_', []).tolist()  # Convert to list
        })
    except Exception as e:
        logging.error(f"Error loading crop model for info: {str(e)}")
    
    return jsonify({
        'crop_model': crop_model_info,
        'sklearn_version': sklearn.__version__
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)