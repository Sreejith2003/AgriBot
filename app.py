# merged_app.py
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import joblib
import pandas as pd
from googletrans import Translator
import random
import numpy as np
import mysql.connector
import bcrypt
from gtts import gTTS
import time
import google.generativeai as genai
import sys

# Initialize Flask app
app = Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Allow CORS for chat API

# --- MySQL Configuration ---
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Sathish2003$', 
    'database': 'agribot_db'
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

# --- Initialize Translator ---
translator = Translator(timeout=10)

# --- Model Paths ---
SOIL_MODEL_PATH = "models/efficientnet_soil.pth"
CROP_MODEL_PATH = "models/crop_model.pkl"
IRRIGATION_MODEL_PATH = "models/irrigation_model.pkl"

# --- Supported Languages ---
SUPPORTED_LANGUAGES = {
    'en': 'English', 'hi': 'Hindi', 'te': 'Telugu', 'ta': 'Tamil',
    'kn': 'Kannada', 'ml': 'Malayalam', 'mr': 'Marathi', 'bn': 'Bengali',
    'gu': 'Gujarati', 'pa': 'Punjabi'
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

# --- Load Models ---
soil_model = None
crop_model = None
try:
    soil_model = CustomEfficientNet()
    soil_model.load_state_dict(torch.load(SOIL_MODEL_PATH, map_location=torch.device('cpu')))
    soil_model.eval()
    crop_model = joblib.load(CROP_MODEL_PATH)
    if hasattr(crop_model, 'classes_'):
        print(f"Crop model classes: {crop_model.classes_}")
    else:
        print("Warning: Crop model has no classes_ attribute")
except Exception as e:
    print(f"Error loading models: {e}")

# --- Fallback Crop Labels ---
crop_labels = {
    0: "barley", 1: "cotton", 2: "groundnut", 3: "maize", 4: "millet",
    5: "rice", 6: "sugarcane", 7: "wheat", 8: "sorghum", 9: "soybean",
    10: "sunflower", 11: "lentil", 12: "chickpea", 13: "pea", 14: "mustard",
    15: "safflower", 16: "sesame", 17: "jute", 18: "tobacco", 19: "sugarcane",
    20: "rice", 21: "wheat"
}

# --- Translation Functions ---
def translate_text(text, dest_lang='en'):
    if not text or dest_lang == 'en':
        return text
    try:
        return translator.translate(str(text), dest=dest_lang).text
    except Exception as e:
        print(f"Translation error ({dest_lang}): {str(e)}")
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
        print(f"Translation error in response ({lang}): {str(e)}")
        return data

# --- Irrigation Check ---
def check_irrigation(crop, soil_type, features):
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

    if irrigation_score >= 4:
        return "Very high irrigation required"
    elif irrigation_score == 3:
        return "High irrigation required"
    elif irrigation_score == 2:
        return "Moderate irrigation required"
    elif irrigation_score == 1:
        return "Low irrigation required"
    else:
        return "Very low irrigation required"

# --- Yield Estimation ---
def estimate_yield(crop, features):
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
    return round(yield_value, 2)

# --- API Key Handling for Gemini ---
API_KEY_FILE = "API_KEY.py"
api_key = None

try:
    if not os.path.exists(API_KEY_FILE):
        raise FileNotFoundError(f"API key file '{API_KEY_FILE}' not found")
    from API_KEY import key as imported_key
    api_key = imported_key
    if not api_key or not isinstance(api_key, str) or "YOUR_API_KEY" in api_key:
        raise ValueError("Invalid API key format")
except Exception as e:
    print(f"Error loading API key: {e}")
    sys.exit(1)

# --- Initialize Gemini ---
gemini_model = None
gemini_chat = None
model_name_to_use = 'gemini-1.5-flash'

try:
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model_name_to_use)
    gemini_chat = gemini_model.start_chat(history=[])
except Exception as e:
    print(f"Error initializing Gemini: {e}")
    sys.exit(1)

# --- Helper Functions for Chatbot ---
def clean_text(text):
    return text.replace('*', '').strip() if text else text

def translate_to_language(text, lang_code):
    if lang_code == 'en' or not text:
        return text
    try:
        return translator.translate(str(text), dest=lang_code).text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def translate_to_english(text, source_lang_code):
    if source_lang_code == 'en' or not text:
        return text
    try:
        return translator.translate(str(text), dest='en').text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def get_gemini_response(query_en):
    if not query_en or len(query_en.strip()) < 2:
        return "Please provide a more detailed question."
    
    if gemini_chat is None:
        return "Chat service is currently unavailable."

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
        print(f"Gemini API error: {e}")
        return "Sorry, I encountered an error processing your request."

# --- Routes from app.py ---
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    full_name = data.get('full_name')
    email = data.get('email')
    password = data.get('password')

    if not full_name or not email or not password:
        return jsonify({'error': 'All fields are required'}), 400

    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        if cursor.fetchone():
            return jsonify({'error': 'Email already registered'}), 400

        cursor.execute('INSERT INTO users (full_name, email, password) VALUES (%s, %s, %s)',
                       (full_name, email, hashed_pw))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            return jsonify({'success': True, 'user': {'full_name': user['full_name'], 'email': user['email']}})
        else:
            return jsonify({'error': 'Invalid email or password'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return render_template('login.html', languages=SUPPORTED_LANGUAGES)

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict_soil', methods=['POST'])
def predict_soil():
    print("Received /predict_soil request")
    if 'image' not in request.files:
        print("Error: No image provided")
        return jsonify({"error": "No image provided"}), 400

    lang = request.form.get("language", "en")
    if lang not in SUPPORTED_LANGUAGES:
        lang = "en"

    try:
        image_file = request.files['image']
        print(f"Processing image: {image_file.filename}")
        img = Image.open(image_file).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = soil_model(img_tensor)
            predicted = torch.argmax(output, 1).item()

        soil_type = soil_classes[predicted]
        pest_detection = random.choice(["None", "Locusts", "Aphids", "Armyworm"])
        print(f"Prediction: soil_type={soil_type}, pest_detection={pest_detection}")

        response = {
            "success": True,
            "data": {
                "soil_type": soil_type,
                "pest_detection": pest_detection
            }
        }
        return jsonify(translate_response(response, lang))

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify(translate_response({"error": f"Image processing failed: {str(e)}"}, lang)), 400

@app.route('/recommend_crop', methods=['POST'])
def recommend_crop():
    print("Received /recommend_crop request")
    data = request.get_json()
    print(f"Request data: {data}")
    lang = data.get("lang", "en")
    try:
        features = [
            float(data.get("nitrogen", 0)),
            float(data.get("phosphorus", 0)),
            float(data.get("potassium", 0)),
            float(data.get("temperature", 0)),
            float(data.get("humidity", 0)),
            float(data.get("ph", 0)),
            float(data.get("rainfall", 0))
        ]
        soil_type = data.get("soil_type", "").capitalize()
        print(f"Features: {features}, Soil Type: {soil_type}")
        valid_soils = ["Alluvial", "Black", "Clay", "Red"]
        if soil_type not in valid_soils:
            print(f"Error: Invalid soil type {soil_type}")
            return jsonify(translate_response({"error": "Invalid soil type. Must be Alluvial, Black, Clay, or Red."}, lang)), 400

        feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        input_df = pd.DataFrame([features], columns=feature_names)
        
        if not hasattr(crop_model, 'predict_proba'):
            print("Error: Model does not support predict_proba")
            return jsonify(translate_response({"error": "Model does not support probability prediction"}, lang)), 400

        probs = crop_model.predict_proba(input_df)[0]
        print(f"Probabilities: {probs}")
        
        print("Using fallback crop labels")
        classes = [crop_labels.get(i, f"crop_{i}") for i in range(len(probs))]
        print(f"Classes: {classes}")
        
        if len(probs) != len(classes):
            print(f"Error: Mismatch between probs ({len(probs)}) and classes ({len(classes)})")
            return jsonify(translate_response({"error": "Model classes and probabilities mismatch"}, lang)), 400

        indices = np.argsort(probs)[::-1][:5]
        print(f"Top indices: {indices}")
        top_crops = [
            {
                "crop": crop_labels.get(idx, classes[idx]),
                "probability": round(float(probs[idx]), 3)
            }
            for idx in indices
        ]
        print(f"Top crops: {top_crops}")

        top_crop = top_crops[0]["crop"] if top_crops else "unknown"
        irrigation_status = check_irrigation(top_crop, soil_type, features)
        estimated_yield = estimate_yield(top_crop, features)
        print(f"Top crop: {top_crop}, irrigation={irrigation_status}, yield={estimated_yield}")

        response = {
            "crop": top_crop,
            "crops": top_crops,
            "irrigation": irrigation_status,
            "estimated_yield": f"{estimated_yield} tons/ha"
        }
        print(f"Response: {response}")
        return jsonify(translate_response(response, lang))

    except Exception as e:
        print(f"Error processing crop recommendation: {str(e)}")
        return jsonify(translate_response({"error": str(e)}, lang)), 400

@app.route('/government_aids', methods=['POST'])
def government_aids():
    print("Received /government_aids request")
    data = request.get_json()
    print(f"Request data: {data}")
    lang = data.get("lang", "en")
    try:
        state = data.get("state", "").strip().lower()
        land_size = float(data.get("land_size", 0))
        print(f"State: {state}, Land Size: {land_size}")
        if not state:
            print("Error: State name is required")
            return jsonify(translate_response({"error": "State name is required."}, lang)), 400
        if lang not in SUPPORTED_LANGUAGES:
            lang = "en"

        all_states = [
            {
                "state": "andhra pradesh",
                "available_schemes": [
                    "Rythu Bharosa - Financial assistance to farmers",
                    "Andhra Pradesh Micro Irrigation Project - Subsidy for irrigation systems",
                    "Pradhan Mantri Fasal Bima Yojana - Crop insurance"
                ],
                "eligibility": "Farmers with land records in Andhra Pradesh are eligible, subject to scheme criteria.",
                "contact": "Visit the nearest Mandal Agricultural Office or Andhra Pradesh Department of Agriculture website."
            },
            {
                "state": "arunachal pradesh",
                "available_schemes": [
                    "Organic Farming Promotion - Support for organic cultivation",
                    "Pradhan Mantri Kisan Samman Nidhi - Income support",
                    "National Food Security Mission - Boost food grain production"
                ],
                "eligibility": "Registered farmers in Arunachal Pradesh are eligible based on landholding and scheme rules.",
                "contact": "Contact the Directorate of Agriculture, Arunachal Pradesh."
            },
            {
                "state": "assam",
                "available_schemes": [
                    "Assam Agri-Business Incentive Scheme - Support for agri-entrepreneurship",
                    "Pradhan Mantri Krishi Sinchayee Yojana - Irrigation support",
                    "Soil Health Card Scheme - Soil nutrient management"
                ],
                "eligibility": "Farmers with valid land records in Assam are eligible.",
                "contact": "Visit the Assam Agriculture Department or local Krishi Vigyan Kendra."
            },
            {
                "state": "bihar",
                "available_schemes": [
                    "Krishi Input Subsidy - Support for inputs",
                    "Pradhan Mantri Fasal Bima Yojana - Crop insurance",
                    "National Mission on Sustainable Agriculture - Climate resilience"
                ],
                "eligibility": "Registered farmers in Bihar with land up to 2 hectares are eligible.",
                "contact": "Contact the Bihar Agriculture Department or district offices."
            },
            {
                "state": "chhattisgarh",
                "available_schemes": [
                    "Rajiv Gandhi Kisan Nyay Yojana - Financial aid",
                    "Chhattisgarh Micro Irrigation Project - Irrigation subsidies",
                    "Paramparagat Krishi Vikas Yojana - Organic farming support"
                ],
                "eligibility": "Farmers with land records in Chhattisgarh are eligible.",
                "contact": "Visit the Chhattisgarh Agriculture Department website or local offices."
            },
            {
                "state": "goa",
                "available_schemes": [
                    "Goa State Organic Farming Scheme - Promotion of organic practices",
                    "Pradhan Mantri Kisan Maan-Dhan Yojana - Pension for farmers",
                    "National Agriculture Market (eNAM) - Market linkage"
                ],
                "eligibility": "Registered farmers in Goa are eligible based on age and landholding.",
                "contact": "Contact the Goa Directorate of Agriculture."
            },
            {
                "state": "gujarat",
                "available_schemes": [
                    "Krishi Mahotsav - Farmer training and input support",
                    "Gujarat Green Revolution Scheme - Productivity enhancement",
                    "Pradhan Mantri Fasal Bima Yojana - Crop insurance"
                ],
                "eligibility": "Farmers with land in Gujarat are eligible, subject to scheme terms.",
                "contact": "Visit the Gujarat Agriculture Department or local taluka offices."
            },
            {
                "state": "haryana",
                "available_schemes": [
                    "Meri Fasal Mera Byora - Crop registration and support",
                    "Haryana Agri-Business Promotion Scheme - Market support",
                    "Pradhan Mantri Krishi Sinchayee Yojana - Irrigation"
                ],
                "eligibility": "Registered farmers in Haryana with land records are eligible.",
                "contact": "Contact the Haryana Agriculture Department or Krishi Vigyan Kendra."
            },
            {
                "state": "himachal pradesh",
                "available_schemes": [
                    "Himachal Pradesh Horticulture Development Scheme - Fruit and vegetable support",
                    "Pradhan Mantri Kisan Samman Nidhi - Income support",
                    "National Mission for Sustainable Agriculture - Climate resilience"
                ],
                "eligibility": "Farmers in Himachal Pradesh with valid land records are eligible.",
                "contact": "Visit the Himachal Pradesh Agriculture Department."
            },
            {
                "state": "jammu and kashmir",
                "available_schemes": [
                    "JK Farmer Welfare Scheme - Financial assistance",
                    "Pradhan Mantri Fasal Bima Yojana - Crop insurance",
                    "Holistic Agriculture Development Programme - Infrastructure support"
                ],
                "eligibility": "Registered farmers in Jammu and Kashmir are eligible.",
                "contact": "Contact the Jammu and Kashmir Agriculture Department."
            },
            {
                "state": "jharkhand",
                "available_schemes": [
                    "Jharkhand Mukhyamantri Krishi Ashirwad Yojana - Input subsidy",
                    "Pradhan Mantri Krishi Sinchayee Yojana - Irrigation",
                    "Soil Health Card Scheme - Soil management"
                ],
                "eligibility": "Farmers with land in Jharkhand are eligible based on scheme criteria.",
                "contact": "Visit the Jharkhand Agriculture Department or local offices."
            },
            {
                "state": "karnataka",
                "available_schemes": [
                    "Raitha Vidya Nidhi - Education support for farmers' children",
                    "Kisan Samman Scheme - Financial aid",
                    "Pradhan Mantri Fasal Bima Yojana - Crop insurance"
                ],
                "eligibility": "Registered farmers in Karnataka with land records are eligible.",
                "contact": "Contact the Karnataka Agriculture Department or local Krishi Kendra."
            },
            {
                "state": "kerala",
                "available_schemes": [
                    "Karshaka Pension Scheme - Pension for elderly farmers",
                    "Subhiksha Keralam - Promotion of self-sufficiency in food production",
                    "Pradhan Mantri Kisan Samman Nidhi - Income support"
                ],
                "eligibility": "Registered farmers in Kerala are eligible for these schemes.",
                "contact": "Visit the Krishi Bhavan in your locality in Kerala."
            },
            {
                "state": "madhya pradesh",
                "available_schemes": [
                    "Mukhyamantri Kisan Kalyan Yojana - Financial assistance",
                    "Madhya Pradesh Micro Irrigation Scheme - Irrigation support",
                    "Pradhan Mantri Fasal Bima Yojana - Crop insurance"
                ],
                "eligibility": "Farmers with land in Madhya Pradesh are eligible.",
                "contact": "Contact the Madhya Pradesh Agriculture Department."
            },
            {
                "state": "maharashtra",
                "available_schemes": [
                    "Mahatma Phule Krishi Vikas Yojana - Farmer welfare",
                    "Maharashtra Agri-Input Subsidy - Support for inputs",
                    "Pradhan Mantri Krishi Sinchayee Yojana - Irrigation"
                ],
                "eligibility": "Registered farmers in Maharashtra with land records are eligible.",
                "contact": "Visit the Maharashtra Agriculture Department or local offices."
            },
            {
                "state": "manipur",
                "available_schemes": [
                    "Manipur Organic Farming Scheme - Organic support",
                    "Pradhan Mantri Kisan Samman Nidhi - Income support",
                    "National Mission on Agricultural Extension - Training"
                ],
                "eligibility": "Farmers in Manipur with valid land records are eligible.",
                "contact": "Contact the Manipur Agriculture Department."
            },
            {
                "state": "meghalaya",
                "available_schemes": [
                    "Meghalaya Integrated Basin Development Scheme - Water management",
                    "Pradhan Mantri Fasal Bima Yojana - Crop insurance",
                    "Paramparagat Krishi Vikas Yojana - Organic farming"
                ],
                "eligibility": "Registered farmers in Meghalaya are eligible based on scheme rules.",
                "contact": "Visit the Meghalaya Agriculture Department."
            },
            {
                "state": "mizoram",
                "available_schemes": [
                    "Mizoram Agri-Horti Scheme - Horticulture support",
                    "Pradhan Mantri Kisan Maan-Dhan Yojana - Pension",
                    "National Food Security Mission - Productivity enhancement"
                ],
                "eligibility": "Farmers with land in Mizoram are eligible.",
                "contact": "Contact the Mizoram Agriculture Department."
            },
            {
                "state": "nagaland",
                "available_schemes": [
                    "Nagaland Organic Mission - Organic farming promotion",
                    "Pradhan Mantri Krishi Sinchayee Yojana - Irrigation",
                    "Soil Health Card Scheme - Soil health management"
                ],
                "eligibility": "Registered farmers in Nagaland are eligible.",
                "contact": "Visit the Nagaland Agriculture Department."
            },
            {
                "state": "odisha",
                "available_schemes": [
                    "Krushak Assistance for Livelihood and Income Augmentation - Financial aid",
                    "Odisha Micro Irrigation Project - Irrigation subsidies",
                    "Pradhan Mantri Fasal Bima Yojana - Crop insurance"
                ],
                "eligibility": "Farmers with land records in Odisha are eligible.",
                "contact": "Contact the Odisha Agriculture Department or local offices."
            },
            {
                "state": "punjab",
                "available_schemes": [
                    "Punjab Agri-Infrastructure Fund - Storage and marketing",
                    "Pradhan Mantri Kisan Samman Nidhi - Income support",
                    "Crop Diversification Scheme - Support for diversification"
                ],
                "eligibility": "Registered farmers in Punjab with land up to 2 hectares are eligible.",
                "contact": "Visit the Punjab Agriculture Department."
            },
            {
                "state": "rajasthan",
                "available_schemes": [
                    "Mukhyamantri Krishak Sathi Yojana - Farmer welfare",
                    "Rajasthan Micro Irrigation Scheme - Irrigation support",
                    "Pradhan Mantri Fasal Bima Yojana - Crop insurance"
                ],
                "eligibility": "Farmers in Rajasthan with valid land records are eligible.",
                "contact": "Contact the Rajasthan Agriculture Department."
            },
            {
                "state": "sikkim",
                "available_schemes": [
                    "Sikkim Organic Mission - Full organic state initiative",
                    "Pradhan Mantri Kisan Maan-Dhan Yojana - Pension",
                    "National Mission on Sustainable Agriculture - Climate resilience"
                ],
                "eligibility": "Registered farmers in Sikkim are eligible.",
                "contact": "Visit the Sikkim Agriculture Department."
            },
            {
                "state": "tamil nadu",
                "available_schemes": [
                    "Uzhavar Aluvalar Thittam - Training and capacity building for farmers",
                    "Tamil Nadu Farmers' Insurance Scheme - Free crop insurance for farmers",
                    "Micro Irrigation Scheme - Subsidy for drip and sprinkler systems"
                ],
                "eligibility": "Registered farmers in Tamil Nadu are eligible based on crop and scheme-specific criteria.",
                "contact": "Contact your local Agricultural Extension Centre or visit the Tamil Nadu Department of Agriculture website."
            },
            {
                "state": "telangana",
                "available_schemes": [
                    "Rythu Bandhu - Investment support for farmers",
                    "Rythu Bhima - Crop insurance",
                    "Telangana Micro Irrigation Project - Irrigation subsidies"
                ],
                "eligibility": "Farmers with land records in Telangana are eligible.",
                "contact": "Visit the Telangana Agriculture Department or local Rythu Vedikas."
            },
            {
                "state": "tripura",
                "available_schemes": [
                    "Tripura Agri-Development Scheme - Infrastructure support",
                    "Pradhan Mantri Kisan Samman Nidhi - Income support",
                    "Paramparagat Krishi Vikas Yojana - Organic farming"
                ],
                "eligibility": "Registered farmers in Tripura are eligible based on landholding.",
                "contact": "Contact the Tripura Agriculture Department."
            },
            {
                "state": "uttar pradesh",
                "available_schemes": [
                    "Kisan Samman Nidhi UP - Financial assistance",
                    "UP Agri-Infrastructure Scheme - Storage and marketing",
                    "Pradhan Mantri Fasal Bima Yojana - Crop insurance"
                ],
                "eligibility": "Farmers with land in Uttar Pradesh are eligible.",
                "contact": "Visit the Uttar Pradesh Agriculture Department or local offices."
            },
            {
                "state": "uttarakhand",
                "available_schemes": [
                    "Uttarakhand Organic Farming Scheme - Organic support",
                    "Pradhan Mantri Krishi Sinchayee Yojana - Irrigation",
                    "National Mission for Sustainable Agriculture - Climate resilience"
                ],
                "eligibility": "Registered farmers in Uttarakhand are eligible.",
                "contact": "Contact the Uttarakhand Agriculture Department."
            },
            {
                "state": "west bengal",
                "available_schemes": [
                    "Krishak Bandhu - Financial and insurance support",
                    "West Bengal Micro Irrigation Scheme - Irrigation subsidies",
                    "Pradhan Mantri Kisan Samman Nidhi - Income support"
                ],
                "eligibility": "Farmers with land records in West Bengal are eligible.",
                "contact": "Visit the West Bengal Agriculture Department or local Krishi Sahayak Kendra."
            },
            {
                "state": "andaman and nicobar islands",
                "available_schemes": [
                    "Island Agriculture Development Scheme - Support for crops",
                    "Pradhan Mantri Fasal Bima Yojana - Crop insurance",
                    "Paramparagat Krishi Vikas Yojana - Organic farming"
                ],
                "eligibility": "Registered farmers in Andaman and Nicobar Islands are eligible.",
                "contact": "Contact the Andaman and Nicobar Agriculture Department."
            },
            {
                "state": "chandigarh",
                "available_schemes": [
                    "Chandigarh Farmer Support Scheme - Input subsidies",
                    "Pradhan Mantri Kisan Samman Nidhi - Income support",
                    "Soil Health Card Scheme - Soil management"
                ],
                "eligibility": "Farmers with land in Chandigarh are eligible.",
                "contact": "Visit the Chandigarh Agriculture Office."
            },
            {
                "state": "dadra and nagar haveli and daman and diu",
                "available_schemes": [
                    "UT Farmer Welfare Scheme - Financial assistance",
                    "Pradhan Mantri Krishi Sinchayee Yojana - Irrigation",
                    "National Agriculture Market (eNAM) - Market linkage"
                ],
                "eligibility": "Registered farmers in Dadra and Nagar Haveli and Daman and Diu are eligible.",
                "contact": "Contact the local Agriculture Office."
            },
            {
                "state": "delhi",
                "available_schemes": [
                    "Delhi Farmer Support Program - Input subsidies",
                    "Pradhan Mantri Fasal Bima Yojana - Crop insurance",
                    "Pradhan Mantri Kisan Maan-Dhan Yojana - Pension"
                ],
                "eligibility": "Farmers with land records in Delhi are eligible.",
                "contact": "Visit the Delhi Agriculture Department."
            },
            {
                "state": "lakshadweep",
                "available_schemes": [
                    "Lakshadweep Agri-Development Scheme - Support for crops",
                    "Pradhan Mantri Kisan Samman Nidhi - Income support",
                    "National Mission on Agricultural Extension - Training"
                ],
                "eligibility": "Registered farmers in Lakshadweep are eligible.",
                "contact": "Contact the Lakshadweep Agriculture Department."
            },
            {
                "state": "ladakh",
                "available_schemes": [
                    "Ladakh Organic Farming Initiative - Organic support",
                    "Pradhan Mantri Krishi Sinchayee Yojana - Irrigation",
                    "Soil Health Card Scheme - Soil management"
                ],
                "eligibility": "Farmers in Ladakh with valid land records are eligible.",
                "contact": "Visit the Ladakh Agriculture Department."
            },
            {
                "state": "puducherry",
                "available_schemes": [
                    "Puducherry Farmer Welfare Scheme - Financial aid",
                    "Pradhan Mantri Fasal Bima Yojana - Crop insurance",
                    "Paramparagat Krishi Vikas Yojana - Organic farming"
                ],
                "eligibility": "Registered farmers in Puducherry are eligible.",
                "contact": "Contact the Puducherry Agriculture Department."
            }
        ]

        for s in all_states:
            if s["state"].lower() == state:
                eligibility_msg = f"{s['eligibility']} You have {land_size} acres of land."
                response = {
                    "success": True,
                    "data": {
                        "state": state.title(),
                        "land_size": land_size,
                        "available_schemes": s["available_schemes"],
                        "eligibility": eligibility_msg,
                        "contact": s["contact"]
                    }
                }
                print(f"Response: {response}")
                return jsonify(translate_response(response, lang))

        response = {
            "success": True,
            "data": {
                "state": state.title(),
                "land_size": land_size,
                "available_schemes": ["No official schemes found for your state."],
                "eligibility": "Unknown",
                "contact": "Visit your local agriculture office for accurate details."
            }
        }
        print(f"Response: {response}")
        return jsonify(translate_response(response, lang))

    except Exception as e:
        print(f"Error processing government aids: {str(e)}")
        return jsonify(translate_response({"error": str(e)}, lang)), 400

# --- Routes from chatapp.py ---
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '').strip()
    lang = data.get('language', 'en')
    
    if not message:
        return jsonify({'error': 'Empty message'}), 400
    
    query_en = translate_to_english(message, lang)
    response_en = get_gemini_response(query_en)
    response = translate_to_language(response_en, lang)
    
    return jsonify({
        'response': response,
        'language': lang
    })

@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    data = request.json
    text = data.get('text', '').strip()
    lang = data.get('language', 'en')
    
    if not text:
        return jsonify({'error': 'Empty text'}), 400
    
    audio_dir = os.path.join(app.root_path, 'audio_output')
    os.makedirs(audio_dir, exist_ok=True)
    
    timestamp = int(time.time())
    filename = f"tts_{lang}_{timestamp}.mp3"
    filepath = os.path.join(audio_dir, filename)
    
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(filepath)
        print(f"Audio saved to: {filepath}")
        return jsonify({
            'audio_url': f'/audio/{filename}',
            'filename': filename
        })
    except Exception as e:
        print(f"TTS error: {e}")
        return jsonify({'error': 'Failed to generate speech'}), 500
    
    
@app.route('/translations.json')
def serve_translations():
    try:
        return send_from_directory('static', 'translations.json')
    except Exception as e:
        print(f"Error serving translations: {str(e)}")
        return jsonify({"error": "Translations not found"}), 404

@app.route('/audio/<filename>')
def serve_audio(filename):
    audio_dir = os.path.join(app.root_path, 'audio_output')
    try:
        return send_from_directory(audio_dir, filename)
    except FileNotFoundError:
        print(f"Audio file not found: {filename}")
        return jsonify({'error': 'Audio file not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)