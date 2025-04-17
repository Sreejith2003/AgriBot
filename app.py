from flask import Flask, request, jsonify, render_template
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

app = Flask(__name__)

# Initialize translator
translator = Translator(timeout=10)

# Model paths
SOIL_MODEL_PATH = "AgriBot-main/model/efficientnet_soil (1).pth"
CROP_MODEL_PATH = "D:\AgriBot-main\AgriBot-main\model\crop_model (2) (1).pkl"
IRRIGATION_MODEL_PATH = "D:\AgriBot-main\AgriBot-main\model\irrigation_model (1).pkl"

# Supported languages
SUPPORTED_LANGUAGES = {
    'en': 'English', 'hi': 'Hindi', 'te': 'Telugu', 'ta': 'Tamil',
    'kn': 'Kannada', 'ml': 'Malayalam', 'mr': 'Marathi', 'bn': 'Bengali',
    'gu': 'Gujarati', 'pa': 'Punjabi'
}

# Define CustomEfficientNet
class CustomEfficientNet(torch.nn.Module):
    def _init_(self):
        super(CustomEfficientNet, self)._init_()
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

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

soil_classes = ["Alluvial Soil", "Black Soil", "Clay Soil", "Red Soil"]

# Load models
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

# Fallback crop labels
crop_labels = {
    0: "barley",
    1: "cotton",
    2: "groundnut",
    3: "maize",
    4: "millet",
    5: "rice",
    6: "sugarcane",
    7: "wheat",
    8: "sorghum",
    9: "soybean",
    10: "sunflower",
    11: "lentil",
    12: "chickpea",
    13: "pea",
    14: "mustard",
    15: "safflower",
    16: "sesame",
    17: "jute",
    18: "tobacco",
    19: "sugarcane",
    20: "rice",
    21: "wheat"
}

# Translation functions
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

# Irrigation check
def check_irrigation(crop, soil_type, features):
    crop = str(crop).lower()
    soil_type = soil_type.lower()
    nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall = features

    crop_irrigation = {
        "rice": "high",
        "sugarcane": "high",
        "maize": "moderate",
        "wheat": "moderate",
        "cotton": "moderate",
        "groundnut": "low",
        "millet": "low",
        "barley": "low",
        "sorghum": "low",
        "soybean": "moderate",
        "sunflower": "moderate",
        "lentil": "low",
        "chickpea": "low",
        "pea": "low",
        "mustard": "low",
        "safflower": "low",
        "sesame": "low",
        "jute": "high",
        "tobacco": "moderate"
    }
    soil_retention = {
        "clay": "high",
        "alluvial": "moderate",
        "black": "high",
        "red": "low"
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

# Yield estimation
def estimate_yield(crop, features):
    rainfall = features[6]
    base_yield = {
        "rice": 3.5,
        "wheat": 2.8,
        "maize": 2.2,
        "sugarcane": 6.5,
        "cotton": 1.5,
        "groundnut": 1.2,
        "barley": 2.0,
        "millet": 1.8,
        "sorghum": 2.0,
        "soybean": 2.5,
        "sunflower": 1.8,
        "lentil": 1.5,
        "chickpea": 1.6,
        "pea": 1.7,
        "mustard": 1.4,
        "safflower": 1.3,
        "sesame": 1.2,
        "jute": 2.0,
        "tobacco": 2.2
    }
    crop = str(crop).lower()
    yield_value = base_yield.get(crop, 2.0)
    if rainfall > 200:
        yield_value *= 1.2
    elif rainfall < 50:
        yield_value *= 0.8
    return round(yield_value, 2)

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

# @app.route('/recommend_crop', methods=['POST'])
# def recommend_crop():
#     print("Received /recommend_crop request")
#     data = request.get_json()
#     print(f"Request data: {data}")
#     lang = data.get("lang", "en")
#     try:
#         features = [
#             float(data.get("nitrogen", 0)),
#             float(data.get("phosphorus", 0)),
#             float(data.get("potassium", 0)),
#             float(data.get("temperature", 0)),
#             float(data.get("humidity", 0)),
#             float(data.get("ph", 0)),
#             float(data.get("rainfall", 0))
#         ]
#         soil_type = data.get("soil_type", "").capitalize()
#         print(f"Features: {features}, Soil Type: {soil_type}")
#         valid_soils = ["Alluvial", "Black", "Clay", "Red"]
#         if soil_type not in valid_soils:
#             print(f"Error: Invalid soil type {soil_type}")
#             return jsonify(translate_response({"error": "Invalid soil type. Must be Alluvial, Black, Clay, or Red."}, lang)), 400

#         # Crop prediction using 7 features
#         feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
#         input_df = pd.DataFrame([features], columns=feature_names)
        
#         if not hasattr(crop_model, 'predict_proba'):
#             print("Error: Model does not support predict_proba")
#             return jsonify(translate_response({"error": "Model does not support probability prediction"}, lang)), 400

#         # Get probabilities
#         probs = crop_model.predict_proba(input_df)[0]
#         print(f"Probabilities: {probs}")
        
#         # Force fallback to crop_labels
#         print("Using fallback crop labels")
#         classes = [crop_labels.get(i, f"crop_{i}") for i in range(len(probs))]
#         print(f"Classes: {classes}")
        
#         # Validate lengths
#         if len(probs) != len(classes):
#             print(f"Error: Mismatch between probs ({len(probs)}) and classes ({len(classes)})")
#             return jsonify(translate_response({"error": "Model classes and probabilities mismatch"}, lang)), 400

#         # Get top 5 crops
#         indices = np.argsort(probs)[::-1][:5]
#         print(f"Top indices: {indices}")
#         top_crops = [
#             {
#                 "crop": crop_labels.get(idx, classes[idx]),
#                 "probability": round(float(probs[idx]), 3)
#             }
#             for idx in indices
#         ]
#         print(f"Top crops: {top_crops}")

#         # Use top crop for irrigation and yield
#         top_crop = top_crops[0]["crop"] if top_crops else "unknown"
#         irrigation_status = check_irrigation(top_crop, soil_type, features)
#         estimated_yield = estimate_yield(top_crop, features)
#         print(f"Top crop: {top_crop}, irrigation={irrigation_status}, yield={estimated_yield}")

#         response = {
#             "crops": top_crops,
#             "irrigation": irrigation_status,
#             "estimated_yield": f"{estimated_yield} tons/ha"
#         }
#         print(f"Response: {response}")
#         return jsonify(translate_response(response, lang))

#     except Exception as e:
#         print(f"Error processing crop recommendation: {str(e)}")
#         return jsonify(translate_response({"error": str(e)}, lang)), 400

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

        # Crop prediction using 7 features
        feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        input_df = pd.DataFrame([features], columns=feature_names)
        
        if not hasattr(crop_model, 'predict_proba'):
            print("Error: Model does not support predict_proba")
            return jsonify(translate_response({"error": "Model does not support probability prediction"}, lang)), 400

        # Get probabilities
        probs = crop_model.predict_proba(input_df)[0]
        print(f"Probabilities: {probs}")
        
        # Force fallback to crop_labels
        print("Using fallback crop labels")
        classes = [crop_labels.get(i, f"crop_{i}") for i in range(len(probs))]
        print(f"Classes: {classes}")
        
        # Validate lengths
        if len(probs) != len(classes):
            print(f"Error: Mismatch between probs ({len(probs)}) and classes ({len(classes)})")
            return jsonify(translate_response({"error": "Model classes and probabilities mismatch"}, lang)), 400

        # Get top 5 crops
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

        # Use top crop for irrigation, yield, and response
        top_crop = top_crops[0]["crop"] if top_crops else "unknown"
        irrigation_status = check_irrigation(top_crop, soil_type, features)
        estimated_yield = estimate_yield(top_crop, features)
        print(f"Top crop: {top_crop}, irrigation={irrigation_status}, yield={estimated_yield}")

        response = {
            "crop": top_crop,  # Add top crop as a single field
            "crops": top_crops,  # Keep the array for potential future use
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
                "state": "kerala",
                "available_schemes": [
                    "Karshaka Pension Scheme - Pension for elderly farmers",
                    "Subhiksha Keralam - Promotion of self-sufficiency in food production"
                ],
                "eligibility": "Registered farmers in Kerala are eligible for these schemes.",
                "contact": "Visit the Krishi Bhavan in your locality in Kerala."
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

if __name__ == "__main__":
    app.run(debug=True)