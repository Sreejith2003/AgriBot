from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import random
import joblib
import requests
from bs4 import BeautifulSoup
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from googletrans import Translator

app = Flask(__name__)
CORS(app)

translator = Translator(timeout=10)

# Paths
CROP_MODEL_PATH = "C://Users//sreej//OneDrive//Documents//AgriBot//model//crop_model (2) (1).pkl"
IRRIGATION_MODEL_PATH = "C://Users//sreej//OneDrive//Documents//AgriBot//model//irrigation_model (1).pkl"
SOIL_MODEL_PATH = "C://Users//sreej//OneDrive//Documents//AgriBot//model//efficientnet_soil (1).pth"

SUPPORTED_LANGUAGES = {
    'en': 'English', 'hi': 'Hindi', 'te': 'Telugu', 'ta': 'Tamil',
    'kn': 'Kannada', 'ml': 'Malayalam', 'mr': 'Marathi', 'bn': 'Bengali',
    'gu': 'Gujarati', 'pa': 'Punjabi'
}

# Load EfficientNet-B0 for soil classification
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

soil_model = CustomEfficientNet()
soil_model.load_state_dict(torch.load(SOIL_MODEL_PATH, map_location=torch.device('cpu')))
soil_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

soil_classes = ["Alluvial Soil", "Black Soil", "Clay Soil", "Red Soil"]

# Load ML models
try:
    crop_model = joblib.load(CROP_MODEL_PATH)
    irrigation_model = joblib.load(IRRIGATION_MODEL_PATH)
except Exception as e:
    print("Error loading models:", e)

@app.route('/')
def home():
    return render_template('login.html', languages=SUPPORTED_LANGUAGES)

@app.route('/index')
def index():
    return render_template('index.html')

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
    if isinstance(data, dict):
        return {k: translate_response(v, lang) for k, v in data.items()}  # Keep keys as-is
    elif isinstance(data, list):
        return [translate_response(i, lang) for i in data]
    elif isinstance(data, str):
        return translate_text(data, lang)
    else:
        return data

def check_irrigation(crop, soil_type):
    crop = crop.lower()
    soil_type = soil_type.lower()

    # Define crop water needs (scale: high, moderate, low)
    crop_irrigation = {
        "rice": "high",
        "sugarcane": "high",
        "maize": "moderate",
        "wheat": "moderate",
        "cotton": "moderate",
        "groundnut": "low",
        "millet": "low",
        "barley": "low"
    }

    # Define soil water retention
    soil_retention = {
        "clay": "high",
        "alluvial": "moderate",
        "black": "high",
        "red": "low"
    }

    crop_need = crop_irrigation.get(crop, "moderate")
    soil_hold = soil_retention.get(soil_type, "moderate")

    # Decision logic
    if crop_need == "high" and soil_hold == "low":
        return "Very high irrigation required"
    elif crop_need == "high" and soil_hold == "moderate":
        return "High irrigation required"
    elif crop_need == "high" and soil_hold == "high":
        return "Moderate irrigation required"

    elif crop_need == "moderate" and soil_hold == "low":
        return "High irrigation required"
    elif crop_need == "moderate" and soil_hold == "moderate":
        return "Moderate irrigation required"
    elif crop_need == "moderate" and soil_hold == "high":
        return "Low irrigation required"

    elif crop_need == "low" and soil_hold == "low":
        return "Moderate irrigation required"
    elif crop_need == "low" and soil_hold == "moderate":
        return "Low irrigation required"
    elif crop_need == "low" and soil_hold == "high":
        return "Very low irrigation required"

    return "Irrigation info not available"


def estimate_yield(crop, features):
    # Dummy logic to simulate yield estimation
    rainfall = features[6]
    base_yield = {
        "rice": 3.5,
        "wheat": 2.8,
        "maize": 2.2,
        "sugarcane": 6.5,
        "cotton": 1.5,
        "groundnut": 1.2
    }

    crop = crop.lower()
    yield_value = base_yield.get(crop, 2.0)  # default base

    if rainfall > 200:
        yield_value *= 1.2
    elif rainfall < 50:
        yield_value *= 0.8

    return round(yield_value, 2)

# @app.route('/recommend_crop', methods=['POST'])
# def recommend_crop():
#     data = request.get_json()
#     lang = data.get("language", "en")

#     try:
#         # Extract 7 input features
#         features = [
#             float(data.get("nitrogen", 0)),
#             float(data.get("phosphorus", 0)),
#             float(data.get("potassium", 0)),
#             float(data.get("temperature", 0)),
#             float(data.get("humidity", 0)),
#             float(data.get("ph", 0)),
#             float(data.get("rainfall", 0))
#         ]
#     except (TypeError, ValueError):
#         response = {
#             "success": False,
#             "error": "Invalid input values provided"
#         }
#         return jsonify(translate_response(response, lang)), 400

#     if len(features) != 7:
#         response = {
#             "success": False,
#             "error": "All 7 features must be provided"
#         }
#         return jsonify(translate_response(response, lang)), 400

#     try:
#         # Predict top 3 crop recommendations
#         probabilities = crop_model.predict_proba([features])[0]
#         class_indices = probabilities.argsort()[::-1][:3]  # Top 3 indices
#         top_crops = [crop_model.classes_[i] for i in class_indices]

#         # Simple logic for irrigation status
#         avg_temp = features[3]
#         avg_rain = features[6]
#         irrigation_status = "Irrigation Needed" if avg_temp > 30 and avg_rain < 50 else "No Irrigation Required"

#         # Dummy yield estimate logic
#         yield_estimate = f"{round(probabilities[class_indices[0]] * 4.0, 2)} tons/hectare"

#         # Form final response
#         response = {
#             "success": True,
#             "data": {
#                 "recommended_crops": top_crops,
#                 "irrigation_status": irrigation_status,
#                 "estimated_yield": yield_estimate
#             }
#         }

#         return jsonify(translate_response(response, lang))

#     except Exception as e:
#         response = {
#             "success": False,
#             "error": f"Prediction failed: {str(e)}"
#         }
#         return jsonify(translate_response(response, lang)), 500

@app.route('/recommend_crop', methods=['POST'])
def recommend_crop():
    data = request.get_json()
    lang = data.get("lang", "en")

    try:
        # Step 1: Extract features
        features = [
            float(data.get("nitrogen", 0)),
            float(data.get("phosphorus", 0)),
            float(data.get("potassium", 0)),
            float(data.get("temperature", 0)),
            float(data.get("humidity", 0)),
            float(data.get("ph", 0)),
            float(data.get("rainfall", 0))
        ]

        # Step 2: Extract & validate soil_type
        soil_type = data.get("soil_type", "").capitalize()
        valid_soils = ["Alluvial", "Black", "Clay", "Red"]
        if soil_type not in valid_soils:
            return jsonify(translate_response({"error": "Invalid soil type. Must be Alluvial, Black, Clay, or Red."}, lang)), 400

        # Step 3: Crop prediction
        crop = crop_model.predict([features])[0]

        # Step 4: Irrigation suggestion
        irrigation_status = check_irrigation(crop, soil_type)

        # Step 5: Yield prediction (rule-based for now)
        estimated_yield = estimate_yield(crop, features)

        response = {
            "crop": crop,
            "irrigation": irrigation_status,
            "estimated_yield": f"{estimated_yield} tons/ha"
        }

        return jsonify(translate_response(response, lang))

    except Exception as e:
        return jsonify(translate_response({"error": str(e)}, lang)), 400

@app.route('/predict_soil', methods=['POST'])
def soil_prediction():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided."}), 400

    lang = request.form.get('language', 'en').lower()
    if lang not in SUPPORTED_LANGUAGES:
        lang = 'en'

    image = request.files['image']
    if image.filename == '':
        return jsonify(translate_response({"error": "No selected file"}, lang)), 400

    filename = secure_filename(image.filename)
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    image_path = os.path.join(upload_dir, filename)
    image.save(image_path)

    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = soil_model(img_tensor)
            predicted = torch.argmax(output, 1).item()

        soil_type = soil_classes[predicted]
        pest_detection = random.choice(["None", "Locusts", "Aphids", "Armyworm"])

        response = {
            "success": True,
            "data": {
                "soil_type": soil_type,
                "pest_detection": pest_detection
            }
        }
        return jsonify(translate_response(response, lang))
    except Exception as e:
        error_msg = {"error": f"Image processing failed: {str(e)}"}
        return jsonify(translate_response(error_msg, lang)), 500
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

@app.route("/get_schemes", methods=["POST"])
@app.route("/government_aids", methods=["POST"])
def government_aid():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    state = data.get("state", "").strip()
    lang = data.get("language", "en").lower()

    # Validate land size
    try:
        land_size = float(data.get("land_size", 0))
    except (ValueError, TypeError):
        return jsonify(translate_response({"error": "Invalid land size format."}, lang)), 400

    if lang not in SUPPORTED_LANGUAGES:
        lang = "en"

    if not state:
        return jsonify(translate_response({"error": "State name is required."}, lang)), 400

    schemes = fetch_govt_schemes(state, land_size)

    response = {
        "success": True,
        "data": {
            "state": state.title(),
            "land_size": land_size,
            "available_schemes": schemes["schemes"],
            "eligibility": schemes["eligibility"],
            "contact": schemes["contact"]
        }
    }
    return jsonify(translate_response(response, lang))


def fetch_govt_schemes(state, land_size):
    state = state.lower()

    all_states = [
        {
    
        "state": "Andhra Pradesh",
        "available_schemes": [
            "YSR Rythu Bharosa - ₹13,500/year financial assistance",
            "YSR Free Crop Insurance - No premium crop insurance",
            "YSR Sunna Vaddi Panta Runalu - Interest-free crop loans"
        ],
        "eligibility": "Farmers with any landholding in Andhra Pradesh are eligible for these state schemes.",
        "contact": "Contact your local Rythu Bharosa Kendram for more details."
        },
        {
        "state": "Arunachal Pradesh",
        "available_schemes": [
            "Chief Minister's Sashakt Kisan Yojana - Financial assistance for farm inputs",
            "Chief Minister's Krishi Rinn Yojana - Interest subvention on crop loans"
        ],
        "eligibility": "All farmers in Arunachal Pradesh are eligible for these schemes.",
        "contact": "Reach out to the Department of Agriculture, Arunachal Pradesh for more information."
        },
        {
        "state": "Assam",
        "available_schemes": [
            "Assam Agribusiness and Rural Transformation Project (APART) - Support for agribusiness",
            "Chief Minister's Samagra Gramya Unnayan Yojana - Comprehensive village development"
        ],
        "eligibility": "Farmers involved in agribusiness in Assam are eligible for these schemes.",
        "contact": "Contact your nearest Krishi Vigyan Kendra in Assam."
        },
        {
        "state": "Bihar",
        "available_schemes": [
            "Bihar Rajya Fasal Sahayata Yojana - Financial assistance for crop loss",
            "Diesel Subsidy Scheme - Subsidy on diesel for irrigation"
        ],
        "eligibility": "Farmers with landholding in Bihar are eligible for these schemes.",
        "contact": "Visit the Bihar Agriculture Department for more details."
        },
        {
        "state": "Chhattisgarh",
        "available_schemes": [
            "Rajiv Gandhi Kisan Nyay Yojana - Direct benefit transfer to farmers",
            "Godhan Nyay Yojana - Procurement of cow dung from farmers"
        ],
        "eligibility": "All farmers in Chhattisgarh are eligible for these schemes.",
        "contact": "Contact your local cooperative society in Chhattisgarh."
        },
        {
        "state": "Goa",
        "available_schemes": [
            "Shetkari Adhar Nidhi - Financial assistance to farmers",
            "Krishi Card Scheme - Credit facility for farmers"
        ],
        "eligibility": "Farmers registered in Goa are eligible for these schemes.",
        "contact": "Reach out to the Directorate of Agriculture, Goa."
        },
        {
        "state": "Gujarat",
        "available_schemes": [
            "Mukhyamantri Kisan Sahay Yojana - Financial assistance during natural calamities",
            "Jyotigram Yojana - 24-hour electricity supply to farmers"
        ],
        "eligibility": "All farmers in Gujarat are eligible for these schemes.",
        "contact": "Contact your local Agricultural Technology Management Agency (ATMA) in Gujarat."
        },
        {
        "state": "Haryana",
        "available_schemes": [
            "Meri Fasal Mera Byora - Online crop registration and assistance",
            "Bhavantar Bharpai Yojana - Price deficiency support"
        ],
        "eligibility": "Farmers with landholding in Haryana are eligible for these schemes.",
        "contact": "Visit the Haryana Agriculture Department for more information."
        },
        {
        "state": "Himachal Pradesh",
        "available_schemes": [
            "Krishi Seva Kendra - One-stop centers for farmers",
            "Mukhya Mantri Khet Sanrakshan Yojana - Subsidy for solar fencing"
        ],
        "eligibility": "All farmers in Himachal Pradesh are eligible for these schemes.",
        "contact": "Contact your local Krishi Seva Kendra in Himachal Pradesh."
        },
        {
        "state": "Jharkhand",
        "available_schemes": [
            "Mukhyamantri Krishi Ashirwad Yojana - Financial assistance per acre",
            "Jharkhand State Crop Relief Scheme - Compensation for crop loss"
        ],
        "eligibility": "Farmers with landholding in Jharkhand are eligible for these schemes.",
        "contact": "Reach out to the Department of Agriculture, Jharkhand."
        },
        {
        "state": "Karnataka",
        "available_schemes": [
            "Raitha Siri - Financial assistance for millet cultivation",
            "Krishi Bhagya Scheme - Support for rainwater harvesting structures"
        ],
        "eligibility": "Farmers in Karnataka engaged in specified crops are eligible.",
        "contact": "Contact your local Raitha Samparka Kendra in Karnataka."
        },
        {
        "state": "Kerala",
        "available_schemes": [
            "Karshaka Pension Scheme - Pension for elderly farmers",
            "Subhiksha Keralam - Promotion of self-sufficiency in food production"
        ],
        "eligibility": "Registered farmers in Kerala are eligible for these schemes.",
        "contact": "Visit the Krishi Bhavan in your locality in Kerala."
        },
        {
        "state": "Madhya Pradesh",
        "available_schemes": [
            "Bhavantar Bhugtan Yojana - Price deficiency payment scheme",
            "Mukhya Mantri Krishak Samriddhi Yojana - Incentives for wheat and paddy"
        ],
        "eligibility": "Farmers growing specified crops in Madhya Pradesh are eligible.",
        "contact": "Contact your local Krishi Upaj Mandi in Madhya Pradesh."
        },
        {
        "state": "Maharashtra",
        "available_schemes": [
            "Nanaji Deshmukh Krishi Sanjivani Yojana - Climate-resilient agriculture",
            "Gopinath Munde Shetkari Apghat Vima Yojana - Accident insurance for farmers"
        ],
        "eligibility": "All farmers in Maharashtra are eligible for these schemes.",
        "contact": "Reach out to the Maharashtra Agriculture Department."
        },
        {
        "state": "Manipur",
        "available_schemes": [
            "Mission Organic Value Chain Development - Support for organic farming",
            "Rashtriya Krishi Vikas Yojana - Holistic development of agriculture"
        ],
        "eligibility": "Farmers practicing organic farming in Manipur are eligible.",
        "contact": "Contact the Department of Agriculture, Manipur."
        },
        {
        "state": "Meghalaya",
        "available_schemes": [
            "Integrated Basin Development and Livelihood Promotion - Sustainable livelihoods",
            "Mission Organic - Promotion of organic farming"
        ],
        "eligibility": "Farmers in Meghalaya engaged in sustainable practices are eligible.",
        "contact": "Visit the Meghalaya Basin."
        },
        {
        "state": "Tamil Nadu",
        "available_schemes": [
            "Uzhavar Aluvalar Thittam - Training and capacity building for farmers",
            "Tamil Nadu Farmers' Insurance Scheme - Free crop insurance for farmers",
            "Micro Irrigation Scheme - Subsidy for drip and sprinkler systems"
        ],
        "eligibility": "Registered farmers in Tamil Nadu are eligible based on crop and scheme-specific criteria.",
        "contact": "Contact your local Agricultural Extension Centre or visit the Tamil Nadu Department of Agriculture website."
        },
        {
        "state": "Delhi",
        "available_schemes": [
            "Soil Health Card Scheme - Free soil testing and nutrient advice",
            "Subsidy on Agricultural Equipment - Financial aid for mechanized tools",
            "Organic Farming Promotion Scheme - Support for adopting organic practices"
        ],
        "eligibility": "Farmers owning agricultural land in the National Capital Territory of Delhi are eligible.",
        "contact": "Visit the Development Department (Agriculture), Government of NCT of Delhi for more information."
    },
]
    
    for s in all_states:
        if s["state"].lower() == state:
            # Optionally include land size in eligibility message
            eligibility_msg = f"{s['eligibility']} You have {land_size} acres of land."
            return {
                "schemes": s["available_schemes"],
                "eligibility": eligibility_msg,
                "contact": s["contact"]
            }

    # If no state match found
    return {
        "schemes": ["No official schemes found for your state."],
        "eligibility": "Unknown",
        "contact": "Visit your local agriculture office for accurate details."
    }

    # query = f"{state} government farming schemes for {land_size} acres site:gov.in"
    # url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    # headers = {"User-Agent": "Mozilla/5.0"}

    # try:
    #     response = requests.get(url, headers=headers, timeout=10)
    #     response.raise_for_status()
    #     soup = BeautifulSoup(response.text, "html.parser")
    #     titles = [g.get_text() for g in soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd")]
    #     schemes = list({title for title in titles if title})

    #     if not schemes:
    #         return ["No official schemes found. Please check your state agricultural website."]
    #     return schemes[:5]

    # except requests.RequestException:
    #     return ["Error fetching data. Please check your internet connection or try again later."]
    # except Exception as e:
    #     print(f"Error scraping schemes: {str(e)}")
    #     return ["Error retrieving scheme information. Please try again later."]

if __name__ == '__main__':
    app.run(debug=True)
