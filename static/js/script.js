// DOM Elements
const toast = document.getElementById('toast');

// Language Support
let currentLanguage = 'en';
fetch('./translations.json')
    .then(response => response.json())
    .then(data => {
        window.translations = data;
        initializeApp();
    })
    .catch(error => console.error('Error loading translations:', error));

// Initialize App
function initializeApp() {
    updateTexts();
    document.addEventListener('DOMContentLoaded', () => {
        updateTexts();
    });
}

// Toast Function
function showToast(message) {
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 3000);
}

// Language and Main App Logic
function changeLanguage() {
    currentLanguage = document.getElementById('language').value;
    updateTexts();
}

function updateTexts() {
    const lang = window.translations[currentLanguage];
    document.getElementById('appTitle').textContent = lang.appTitle;
    document.getElementById('soilOption').textContent = lang.soilOption;
    document.getElementById('cropOption').textContent = lang.cropOption;
    document.getElementById('aidOption').textContent = lang.aidOption;
    document.getElementById('soilTitle').textContent = lang.soilTitle;
    document.getElementById('soilImageLabel').textContent = lang.soilImageLabel;
    document.getElementById('soilButton').textContent = lang.soilButton;
    document.getElementById('soilLoading').textContent = lang.analyzingSoil;
    document.getElementById('cropTitle').textContent = lang.cropTitle;
    document.getElementById('nitrogenLabel').textContent = lang.nitrogenLabel;
    document.getElementById('phosphorousLabel').textContent = lang.phosphorousLabel;
    document.getElementById('potassiumLabel').textContent = lang.potassiumLabel;
    document.getElementById('tempLabel').textContent = lang.tempLabel;
    document.getElementById('humidityLabel').textContent = lang.humidityLabel;
    document.getElementById('phLabel').textContent = lang.phLabel;
    document.getElementById('rainfallLabel').textContent = lang.rainfallLabel;
    document.getElementById('cropButton').textContent = lang.cropButton;
    document.getElementById('cropLoading').textContent = lang.analyzingCrop;
    document.getElementById('aidTitle').textContent = lang.aidTitle;
    document.getElementById('stateLabel').textContent = lang.stateLabel;
    document.getElementById('landLabel').textContent = lang.landLabel;
    document.getElementById('aidButton').textContent = lang.aidButton;
    document.getElementById('aidLoading').textContent = lang.searchingSchemes;
    document.getElementById('backButton1').textContent = lang.backButton;
    document.getElementById('backButton2').textContent = lang.backButton;
    document.getElementById('backButton3').textContent = lang.backButton;
    document.getElementById('nitrogen').placeholder = lang.nitrogenPlaceholder;
    document.getElementById('phosphorous').placeholder = lang.phosphorousPlaceholder;
    document.getElementById('potassium').placeholder = lang.potassiumPlaceholder;
    document.getElementById('temperature').placeholder = lang.tempPlaceholder;
    document.getElementById('humidity').placeholder = lang.humidityPlaceholder;
    document.getElementById('ph').placeholder = lang.phPlaceholder;
    document.getElementById('rainfall').placeholder = lang.rainfallPlaceholder;
    document.getElementById('state').placeholder = lang.statePlaceholder;
    document.getElementById('land_size').placeholder = lang.landPlaceholder;
}

function showForm(formId) {
    document.getElementById('options').classList.add('hidden');
    ['soil_prediction', 'crop_mgmt', 'govt_aid'].forEach(form => {
        document.getElementById(form).classList.add('hidden');
    });
    document.getElementById(formId).classList.remove('hidden');
}

function goBack() {
    ['soil_prediction', 'crop_mgmt', 'govt_aid'].forEach(form => {
        document.getElementById(form).classList.add('hidden');
    });
    document.getElementById('options').classList.remove('hidden');
    document.querySelectorAll('.result').forEach(el => el.innerHTML = '');
}

function updateBotMessage(section) {
    const messageMap = {
        "soil": "This section lets you detect the soil type and any pests using an image!",
        "crop": "This helps you find the best crop and irrigation needs based on your soil data.",
        "aids": "Here you'll find government schemes based on your state and land area.",
        "home": "Welcome back! Choose any option to begin your agricultural journey."
};
document.getElementById("botMessage").textContent = messageMap[section] || messageMap["home"];
}

// Form Handlers
document.getElementById('cropForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const loading = document.getElementById('cropLoading');
    const resultDiv = document.getElementById('cropResult');
    const lang = window.translations[currentLanguage];

    // Get soil type separately
    const soilType = document.getElementById('soil_type').value;

    // Only pass the 7 crop features here
    const features = [
        document.getElementById('nitrogen').value,
        document.getElementById('phosphorous').value,
        document.getElementById('potassium').value,
        document.getElementById('temperature').value,
        document.getElementById('humidity').value,
        document.getElementById('ph').value,
        document.getElementById('rainfall').value
    ].map(Number);

    loading.style.display = 'block';
    resultDiv.innerHTML = '';

    try {
        const response = await fetch('/recommend_crop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                features: features,
                soil_type: soilType,
                lang: currentLanguage
            })
        });

        const data = await response.json();
        loading.style.display = 'none';

        if (response.ok) {
            resultDiv.innerHTML = `
                <div class="result">
                    <div class="result-item"><span class="result-title">${lang.resultFields['Recommended Crops']}:</span> ${data.crop}</div>
                    <div class="result-item"><span class="result-title">${lang.resultFields['Irrigation Status']}:</span> ${data.irrigation}</div>
                    <div class="result-item"><span class="result-title">${lang.resultFields['Estimated Yield']}:</span> ${data.estimated_yield}</div>
                </div>
            `;
        } else {
            resultDiv.innerHTML = `<div class="error">${data.error || lang.cropError}</div>`;
        }
    } catch (error) {
        loading.style.display = 'none';
        resultDiv.innerHTML = `<div class="error">${lang.networkError}</div>`;
    }
});

document.getElementById('aidForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const loading = document.getElementById('aidLoading');
    const resultDiv = document.getElementById('aidResult');

    const state = document.getElementById('state').value;
    const landSize = document.getElementById('land_size').value;

    loading.style.display = 'block';
    resultDiv.innerHTML = '';

    try {
        const response = await fetch('/government_aids', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ state, land_size: Number(landSize) })
        });
        const data = await response.json();
        loading.style.display = 'none';

        if (response.ok) {
            const lang = window.translations[currentLanguage];
            resultDiv.innerHTML = `
                <div class="result">
                    <div class="result-item"><span class="result-title">${lang.resultFields['State']}:</span> ${data['State']}</div>
                    <div class="result-item"><span class="result-title">${lang.resultFields['Land Size (acres)']}:</span> ${data['Land Size (acres)']}</div>
                    <div class="result-item"><span class="result-title">${lang.resultFields['Available Schemes']}:</span>
                        <ul>${data['Available Schemes'].map(s => `<li>${s}</li>`).join('')}</ul>
                    </div>
                </div>
            `;
        } else {
            resultDiv.innerHTML = `<div class="error">${window.translations[currentLanguage].aidError}</div>`;
        }
    } catch (error) {
        loading.style.display = 'none';
        resultDiv.innerHTML = `<div class="error">${window.translations[currentLanguage].networkError}</div>`;
    }
});