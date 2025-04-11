const slogans = [
    { text: "விவசாயி வளம் பெற்றால், நாடு வளம் பெறும்", lang: "Tamil" },
    { text: "കർഷകൻ സമ്പന്നനാണെങ്കിൽ രാജ്യവും സമ്പന്നമാണ്", lang: "Malayalam" },
    { text: "రైతు సంపన్నుడైతే, దేశం సంపన్నం", lang: "Telugu" },
    { text: "ರೈತ ಶ್ರೀಮಂತನಾದರೆ, ದೇಶವೂ ಶ್ರೀಮಂತ", lang: "Kannada" },
    { text: "किसान समृद्ध है तो देश समृद्ध है", lang: "Hindi" },
    { text: "If the farmer is rich, then so is the nation", lang: "English" }
];

const sloganText = document.getElementById('dynamic-slogan');
const sloganLanguage = document.getElementById('slogan-language');
const tabButtons = document.querySelectorAll('.tab-btn');
const authForms = document.querySelectorAll('.auth-form');

let currentSloganIndex = 0;


function typeSlogan(text, lang, callback) {
    sloganText.textContent = '';
    sloganLanguage.textContent = '';
    sloganText.classList.remove('fade-out');
    sloganLanguage.classList.remove('visible');

    let i = 0;
    const speed = 50;

    function type() {
        if (i < text.length) {
            sloganText.textContent += text.charAt(i);
            i++;
            setTimeout(type, speed);
        } else {
            sloganLanguage.textContent = lang;
            sloganLanguage.classList.add('visible');
            callback();
        }
    }

    type();
}

function updateSlogan() {
    const currentSlogan = slogans[currentSloganIndex];
    sloganText.classList.add('fade-out');
    sloganLanguage.classList.remove('visible');

    setTimeout(() => {
        typeSlogan(currentSlogan.text, currentSlogan.lang, () => {
            currentSloganIndex = (currentSloganIndex + 1) % slogans.length;
            setTimeout(updateSlogan, 3500);
        });
    }, 1000);
}

// Initial call
updateSlogan();

// Tab switching logic
tabButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
        tabButtons.forEach((b) => b.classList.remove('active'));
        authForms.forEach((form) => form.classList.remove('active'));

        btn.classList.add('active');
        document.getElementById(btn.dataset.tab + 'Form').classList.add('active');
    });
});

// ------------------------------
// Login validation functionality
// ------------------------------
document.getElementById('loginForm').addEventListener('submit', function (e) {
    e.preventDefault();

    const email = document.getElementById('loginEmail').value.trim();
    const password = document.getElementById('loginPassword').value.trim();

    const emailPattern = /^[a-zA-Z0-9._%+-]+@gmail\.com$/;

    if (!emailPattern.test(email)) {
        alert("Please enter a valid Gmail address (e.g. user@gmail.com).");
        return;
    }

    if (password.length < 6) {
        alert("Password should be at least 6 characters long.");
        return;
    }

    // Store email and password in localStorage (simulate login session)
    localStorage.setItem('userEmail', email);
    localStorage.setItem('userPassword', password);

    alert("Login successful!");

    // You can redirect to another page here:
    // window.location.href = "dashboard.html";
    window.location.href = "/index";

});

// ------------------------------
// Auto-fill login if data exists
// ------------------------------
window.addEventListener('DOMContentLoaded', () => {
    const savedEmail = localStorage.getItem('userEmail');
    const savedPassword = localStorage.getItem('userPassword');

    if (savedEmail && savedPassword) {
        document.getElementById('loginEmail').value = savedEmail;
        document.getElementById('loginPassword').value = savedPassword;
    }
});
