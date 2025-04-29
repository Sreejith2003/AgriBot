document.addEventListener('DOMContentLoaded', () => {
    // Tab switching functionality
    const tabButtons = document.querySelectorAll(".tab-btn");
    const authForms = document.querySelectorAll(".auth-form");

    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            tabButtons.forEach(b => b.classList.remove("active"));
            authForms.forEach(form => form.classList.remove("active"));
            btn.classList.add("active");
            document.getElementById(`${btn.dataset.tab}Form`).classList.add("active");
        });
    });

    // LOGIN form submission
    document.getElementById("loginForm").addEventListener("submit", async function(e) {
        e.preventDefault();
        const email = document.getElementById("loginEmail").value.trim();
        const password = document.getElementById("loginPassword").value.trim();

        const emailRegex = /^[a-zA-Z0-9._%+-]+@gmail\.com$/;
        if (!emailRegex.test(email)) {
            alert("Please enter a valid Gmail address (must end with @gmail.com)");
            return;
        }

        try {
            const response = await fetch('/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            if (result.success) {
                alert("Login successful! Welcome, " + result.user.full_name);
                window.location.href = "/index";
            } else {
                alert(result.error || "Login failed. Please try again.");
            }
        } catch (err) {
            alert("Network error during login: " + err.message);
        }
    });

    // REGISTER form submission
    document.getElementById("registerForm").addEventListener("submit", async function(e) {
        e.preventDefault();
        const full_name = document.getElementById("registerName").value.trim();
        const email = document.getElementById("registerEmail").value.trim();
        const password = document.getElementById("registerPassword").value.trim();
        const confirmPassword = document.getElementById("confirmPassword").value.trim();

        const emailRegex = /^[a-zA-Z0-9._%+-]+@gmail\.com$/;
        if (!emailRegex.test(email)) {
            alert("Please enter a valid Gmail address (must end with @gmail.com)");
            return;
        }

        if (password !== confirmPassword) {
            alert("Passwords do not match");
            return;
        }

        try {
            const response = await fetch('/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ full_name, email, password })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            if (result.success) {
                alert("Registration successful! Please log in.");
                document.querySelector('.tab-btn[data-tab="login"]').click();
            } else {
                alert(result.error || "Registration failed. Please try again.");
            }
        } catch (err) {
            alert("Network error during registration: " + err.message);
        }
    });

    // Slogan rotation logic
    const slogans = [
        { text: "உழுதுண்டு வாழ்வாரே வாழ்வார் மற்றெல்லாம் தொழுதுண்டு பின்செல் பவர்", lang: "Tamil" },
        { text: "ഉഴുതുണ്ട് ജീവിക്കുന്നവരാണ് ജീവിക്കുന്നവർ മറ്റെല്ലാവരും അനുഗമിച്ച് ജീവിക്കുന്നവരാണ്", lang: "Malayalam" },
        { text: "ఉధృతి తిని జీవించేవారే నిజంగా జీవించేవారు మిగతావారంతా ఇతరులను ఆధారపడి జీవించేవారు", lang: "Telugu" },
        { text: "ಉಳುಮೆ ಮಾಡಿ ತಿನ್ನುವವರೇ ನಿಜವಾಗಿ ಬದುಕುವವರು ಉಳಿದವರೆಲ್ಲಾ ಇತರರನ್ನು ಅವಲಂಬಿಸಿ ಬದುಕುವವರು", lang: "Kannada" },
        { text: "जो हल चलाकर खाते हैं, वे ही सच्चे जीवन जीते हैं बाकी सब दूसरों के आगे हाथ फैलाकर जीते हैं", lang: "Hindi" },
        { text: "Only those who plough and eat shall truly live All others are but followers, eating from their hands", lang: "English" }
    ];

    const sloganText = document.getElementById('dynamic-slogan');
    const sloganLanguage = document.getElementById('slogan-language');

    // Add CSS for line breaks and text alignment
    const style = document.createElement('style');
    style.textContent = `
        #dynamic-slogan {
            white-space: pre-wrap;
            text-align: center;
            line-height: 1.2; /* Reduced to tighten lines */
            min-height: 3em;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        
        .slogan-line {
            display: block;
            width: 100%;
            text-align: center;
            word-break: break-word;
            margin: 0; /* Removed margin to reduce space */
        }
        
        #slogan-language {
            font-style: italic;
            margin-top: 0.5em;
            opacity: 0;
            transition: opacity 0.5s ease;
        }
        
        #slogan-language.visible {
            opacity: 1;
        }
        
        .fade-out {
            opacity: 0;
            transition: opacity 0.7s ease;
        }
    `;
    document.head.appendChild(style);

    let currentSloganIndex = 0;

    function typeSlogan(slogan, callback) {
        sloganText.innerHTML = ''; // Clear previous content
        sloganLanguage.textContent = '';
        sloganText.classList.remove('fade-out');
        sloganLanguage.classList.remove('visible');

        // Split into two lines based on language
        let lines;
        if (slogan.lang === "Tamil") {
            // Explicit split for Tamil to ensure correct alignment
            lines = [
                "உழுதுண்டு வாழ்வாரே வாழ்வார்",
                "மற்றெல்லாம் தொழுதுண்டு பின்செல் பவர்"
            ];
        } else {
            // For other languages, split at a logical point
            const splitPoint = slogan.text.indexOf("मറ്റെല്ലാവരും") || 
                              slogan.text.indexOf("మిగతావారంతా") || 
                              slogan.text.indexOf("ಉಳಿದವರೆಲ್ಲಾ") || 
                              slogan.text.indexOf("बाकी सब") || 
                              slogan.text.indexOf("All others") || 
                              (slogan.text.length / 2);
            lines = [
                slogan.text.substring(0, splitPoint).trim(),
                slogan.text.substring(splitPoint).trim()
            ];
        }

        const lineElements = [];

        // Create line spans
        lines.forEach((line, index) => {
            const lineSpan = document.createElement('span');
            lineSpan.className = 'slogan-line';
            sloganText.appendChild(lineSpan);
            lineElements.push(lineSpan);

            // Add line break after the first line
            if (index === 0) {
                sloganText.appendChild(document.createElement('br'));
            }
        });

        let currentLine = 0;

        function typeNextLine() {
            if (currentLine < lines.length) {
                lineElements[currentLine].textContent = lines[currentLine];
                currentLine++;
                if (currentLine < lines.length) {
                    setTimeout(typeNextLine, 1000); // Pause between lines
                } else {
                    sloganLanguage.textContent = `— ${slogan.lang}`;
                    sloganLanguage.classList.add('visible');
                    callback();
                }
            }
        }

        typeNextLine();
    }

    function updateSlogan() {
        const currentSlogan = slogans[currentSloganIndex];
        sloganText.classList.add('fade-out');
        sloganLanguage.classList.remove('visible');

        setTimeout(() => {
            typeSlogan(currentSlogan, () => {
                currentSloganIndex = (currentSloganIndex + 1) % slogans.length;
                setTimeout(updateSlogan, 5000); // Show each slogan for 5 seconds
            });
        }, 700); // Match fade-out transition
    }

    // Initial call
    updateSlogan();
});
