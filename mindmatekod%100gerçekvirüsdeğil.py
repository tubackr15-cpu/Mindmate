from flask import Flask, render_template_string, request, jsonify
import json, random, re, math, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------
# 1. HTML/CSS/JS ARAYÜZÜ (Python içine gömülü)
# ---------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mindmate AI</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        :root { --bg: #343541; --sidebar: #202123; --text: #ececf1; --input: #40414f; --bot: #444654; --user: #343541; --green: #10a37f; }
        body { background-color: var(--bg); color: var(--text); font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
        
        /* Sohbet Alanı */
        #chat-container { flex: 1; overflow-y: auto; padding: 20px 10%; display: flex; flex-direction: column; gap: 0; scroll-behavior: smooth; }
        .message-row { display: flex; padding: 25px; border-bottom: 1px solid rgba(0,0,0,0.1); }
        .bot-row { background-color: var(--bot); }
        .user-row { background-color: var(--user); }
        
        .avatar { width: 35px; height: 35px; border-radius: 4px; margin-right: 15px; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0; }
        .bot-avatar { background: var(--green); color: white; }
        .user-avatar { background: #5436DA; color: white; }
        
        .message-text { line-height: 1.6; font-size: 16px; margin-top: 5px; }

        /* Giriş Alanı */
        .input-area { background: var(--bg); padding: 30px 10%; border-top: 1px solid #5d5d67; position: relative; }
        .input-box { background: var(--input); border-radius: 10px; display: flex; padding: 10px 15px; box-shadow: 0 0 10px rgba(0,0,0,0.3); border: 1px solid #565869; }
        input { flex: 1; background: transparent; border: none; color: white; outline: none; font-size: 16px; padding: 5px; }
        button { background: transparent; border: none; color: #acacbe; cursor: pointer; transition: 0.3s; padding: 5px; }
        button:hover { color: white; }
    </style>
</head>
<body>

    <div id="chat-container">
        <div class="message-row bot-row">
            <div class="avatar bot-avatar">AI</div>
            <div class="message-text">Merhaba! Ben Mindmate. Sana nasıl yardımcı olabilirim?</div>
        </div>
    </div>

    <div class="input-area">
        <div class="input-box">
            <input type="text" id="userInput" placeholder="Bir mesaj gönder..." autocomplete="off">
            <button onclick="sendMessage()">➤</button>
        </div>
        <p style="text-align:center; font-size:11px; color:#8e8ea0; margin-top:10px;">Mindmate henüz geliştirme aşamasındadır.</p>
    </div>

    <script>
        const chatBox = document.getElementById("chat-container");

        // Enter tuşu kontrolü
        $("#userInput").keypress(function(event) {
            if (event.which == 13) sendMessage();
        });

        function sendMessage() {
            let text = $("#userInput").val().trim();
            if (!text) return;

            // Kullanıcı mesajını ekle
            $("#chat-container").append(`
                <div class="message-row user-row">
                    <div class="avatar user-avatar">S</div>
                    <div class="message-text">${text}</div>
                </div>
            `);
            $("#userInput").val("");
            chatBox.scrollTop = chatBox.scrollHeight;

            // Python'a gönder
            $.ajax({
                url: "/get_response",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({ msg: text }),
                success: function(data) {
                    // Bot cevabını ekle
                    $("#chat-container").append(`
                        <div class="message-row bot-row">
                            <div class="avatar bot-avatar">AI</div>
                            <div class="message-text">${data.reply}</div>
                        </div>
                    `);
                    chatBox.scrollTop = chatBox.scrollHeight;

                    // Eğer öğretme modu açıldıysa input'a odaklan
                    if (data.learn_mode) {
                        $("#userInput").attr("placeholder", "Bana doğru cevabı öğret...");
                    } else {
                        $("#userInput").attr("placeholder", "Bir mesaj gönder...");
                    }
                },
                error: function() {
                    alert("Bir hata oluştu.");
                }
            });
        }
    </script>
</body>
</html>
"""

# ---------------------------------------------------------
# 2. PYTHON BACKEND (FLASK & AI)
# ---------------------------------------------------------
app = Flask(__name__)
DATA_FILE = "mindmate_data.json"

# Değişkenler
vectorizer = TfidfVectorizer(ngram_range=(1,2))
clf = LogisticRegression(max_iter=1000)
data = {}
last_user_input = {} # Kullanıcı bazlı son mesajı tutmak için (Basit hafıza)

def load_and_train():
    global data, clf, vectorizer
    # Dosya yoksa oluştur
    if not os.path.exists(DATA_FILE):
        initial_data = {
            "intents": [
                {"tag": "selam", "patterns": ["selam", "merhaba", "slm"], "responses": ["Selam!", "Merhaba, hoş geldin!"]},
                {"tag": "nasılsın", "patterns": ["nasılsın", "naber"], "responses": ["Ben bir yapay zekayım, duygularım yok ama sistemlerim harika çalışıyor!"]}
            ]
        }
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, ensure_ascii=False, indent=2)
    
    # Yükle ve Eğit
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    texts = []
    tags = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            texts.append(pattern.lower())
            tags.append(intent["tag"])
            
    if texts:
        X = vectorizer.fit_transform(texts)
        clf.fit(X, tags)

# İlk başlatmada eğit
load_and_train()

def normalize(text):
    return re.sub(r'[^a-zA-Z0-9ğüşöçİĞÜŞÖÇ\s]', '', text.lower()).strip()

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/get_response", methods=["POST"])
def get_response():
    global last_user_input
    user_ip = request.remote_addr # Basit kullanıcı ayrımı
    user_msg = request.json.get("msg", "")
    norm_msg = normalize(user_msg)
    
    # 1. ÖĞRETME MODU KONTROLÜ (Eğer önceki cevap "Bilmiyorum" ise)
    if last_user_input.get(user_ip) == "WAITING_TEACH":
        if norm_msg in ["hayır", "iptal", "yok"]:
            last_user_input[user_ip] = None
            return jsonify({"reply": "Tamam, iptal ettim. Başka ne sormak istersin?", "learn_mode": False})
        
        # Yeni bilgiyi kaydet
        prev_question = last_user_input.get(user_ip + "_q")
        new_tag = f"user_taught_{random.randint(1000,9999)}"
        
        # JSON'a ekle
        data["intents"].append({
            "tag": new_tag,
            "patterns": [prev_question],
            "responses": [user_msg] # Kullanıcının öğrettiği cevap
        })
        
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Yeniden eğit
        load_and_train()
        
        last_user_input[user_ip] = None
        return jsonify({"reply": f"Teşekkürler! '{prev_question}' sorusuna '{user_msg}' demeyi öğrendim.", "learn_mode": False})

    # 2. NORMAL CEVAPLAMA
    # Matematik
    if re.search(r"\d", norm_msg) and any(x in norm_msg for x in ["+", "-", "*", "/", "x"]):
        try:
            clean_math = norm_msg.replace("x", "*").replace(",", ".")
            res = eval(clean_math, {"__builtins__": {}})
            return jsonify({"reply": f"Hesapladım: {res}", "learn_mode": False})
        except: pass

    # AI Tahmin
    try:
        vec = vectorizer.transform([norm_msg])
        probs = clf.predict_proba(vec)[0]
        max_prob = max(probs)
        best_tag = clf.classes_[probs.argmax()]

        if max_prob > 0.6:
            for intent in data["intents"]:
                if intent["tag"] == best_tag:
                    return jsonify({"reply": random.choice(intent["responses"]), "learn_mode": False})
    except: pass
    
    # Bilmiyorsa Öğrenme Moduna Geç
    last_user_input[user_ip] = "WAITING_TEACH"
    last_user_input[user_ip + "_q"] = norm_msg
    return jsonify({
        "reply": "Bunu henüz bilmiyorum. Bana ne cevap vermem gerektiğini öğretir misin? (Veya 'hayır' yaz)",
        "learn_mode": True
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)