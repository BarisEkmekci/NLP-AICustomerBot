from flask import Flask, request, jsonify
from flask_cors import CORS
from chat import get_response

# Flask uygulamasını oluştur
app = Flask(__name__)
# CORS (Cross-Origin Resource Sharing) ayarlarını yap
CORS(app)

@app.route("/chat", methods=["POST"])
def chat_api():
    """
    Bu fonksiyon, /chat adresine gelen POST isteklerini karşılar.
    İstek içindeki mesajı alır, chatbot'tan cevabı alır ve geri gönderir.
    """
    # İstekten JSON verisini al
    data = request.get_json()
    message = data.get("message")

    # Mesaj boş ise hata döndür
    if not message:
        return jsonify({"error": "Message field is required"}), 400

    # Chatbot'tan cevabı al
    response = get_response(message)

    # Cevabı JSON formatında geri gönder
    return jsonify({"answer": response})

if __name__ == '__main__':
    # Sunucuyu başlat. http://127.0.0.1:5000 adresinde çalışacak.
    app.run(port=5000, debug=True)