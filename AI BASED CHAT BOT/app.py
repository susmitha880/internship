from flask import Flask, request, jsonify, render_template_string
from chatbot import get_response

app = Flask(__name__)

# ✅ Home route (browser-friendly)
@app.route("/")
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Chatbot</title>
        <style>
            body { font-family: Arial; background: #f4f4f4; }
            .chatbox { width: 400px; margin: 50px auto; background: white; padding: 20px; }
            input { width: 75%; padding: 10px; }
            button { padding: 10px; }
            #chat { margin-top: 15px; }
        </style>
    </head>
    <body>
        <div class="chatbox">
            <h2>AI Customer Support Chatbot</h2>
            <input id="msg" placeholder="Type your message..." />
            <button onclick="send()">Send</button>
            <div id="chat"></div>
        </div>

        <script>
            function send() {
                let msg = document.getElementById("msg").value;
                fetch("/chat", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({message: msg})
                })
                .then(res => res.json())
                .then(data => {
                    document.getElementById("chat").innerHTML += 
                        "<p><b>You:</b> " + msg + "</p>" +
                        "<p><b>Bot:</b> " + data.response + "</p>";
                });
            }
        </script>
    </body>
    </html>
    """)

# ✅ Chat API route
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    reply = get_response(user_message)
    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(debug=True)

