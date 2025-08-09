from flask import Flask, request, render_template_string
import joblib

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

# Simple HTML template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Spam Classifier</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f4f4; }
        .container { max-width: 500px; margin: auto; background: white; padding: 20px; border-radius: 8px; }
        h2 { text-align: center; }
        textarea { width: 100%; padding: 10px; margin: 10px 0; }
        button { width: 100%; padding: 10px; background: #007bff; color: white; border: none; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { font-size: 18px; margin-top: 15px; text-align: center; }
        .spam { color: red; }
        .not-spam { color: green; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Spam Classifier</h2>
        <form method="post">
            <textarea name="message" rows="5" placeholder="Enter your message here..." required></textarea>
            <button type="submit">Check</button>
        </form>
        {% if result %}
            <div class="result {{ css_class }}">{{ result }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    css_class = ""
    if request.method == "POST":
        message = request.form["message"]
        message_vec = vectorizer.transform([message])
        prediction = model.predict(message_vec)[0]
        if prediction == 1:
            result = "Spam Message 🚫"
            css_class = "spam"
        else:
            result = "Not Spam ✅"
            css_class = "not-spam"
    return render_template_string(html_template, result=result, css_class=css_class)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
