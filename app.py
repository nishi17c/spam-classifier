from flask import Flask, render_template, request, jsonify
import joblib
import os

app = Flask(__name__)

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    email_text = data.get("text", "")
    X = vectorizer.transform([email_text])
    prediction = model.predict(X)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=8080)
