from flask import Flask, request, jsonify, render_template
import joblib, json, os

# Load model & vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

# ✅ Single prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]

    confidence = round(float(proba[prediction]) * 100, 2)

    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = vec.toarray()[0]
    spammy_words = [feature_names[i] for i, score in enumerate(tfidf_scores) if score > 0.2]

    highlighted_text = text
    for w in spammy_words:
        highlighted_text = highlighted_text.replace(
            w, f"<span style='color:red;font-weight:bold;'>{w}</span>"
        )

    return jsonify({
        "prediction": int(prediction),
        "confidence": confidence,
        "spammy_words": spammy_words,
        "highlighted_text": highlighted_text
    })

# ✅ Batch prediction endpoint
@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    data = request.get_json()
    texts = data.get("texts", [])
    results = []

    for text in texts:
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        confidence = round(float(proba[prediction]) * 100, 2)

        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = vec.toarray()[0]
        spammy_words = [feature_names[i] for i, score in enumerate(tfidf_scores) if score > 0.2]

        highlighted_text = text
        for w in spammy_words:
            highlighted_text = highlighted_text.replace(
                w, f"<span style='color:red;font-weight:bold;'>{w}</span>"
            )

        results.append({
            "text": text,
            "prediction": int(prediction),
            "confidence": confidence,
            "spammy_words": spammy_words,
            "highlighted_text": highlighted_text
        })

    return jsonify(results)

# ✅ Insights route
@app.route("/insights")
def insights():
    if not os.path.exists("model_insights.json"):
        return "⚠️ No insights found. Please run train_model.py first.", 500
    with open("model_insights.json", "r") as f:
        insights = json.load(f)
    return render_template("insights.html", insights=insights)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
