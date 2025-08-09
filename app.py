from flask import Flask, render_template, request
import joblib
import os
from waitress import serve

# Load model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), 'spam_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email']
    transformed_text = vectorizer.transform([email_text])
    prediction = model.predict(transformed_text)[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    return render_template('index.html', prediction=result, email=email_text)

if __name__ == '__main__':
    # For local dev: python app.py
    # For Render: waitress will serve it
    port = int(os.environ.get("PORT", 8080))
    serve(app, host='0.0.0.0', port=port)
