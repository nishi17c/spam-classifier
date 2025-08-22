import pandas as pd
import joblib
import urllib.request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os
import numpy as np
import json

# ----------------------------
# Download dataset if not present
# ----------------------------
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
dataset_zip = "smsspamcollection.zip"
dataset_file = "SMSSpamCollection"

if not os.path.exists(dataset_file):
    print("Downloading dataset...")
    urllib.request.urlretrieve(dataset_url, dataset_zip)
    import zipfile
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall()
    os.remove(dataset_zip)
    print("Dataset downloaded and extracted.")

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv(dataset_file, sep='\t', header=None, names=['label', 'message'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# ----------------------------
# TF-IDF Vectorization
# ----------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------
# Train Model (Naive Bayes)
# ----------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ----------------------------
# Save model and vectorizer
# ----------------------------
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# ----------------------------
# Evaluate Model
# ----------------------------
y_pred = model.predict(X_test_vec)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))

# ----------------------------
# Feature Insights (Top Spam vs Ham Words)
# ----------------------------
feature_names = vectorizer.get_feature_names_out()
log_prob = model.feature_log_prob_  # [ham, spam]

spam_strength = log_prob[1] - log_prob[0]   # higher = more spammy
ham_strength = log_prob[0] - log_prob[1]    # higher = more hammy

# Get top 15 spammy & hammy words
top_spam_idx = np.argsort(spam_strength)[-15:]
top_ham_idx = np.argsort(ham_strength)[-15:]

top_spam_words = [(feature_names[i], round(spam_strength[i], 3)) for i in top_spam_idx]
top_ham_words = [(feature_names[i], round(ham_strength[i], 3)) for i in top_ham_idx]

print("\n🔴 Top Spam Words:", top_spam_words)
print("\n🟢 Top Ham Words:", top_ham_words)

# Save insights as JSON for Flask
insights = {
    "top_spam_words": top_spam_words[::-1],  # descending
    "top_ham_words": top_ham_words[::-1]
}
with open("model_insights.json", "w") as f:
    json.dump(insights, f, indent=2)

print("\nModel, vectorizer, and insights saved successfully.")
