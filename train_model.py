import pandas as pd
import joblib
import urllib.request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

# Download dataset from UCI if not present
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

# Load dataset
df = pd.read_csv(dataset_file, sep='\t', header=None, names=['label', 'message'])

# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Evaluate model
y_pred = model.predict(X_test_vec)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))

print("\nModel and vectorizer saved successfully.")
