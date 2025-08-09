import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

data = [
    ('Free entry in 2 a wkly comp to win FA Cup final tkts', 1),
    ('Upto 20% discount on your next purchase', 1),
    ('Hey, are we still meeting for lunch today?', 0),
    ('Please find the attached report for your review', 0),
    ('WINNER!! Click here to claim your prize', 1),
    ('Can you call me when you are free?', 0),
    ('Lowest prices guaranteed, buy now', 1),
    ('Project deadline is next Monday — please submit', 0),
    ('Congratulations, you have been selected for a cash prize', 1),
    ('Let us catch up this weekend', 0)
]

df = pd.DataFrame(data, columns=['message', 'label'])

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved!")
