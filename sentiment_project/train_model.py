# train_model.py

import pandas as pd
import numpy as np
import re
import nltk
import pickle

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# 🔹 Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# 🔹 Load Dataset
# Replace with your dataset path
data = pd.read_csv("dataset.csv")

# Ensure columns exist
data = data[['text', 'sentiment']]

# Clean text
data['clean_text'] = data['text'].apply(clean_text)

# 🔹 Convert text to vectors
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['clean_text'])

# 🔹 Encode labels
# positive=2, neutral=1, negative=0
label_map = {"negative": 0, "neutral": 1, "positive": 2}
data['sentiment'] = data['sentiment'].map(label_map)

y = data['sentiment']

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 🔹 Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 🔹 Save Model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model saved successfully!")