import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

with open("intents.json") as file:
    intents = json.load(file)

def get_response(user_input):
    tokens = nltk.word_tokenize(user_input.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    text = " ".join(tokens)

    X = vectorizer.transform([text])
    tag = model.predict(X)[0]

    # Find matching intent
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    # Fallback safety
    for intent in intents["intents"]:
        if intent["tag"] == "fallback":
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand that."
