from flask import Flask, request, render_template, jsonify
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

app = Flask(__name__)

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(rf"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Load dataset
file_path = r"C:\Users\PC\OneDrive\Documents\demo\sms_spam.xlsx"
data = pd.read_excel(file_path)

# Prepare dataset
data = data.iloc[:, :2]
data.columns = ["label", "message"]
data["label"] = data["label"].map({'ham': 0, 'spam': 1})
data.dropna(subset=["message"], inplace=True)
data["message"] = data["message"].apply(clean_text)

# Train spam detection model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])
pipeline.fit(data["message"], data["label"])

# Save the model
joblib.dump(pipeline, "spam_detector.pkl")

# Load trained model
model = joblib.load("spam_detector.pkl")

# Define home route
@app.route("/")
def home():
    return render_template("index.html")

# Define prediction API route
@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    cleaned_message = clean_text(message)
    prediction = model.predict([cleaned_message])[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
