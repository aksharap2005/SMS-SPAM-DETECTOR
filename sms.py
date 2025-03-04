import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path = r"C:\Users\PC\OneDrive\Documents\demo\sms_spam.xlsx"
data = pd.read_excel(file_path)

# Ensure the dataset has at least two columns
expected_columns = ["label", "message"]
if len(data.columns) < 2:
    raise ValueError("Dataset does not have enough columns!")

# Keep only the first two relevant columns
data = data.iloc[:, :2]
data.columns = expected_columns

# Validate labels before mapping
valid_labels = {"ham", "spam"}
if not set(data["label"].unique()).issubset(valid_labels):
    raise ValueError("Unexpected values found in the 'label' column!")

# Convert labels to binary (ham = 0, spam = 1)
data["label"] = data["label"].map({'ham': 0, 'spam': 1})

# Handle missing values
data.dropna(subset=["message"], inplace=True)

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(rf"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Apply text cleaning
data["message"] = data["message"].apply(clean_text)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data["message"], data["label"], test_size=0.2, random_state=42
)

# Create a pipeline for text processing and model training
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to predict new messages
def predict_spam(text):
    text = clean_text(text)
    return "Spam" if pipeline.predict([text])[0] else "Not Spam"

# Example usage
print(predict_spam("Congratulations! You have won a free lottery! Call now."))
print(predict_spam("Hey, let's meet up for lunch tomorrow."))
