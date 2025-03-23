import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# ✅ Load dataset
df = pd.read_csv("data/dataset.csv")

# ✅ Rename columns to match expected format
df = df.rename(columns={"tweets": "text", "target": "label"})

# ✅ Ensure correct column names
if not {"text", "label"}.issubset(df.columns):
    raise ValueError(f"Dataset must contain 'text' and 'label' columns. Found columns: {df.columns}")

# ✅ Split dataset
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# ✅ Load trained TF-IDF vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
X_test_tfidf = vectorizer.transform(X_test)

# ✅ Load trained models
models = {
    "Naive Bayes": joblib.load("models/naive_bayes_model.pkl"),
    "Logistic Regression": joblib.load("models/logistic_regression_model.pkl"),
    "KNN": joblib.load("models/knn_model.pkl"),
}

# ✅ Dictionary to store accuracy scores
accuracy_scores = {}

# ✅ Evaluate models and save confusion matrices
for model_name, model in models.items():
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores[model_name] = accuracy

    # ✅ Generate and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"results/confusion_matrix_{model_name}.png")  # Save confusion matrix
    plt.close()

# ✅ Save accuracy scores
with open("results/accuracy_scores.txt", "w") as f:
    for model_name, accuracy in accuracy_scores.items():
        f.write(f"{model_name} Accuracy: {accuracy:.4f}\n")

print("✅ Model evaluation complete! Accuracy scores saved in results/accuracy_scores.txt")
print("✅ Confusion matrices saved in results/ directory")

