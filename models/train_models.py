import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ✅ Load dataset
df = pd.read_csv("data/processed_data.csv")

# ✅ Check required columns
if not all(col in df.columns for col in ["clean_text", "target"]):
    raise ValueError(f"Dataset must contain 'clean_text' and 'target' columns. Found columns: {df.columns}")

X = df["clean_text"]
y = df["target"]

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# ✅ Save vectorizer
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

# ✅ Train Models
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)
nb_accuracy = accuracy_score(y_test, nb_pred)
pickle.dump(nb_model, open("models/naive_bayes_model.pkl", "wb"))

log_model = LogisticRegression(max_iter=500)
log_model.fit(X_train_tfidf, y_train)
log_pred = log_model.predict(X_test_tfidf)
log_accuracy = accuracy_score(y_test, log_pred)
pickle.dump(log_model, open("models/logistic_regression_model.pkl", "wb"))

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_tfidf, y_train)
knn_pred = knn_model.predict(X_test_tfidf)
knn_accuracy = accuracy_score(y_test, knn_pred)
pickle.dump(knn_model, open("models/knn_model.pkl", "wb"))

# ✅ Save accuracy scores
with open("results/accuracy_scores.txt", "w") as f:
    f.write(f"Naive Bayes Accuracy: {nb_accuracy:.4f}\n")
    f.write(f"Logistic Regression Accuracy: {log_accuracy:.4f}\n")
    f.write(f"KNN Accuracy: {knn_accuracy:.4f}\n")

# ✅ Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Disaster", "Disaster"], yticklabels=["Non-Disaster", "Disaster"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"results/confusion_matrix_{model_name}.png")
    plt.close()

# ✅ Generate Confusion Matrices
plot_confusion_matrix(y_test, nb_pred, "Naive_Bayes")
plot_confusion_matrix(y_test, log_pred, "Logistic_Regression")
plot_confusion_matrix(y_test, knn_pred, "KNN")

print("✅ Models trained, accuracy saved, and confusion matrices generated successfully!")

