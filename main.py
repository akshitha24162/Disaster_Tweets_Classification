import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords")

# ✅ Load the same TF-IDF vectorizer used during training
try:
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    print("✅ Loaded tfidf_vectorizer.pkl")
except FileNotFoundError:
    raise FileNotFoundError("❌ ERROR: 'models/tfidf_vectorizer.pkl' not found! Train your model first.")

# ✅ Load trained models
try:
    nb_model = joblib.load("models/naive_bayes_model.pkl")
    lr_model = joblib.load("models/logistic_regression_model.pkl")
    knn_model = joblib.load("models/knn_model.pkl")
    print("✅ Loaded all models successfully!")
except FileNotFoundError as e:
    raise FileNotFoundError(f"❌ ERROR: {e}. Train your models first.")

# ✅ Example input tweet
example_text = ["Breaking: Fire outbreak in California, thousands evacuated!"]

# ✅ Transform input text using the SAME vectorizer
example_tfidf = vectorizer.transform(example_text)  # ✅ Fix: Use the same TF-IDF vectorizer

# ✅ Make predictions
print("\n🔍 Predictions:")
print(f"Naive Bayes: {nb_model.predict(example_tfidf)[0]}")
print(f"Logistic Regression: {lr_model.predict(example_tfidf)[0]}")
print(f"KNN: {knn_model.predict(example_tfidf)[0]}")
