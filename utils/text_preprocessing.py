import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords (run only once)
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    """Function to clean tweet text"""
    if not isinstance(text, str):  # Handle NaN values or non-string values
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions (@username)
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = " ".join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

if __name__ == "__main__":
    try:
        # Load dataset
        df = pd.read_csv("data/dataset.csv")

        # Check if dataset is empty
        if df.empty:
            print("❌ ERROR: dataset.csv is empty!")
        else:
            print("✅ dataset.csv loaded successfully.")

         # Apply text preprocessing
        df["clean_text"] = df["tweets"].apply(preprocess_text)  # Make sure the column name matches!

        # Save cleaned dataset
        df.to_csv("data/processed_data.csv", index=False)
        print("✅ Processed data saved as 'data/processed_data.csv'")

    except FileNotFoundError:
        print("❌ ERROR: dataset.csv not found in 'data/' folder!")