# Disaster Tweets Classification

## 📌 Project Overview
This project focuses on classifying tweets related to disasters using **Machine Learning and Natural Language Processing (NLP)**. The goal is to differentiate between tweets that describe real disaster events and those that do not.

## 🛠️ Technologies Used
- **Python**
- **Scikit-Learn**
- **Natural Language Processing (NLP)**
- **Pandas, NumPy**
- **NLTK, TF-IDF Vectorization**
- **Machine Learning Models (Logistic Regression, Random Forest, etc.)**

## 📂 Project Structure
```
📦 Disaster_Tweets_Classification
 ┣ 📂 data
 ┃ ┣ 📜 train.csv
 ┃ ┣ 📜 test.csv
 ┣ 📂 notebooks
 ┃ ┣ 📜 EDA.ipynb
 ┃ ┣ 📜 Model_Training.ipynb
 ┣ 📂 src
 ┃ ┣ 📜 preprocessing.py
 ┃ ┣ 📜 model.py
 ┣ 📜 README.md
 ┣ 📜 requirements.txt
 ┣ 📜 app.py (if applicable)
```

## 🚀 Installation & Setup
1. **Clone the repository**
   ```sh
   git clone https://github.com/your-username/Disaster_Tweets_Classification.git
   cd Disaster_Tweets_Classification
   ```
2. **Create a virtual environment** (optional but recommended)
   ```sh
   python -m venv venv
   source venv/bin/activate  # For Mac/Linux
   venv\Scripts\activate  # For Windows
   ```
3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the model training script**
   ```sh
   python src/model.py
   ```

## 🔍 Dataset
The dataset used in this project comes from **Kaggle’s Disaster Tweets dataset**. It contains tweets labeled as **1 (real disaster)** or **0 (not a disaster)**.

## 📊 Exploratory Data Analysis (EDA)
- Data Cleaning (Removing special characters, stopwords, etc.)
- Text Preprocessing (Tokenization, Lemmatization)
- Feature Engineering (TF-IDF, Word Embeddings)

## 🏆 Model Training & Evaluation
- **Supervised Learning Models:**
  - Logistic Regression
  - Random Forest
  - Naïve Bayes
  - Support Vector Machine (SVM)
- **Performance Metrics:**
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix

## 📌 Future Improvements
- Implement **Deep Learning (LSTMs, BERT)** for better accuracy.
- Deploy the model as a **Web Application** using Flask or FastAPI.

## 🤝 Contributing
Feel free to submit pull requests or suggest improvements!

## 📜 License
This project is open-source under the **MIT License**.
