# 🌍 Fake World News Detection Model

A machine learning project that detects **fake world news** using Natural Language Processing (NLP) and classification algorithms.

## 🧠 Overview
Fake news is a growing issue, especially in global and political contexts. This model analyzes news headlines and articles to predict if they're **real** or **fake**, focusing on *world news* content.

## ⚙️ Technologies Used
- Python
- Scikit-learn
- NLTK
- TfidfVectorizer
- Logistic Regression / Naive Bayes
- Streamlit (for UI)

## 📁 Dataset
[Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  
*Filtered for world news headlines and stories*

## 🚀 Features
- Preprocessing of text (removal of stop words, stemming, etc.)
- Vectorization using TF-IDF
- Fake/Real classification using ML models
- Simple Streamlit UI for prediction

## 🔮 Example
Input:
> "Breaking: Global leaders gather in Rome for secret summit"

Prediction:
> **Fake**

## How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
---
