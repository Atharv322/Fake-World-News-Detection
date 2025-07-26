import streamlit as st
import pickle
import nltk
nltk.data.path.append("/home/appuser/nltk_data")
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Load model and vectorizer
model = pickle.load(open("model_news.sav", "rb"))
vectorizer = pickle.load(open("feature_extraction.pkl", "rb"))

# Streamlit interface
st.title("📰 Fake News Detector")
text_input = st.text_area("Enter news text:")

if st.button("Predict"):
    if text_input:
        text_transformed = vectorizer.transform([text_input])
        prediction = model.predict(text_transformed)
        st.success("✅ Real News" if prediction[0] == 0 else "⚠️ Fake News")
