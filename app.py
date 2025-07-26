import streamlit as st
import pickle
import re
import nltk
import os

# Add tmp path for nltk
nltk.data.path.append("/tmp")

# Download required packages (only once per deployment)
nltk.download('punkt', download_dir="/tmp")
nltk.download('stopwords', download_dir="/tmp")


from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Download stopwords once
nltk.download('stopwords')

# Load trained model and vectorizer
model = pickle.load(open(r"C:\Users\athar\OneDrive\Documents\Desktop\streamlit\fake news\model_news.sav", "rb"))
vectorizer = pickle.load(open(r"C:\Users\athar\OneDrive\Documents\Desktop\streamlit\fake news\feature_extraction.pkl", "rb"))

# Initialize stemmer and stopwords
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))

# Define preprocessing function
def clean_and_stem(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(stemmed_words)

# Streamlit app interface
st.title("📰 Fake News Prediction System")
st.markdown("Enter a news headline or article and click **Predict** to check if it's Fake or Real.")

# User input
user_input = st.text_area("Enter news text:", height=200)

# Prediction
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess and vectorize input
        processed_input = clean_and_stem(user_input)
        vector_input = vectorizer.transform([processed_input])
        prediction = model.predict(vector_input)

        # Output result
        if prediction[0] == 0:
            st.error("✅Real News Detected!")
        else:
            st.success("❌ Fake News Detected!")
