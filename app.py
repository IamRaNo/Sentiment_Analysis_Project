import re
import dill
import pickle
import contractions
import spacy
import sklearn
import streamlit as st

# Page configuration
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title('🧠 Sentiment Analysis App')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    import en_core_web_md
    nlp = en_core_web_md.load()

# Load necessary files
with open('slang_dict.dill', 'rb') as f:
    slang_dict = dill.load(f)

with open('punctuations.dill', 'rb') as f:
    punctuations = dill.load(f)

with open('stop_words.dill', 'rb') as f:
    stop_words = dill.load(f)

with open('predict_new.dill', 'rb') as f:
    predict_new = dill.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)


# Preprocessing Function
def pre_processing(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = contractions.fix(text).strip()
    words = text.split()
    words = [slang_dict.get(word, word) for word in words]
    text = " ".join(words)

    doc = nlp(text)
    processed = [
        token.lemma_.lower()
        for token in doc
        if token.text not in stop_words and token.text not in punctuations
    ]
    return " ".join(processed)


# Prediction Function
def predict_sentiment(text):
    cleaned_text = pre_processing(text)
    vector = vectorizer.transform([cleaned_text]).toarray()
    prediction = classifier.predict(vector)
    return encoder.classes_[prediction][0]


# Streamlit UI
user_input = st.text_area("📝 Enter your text below:", height=200)

if st.button('🔍 Predict Sentiment'):
    if user_input.strip() == "":
        st.warning("Please enter some text before predicting.")
    else:
        result = predict_sentiment(user_input)
        st.success(f"💬 Sentiment: **{result}**")
