import contractions
import re
import dill
import pickle
import spacy
import sklearn

import streamlit as st

st.title('Sentiment Analysis')


def pre_processing(text):
    text = text.lower()
    url_regex = re.compile(r"https?://\S+|www\.\S+")
    text = url_regex.sub('', text)
    text = text.replace('`', "'").strip()
    text = text.strip()
    text = contractions.fix(text).lower()
    for item in text.split():
        if item in slang_dict.keys():
            text = text.replace(item, slang_dict.get(item))
    doc = nlp(text)
    processed_text = []
    for tokens in doc:
        if tokens.text not in stop_words and tokens.text not in punctuations:
            processed_text.append(tokens.lemma_.lower())
    return " ".join(processed_text)


nlp = spacy.load('en_core_web_md')
slang_dict = dill.load(open('slang_dict.dill', 'rb'))
punctuations = dill.load(open('punctuations.dill', 'rb'))
stop_words = dill.load(open('stop_words.dill', 'rb'))
predict_new = dill.load(open('predict_new.dill', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
classifier = pickle.load(open('classifier.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))


def predictNew(new_text):
    processed_text = pre_processing(new_text)
    processed_text = vectorizer.transform([processed_text]).toarray()
    pred = classifier.predict(processed_text)
    return encoder.classes_[pred][0]


element = st.text_area("Enter your text")

if st.button('Predict'):
    prediction = (predictNew(pre_processing(element)))
    st.write(prediction)
