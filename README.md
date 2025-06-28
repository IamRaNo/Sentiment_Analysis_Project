# 🧠 Sentiment Analysis with NLP & Streamlit

A **Sentiment Analysis** app to classify text as **Positive**, **Negative**, or **Neutral** using advanced NLP preprocessing and a custom **Stacking Classifier**.

## 📌 Project Objective

> **Goal:** Build an end-to-end sentiment analysis workflow — from raw text preprocessing to live predictions — and deploy it with an interactive **Streamlit** web app.

## 📂 Dataset

- 📥 **Source:** [Kaggle](https://www.kaggle.com/)  
- Contains user-generated text samples labeled by sentiment.

## 🧩 Model

- A **Stacking Classifier** combining:
  - **Logistic Regression**
  - **SGD Classifier**
  - **Multinomial Naive Bayes**

## ⚙️ Highlights

- 📚 Custom **text cleaning pipeline** with:
  - Slang dictionary expansion
  - Contraction fixing (e.g., *don't → do not*)
  - Stop word & punctuation removal
  - Lemmatization using **spaCy**
- ✅ Handles user short forms and real-world messy text
- 🔍 Simple, clear prediction interface powered by **Streamlit**

## 🚀 How to Run Locally

1️⃣ **Clone the repo**
```bash
git clone https://github.com/yourusername/sentiment-analysis-app.git
cd sentiment-analysis-app

