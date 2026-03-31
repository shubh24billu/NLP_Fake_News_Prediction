import streamlit as st
import pickle
import os


def load_models():
    model_path = "models/best_ml_model.pkl"
    vect_path = "models/vectorizer.pkl"

    if not os.path.exists(model_path) or not os.path.exists(vect_path):
        raise FileNotFoundError("Trained model/vectorizer not found. Run main.py first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vect_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def predict_news(model, vectorizer, text):
    x = vectorizer.transform([text])
    pred = model.predict(x)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0][pred]
    return pred, proba


def main():
    st.title("Fake News Detection (NLP)")
    st.write("Enter news title + body text and get fake/real prediction.")

    text = st.text_area("News text", "", height=250)

    if st.button("Predict"):
        try:
            model, vectorizer = load_models()
            pred, proba = predict_news(model, vectorizer, text)

            label = "Real" if pred == 1 else "Fake"
            st.subheader(f"Prediction: {label}")
            if proba is not None:
                st.write(f"Confidence: {proba:.3f}")
        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
