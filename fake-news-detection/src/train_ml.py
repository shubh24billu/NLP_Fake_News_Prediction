import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def train_and_save_ml_models(clean_path: str):
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"Cleaned dataset missing: {clean_path}")

    df = pd.read_csv(clean_path)
    if "content" not in df.columns:
        raise ValueError("cleaned dataset must contain content column")

    X = df["content"].astype(str)
    y = df["label"].astype(int)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vect = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vect, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "MultinomialNB": MultinomialNB(),
        "SVM": LinearSVC(max_iter=5000, random_state=42),
    }

    results = {}
    best_model = None
    best_acc = 0.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"\n=== {name} Evaluation ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 score: {f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    os.makedirs("models", exist_ok=True)

    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("models/best_ml_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print(f"\nBest ML model: {best_name} with Accuracy {best_acc:.4f}")

    return {
        "best_model": best_model,
        "vectorizer": vectorizer,
        "X_test": X_test,
        "y_test": y_test,
        "metrics": results,
    }
