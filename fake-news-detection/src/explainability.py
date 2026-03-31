import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from lime.lime_text import LimeTextExplainer


def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(y_pred, "toarray"):
        y_pred = y_pred.toarray()
    y_pred_labels = (y_pred > 0.5).astype(int).flatten() if y_pred.ndim > 1 and y_pred.shape[1] == 1 else y_pred

    cm = confusion_matrix(y_test, y_pred_labels)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/confusion_matrix.png")
    plt.close()
    print("Saved confusion matrix to plots/confusion_matrix.png")


def plot_accuracy_comparison(ml_metrics: dict, dl_metrics: dict):
    names = []
    accuracies = []

    for key, details in ml_metrics.items():
        names.append(key)
        accuracies.append(details.get("accuracy", 0))

    for key, details in dl_metrics.items():
        names.append(key)
        accuracies.append(details.get("accuracy", 0))

    plt.figure(figsize=(8, 5))
    sns.barplot(x=names, y=accuracies)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    os.makedirs("plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig("plots/accuracy_comparison.png")
    plt.close()
    print("Saved accuracy comparison to plots/accuracy_comparison.png")


def explain_instance(text: str):
    with open("models/best_ml_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)

    class_names = ["Fake", "Real"]
    explainer = LimeTextExplainer(class_names=class_names)

    def predict_proba(texts):
        X = vec.transform(texts)
        probs = model.predict_proba(X)
        return probs

    exp = explainer.explain_instance(text, predict_proba, num_features=10)
    print("LIME explanation for input text:")
    print(exp.as_list())
    return exp
