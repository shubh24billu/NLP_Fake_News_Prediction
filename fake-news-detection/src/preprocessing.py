import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def ensure_nltk():
    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')


def clean_text(text: str, stop_words: set) -> str:
    """Clean a single string with NLP preprocessing."""
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [tok for tok in tokens if tok not in stop_words and len(tok) > 1]

    return " ".join(tokens)


def prepare_data(fake_path: str, true_path: str, output_path: str) -> pd.DataFrame:
    """Load raw data, label, merge, shuffle, clean, and save."""
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Fake data file not found: {fake_path}")
    if not os.path.exists(true_path):
        raise FileNotFoundError(f"True data file not found: {true_path}")

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if not {"title", "text", "label"}.issubset(df.columns):
        raise ValueError("Input files must contain 'title' and 'text' columns")

    df = df[["title", "text", "label"]].copy()
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["content"] = (df["title"] + " " + df["text"]).str.strip()

    ensure_nltk()
    stop_words = set(stopwords.words("english"))

    df["content"] = df["content"].apply(lambda x: clean_text(x, stop_words))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved cleaned data to {output_path}")
    print(f"Data shape after preprocessing: {df.shape}")

    return df


def load_clean_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cleaned data not found: {path}")
    return pd.read_csv(path)
