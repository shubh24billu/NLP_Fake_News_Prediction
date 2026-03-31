import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def train_and_save_lstm(clean_path: str):
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"Cleaned dataset missing: {clean_path}")

    df = pd.read_csv(clean_path)
    if "content" not in df.columns:
        raise ValueError("cleaned dataset must contain content column")

    X = df["content"].astype(str).values
    y = df["label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    max_words = 10000
    max_len = 250

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    train_seq = tokenizer.texts_to_sequences(X_train)
    test_seq = tokenizer.texts_to_sequences(X_test)

    x_train_pad = pad_sequences(train_seq, maxlen=max_len, padding="post", truncating="post")
    x_test_pad = pad_sequences(test_seq, maxlen=max_len, padding="post", truncating="post")

    model = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(
        x_train_pad,
        y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=2,
    )

    loss, accuracy = model.evaluate(x_test_pad, y_test, verbose=0)
    print(f"LSTM test accuracy: {accuracy:.4f}, test loss: {loss:.4f}")

    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_model.h5")

    import pickle
    with open("models/lstm_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    metrics = {
        "LSTM": {
            "accuracy": float(accuracy),
            "loss": float(loss)
        }
    }

    return {"model": model, "tokenizer": tokenizer, "metrics": metrics}
