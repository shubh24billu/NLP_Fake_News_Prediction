# Fake News Detection using NLP

End-to-end project for fake news detection with structured Day 1-5 implementation.

## Project Structure

```
fake-news-detection/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── models/
│
├── src/
│   ├── preprocessing.py
│   ├── train_ml.py
│   ├── train_dl.py
│   ├── explainability.py
│
├── app/
│   └── app.py
│
├── main.py
├── requirements.txt
└── README.md
```

## Setup

1. Place `Fake.csv` and `True.csv` in `data/raw/`.
2. Create virtualenv and install dependencies:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

3. Run the pipeline:

```bash
python main.py
```

4. Run Streamlit app:

```bash
streamlit run app/app.py
```
