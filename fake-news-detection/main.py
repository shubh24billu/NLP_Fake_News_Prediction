import os
from src.preprocessing import (prepare_data, load_clean_data)
from src.train_ml import train_and_save_ml_models
# from src.train_dl import train_and_save_lstm  # TensorFlow not compatible with Python 3.14
from src.explainability import plot_confusion_matrix, plot_accuracy_comparison


def main():
    # Change to the project root directory
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    os.makedirs("fake-news-detection/data/processed", exist_ok=True)
    os.makedirs("fake-news-detection/models", exist_ok=True)

    # Day 1: Data + Preprocessing
    df = prepare_data(
        fake_path="fake-news-detection/data/raw/Fake.csv",
        true_path="fake-news-detection/data/raw/True.csv",
        output_path="fake-news-detection/data/processed/cleaned_fake_news.csv",
    )

    # Day 2: TF-IDF + ML Models
    ml_results = train_and_save_ml_models("fake-news-detection/data/processed/cleaned_fake_news.csv")

    # Day 3: LSTM model - SKIPPED (TensorFlow not compatible with Python 3.14)
    # dl_results = train_and_save_lstm("data/processed/cleaned_fake_news.csv")

    # Day 4: Evaluation/Explainability charts
    plot_confusion_matrix(ml_results['best_model'], ml_results['X_test'], ml_results['y_test'])
    # plot_accuracy_comparison(ml_results['metrics'], dl_results['metrics'])  # Skipped DL results

    print("\nPipeline completed. Day 5: run streamlit app/app.py for deployment")


if __name__ == "__main__":
    main()
