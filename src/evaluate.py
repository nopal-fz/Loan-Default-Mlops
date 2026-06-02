import logging
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DATA_PATH = "data/german_credit_data_preprocessed.csv"
MODEL_PATH = "model/credit_risk_pipeline.joblib"

def main():
    logging.info("Loading dataset")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["target"])
    y = df["target"]

    # mapp: 1 = good credit, 2 = bad credit
    y = y.map({1: 1, 2: 0})

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logging.info("Loading trained pipeline")
    pipeline = joblib.load(MODEL_PATH)

    logging.info("Running prediction")
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Quality gate untuk CI/CD
    assert accuracy >= 0.65, "Accuracy below threshold"
    assert f1 >= 0.70, "F1 score below threshold"

    logging.info("Model evaluation passed")


if __name__ == "__main__":
    main()