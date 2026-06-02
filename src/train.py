import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.preprocess import get_preprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DATA_PATH = "data/german_credit_data_preprocessed.csv"
MODEL_PATH = "Loan-Default-Mlops/model/credit_risk_pipeline.joblib"

def main():
    logging.info("Loading dataset")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["target"])
    y = df["target"]
    
    # mapping: 1 = good credit, 2 = bad credit
    y = y.map({1: 1, 2: 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = get_preprocessor(df)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    logging.info("Training model")
    pipeline.fit(X_train, y_train)

    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    logging.info(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()