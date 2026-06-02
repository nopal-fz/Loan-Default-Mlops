import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.load_data import load_data, save_data

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# function untuk membuat preprocessor
def get_preprocessor(df):
    X = df.drop(columns=["target"])

    categorical_features = (
        X.select_dtypes(
            include=["object"]
        )
        .columns
        .tolist()
    )

    numerical_features = (
        X.select_dtypes(
            exclude=["object"]
        )
        .columns
        .tolist()
    )

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numerical_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    return preprocessor

# main function untuk preprocessing data
def main():
    df = load_data()

    data_path = "data"

    os.makedirs(
        data_path,
        exist_ok=True
    )

    save_data(
        df,
        os.path.join(
            data_path,
            "german_credit_data.csv"
        )
    )

    df.columns = [
        "checkingAccount",
        "duration",
        "creditHistory",
        "purpose",
        "creditAmount",
        "savingsAcc",
        "employmentTime",
        "installmentPercIncome",
        "status",
        "otherFin",
        "timeResidence",
        "property",
        "age",
        "otherInstallPlans",
        "housing",
        "numCredits",
        "job",
        "numOfMant",
        "phone",
        "foreignWorker",
        "target"
    ]

    print(df.head())

    print(df.isnull().sum())

    num_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    cat_cols = df.select_dtypes(
        include=["object"]
    ).columns

    print("Numerical columns:", num_cols)
    print("Categorical columns:", cat_cols)

    df.to_csv(
        os.path.join(
            data_path,
            "german_credit_data_preprocessed.csv"
        ),
        index=False
    )

if __name__ == "__main__":
    main()