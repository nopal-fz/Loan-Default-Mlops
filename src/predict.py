import joblib
import pandas as pd

MODEL_PATH = "model/credit_risk_pipeline.joblib"

# function load model
def load_model():
    return joblib.load(MODEL_PATH)

# function predict single data
def predict_single(input_data: dict) -> dict:
    pipeline = load_model()

    input_df = pd.DataFrame([input_data])
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0]

    label = "good_credit" if prediction == 1 else "bad_credit"

    return {
        "prediction": int(prediction),
        "label": label,
        "probability_good_credit": float(probability[1]),
        "probability_bad_credit": float(probability[0])
    }