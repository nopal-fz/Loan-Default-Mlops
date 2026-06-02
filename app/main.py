from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_single

app = FastAPI(
    title="Credit Risk Prediction API",
    version="1.0.0"
)

# pydantic model untuk request body
class CreditRequest(BaseModel):
    checkingAccount: str
    duration: int
    creditHistory: str
    purpose: str
    creditAmount: int
    savingsAcc: str
    employmentTime: str
    installmentPercIncome: int
    status: str
    otherFin: str
    timeResidence: int
    property: str
    age: int
    otherInstallPlans: str
    housing: str
    numCredits: int
    job: str
    numOfMant: int
    phone: str
    foreignWorker: str

# endpoint untuk health check
@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "message": "Credit Risk Prediction API is running"
    }

# endpoint untuk prediksi single data
@app.post("/predict")
def predict(data: CreditRequest):
    result = predict_single(data.model_dump())
    return result