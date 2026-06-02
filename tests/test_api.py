from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# test untuk health check endpoint
def test_health_check():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

# test untuk predict endpoint
def test_predict_endpoint():
    payload = {
        "checkingAccount": "A11",
        "duration": 6,
        "creditHistory": "A34",
        "purpose": "A43",
        "creditAmount": 1169,
        "savingsAcc": "A65",
        "employmentTime": "A75",
        "installmentPercIncome": 4,
        "status": "A93",
        "otherFin": "A101",
        "timeResidence": 4,
        "property": "A121",
        "age": 67,
        "otherInstallPlans": "A143",
        "housing": "A152",
        "numCredits": 2,
        "job": "A173",
        "numOfMant": 1,
        "phone": "A192",
        "foreignWorker": "A201"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "label" in result
    assert "probability_good_credit" in result
    assert "probability_bad_credit" in result

    assert result["prediction"] in [0, 1]
    assert result["label"] in ["good_credit", "bad_credit"]