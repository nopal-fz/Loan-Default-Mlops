from src.predict import predict_single

# test untuk memastikan fungsi predict_single mengembalikan hasil dengan schema yang benar
def test_predict_single_return_schema():
    sample_input = {
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

    result = predict_single(sample_input)

    assert isinstance(result, dict)

    assert "prediction" in result
    assert "label" in result
    assert "probability_good_credit" in result
    assert "probability_bad_credit" in result

    assert result["prediction"] in [0, 1]
    assert result["label"] in ["good_credit", "bad_credit"]

    assert 0 <= result["probability_good_credit"] <= 1
    assert 0 <= result["probability_bad_credit"] <= 1