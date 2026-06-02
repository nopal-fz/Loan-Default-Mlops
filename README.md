# Credit Risk Prediction API with CI/CD

## Overview

This project demonstrates how machine learning models can be integrated into a production-style workflow using FastAPI, automated testing, and GitHub Actions.

The application predicts whether a loan applicant is likely to be a good or bad credit risk based on financial and demographic information.

Beyond model training, this project focuses on software engineering and MLOps practices such as:

* Reusable preprocessing pipelines
* Model evaluation and quality gates
* REST API serving with FastAPI
* Automated testing using Pytest
* Continuous Integration (CI) with GitHub Actions
* Branch and Pull Request workflow

---

## Problem Statement

Financial institutions need a reliable way to assess credit risk before approving loans.

This project uses the German Credit dataset to build a machine learning model capable of predicting creditworthiness and exposing predictions through a REST API.

---

## Project Architecture

Dataset
→ Preprocessing Pipeline
→ Model Training
→ Model Evaluation
→ FastAPI Service
→ Automated Tests
→ GitHub Actions CI Pipeline

---

## Project Structure

```text
loan-default-mlops/
│
├── app/
│   └── main.py
│
├── data/
│
├── model/
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
│   ├── test_api.py
│   └── test_predict.py
│
├── .github/workflows/
│   └── ci.yml
│
└── README.md
```

---

## Features

### Model Training

* Data preprocessing using Scikit-Learn pipelines
* One-hot encoding for categorical features
* Random Forest classification

### Model Evaluation

Quality gates are implemented during evaluation:

* Accuracy threshold validation
* F1-score threshold validation

If performance drops below the threshold, the CI pipeline fails automatically.

### REST API

Available endpoints:

#### Health Check

```http
GET /
```

#### Credit Risk Prediction

```http
POST /predict
```

Example request:

```json
{
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
```

---

## Continuous Integration

GitHub Actions automatically runs:

1. Model Training
2. Model Evaluation
3. API Tests
4. Prediction Tests

The workflow prevents broken code or failing tests from being merged into the main branch.

---

## Running Locally

### Create Virtual Environment

```bash
uv venv
```

### Activate Environment

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python -m src.train
```

### Evaluate Model

```bash
python -m src.evaluate
```

### Run API

```bash
uvicorn app.main:app --reload
```

### Run Tests

```bash
python -m pytest
```

---

## Future Improvements

* Docker containerization
* Automated deployment
* Model versioning
* Monitoring and logging
* Model performance tracking

---

## Tech Stack

* Python
* Scikit-Learn
* FastAPI
* Pytest
* GitHub Actions
* uv
