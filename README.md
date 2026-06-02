# Loan Default MLOps

## Project Overview
Credit risk prediction using machine learning.

## Tech Stack
- Python
- Scikit-Learn
- FastAPI
- Pytest
- GitHub Actions

## Run Locally

uv venv
.venv\Scripts\activate

pip install -r requirements.txt

python -m src.train
python -m src.evaluate

uvicorn app.main:app --reload

## Run Test

python -m pytest