# Credit Card Fraud Detection - Django App

This project now includes a Django web application for the existing fraud detection model and data artifacts.

## Run locally

```powershell
pip install -r requirements.txt
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000/
```

## Pages

- `/` - professional fraud monitoring dashboard
- `/predict/` - transaction scoring form using the trained model
- `/history/` - saved prediction history from `database/fraud.db`
- `/api/predict/` - JSON prediction endpoint

## JSON prediction example

Send a POST request to `/api/predict/` with the 30 model features:

```json
{
  "Time": 0,
  "V1": 0,
  "V2": 0,
  "V3": 0,
  "V4": 0,
  "V5": 0,
  "V6": 0,
  "V7": 0,
  "V8": 0,
  "V9": 0,
  "V10": 0,
  "V11": 0,
  "V12": 0,
  "V13": 0,
  "V14": 0,
  "V15": 0,
  "V16": 0,
  "V17": 0,
  "V18": 0,
  "V19": 0,
  "V20": 0,
  "V21": 0,
  "V22": 0,
  "V23": 0,
  "V24": 0,
  "V25": 0,
  "V26": 0,
  "V27": 0,
  "V28": 0,
  "Amount": 120.0
}
```

The original training pipeline under `src/` is still available for retraining.
