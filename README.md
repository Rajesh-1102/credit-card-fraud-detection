# Credit Card Fraud Detection

A Django-based web application for detecting fraudulent credit card transactions using a trained machine learning model. The project includes a dashboard, prediction form, prediction history, JSON API endpoint, and the original model training pipeline.

## Features

- Fraud monitoring dashboard
- Transaction fraud prediction form
- Prediction history stored in SQLite
- JSON API endpoint for transaction scoring
- Login and logout system using Django authentication
- Light and dark dashboard theme toggle
- Risk category and analyst guidance after prediction
- History filter for fraud and legitimate predictions
- Machine learning pipeline for preprocessing, training, and evaluation
- Django frontend templates with custom styling

## Tech Stack

- Python
- Django
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Matplotlib
- Seaborn

## Project Structure

```text
credit card fraud detection/
|-- fraud_app/              # Django app
|-- fraud_project/          # Django project settings
|-- src/                    # ML pipeline source code
|-- static/                 # CSS and static assets
|-- templates/              # Django HTML templates
|-- manage.py               # Django management script
|-- requirements.txt        # Python dependencies
`-- README_DJANGO.md        # Extra Django usage notes
```

## Installation

1. Clone the repository:

```powershell
git clone https://github.com/Rajesh-1102/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Apply database migrations:

```powershell
python manage.py migrate
```

5. Create a login user:

```powershell
python manage.py createsuperuser
```

## Run the Django App

Start the development server:

```powershell
python manage.py runserver
```

Open the app in your browser:

```text
http://127.0.0.1:8000/
```

Login with the username and password created using `createsuperuser`.

## Pages

- `/` - Fraud monitoring dashboard
- `/predict/` - Transaction scoring form
- `/history/` - Saved prediction history
- `/api/predict/` - JSON prediction API

The prediction page includes one-click sample loading for legitimate, fraud, and high-amount transactions. This helps fill all model feature values without typing them manually.

## API Example

Send a POST request to:

```text
http://127.0.0.1:8000/api/predict/
```

Example JSON body:

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

## Machine Learning Pipeline

The original training pipeline is available in the `src/` directory. It contains modules for:

- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Prediction logic
- Database handling

## Notes

Large generated files such as datasets, model artifacts, logs, database files, CSV files, PNG files, and pickle files are ignored using `.gitignore`.

## Author

Rajesh
