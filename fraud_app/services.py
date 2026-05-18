from functools import lru_cache
import sqlite3

from django.conf import settings
from django.utils import timezone

FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

MODEL_PATH = settings.BASE_DIR / "models" / "fraud_model.pkl"
SCALER_PATH = settings.BASE_DIR / "models" / "scaler.pkl"
RAW_DATA_PATH = settings.BASE_DIR / "data" / "raw" / "creditcard.csv"
PROCESSED_DATA_PATH = settings.BASE_DIR / "data" / "processed" / "processed.csv"
DB_PATH = settings.BASE_DIR / "database" / "fraud.db"
CLASS_OPTIONS = {
    "all": "All transactions",
    "fraud": "Fraud only",
    "legitimate": "Legitimate only",
}

ASSISTANT_PROMPTS = [
    {
        "key": "dashboard",
        "label": "Explain dashboard",
        "question": "Explain the dashboard in simple words",
    },
    {
        "key": "prediction",
        "label": "Explain prediction",
        "question": "How does the prediction page work?",
    },
    {
        "key": "features",
        "label": "Explain V1-V28",
        "question": "What are V1 to V28 features?",
    },
    {
        "key": "model",
        "label": "Explain model",
        "question": "How does the machine learning model work?",
    },
    {
        "key": "viva",
        "label": "Viva answer",
        "question": "How should I explain this project to external examiner?",
    },
]


class ProjectDependencyError(RuntimeError):
    pass


def _import_or_raise(package_name, import_name=None):
    try:
        module = __import__(import_name or package_name)
    except ImportError as exc:
        raise ProjectDependencyError(
            f"Could not import {package_name}: {exc}. "
            "Install dependencies with `python -m pip install -r requirements.txt`."
        ) from exc
    return module


@lru_cache(maxsize=1)
def load_model():
    joblib = _import_or_raise("joblib")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_scaler():
    joblib = _import_or_raise("joblib")
    return joblib.load(SCALER_PATH)


def normalize_features(payload):
    features = {}
    for name in FEATURE_ORDER:
        raw_value = payload.get(name, 0)
        if raw_value in ("", None):
            raw_value = 0
        try:
            features[name] = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number.") from exc
    return features


def predict_transaction(payload, save=True):
    pandas = _import_or_raise("pandas")
    features = normalize_features(payload)
    df = pandas.DataFrame([features], columns=FEATURE_ORDER)

    scaler = load_scaler()
    model = load_model()
    scaled = scaler.transform(df)

    prediction = int(model.predict(scaled)[0])
    probability = float(model.predict_proba(scaled)[0][1])

    if save:
        save_transaction(prediction, probability)

    return {
        "prediction": prediction,
        "label": "Fraud" if prediction else "Legitimate",
        "probability": probability,
        "risk_category": risk_category(probability),
        "risk_message": risk_message(prediction, probability),
        "features": features,
    }


def risk_category(probability):
    if probability >= 0.75:
        return "High risk"
    if probability >= 0.35:
        return "Medium risk"
    return "Low risk"


def risk_message(prediction, probability):
    if prediction:
        return "Review this transaction before approval and verify customer identity."
    if probability >= 0.35:
        return "Prediction is legitimate, but the score is elevated enough for analyst review."
    return "Transaction pattern looks consistent with legitimate activity."


def _dataset_path():
    path = RAW_DATA_PATH if RAW_DATA_PATH.exists() else PROCESSED_DATA_PATH
    if not path.exists():
        raise FileNotFoundError("Credit card dataset was not found in data/raw or data/processed.")
    return path


def load_transaction_dataset(usecols=None):
    pandas = _import_or_raise("pandas")
    return pandas.read_csv(_dataset_path(), usecols=usecols)


def get_sample_choices():
    df = load_transaction_dataset(usecols=FEATURE_ORDER + ["Class"])
    choices = []

    sample_specs = [
        ("legitimate", "Legitimate sample", df[df["Class"] == 0].head(3)),
        ("fraud", "Fraud sample", df[df["Class"] == 1].head(3)),
        ("high", "High amount sample", df.sort_values("Amount", ascending=False).head(3)),
    ]

    for group, label, rows in sample_specs:
        for position, (_, row) in enumerate(rows.iterrows(), start=1):
            choices.append(
                {
                    "key": f"{group}:{position}",
                    "group": group,
                    "label": f"{label} {position}",
                    "amount": float(row["Amount"]),
                    "actual_class": int(row["Class"]),
                    "features": {name: float(row[name]) for name in FEATURE_ORDER},
                }
            )
    return choices


def get_sample_features(sample_key):
    if not sample_key:
        return None
    for choice in get_sample_choices():
        if choice["key"] == sample_key:
            return choice
    return None


def ensure_transaction_table():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction INTEGER,
                probability REAL,
                timestamp TEXT
            )
            """
        )
        conn.commit()


def save_transaction(prediction, probability):
    ensure_transaction_table()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO transactions (prediction, probability, timestamp)
            VALUES (?, ?, ?)
            """,
            (prediction, probability, timezone.localtime().isoformat()),
        )
        conn.commit()


def get_recent_transactions(limit=8, prediction_filter="all"):
    ensure_transaction_table()
    where_clause = ""
    params = []
    if prediction_filter == "fraud":
        where_clause = "WHERE prediction = ?"
        params.append(1)
    elif prediction_filter == "legitimate":
        where_clause = "WHERE prediction = ?"
        params.append(0)
    params.append(limit)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT id, prediction, probability, timestamp
            FROM transactions
            {where_clause}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def get_prediction_counts():
    ensure_transaction_table()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT prediction, COUNT(*) AS total
            FROM transactions
            GROUP BY prediction
            """
        ).fetchall()
    counts = {0: 0, 1: 0}
    counts.update({int(prediction): int(total) for prediction, total in rows})
    return counts


def load_dashboard_metrics(filters=None):
    pandas = _import_or_raise("pandas")
    filters = filters or {}
    usecols = ["Time", "Amount", "Class"]
    df = load_transaction_dataset(usecols=usecols)

    amount_min = _safe_float(filters.get("amount_min"), None)
    amount_max = _safe_float(filters.get("amount_max"), None)
    class_filter = filters.get("class_filter", "all")

    if amount_min is not None:
        df = df[df["Amount"] >= amount_min]
    if amount_max is not None:
        df = df[df["Amount"] <= amount_max]
    if class_filter == "fraud":
        df = df[df["Class"] == 1]
    elif class_filter == "legitimate":
        df = df[df["Class"] == 0]

    total = int(len(df))
    fraud = int(df["Class"].sum())
    legitimate = total - fraud
    fraud_rate = fraud / total if total else 0
    fraud_amount = float(df.loc[df["Class"] == 1, "Amount"].sum())
    avg_amount = float(df["Amount"].mean())

    amount_bins = pandas.cut(
        df["Amount"],
        bins=[-0.01, 10, 50, 100, 250, 500, 1000, float("inf")],
        labels=["0-10", "10-50", "50-100", "100-250", "250-500", "500-1k", "1k+"],
    )
    amount_distribution = [
        {"label": str(label), "count": int(count)}
        for label, count in amount_bins.value_counts(sort=False).items()
    ]

    class_distribution = [
        {"label": "Legitimate", "count": legitimate},
        {"label": "Fraud", "count": fraud},
    ]

    hour_series = (df["Time"] // 3600).astype(int) % 24
    fraud_by_hour = (
        df.loc[df["Class"] == 1]
        .assign(hour=hour_series[df["Class"] == 1])
        .groupby("hour")
        .size()
        .reindex(range(24), fill_value=0)
    )
    hourly_fraud = [{"hour": int(hour), "count": int(count)} for hour, count in fraud_by_hour.items()]

    time_bins = pandas.cut(df["Time"], bins=12, duplicates="drop")
    transaction_trend = [
        {"label": f"T{index + 1}", "count": int(count)}
        for index, count in enumerate(time_bins.value_counts(sort=False).tolist())
    ]

    top_transactions = (
        df.sort_values("Amount", ascending=False)
        .head(8)[["Time", "Amount", "Class"]]
        .to_dict("records")
    )

    prediction_counts = get_prediction_counts()

    return {
        "total": total,
        "fraud": fraud,
        "legitimate": legitimate,
        "fraud_rate": fraud_rate,
        "fraud_amount": fraud_amount,
        "avg_amount": avg_amount,
        "class_distribution": class_distribution,
        "amount_distribution": amount_distribution,
        "hourly_fraud": hourly_fraud,
        "transaction_trend": transaction_trend,
        "top_transactions": top_transactions,
        "prediction_counts": prediction_counts,
        "filters": {
            "amount_min": "" if amount_min is None else amount_min,
            "amount_max": "" if amount_max is None else amount_max,
            "class_filter": class_filter,
        },
        "class_options": CLASS_OPTIONS,
    }


def _safe_float(value, default):
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_plot_path(name):
    allowed = {
        "confusion_matrix": "confusion_matrix.png",
        "roc_curve": "roc_curve.png",
        "precision_recall": "precision_recall.png",
    }
    filename = allowed.get(name)
    if not filename:
        return None
    path = settings.BASE_DIR / "artifacts" / "plots" / filename
    return path if path.exists() else None


def get_assistant_answer(question):
    cleaned = (question or "").strip()
    lower_question = cleaned.lower()
    metrics = None

    try:
        metrics = load_dashboard_metrics()
    except (FileNotFoundError, ProjectDependencyError):
        metrics = None

    if not cleaned:
        return "Please select a question button or type a question about the dashboard, prediction page, dataset, model, or viva explanation."

    if any(word in lower_question for word in ["dashboard", "kpi", "chart", "filter"]):
        if metrics:
            return (
                "The dashboard is used to understand fraud patterns visually. "
                f"It analyzes {metrics['total']} transactions, shows {metrics['fraud']} fraud cases, "
                f"{metrics['legitimate']} legitimate transactions, and a fraud rate of {metrics['fraud_rate']:.4%}. "
                "KPIs give quick numbers, charts show patterns, filters help focus on selected transaction groups, "
                "and the table shows important transactions. In a presentation, say that the dashboard converts raw ML data into easy business insights."
            )
        return (
            "The dashboard shows KPIs, charts, filters, model activity, and transaction tables. "
            "It helps a user understand fraud patterns without reading raw CSV files."
        )

    if any(word in lower_question for word in ["predict", "prediction", "score", "probability", "risk"]):
        return (
            "The prediction page takes one transaction with 30 input features: Time, V1 to V28, and Amount. "
            "These values are scaled using the saved scaler, then passed to the trained model. "
            "The model returns a class label, either Fraud or Legitimate, and a fraud probability. "
            "The auto-fill buttons load real dataset rows so the user does not need to manually type all 30 values."
        )

    if any(word in lower_question for word in ["v1", "v2", "v28", "feature", "features", "pca"]):
        return (
            "V1 to V28 are transformed transaction features from the credit card dataset. "
            "They are not normal fields like name or card number. They are numerical features created using PCA for privacy. "
            "Because they are difficult to type manually, the project provides auto-fill sample buttons using real dataset rows."
        )

    if any(word in lower_question for word in ["model", "machine learning", "algorithm", "training", "smote"]):
        return (
            "The machine learning part trains models on historical credit card transactions. "
            "The target column is Class, where 0 means legitimate and 1 means fraud. "
            "The project preprocesses data, scales numerical features, handles class imbalance using SMOTE, trains models, "
            "and saves the best model. During prediction, Django loads that saved model and scaler to classify new transactions."
        )

    if any(word in lower_question for word in ["dataset", "data", "class", "amount", "time"]):
        return (
            "The dataset contains credit card transactions. Time is the transaction time, Amount is the transaction amount, "
            "V1 to V28 are privacy-protected numerical features, and Class is the output label. "
            "Class 0 means legitimate and Class 1 means fraud. The dataset is highly imbalanced because fraud cases are much fewer than normal transactions."
        )

    if any(word in lower_question for word in ["external", "examiner", "viva", "explain", "presentation"]):
        return (
            "You can explain it like this: This project detects credit card fraud using machine learning and presents the result in a Django web application. "
            "The dashboard shows fraud statistics and charts, the prediction page classifies a selected transaction, the history page stores previous predictions, "
            "and the AI Assistant explains project concepts in simple language. The main goal is to help users identify suspicious transactions quickly and clearly."
        )

    return (
        "This assistant is designed for project-related questions. You can ask about the dashboard, prediction page, dataset, V1 to V28 features, "
        "machine learning model, fraud probability, or how to explain the project to an external examiner."
    )
