from pathlib import Path
import json
import urllib.request

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SequentialFeatureSelector


# 1. PROJECT PATHS
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

for folder in (DATA, MODELS, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)


# Existing model used by the Django training/comparison pipeline
MODEL_PATH = MODELS / "best_model.joblib"
META_PATH = MODELS / "model_metadata.json"

# Proposed model trained in Google Colab
PROPOSED_MODEL_PATH = MODELS / "heart_disease_rf_sfs.joblib"
PROPOSED_META_PATH = MODELS / "heart_disease_rf_sfs_metadata.json"

# Comparison report
REPORT_PATH = REPORTS / "model_comparison.csv"

URL = "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"


# 2. FEATURES, LABELS AND VALIDATION RANGES
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

LABELS = {
    "age": "Age (years)",
    "sex": "Sex (0=female, 1=male)",
    "cp": "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting blood sugar >120 (0/1)",
    "restecg": "Resting ECG (0-2)",
    "thalach": "Maximum heart rate",
    "exang": "Exercise angina (0/1)",
    "oldpeak": "ST depression",
    "slope": "ST slope (0-2)",
    "ca": "Major vessels (0-3)",
    "thal": "Thal category",
}

RANGES = {
    "age": (1, 120),
    "sex": (0, 1),
    "cp": (0, 3),
    "trestbps": (50, 300),
    "chol": (50, 800),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (40, 250),
    "exang": (0, 1),
    "oldpeak": (0, 10),
    "slope": (0, 2),
    "ca": (0, 3),
    "thal": (0, 7),
}
BINARY_FEATURES = ("sex", "fbs", "exang")


# 3. LOAD AND VALIDATE DATASET
def load_data():
    """Load and validate data/heart.csv, downloading the default CSV if absent."""
    path = DATA / "heart.csv"

    if not path.exists():
        try:
            with urllib.request.urlopen(URL, timeout=20) as response:
                path.write_bytes(response.read())
        except Exception as exc:
            raise RuntimeError(
                f"Dataset download failed. Save a compatible CSV as {path}. "
                f"Details: {exc}"
            ) from exc

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not read dataset CSV: {exc}") from exc

    df.columns = [str(column).strip().lower() for column in df.columns]
    df = df.rename(columns={
        "target": "num",
        "heartdisease": "num",
        "heart_disease": "num",
    })

    required = FEATURES + ["num"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))

    df = df[required].copy()
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["num"])

    if df.empty:
        raise ValueError("The dataset contains no valid target labels.")

    # Target 0 remains 0; positive target values become 1.
    df["num"] = (df["num"] > 0).astype(int)
    if df["num"].nunique() != 2:
        raise ValueError(
            "After target conversion, the dataset must contain both class 0 and class 1."
        )
    return df


# 4. BUILD MACHINE-LEARNING PIPELINES
def make_pipe(estimator, select=False):
    """Build preprocessing and classifier steps inside each training fold."""
    if select:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("sfs", SequentialFeatureSelector(
                estimator=RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1
                ),
                n_features_to_select=6,
                direction="forward",
                scoring="accuracy",
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                n_jobs=1,
            )),
            ("rf", estimator),
        ])

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


# 5. TRAIN AND EVALUATE ALL EIGHT MODELS
def train(test_size=0.20):
    """
    Train eight classifiers. test_size accepts 0.20, 0.25, or 0.30.
    Cross-validation is performed on training data only; test metrics are
    calculated on the held-out test set. Random Forest + SFS remains the
    model saved for manual and CSV prediction.
    """
    try:
        test_size = float(test_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("Test size must be 0.20, 0.25, or 0.30.") from exc

    if not any(np.isclose(test_size, allowed) for allowed in (0.20, 0.25, 0.30)):
        raise ValueError("Choose a testing ratio of 20%, 25%, or 30%.")

    df = load_data()
    X, y = df[FEATURES], df["num"]

    if y.nunique() != 2 or y.value_counts().min() < 5:
        raise ValueError(
            "The dataset must contain both target classes, with at least five records in each."
        )

    print("=" * 65)
    print("HEART DISEASE MODEL TRAINING")
    print("=" * 65)
    print("Dataset shape:", df.shape)
    print("Duplicate rows:", int(df.duplicated().sum()))
    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Testing ratio:", f"{test_size:.0%}")

    # Train only the seven Django comparison models.
    # Random Forest + SFS is preserved from the existing Colab report.
    models = {
        "Logistic Regression": make_pipe(LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        )),
        "Random Forest": make_pipe(RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        )),
        "Gradient Boosting": make_pipe(HistGradientBoostingClassifier(
            max_iter=100, random_state=42
        )),
        "SVM RBF": make_pipe(SVC(probability=True, random_state=42)),
        "KNN": make_pipe(KNeighborsClassifier(n_neighbors=7)),
        "Decision Tree": make_pipe(DecisionTreeClassifier(random_state=42)),
        "LDA": make_pipe(LinearDiscriminantAnalysis()),
    }

    # Keep the existing Colab RF + SFS report row; never retrain that model.
    existing_proposed_row = None
    if REPORT_PATH.exists():
        try:
            old_report = pd.read_csv(REPORT_PATH)
            if "model" in old_report.columns:
                matches = old_report[old_report["model"].astype(str) == "Random Forest + SFS"]
                if not matches.empty:
                    existing_proposed_row = matches.iloc[0].to_dict()
        except Exception:
            existing_proposed_row = None

    if existing_proposed_row is None:
        raise FileNotFoundError(
            f"Could not find 'Random Forest + SFS' in {REPORT_PATH}. "
            "Restore the Colab-generated model_comparison.csv before training."
        )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []

    for name, model in models.items():
        print("\n" + "-" * 60, flush=True)
        print("Training:", name, flush=True)

        try:
            # Keep the held-out test set out of cross-validation.
            scores = cross_validate(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring={
                    "accuracy": "accuracy",
                    "f1": "f1",
                    "roc_auc": "roc_auc",
                },
                n_jobs=1,
                error_score="raise",
            )

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            probabilities = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba") else None
            )

            tn, fp, fn, tp = confusion_matrix(
                y_test, predictions, labels=[0, 1]
            ).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
            specificity = tn / (tn + fp) if (tn + fp) else 0.0

            result = {
                "model": name,
                "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
                "cv_f1_mean": float(scores["test_f1"].mean()),
                "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
                "test_accuracy": float(accuracy_score(y_test, predictions)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
                "precision": float(precision_score(y_test, predictions, zero_division=0)),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "recall": float(recall_score(y_test, predictions, zero_division=0)),
                "f1": float(f1_score(y_test, predictions, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, probabilities))
                    if probabilities is not None else float("nan"),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
            rows.append(result)

            # Save progress alongside the preserved Colab RF + SFS row.
            progress_report = pd.DataFrame([existing_proposed_row] + rows)
            sort_columns = [
                column for column in ("cv_accuracy_mean", "cv_f1_mean")
                if column in progress_report.columns
            ]
            if sort_columns:
                progress_report = progress_report.sort_values(
                    sort_columns, ascending=False, na_position="last"
                )
            progress_report.to_csv(REPORT_PATH, index=False)

            print("CV accuracy:", round(result["cv_accuracy_mean"], 4), flush=True)
            print("Test accuracy:", round(result["test_accuracy"], 4), flush=True)
            print("Sensitivity:", round(result["sensitivity"], 4), flush=True)
            print("Specificity:", round(result["specificity"], 4), flush=True)
            print("ROC-AUC:", round(result["roc_auc"], 4), flush=True)
            print(
                f"Saved progress: {len(rows)}/{len(models)} Django models plus existing Colab RF + SFS",
                flush=True,
            )

        except Exception:
            # Preserve the existing Colab row and any completed Django rows.
            pd.DataFrame([existing_proposed_row] + rows).to_csv(REPORT_PATH, index=False)
            raise

    report = pd.DataFrame([existing_proposed_row] + rows)
    sort_columns = [
        column for column in ("cv_accuracy_mean", "cv_f1_mean")
        if column in report.columns
    ]
    if sort_columns:
        report = report.sort_values(
            sort_columns, ascending=False, na_position="last"
        )
    report.to_csv(REPORT_PATH, index=False)

    print("\\n" + "=" * 65)
    print("MODEL COMPARISON (7 Django models + existing Colab RF + SFS)")
    print("=" * 65)
    print(report.to_string(index=False))
    print("\\nComparison report saved:", REPORT_PATH)
    print("Existing Colab prediction model files were not modified.")
    return report


# 6. READ MODEL COMPARISON REPORT

def get_report():
    """Load model comparison metrics from the saved CSV report."""
    if not REPORT_PATH.exists():
        return []

    try:
        report = pd.read_csv(REPORT_PATH)
        report.columns = report.columns.str.strip().str.lower()

        # Support alternative column names used by older reports.
        column_aliases = {
            "accuracy": "test_accuracy",
            "test_f1": "f1",
            "f_score": "f1",
            "f-score": "f1",
            "recall": "sensitivity",
        }

        for old_name, new_name in column_aliases.items():
            if old_name in report.columns and new_name not in report.columns:
                report.rename(columns={old_name: new_name}, inplace=True)

        # Replace missing values without changing valid metric values.
        report = report.astype(object).where(pd.notna(report), "")

        return report.to_dict("records")

    except Exception as exc:
        print(f"Error loading model comparison report: {exc}")
        return []


# 7. PREDICT HEART DISEASE CLASS


def predict(values):
    """
    Predict using the proposed Colab-trained RF-SFS model.
    """
    if not PROPOSED_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Proposed model not found: {PROPOSED_MODEL_PATH}"
        )

    if not PROPOSED_META_PATH.exists():
        raise FileNotFoundError(
            f"Proposed model metadata not found: {PROPOSED_META_PATH}"
        )

    if not isinstance(values, dict):
        raise ValueError("Prediction input must be a dictionary.")

    validated = {}

    for feature in FEATURES:
        if feature not in values:
            raise ValueError("Missing feature: " + feature)

        try:
            value = float(values[feature])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{LABELS[feature]} must be numeric."
            ) from exc

        if not np.isfinite(value):
            raise ValueError(
                f"{LABELS[feature]} must be a finite number."
            )

        lower, upper = RANGES[feature]

        if not lower <= value <= upper:
            raise ValueError(
                f"{LABELS[feature]} must be between {lower} and {upper}."
            )

        if feature in BINARY_FEATURES and value not in (0, 1):
            raise ValueError(
                f"{LABELS[feature]} must be 0 or 1."
            )

        validated[feature] = value

    with PROPOSED_META_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    expected_features = metadata.get("input_features", [])

    if expected_features and expected_features != FEATURES:
        raise ValueError(
            "The Colab model feature order does not match Django FEATURES. "
            f"Expected: {expected_features}; Django: {FEATURES}"
        )

    model = joblib.load(PROPOSED_MODEL_PATH)

    row = pd.DataFrame(
        [[validated[feature] for feature in FEATURES]],
        columns=FEATURES
    )

    prediction = int(model.predict(row)[0])

    probability = None

    if hasattr(model, "predict_proba"):
        class_probabilities = model.predict_proba(row)[0]
        classes = list(model.classes_)

        if 1 in classes:
            probability = float(
                class_probabilities[classes.index(1)]
            )

    return {
        "label": (
            "Heart Disease Predicted"
            if prediction == 1
            else "Normal"
        ),
        "prediction": prediction,
        "probability": probability,
        "model": metadata.get(
            "model_name",
            "Random Forest + Sequential Feature Selection"
        ),
    }


