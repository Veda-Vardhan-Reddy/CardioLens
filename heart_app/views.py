
import json
import joblib
import pandas as pd

from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.http import HttpResponse
from django.utils.http import url_has_allowed_host_and_scheme

from sklearn.model_selection import train_test_split

from ml_engine import pipeline
from ml_engine.pipeline import FEATURES, LABELS, predict, get_report


# =========================================================
# SIGNUP
# =========================================================

@require_http_methods(["GET", "POST"])
def signup_view(request):
    if request.user.is_authenticated:
        return redirect("dashboard")

    form = UserCreationForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        user = form.save()
        login(request, user)
        messages.success(request, "Account created successfully.")
        return redirect("dashboard")

    return render(request, "heart_app/signup.html", {
        "form": form,
    })


# =========================================================
# LOGIN
# =========================================================

@require_http_methods(["GET", "POST"])
def login_view(request):
    if request.user.is_authenticated:
        return redirect("dashboard")

    form = AuthenticationForm(request, data=request.POST or None)

    if request.method == "POST" and form.is_valid():
        login(request, form.get_user())

        next_url = request.POST.get("next") or request.GET.get("next")

        if next_url and url_has_allowed_host_and_scheme(
            next_url,
            allowed_hosts={request.get_host()},
        ):
            return redirect(next_url)

        return redirect("dashboard")

    return render(request, "heart_app/login.html", {
        "form": form,
        "next": request.GET.get("next", ""),
    })


# =========================================================
# LOGOUT
# =========================================================

@require_http_methods(["POST"])
@login_required
def logout_view(request):
    logout(request)
    messages.success(request, "You have been logged out.")
    return redirect("login")


# =========================================================
# DASHBOARD
# =========================================================

@login_required
def dashboard(request):
    summary = None
    rows = get_report()
    error = None

    try:
        df = pipeline.load_data()

        summary = {
            "total_records": len(df),
            "total_features": len(FEATURES),
            "missing_values": int(df[FEATURES].isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "class_0": int((df["num"] == 0).sum()),
            "class_1": int((df["num"] == 1).sum()),
        }

    except Exception as exc:
        error = str(exc)

    chart_labels = [row.get("model", "") for row in rows]

    chart_accuracy = [
        round(float(row.get("test_accuracy", 0)) * 100, 2)
        for row in rows
    ]

    chart_auc = [
        round(float(row.get("roc_auc", 0)) * 100, 2)
        for row in rows
    ]

    return render(request, "heart_app/dashboard.html", {
        "summary": summary,
        "rows": rows,
        "error": error,
        "chart_labels": json.dumps(chart_labels),
        "chart_accuracy": json.dumps(chart_accuracy),
        "chart_auc": json.dumps(chart_auc),
    })


# =========================================================
# STEP 1: UPLOAD DATASET
# =========================================================


@require_http_methods(["GET", "POST"])
@login_required
def upload_dataset(request):
    if request.method == "GET":
        return render(
            request,
            "heart_app/upload.html",
            {"error": None},
        )

    uploaded_file = request.FILES.get("dataset")

    try:
        if not uploaded_file:
            raise ValueError("Please select a CSV dataset to upload.")

        if not uploaded_file.name.lower().endswith(".csv"):
            raise ValueError("Only CSV files are allowed.")

        if uploaded_file.size > 25 * 1024 * 1024:
            raise ValueError("File size must not exceed 25 MB.")

        df = pd.read_csv(uploaded_file)

        if df.empty:
            raise ValueError("The uploaded CSV contains no records.")

        df.columns = [
            str(column).strip().lower()
            for column in df.columns
        ]

        # Support common target-column names.
        target_aliases = [
            "target",
            "heartdisease",
            "heart_disease",
        ]

        for alias in target_aliases:
            if alias in df.columns and "num" not in df.columns:
                df = df.rename(columns={alias: "num"})
                break

        required_columns = list(FEATURES) + ["num"]
        missing_columns = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        # Validate that feature and target columns are numeric.
        for column in required_columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

        if df[required_columns].isna().any().any():
            raise ValueError(
                "The dataset contains missing or non-numeric values "
                "in required columns. Please correct the CSV and retry."
            )

        # Convert heart disease target values to binary labels.
        if not df["num"].isin([0, 1]).all():
            df["num"] = (df["num"] > 0).astype(int)

        # Save the dataset without deleting existing model files.
        pipeline.DATA.mkdir(parents=True, exist_ok=True)
        data_path = pipeline.DATA / "heart.csv"
        df.to_csv(data_path, index=False)

        # Keep existing models, metadata and comparison report.
        # They can be updated after successful model training.

        messages.success(
            request,
            f"Dataset uploaded successfully: {len(df)} records."
        )

        return render(
            request,
            "heart_app/upload.html",
            {
                "error": None,
                "success": (
                    f"Dataset uploaded successfully: {len(df)} records."
                ),
            },
        )

    except Exception as exc:
        return render(
            request,
            "heart_app/upload.html",
            {
                "error": str(exc),
                "success": None,
            },
        )




# =========================================================
# STEP 2: PREPROCESS AND INSPECT DATASET
# =========================================================


@login_required
def preprocess(request):
    try:
        data_path = pipeline.DATA / "heart.csv"

        if not data_path.exists():
            raise ValueError("Upload a dataset first.")

        df = pd.read_csv(data_path)

        df.columns = [
            str(column).strip().lower()
            for column in df.columns
        ]

        required_columns = list(FEATURES) + ["num"]
        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        if df.empty:
            raise ValueError("The dataset contains no records.")

        # Convert required columns to numeric values for validation.
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Count invalid or missing target values.
        invalid_targets = int(
            (df["num"].isna() | ~df["num"].isin([0, 1])).sum()
        )

        # Prepare a summary without modifying the source dataset.
        summary = {
            "total_records": len(df),
            "total_features": len(FEATURES),
            "missing_values": int(
                df[FEATURES].isna().sum().sum()
            ),
            "duplicate_rows": int(df.duplicated().sum()),
            "class_0": int((df["num"] == 0).sum()),
            "class_1": int((df["num"] == 1).sum()),
            "invalid_target_rows": invalid_targets,
            "columns": list(df.columns),
            "preview": df.head(10).fillna("").to_dict("records"),
        }

        if request.method == "POST":
            messages.success(
                request,
                "Dataset validation completed. "
                "Review the missing values and duplicate rows below."
            )

        return render(
            request,
            "heart_app/preprocess.html",
            {
                "summary": summary,
                "error": None,
            },
        )

    except Exception as exc:
        return render(
            request,
            "heart_app/preprocess.html",
            {
                "summary": None,
                "error": str(exc),
            },
        )




# =========================================================
# STEP 3: TRAIN / TEST SPLIT
# =========================================================


@login_required
def split_dataset(request):
    try:
        df = pipeline.load_data()

        if df is None or df.empty:
            raise ValueError("Please upload a valid dataset first.")

        if "num" not in df.columns:
            raise ValueError("Target column 'num' was not found.")

        X = df[FEATURES]
        y = df["num"]

        if y.nunique() != 2:
            raise ValueError(
                "The dataset must contain both target classes: 0 and 1."
            )

        if y.value_counts().min() < 2:
            raise ValueError(
                "Each target class must contain at least two records."
            )

        # Read the selected test ratio from the form.
        test_size = 0.20

        if request.method == "POST":
            try:
                test_size = float(request.POST.get("test_size", "0.20"))
            except (TypeError, ValueError):
                test_size = 0.20

        if test_size not in (0.20, 0.25, 0.30):
            raise ValueError("Please select a valid testing ratio.")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        summary = {
            "total": len(df),
            "train": len(X_train),
            "test": len(X_test),
            "train_class_0": int((y_train == 0).sum()),
            "train_class_1": int((y_train == 1).sum()),
            "test_class_0": int((y_test == 0).sum()),
            "test_class_1": int((y_test == 1).sum()),
            "test_size": test_size,
            "train_percent": int((1 - test_size) * 100),
            "test_percent": int(test_size * 100),
        }

        return render(request, "heart_app/split.html", {
            "summary": summary,
            "error": None,
        })

    except Exception as exc:
        return render(request, "heart_app/split.html", {
            "summary": None,
            "error": str(exc),
        })

# =========================================================
# STEP 4: TRAIN ALL EIGHT MODELS
# =========================================================





@require_http_methods(["GET", "POST"])
@login_required
def train_models(request):
    error = None
    rows = get_report()

    model_exists = (
        pipeline.PROPOSED_MODEL_PATH.exists()
        and pipeline.PROPOSED_META_PATH.exists()
    )

    if model_exists:
        messages.info(
            request,
            "The existing Colab-trained model is available. "
            "It will be used for predictions without retraining."
        )
    else:
        error = (
            "The Colab-trained model or its metadata is missing. "
            "Please check the models folder."
        )

    return render(request, "heart_app/train.html", {
        "error": error,
        "rows": rows,
        "trained": False,
        "model_available": model_exists,
    })



# =========================================================
# STEP 5: MODEL COMPARISON AND EVALUATION GRAPHS
# =========================================================

@login_required
def comparison(request):
    try:
        rows = get_report()
        error = None

    except Exception as exc:
        rows = []
        error = str(exc)

    labels = [row.get("model", "") for row in rows]

    def metric_values(key):
        return [
            round(float(row.get(key, 0)) * 100, 2)
            for row in rows
        ]

    return render(request, "heart_app/comparison.html", {
        "rows": rows,
        "error": error,
        "chart_labels": json.dumps(labels),
        "chart_accuracy": json.dumps(
            metric_values("test_accuracy")
        ),
        "chart_sensitivity": json.dumps(
            metric_values("sensitivity")
        ),
        "chart_specificity": json.dumps(
            metric_values("specificity")
        ),
        "chart_auc": json.dumps(
            metric_values("roc_auc")
        ),
    })


# =========================================================
# STEP 6A: MANUAL PREDICTION
# =========================================================

@require_http_methods(["GET", "POST"])
@login_required
def predict_view(request):
    result = None
    error = None
    precautions = []

    if request.method == "POST":
        try:
            values = {}

            # 1. Read and validate all required features
            for feature in FEATURES:
                raw_value = request.POST.get(feature, "").strip()

                if not raw_value:
                    label = LABELS.get(feature, feature)
                    raise ValueError(
                        f"Please enter a value for {label}."
                    )

                try:
                    values[feature] = float(raw_value)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Please enter a valid numeric value for "
                        f"{LABELS.get(feature, feature)}."
                    )

            # 2. Generate a prediction using the existing ML pipeline
            prediction_result = predict(values)

            if not isinstance(prediction_result, dict):
                raise ValueError(
                    "The prediction pipeline did not return a valid result."
                )

            if "prediction" not in prediction_result:
                raise ValueError(
                    "The prediction pipeline result is missing "
                    "the 'prediction' field. Check ml_engine/pipeline.py."
                )

            predicted_class = int(prediction_result["prediction"])

            if predicted_class not in (0, 1):
                raise ValueError(
                    "The model returned an unexpected prediction class."
                )

            # 3. Normalize the result keys expected by predict.html
            result = dict(prediction_result)
            result["prediction"] = predicted_class

            result.setdefault(
                "label",
                (
                    "Heart Disease Predicted"
                    if predicted_class == 1
                    else "Normal"
                )
            )

            result.setdefault(
                "model",
                "Random Forest + SFS"
            )

            # Convert a returned probability to a numeric value if present.
            probability = result.get("probability")

            if probability is not None:
                result["probability"] = float(probability)

                if not 0 <= result["probability"] <= 1:
                    raise ValueError(
                        "The model returned a probability outside 0 to 1."
                    )

            # 4. Provide general educational guidance
            if predicted_class == 1:
                precautions = [
                    "Discuss this result with a qualified healthcare professional.",
                    "Ask your clinician about appropriate blood pressure, "
                    "cholesterol and blood-sugar checks.",
                    "Follow professional advice about diet, physical activity "
                    "and any prescribed treatment.",
                    "Seek emergency help for chest pain, severe breathlessness "
                    "or other possible emergency symptoms."
                ]
            else:
                precautions = [
                    "Maintain heart-healthy lifestyle habits.",
                    "Attend routine health check-ups as appropriate.",
                    "Seek medical advice if symptoms or concerns persist.",
                    "A Normal model prediction does not rule out heart disease."
                ]

        except (ValueError, TypeError, FileNotFoundError) as exc:
            error = str(exc)
            result = None

        except Exception as exc:
            # Keep the actual error visible during development.
            error = f"Prediction failed: {exc}"
            result = None

    context = {
        "features": FEATURES,
        "labels": LABELS,
        "result": result,
        "error": error,
        "precautions": precautions,
    }

    return render(
        request,
        "heart_app/predict.html",
        context
    )


# =========================================================
# STEP 6B: CSV / TESTDATA PREDICTION
# =========================================================

@require_http_methods(["GET", "POST"])
@login_required
def csv_predict(request):
    if request.method == "POST":
        uploaded = request.FILES.get("prediction_file")

        try:
            if not uploaded:
                raise ValueError("Please select a prediction CSV.")

            if not uploaded.name.lower().endswith(".csv"):
                raise ValueError("Only CSV files are supported.")

            if uploaded.size > 25 * 1024 * 1024:
                raise ValueError("The CSV file must be smaller than 25 MB.")

            if not pipeline.PROPOSED_MODEL_PATH.exists():
                raise ValueError(
                    "The proposed Colab-trained model was not found. "
                    "Please check models/heart_disease_rf_sfs.joblib."
                )

            if not pipeline.PROPOSED_META_PATH.exists():
                raise ValueError(
                    "The proposed model metadata was not found. "
                    "Please check models/heart_disease_rf_sfs_metadata.json."
                )

            df = pd.read_csv(uploaded)

            if df.empty:
                raise ValueError("The prediction CSV is empty.")

            if len(df) > 10000:
                raise ValueError(
                    "Please upload no more than 10,000 records per request."
                )

            df.columns = [
                str(column).strip().lower()
                for column in df.columns
            ]

            missing = [
                feature for feature in FEATURES
                if feature not in df.columns
            ]

            if missing:
                raise ValueError(
                    "Missing feature columns: " + ", ".join(missing)
                )

            X = df[FEATURES].apply(
                pd.to_numeric,
                errors="coerce",
            )

            X = X.replace(
                [float("inf"), float("-inf")],
                float("nan"),
            )

            if X.isnull().any().any():
                raise ValueError(
                    "The prediction CSV contains missing or "
                    "non-numeric feature values."
                )

            for feature in FEATURES:
                low, high = pipeline.RANGES[feature]

                if not X[feature].between(low, high).all():
                    raise ValueError(
                        f"{feature} must be between {low} and {high}."
                    )

            for feature in ("sex", "fbs", "exang"):
                if not X[feature].isin([0, 1]).all():
                    raise ValueError(
                        f"{feature} must contain only 0 or 1."
                    )

            model = joblib.load(pipeline.PROPOSED_MODEL_PATH)

            metadata = json.loads(
                pipeline.PROPOSED_META_PATH.read_text(
                    encoding="utf-8"
                )
            )

            metadata_features = metadata.get("input_features")

            if metadata_features != FEATURES:
                raise ValueError(
                    "The model's input features do not match the "
                    "application's expected feature order."
                )

            predictions = model.predict(X).astype(int)

            output = df.copy()
            output["predicted_target"] = predictions
            output["predicted_label"] = [
                "Heart Disease Predicted" if value == 1 else "Normal"
                for value in predictions
            ]

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X)
                classes = list(model.classes_)

                if 1 in classes:
                    positive_index = classes.index(1)
                    output["model_probability"] = (
                        probabilities[:, positive_index]
                    )

            # Prepare results for displaying in the webpage.
            columns = output.columns.tolist()

            # Round probabilities for easier reading.
            if "model_probability" in output.columns:
                output["model_probability"] = (
                    output["model_probability"].round(4)
                )

            # Convert values to ordinary Python values for the template.
            results = output.astype(object).where(
                pd.notna(output), ""
            ).values.tolist()

            context = {
                "error": None,
                "message": (
                    f"Predictions generated successfully for "
                    f"{len(results)} records."
                ),
                "columns": columns,
                "results": results,
                "total_records": len(results),
                "disease_count": int((predictions == 1).sum()),
                "normal_count": int((predictions == 0).sum()),
            }

            return render(
                request,
                "heart_app/csv_predict.html",
                context,
            )

        except Exception as exc:
            return render(
                request,
                "heart_app/csv_predict.html",
                {
                    "error": str(exc),
                    "message": None,
                    "columns": [],
                    "results": [],
                },
            )

    return render(
        request,
        "heart_app/csv_predict.html",
        {
            "error": None,
            "message": None,
            "columns": [],
            "results": [],
        },
    )


# =========================================================
# ABOUT
# =========================================================

@login_required
def about(request):
    return render(request, "heart_app/about.html")
