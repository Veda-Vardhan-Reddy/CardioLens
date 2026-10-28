
import pandas as pd

from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_validate,
)
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef,
)

from ml_engine.pipeline import load_data, FEATURES, make_pipe
from sklearn.ensemble import RandomForestClassifier


# 1. Load data
df = load_data()
X = df[FEATURES]
y = df["num"]

# Identical feature combinations receive the same group ID.
groups = df.groupby(FEATURES, sort=False).ngroup()

print("=" * 60)
print("DUPLICATE-AWARE EVALUATION: RANDOM FOREST + SFS")
print("=" * 60)
print("Original rows:", len(df))
print("Unique feature groups:", groups.nunique())

# 2. Hold out entire groups.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
train_groups = groups.iloc[train_idx]
test_groups = groups.iloc[test_idx]

assert set(train_groups).isdisjoint(set(test_groups))

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training unique groups:", train_groups.nunique())
print("Testing unique groups:", test_groups.nunique())

# 3. Use the same RF + SFS configuration as the updated pipeline.
model = make_pipe(
    RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    ),
    select=True,
)

# 4. Group-aware cross-validation on training data only.
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scores = cross_validate(
    model,
    X_train,
    y_train,
    groups=train_groups,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
    },
    n_jobs=1,
    error_score="raise",
)

print("\nGroup-aware 5-fold CV:")
print(f"Accuracy: {scores['test_accuracy'].mean():.4f}")
print(f"F1-score: {scores['test_f1'].mean():.4f}")
print(f"ROC-AUC: {scores['test_roc_auc'].mean():.4f}")

# 5. Fit once on training groups; evaluate on held-out groups.
model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

tn, fp, fn, tp = confusion_matrix(
    y_test, pred, labels=[0, 1]
).ravel()

sensitivity = tp / (tp + fn) if tp + fn else 0.0
specificity = tn / (tn + fp) if tn + fp else 0.0

print("\nGroup-held-out test results:")
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
print(f"Balanced accuracy: {balanced_accuracy_score(y_test, pred):.4f}")
print(f"Precision: {precision_score(y_test, pred, zero_division=0):.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-score: {f1_score(y_test, pred, zero_division=0):.4f}")
print(f"MCC: {matthews_corrcoef(y_test, pred):.4f}")

if y_test.nunique() == 2:
    print(f"ROC-AUC: {roc_auc_score(y_test, prob):.4f}")
else:
    print("ROC-AUC unavailable: test split contains only one class.")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, pred, labels=[0, 1]))
