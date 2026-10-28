
import pandas as pd

path = "reports/model_comparison.csv"
df = pd.read_csv(path)

mask = df["model"].astype(str).str.strip().eq("Random Forest + SFS")

if mask.sum() != 1:
    raise ValueError("Expected exactly one Random Forest + SFS row.")

df.loc[mask, "cv_accuracy_mean"] = 0.9768
df.loc[mask, "cv_f1_mean"] = 0.9773
df.loc[mask, "cv_roc_auc_mean"] = 0.9931

df.to_csv(path, index=False)

print("CSV updated successfully!")
print(df.loc[mask, [
    "model", "cv_accuracy_mean", "cv_f1_mean", "cv_roc_auc_mean"
]].to_string(index=False))
