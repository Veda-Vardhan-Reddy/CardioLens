
import pandas as pd
from ml_engine.pipeline import load_data, FEATURES

df = load_data()

print("=" * 55)
print("HEART DATASET QUALITY CHECK")
print("=" * 55)

print("Total records:", len(df))
print("Exact duplicate rows:", df.duplicated().sum())
print("Unique feature combinations:", df[FEATURES].drop_duplicates().shape[0])

# Check whether identical feature values have conflicting labels.
label_counts = df.groupby(FEATURES, dropna=False)["num"].nunique()
conflicts = int((label_counts > 1).sum())

print("Feature combinations with conflicting labels:", conflicts)

# Show how often each target occurs.
print("\nTarget distribution:")
print(df["num"].value_counts().sort_index())

print("\nMissing values per column:")
print(df.isna().sum())

print("\nDataset quality check completed.")
