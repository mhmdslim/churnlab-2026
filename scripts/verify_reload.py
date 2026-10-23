"""TP02 - Verify that the saved pipeline works in a fresh process.

Usage:
    uv run python scripts/verify_reload.py

No fit() call. Loads the artifact and predicts a small batch.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

artifact_path = Path("artifacts/pipeline.joblib")
if not artifact_path.exists():
    raise SystemExit(f"Artifact not found: {artifact_path}. Run scripts/train.py first.")

pipeline = joblib.load(artifact_path)
print(f"Loaded pipeline from {artifact_path}")
print(f"Classifier: {pipeline.named_steps['classifier'].__class__.__name__}")

# Small batch with the documented 6 features
batch = pd.DataFrame([
    {"tenure": 1, "MonthlyCharges": 29.85, "TotalCharges": 29.85,
     "Contract": "Month-to-month", "InternetService": "DSL", "PaymentMethod": "Electronic check"},
    {"tenure": 72, "MonthlyCharges": 110.0, "TotalCharges": 7900.0,
     "Contract": "Two year", "InternetService": "Fiber optic", "PaymentMethod": "Bank transfer (automatic)"},
    {"tenure": 10, "MonthlyCharges": 55.0, "TotalCharges": np.nan,
     "Contract": "One year", "InternetService": "No", "PaymentMethod": "Mailed check"},
])

predictions = pipeline.predict(batch)
probabilities = pipeline.predict_proba(batch)[:, list(pipeline.classes_).index(1)]

print("\nPredictions on test batch:")
for i in range(len(batch)):
    label = "Churn" if predictions[i] == 1 else "No churn"
    print(f"  Customer {i+1}: {label} (p={probabilities[i]:.3f})")

print("\n[OK] Reload and inference work without fit().")
