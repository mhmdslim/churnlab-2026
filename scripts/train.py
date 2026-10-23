"""TP02 - Training pipeline for Telco Customer Churn prediction.

Usage:
    uv run python scripts/train.py

Reads data/raw/telco.csv, trains and compares classifiers,
saves the selected fitted pipeline to artifacts/.
"""

import hashlib
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CSV_PATH = Path("data/raw/telco.csv")
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
RANDOM_STATE = 42
THRESHOLD = 0.5

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = ["Contract", "InternetService", "PaymentMethod"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# 0. Load and validate data
# ---------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != EXPECTED_SHA256:
        raise ValueError(f"SHA-256 mismatch: {sha}")
    print(f"[OK] SHA-256 verified: {sha[:16]}...")

    df = pd.read_csv(path)

    # TotalCharges: whitespace-only -> NaN, then convert to float
    tc = df["TotalCharges"].astype(str).str.strip()
    blank_mask = tc == ""
    n_blank = blank_mask.sum()
    print(f"[INFO] {n_blank} blank TotalCharges values -> NaN")
    df["TotalCharges"] = pd.to_numeric(tc, errors="raise").where(~blank_mask, np.nan)

    # Encode target
    df["ChurnBinary"] = df["Churn"].map({"No": 0, "Yes": 1})
    assert df["ChurnBinary"].isna().sum() == 0, "Unexpected Churn values"

    return df


# ---------------------------------------------------------------------------
# 1. Split: 60 / 20 / 20 stratified
# ---------------------------------------------------------------------------
def split_data(df: pd.DataFrame):
    train_df, remaining = train_test_split(
        df, test_size=0.4, stratify=df["ChurnBinary"], random_state=RANDOM_STATE
    )
    valid_df, test_df = train_test_split(
        remaining, test_size=0.5, stratify=remaining["ChurnBinary"], random_state=RANDOM_STATE
    )
    # Verify no overlap
    assert set(train_df["customerID"]) & set(valid_df["customerID"]) == set()
    assert set(train_df["customerID"]) & set(test_df["customerID"]) == set()
    assert set(valid_df["customerID"]) & set(test_df["customerID"]) == set()
    print(f"[OK] Split: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
    print(f"     Churn rates: train={train_df['ChurnBinary'].mean():.3f}, "
          f"valid={valid_df['ChurnBinary'].mean():.3f}, test={test_df['ChurnBinary'].mean():.3f}")
    return train_df, valid_df, test_df


# ---------------------------------------------------------------------------
# 2. Pipeline builder
# ---------------------------------------------------------------------------
def build_pipeline(classifier) -> Pipeline:
    """Return an unfitted pipeline with preprocessing + classifier."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


# ---------------------------------------------------------------------------
# 3. Evaluate a fitted pipeline
# ---------------------------------------------------------------------------
def evaluate_model(pipe, X, y, name: str) -> dict:
    y_pred = pipe.predict(X)

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y, y_pred, zero_division=0), 4),
    }

    # Average precision (needs probability scores)
    if hasattr(pipe, "predict_proba"):
        y_scores = pipe.predict_proba(X)[:, list(pipe.classes_).index(1)]
        metrics["avg_precision"] = round(average_precision_score(y, y_scores), 4)
    else:
        metrics["avg_precision"] = None

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = {"TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
                                    "FN": int(cm[1, 0]), "TP": int(cm[1, 1])}
    return metrics


# ---------------------------------------------------------------------------
# Milestone 1: first simple model (2 features only)
# ---------------------------------------------------------------------------
def milestone_1(train_df, valid_df):
    print("\n" + "=" * 60)
    print("MILESTONE 1: First model (tenure + MonthlyCharges)")
    print("=" * 60)

    first_features = ["tenure", "MonthlyCharges"]
    X_train = train_df[first_features].values
    y_train = train_df["ChurnBinary"].values
    X_valid = valid_df[first_features].values
    y_valid = valid_df["ChurnBinary"].values

    # Logistic regression
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_valid)
    print(f"  Logistic Regression accuracy: {accuracy_score(y_valid, y_pred):.4f}")
    print(f"  Confusion matrix (labels [0,1]):\n  {confusion_matrix(y_valid, y_pred, labels=[0, 1])}")

    # Baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_valid)
    print(f"  Baseline (majority) accuracy: {accuracy_score(y_valid, y_pred_dummy):.4f}")
    print(f"  -> A model that always predicts 'No' gets {accuracy_score(y_valid, y_pred_dummy):.1%}.")
    print(f"     High accuracy can coexist with 0 churners identified.")

    return {"lr_accuracy": round(accuracy_score(y_valid, y_pred), 4),
            "baseline_accuracy": round(accuracy_score(y_valid, y_pred_dummy), 4)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load
    df = load_data(CSV_PATH)
    train_df, valid_df, test_df = split_data(df)

    X_train = train_df[ALL_FEATURES]
    y_train = train_df["ChurnBinary"].values
    X_valid = valid_df[ALL_FEATURES]
    y_valid = valid_df["ChurnBinary"].values
    X_test = test_df[ALL_FEATURES]
    y_test = test_df["ChurnBinary"].values

    # Milestone 1
    m1 = milestone_1(train_df, valid_df)

    # ------------------------------------------------------------------
    # Milestone 2 & 3: full pipeline comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MILESTONE 2-3: Pipeline comparison (6 features)")
    print("=" * 60)

    models = {
        "baseline_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE
        ),
    }

    results = []
    fitted_pipelines = {}

    for name, clf in models.items():
        pipe = build_pipeline(clf)
        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_time = round(time.time() - t0, 2)

        metrics = evaluate_model(pipe, X_valid, y_valid, name)
        metrics["fit_seconds"] = fit_time
        results.append(metrics)
        fitted_pipelines[name] = pipe

        ap = metrics["avg_precision"] if metrics["avg_precision"] else "N/A"
        print(f"  {name:25s}  acc={metrics['accuracy']:.4f}  "
              f"f1={metrics['f1']:.4f}  avg_prec={ap}  ({fit_time:.1f}s)")

    # ------------------------------------------------------------------
    # Selection: best average precision on validation
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("SELECTION: primary metric = average precision (validation)")
    print("-" * 60)

    scored = [r for r in results if r["avg_precision"] is not None and r["model"] != "baseline_majority"]
    best = max(scored, key=lambda r: r["avg_precision"])
    selected_name = best["model"]
    selected_pipeline = fitted_pipelines[selected_name]
    print(f"  Selected: {selected_name} (avg_precision={best['avg_precision']})")

    # ------------------------------------------------------------------
    # Milestone 4: save artifact + metadata
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MILESTONE 4: Save fitted pipeline")
    print("=" * 60)

    artifact_path = Path("artifacts/pipeline.joblib")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_pipeline, artifact_path)
    print(f"  Saved to {artifact_path} ({artifact_path.stat().st_size:,} bytes)")

    # Metadata
    from importlib.metadata import version as pkg_version
    import platform, sys

    metadata = {
        "selected_model": selected_name,
        "data_sha256": EXPECTED_SHA256,
        "split_seed": RANDOM_STATE,
        "split_sizes": {"train": len(train_df), "valid": len(valid_df), "test": len(test_df)},
        "customer_ids": {
            "train": sorted(train_df["customerID"].tolist()),
            "valid": sorted(valid_df["customerID"].tolist()),
            "test": sorted(test_df["customerID"].tolist()),
        },
        "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
        "positive_class": "Churn=Yes (encoded as 1)",
        "threshold": THRESHOLD,
        "preprocessing": {
            "numeric": "SimpleImputer(median) -> StandardScaler",
            "categorical": "SimpleImputer(constant='missing') -> OneHotEncoder(handle_unknown='ignore')",
        },
        "model_params": selected_pipeline.named_steps["classifier"].get_params(),
        "validation_metrics": best,
        "all_results": results,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "scikit-learn": pkg_version("scikit-learn"),
            "pandas": pkg_version("pandas"),
            "numpy": pkg_version("numpy"),
        },
    }

    meta_path = Path("evidence/tp02/training_metadata.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    # customer_ids are large; save separately
    ids_path = Path("evidence/tp02/customer_splits.json")
    ids_path.write_text(json.dumps(metadata.pop("customer_ids"), indent=2), encoding="utf-8")

    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"  Metadata: {meta_path}")
    print(f"  Customer splits: {ids_path}")

    # Comparison table
    comparison_path = Path("evidence/tp02/comparison_table.json")
    comparison_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  Comparison: {comparison_path}")

    # ------------------------------------------------------------------
    # Milestone 4 (cont): reload and verify
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MILESTONE 4: Reload and verify predictions")
    print("=" * 60)

    loaded = joblib.load(artifact_path)
    orig_proba = selected_pipeline.predict_proba(X_valid)
    loaded_proba = loaded.predict_proba(X_valid)
    max_diff = np.max(np.abs(orig_proba - loaded_proba))
    print(f"  Max probability difference (original vs loaded): {max_diff}")
    assert max_diff < 1e-10, f"Prediction mismatch: {max_diff}"
    print(f"  [OK] Predictions match within tolerance 1e-10")

    orig_pred = selected_pipeline.predict(X_valid)
    loaded_pred = loaded.predict(X_valid)
    assert np.array_equal(orig_pred, loaded_pred), "Class predictions differ!"
    print(f"  [OK] Class predictions identical")

    # ------------------------------------------------------------------
    # Milestone 5: focused checks
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MILESTONE 5: Focused checks")
    print("=" * 60)

    # 5a. No customer ID overlap
    print("  [CHECK 1] Customer ID overlap across splits")
    assert set(train_df["customerID"]) & set(valid_df["customerID"]) == set(), "train/valid overlap!"
    assert set(train_df["customerID"]) & set(test_df["customerID"]) == set(), "train/test overlap!"
    assert set(valid_df["customerID"]) & set(test_df["customerID"]) == set(), "valid/test overlap!"
    print("    -> No overlap. OK")

    # 5b. Imputer statistics match training data
    print("  [CHECK 2] Imputer statistics match training-only data")
    num_pipe = selected_pipeline.named_steps["preprocessor"].named_transformers_["num"]
    learned_medians = num_pipe.named_steps["imputer"].statistics_
    for i, feat in enumerate(NUMERIC_FEATURES):
        train_median = X_train[feat].median()
        assert abs(learned_medians[i] - train_median) < 1e-10, \
            f"{feat}: learned={learned_medians[i]} vs train={train_median}"
        print(f"    {feat}: learned={learned_medians[i]:.2f}, train median={train_median:.2f} -> match")

    # 5c. Prediction does not refit / change statistics
    print("  [CHECK 3] Predict does not change learned statistics")
    stats_before = num_pipe.named_steps["imputer"].statistics_.copy()
    scaler_mean_before = num_pipe.named_steps["scaler"].mean_.copy()
    _ = selected_pipeline.predict(X_valid)
    stats_after = num_pipe.named_steps["imputer"].statistics_
    scaler_mean_after = num_pipe.named_steps["scaler"].mean_
    assert np.array_equal(stats_before, stats_after), "Imputer stats changed after predict!"
    assert np.array_equal(scaler_mean_before, scaler_mean_after), "Scaler means changed after predict!"
    print("    -> Statistics unchanged after predict. OK")

    # 5d. Missing numeric + unknown category
    print("  [CHECK 4] Missing numeric and unknown category handling")
    test_row = pd.DataFrame([{
        "tenure": 12, "MonthlyCharges": 50.0, "TotalCharges": np.nan,
        "Contract": "Two year", "InternetService": "UNKNOWN_SERVICE", "PaymentMethod": "Mailed check",
    }])
    pred = selected_pipeline.predict(test_row)
    print(f"    Missing TotalCharges + unknown InternetService -> predicted class: {pred[0]}")
    print("    -> Handled without error. OK")

    # 5e. Missing required feature raises error
    print("  [CHECK 5] Missing required feature produces error")
    incomplete = pd.DataFrame([{
        "tenure": 12, "MonthlyCharges": 50.0,
        # TotalCharges missing entirely as a column
        "Contract": "Two year", "InternetService": "Fiber optic", "PaymentMethod": "Mailed check",
    }])
    try:
        selected_pipeline.predict(incomplete)
        print("    -> WARNING: no error raised (unexpected)")
    except (KeyError, ValueError) as e:
        print(f"    -> Error raised as expected: {type(e).__name__}")

    # 5f. Reload in "fresh" context
    print("  [CHECK 6] Reload predictions match (already verified above)")
    print("    -> OK")

    print("\n" + "=" * 60)
    print("ALL MILESTONES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
