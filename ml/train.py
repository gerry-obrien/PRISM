"""
train.py — PRISM Random Forest Training Script
===============================================
Run: python ml/train.py  (from project root)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import wandb
import mlflow
import mlflow.sklearn
from datetime import datetime
from generate_data import generate_dataset

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

DATA_FILE = os.path.join(ROOT_DIR, "data", "training_data.csv")
LISTINGS_FILE = os.path.join(ROOT_DIR, "data", "listings.csv")
LISTINGS_PREDICTED_FILE = os.path.join(ROOT_DIR, "data", "listings_predicted.csv")

ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("prism-real-estate")

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
BOOL_COLS = ["has_elevator", "is_new_build"]
TARGET = "price_eur"

NUMERIC_FEATURES = ["area_sqm", "num_rooms", "arrondissement", "floor", "year",
                    "has_elevator", "is_new_build"]
ORDINAL_DPE = ["dpe_rating"]
ORDINAL_CONDITION = ["building_condition"]
CATEGORICAL_FEATURES = ["property_type"]


def cast_booleans(df: pd.DataFrame) -> pd.DataFrame:
    for col in BOOL_COLS:
        df[col] = (df[col].astype(str).str.lower() == "true").astype(int)
    return df


def load_training_data() -> pd.DataFrame:
    print(f"Loading training data from {DATA_FILE} ...")
    df = pd.read_csv(DATA_FILE)
    df = cast_booleans(df)
    return df


def build_pipeline() -> Pipeline:
    numeric_transformer = StandardScaler()

    dpe_encoder = OrdinalEncoder(
        categories=[["G", "F", "E", "D", "C", "B", "A"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    condition_encoder = OrdinalEncoder(
        categories=[["Poor", "Average", "Good"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("dpe", dpe_encoder, ORDINAL_DPE),
            ("condition", condition_encoder, ORDINAL_CONDITION),
            ("cat", ohe, CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ])

    return pipeline


def compute_mape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def log_model_artifact(model_path, artifact_name="trained-model"):
    artifact = wandb.Artifact(artifact_name, type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    ### To be moved to a registry part

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train():
    df = load_training_data()

    config = {
        "test_size": 0.2,
        "random_state": 42,
        "model_type": "RandomForest"
    }

    with wandb.init(project="mlops-simple-project", config=config):
        config = wandb.config

        feature_cols = NUMERIC_FEATURES + ORDINAL_DPE + ORDINAL_CONDITION + CATEGORICAL_FEATURES
        X = df[feature_cols]
        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config["test_size"],
            random_state=config["random_state"]
        )

        print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

        pipeline = build_pipeline()
        print("Training Random Forest ...")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        mape = float(compute_mape(y_test, y_pred))

        print("\n--- Model Metrics ---")
        print(f"  R²   : {r2:.4f}")
        print(f"  MAE  : €{mae:,.0f}")
        print(f"  MAPE : {mape:.2f}%")

        metrics = {
            "r2": r2,
            "mae": mae,
            "mape": mape,
            "train_size": len(X_train),
            "test_size": len(X_test)
        }

        wandb.log(metrics)

        with mlflow.start_run():
            mlflow.log_params({
                "test_size": config["test_size"],
                "random_state": config["random_state"],
                "model_type": config["model_type"],
                "n_estimators": 100,
            })
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, "model")

        # Save model before logging artifact
        joblib.dump(pipeline, MODEL_PATH)
        print(f"\nModel saved to {MODEL_PATH}")

        log_model_artifact(MODEL_PATH)

        # Save metrics locally
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {METRICS_PATH}")

        # Optional plots
        save_plots(pipeline, X_train, X_test, y_test, y_pred, feature_cols)

    # Predict listings
    predict_listings(pipeline) #### Should It be done during the training ?

    return pipeline, metrics


def retrain(months = 1):

    df = load_training_data()
    new_data = generate_dataset(months * 10000)
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")

    final_df = pd.concat([df, new_data],axis = 0)

    config = {
        "test_size": 0.2,
        "random_state": 42,
        "model_type": "RandomForest"
    }

    with wandb.init(project="mlops-simple-project", config=config, job_type="retrain"):
        config = wandb.config

        feature_cols = NUMERIC_FEATURES + ORDINAL_DPE + ORDINAL_CONDITION + CATEGORICAL_FEATURES
        X = final_df[feature_cols]
        y = final_df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config["test_size"],
            random_state=config["random_state"]
        )

        print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

        pipeline = build_pipeline()
        print("Re-Training Random Forest ...")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        mape = float(compute_mape(y_test, y_pred))

        print("\n--- Model Metrics ---")
        print(f"  R²   : {r2:.4f}")
        print(f"  MAE  : €{mae:,.0f}")
        print(f"  MAPE : {mape:.2f}%")

        metrics = {
                    "r2": r2,
                    "mae": mae,
                    "mape": mape,
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "months": months,
                    "new_rows": len(new_data),
                    "total_rows": len(final_df)}

        wandb.log(metrics)

        with mlflow.start_run():
            mlflow.log_params({
                "test_size": config["test_size"],
                "random_state": config["random_state"],
                "model_type": config["model_type"],
                "n_estimators": 100,
                "months": months,
            })
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, "model")

        # Save model before logging artifact
        joblib.dump(pipeline, MODEL_PATH)
        print(f"\nModel saved to {MODEL_PATH}")

        log_model_artifact(MODEL_PATH)

        # Save metrics locally
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {METRICS_PATH}")

        # Optional plots
        save_plots(pipeline, X_train, X_test, y_test, y_pred, feature_cols)

    # Predict listings
    predict_listings(pipeline) #### Should It be done during the training ?

    DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")

    final_df.to_csv(os.path.join(DATA_DIR, f"training_data_{current_time}.csv"), index=False)

    return pipeline, metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def save_plots(pipeline, X_train, X_test, y_test, y_pred, feature_cols):
    rf_model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # --- Feature importance
    ohe_feature_names = (
        preprocessor.named_transformers_["cat"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
        .tolist()
    )
    all_feature_names = (
        NUMERIC_FEATURES + ORDINAL_DPE + ORDINAL_CONDITION + ohe_feature_names
    )
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    n_show = min(15, len(all_feature_names))
    top_idx = indices[:n_show]
    ax.barh(
        [all_feature_names[i] for i in reversed(top_idx)],
        importances[top_idx][::-1],
    )
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Top 15)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=100)
    plt.close()

    # --- Actual vs Predicted
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.3, s=5, color="steelblue")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Price (€)")
    ax.set_ylabel("Predicted Price (€)")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "actual_vs_predicted.png"), dpi=100)
    plt.close()

    # --- Residuals
    residuals = np.array(y_test) - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.3, s=5, color="darkorange")
    ax.axhline(0, color="red", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Predicted Price (€)")
    ax.set_ylabel("Residual (€)")
    ax.set_title("Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "residuals.png"), dpi=100)
    plt.close()

    # --- MAE by arrondissement
    test_df = X_test.copy()
    test_df["y_true"] = np.array(y_test)
    test_df["y_pred"] = y_pred
    test_df["abs_error"] = np.abs(test_df["y_true"] - test_df["y_pred"])
    mae_by_arr = test_df.groupby("arrondissement")["abs_error"].mean().sort_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(mae_by_arr.index, mae_by_arr.values, color="mediumseagreen")
    ax.set_xlabel("Arrondissement")
    ax.set_ylabel("MAE (€)")
    ax.set_title("MAE by Arrondissement")
    ax.set_xticks(mae_by_arr.index)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "mae_by_arrondissement.png"), dpi=100)
    plt.close()

    print(f"Plots saved to {PLOTS_DIR}/")


# ---------------------------------------------------------------------------
# Predict listings
# ---------------------------------------------------------------------------
def predict_listings(pipeline):
    print(f"\nLoading listings from {LISTINGS_FILE} ...")
    df = pd.read_csv(LISTINGS_FILE)
    df = cast_booleans(df)

    feature_cols = NUMERIC_FEATURES + ORDINAL_DPE + ORDINAL_CONDITION + CATEGORICAL_FEATURES
    X = df[feature_cols]
    estimated = pipeline.predict(X)

    df["estimated_price_eur"] = np.round(estimated, 2)

    asking = df["price_eur"]
    df["price_delta_pct"] = np.round(
        (df["estimated_price_eur"] - asking) / asking * 100, 2
    )

    def classify(delta):
        if delta > 10:
            return "Undervalued"
        elif delta < -10:
            return "Overvalued"
        else:
            return "Fair"

    df["valuation"] = df["price_delta_pct"].apply(classify)

    df.to_csv(LISTINGS_PREDICTED_FILE, index=False)
    print(f"Listings with predictions saved to {LISTINGS_PREDICTED_FILE}")

    counts = df["valuation"].value_counts()
    print("\nValuation breakdown:")
    for label, count in counts.items():
        print(f"  {label}: {count}")


def model_exists():
    return os.path.exists(MODEL_PATH)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    if model_exists():
        print("Existing model found in artifacts -> launching retraining...")
        retrain()
    else:
        print("No existing model found -> launching training...")
        train()
