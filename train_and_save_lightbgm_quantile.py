
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import joblib

# Optional dataset downloader (only used if processed file missing)
try:
    from kagglehub import dataset_download
    KAGGLE_AVAILABLE = True
except Exception:
    KAGGLE_AVAILABLE = False

RNG = 42
np.random.seed(RNG)

# -----------------------
# Paths & config
# -----------------------
PROCESSED_PATH = "../results/processed_ev_data.csv"
RESULTS_DIR = "../results"
MODELS_DIR = "./models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Columns expected in raw dataset (adjust if your dataset uses slightly different names)
COL_START = "Charging Start Time"
COL_END = "Charging End Time"
COL_ENERGY = "Energy Consumed (kWh)"
# if dataset has 'Charging Duration (hours)' we can use it if present

# -----------------------
# Utility functions
# -----------------------
def download_and_preprocess():
    """
    Download dataset from Kaggle (if kagglehub available) and preprocess.
    Returns processed DataFrame.
    """
    if not KAGGLE_AVAILABLE:
        raise RuntimeError("kagglehub not available in this environment. Provide processed CSV at ../results/processed_ev_data.csv")

    print("Downloading dataset from Kaggle...")
    path = dataset_download("valakhorasani/electric-vehicle-charging-patterns")
    files = os.listdir(path)
    csv_files = [f for f in files if f.lower().endswith(".csv")]
    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV found in downloaded dataset folder.")
    raw_csv = os.path.join(path, csv_files[0])
    df = pd.read_csv(raw_csv)
    print("Raw data shape:", df.shape)
    df = preprocess_dataframe(df)
    return df

def preprocess_dataframe(df):
    """
    Clean and feature-engineer raw DataFrame.
    Returns processed DataFrame.
    """
    # normalize column names (strip)
    df.columns = [c.strip() for c in df.columns]

    # Basic cleaning
    df = df.drop_duplicates().copy()
    # Drop rows where energy or times are missing
    required_cols = [COL_START, COL_END, COL_ENERGY]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Expected column '{c}' not found in raw data. Available columns: {df.columns.tolist()}")
    df = df.dropna(subset=required_cols)

    # parse datetimes
    df[COL_START] = pd.to_datetime(df[COL_START], errors='coerce')
    df[COL_END] = pd.to_datetime(df[COL_END], errors='coerce')
    df = df.dropna(subset=[COL_START, COL_END])  # drop rows with parse errors

    # energy numeric
    df[COL_ENERGY] = pd.to_numeric(df[COL_ENERGY], errors='coerce')
    df = df.dropna(subset=[COL_ENERGY])

    # derived features
    df['duration_min'] = (df[COL_END] - df[COL_START]).dt.total_seconds() / 60.0
    # if negative durations exist, drop them
    df = df[df['duration_min'] >= 0]

    df['start_hour'] = df[COL_START].dt.hour
    df['day_of_week'] = df[COL_START].dt.dayofweek  # 0 = Mon
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # target: flexible_kW (assumption: 30% of charged energy can be used for regulation)
    df['flexible_kW'] = df[COL_ENERGY] * 0.3

    # Save processed CSV
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed dataset saved to: {PROCESSED_PATH} (shape: {df.shape})")
    return df

def load_or_prepare():
    """
    Load processed CSV if exists, otherwise download+preprocess.
    """
    if os.path.exists(PROCESSED_PATH):
        print("Loading processed data from", PROCESSED_PATH)
        df = pd.read_csv(PROCESSED_PATH)
        # ensure datetimes
        df[COL_START] = pd.to_datetime(df[COL_START])
        df[COL_END] = pd.to_datetime(df[COL_END])
        # recompute derived features to be safe
        if 'duration_min' not in df.columns or df['duration_min'].isnull().any():
            df['duration_min'] = (df[COL_END] - df[COL_START]).dt.total_seconds() / 60.0
        if 'start_hour' not in df.columns:
            df['start_hour'] = df[COL_START].dt.hour
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df[COL_START].dt.dayofweek
        if 'is_weekend' not in df.columns:
            df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        if 'flexible_kW' not in df.columns:
            df['flexible_kW'] = df[COL_ENERGY] * 0.3
        return df
    else:
        return download_and_preprocess()

# -----------------------
# Training & evaluation
# -----------------------
def train_point_model(X_train, y_train, X_val=None, y_val=None):
    """
    Train LightGBM regressor for point estimates.
    Returns trained model.
    """
    params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "random_state": RNG
    }
    model = lgb.LGBMRegressor(**params)
    print("Training LightGBM point model...")

    callbacks = []
    if X_val is not None and y_val is not None:
      callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)] if X_val is not None else None,
              eval_metric="rmse",
              callbacks=callbacks)
    return model

def train_quantile_models(X_train, y_train, quantiles=[0.1, 0.5, 0.9]):
    """
    Train GradientBoostingRegressor for each quantile.
    Returns dict: {quantile: model}
    """
    q_models = {}
    for q in quantiles:
        print(f"Training quantile model for alpha={q} ...")
        gbr = GradientBoostingRegressor(
            loss="quantile",
            alpha=q,
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            random_state=RNG
        )
        gbr.fit(X_train, y_train)
        q_models[q] = gbr
    return q_models

def evaluate_point(y_true, y_pred, model_name="Model"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{model_name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return rmse, mae

def evaluate_probabilistic(y_true, q10, q50, q90):
    # Median evaluation
    median_rmse = np.sqrt(mean_squared_error(y_true, q50))
    median_mae = mean_absolute_error(y_true, q50)
    # Coverage: fraction of true values within [q10, q90]
    coverage = np.mean((y_true >= q10) & (y_true <= q90))
    # Average interval width
    avg_width = np.mean(q90 - q10)
    print(f"Quantile (median) -> RMSE: {median_rmse:.4f}, MAE: {median_mae:.4f}")
    print(f"Interval coverage [10%,90%]: {coverage*100:.2f}%")
    print(f"Average interval width: {avg_width:.4f}")
    return {"median_rmse": median_rmse, "median_mae": median_mae, "coverage": coverage, "avg_width": avg_width}

# -----------------------
# Main pipeline
# -----------------------
def main():
    # 1. Load / preprocess
    df = load_or_prepare()

    # 2. Features & target
    features = ['duration_min', COL_ENERGY, 'start_hour', 'day_of_week', 'is_weekend']
    target = 'flexible_kW'
    for f in features + [target]:
        if f not in df.columns:
            raise KeyError(f"Required column '{f}' not found in processed DataFrame.")

    # Drop any remaining NaNs
    df = df.dropna(subset=features + [target]).reset_index(drop=True)
    print("Using data shape:", df.shape)

    # 3. Chronological split (80/20) - keep temporal ordering
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index].copy().reset_index(drop=True)
    test_df = df.iloc[split_index:].copy().reset_index(drop=True)
    print("Train / Test split:", train_df.shape, test_df.shape)

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # 4. Train point model (LightGBM)
    point_model = train_point_model(X_train, y_train, X_val=X_test, y_val=y_test)
    joblib.dump(point_model, os.path.join(MODELS_DIR, "lightgbm_point_model.pkl"))
    print("Saved LightGBM point model to:", os.path.join(MODELS_DIR, "lightgbm_point_model.pkl"))

    # Predict & evaluate point model
    point_preds = point_model.predict(X_test)
    evaluate_point(y_test, point_preds, model_name="LightGBM (point)")

    # Save point predictions CSV
    df_point_out = test_df[[COL_START, COL_ENERGY, 'duration_min', 'start_hour', 'day_of_week', 'is_weekend', target]].copy()
    df_point_out['LightGBM_Pred'] = point_preds
    df_point_out.to_csv(os.path.join(RESULTS_DIR, "ml_predictions.csv"), index=False)
    print("Saved point predictions to:", os.path.join(RESULTS_DIR, "ml_predictions.csv"))

    # 5. Train quantile models
    quantiles = [0.1, 0.5, 0.9]
    q_models = train_quantile_models(X_train, y_train, quantiles=quantiles)
    for q, m in q_models.items():
        joblib.dump(m, os.path.join(MODELS_DIR, f"quantile_q{int(q*100)}.pkl"))
    print("Saved quantile models to:", MODELS_DIR)

    # 6. Predict quantiles on test set
    preds_q = {}
    for q in quantiles:
        preds_q[q] = q_models[q].predict(X_test)

    # Evaluate probabilistic predictions
    prob_eval = evaluate_probabilistic(y_test.values, preds_q[0.1], preds_q[0.5], preds_q[0.9])

    # Save probabilistic predictions CSV
    df_prob = test_df[[COL_START, COL_ENERGY, 'duration_min', 'start_hour', 'day_of_week', 'is_weekend', target]].copy()
    df_prob['Q10'] = preds_q[0.1]
    df_prob['Q50'] = preds_q[0.5]
    df_prob['Q90'] = preds_q[0.9]
    df_prob.to_csv(os.path.join(RESULTS_DIR, "probabilistic_predictions.csv"), index=False)
    print("Saved probabilistic predictions to:", os.path.join(RESULTS_DIR, "probabilistic_predictions.csv"))

    # 7. Simple visualizations (saved as PNG)
    plt.figure(figsize=(12,5))
    plt.plot(y_test.values, label="True", color="k", linewidth=1.5)
    plt.plot(point_preds, label="LightGBM (point)", alpha=0.8)
    plt.plot(preds_q[0.5], label="Quantile Median (Q50)", linestyle="--")
    plt.fill_between(range(len(y_test)), preds_q[0.1], preds_q[0.9], color="tab:blue", alpha=0.15, label="10-90% interval")
    plt.legend()
    plt.title("Flexible kW: True vs Point & Probabilistic Predictions (test set)")
    plt.xlabel("Test index")
    plt.ylabel("Flexible kW")
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "predictions_plot.png")
    plt.savefig(plot_path, dpi=150)
    print("Saved plot to:", plot_path)
    plt.close()

    # Error distribution for point model
    errors = y_test.values - point_preds
    plt.figure(figsize=(8,4))
    plt.hist(errors, bins=40, alpha=0.7)
    plt.title("Error distribution (True - LightGBM Prediction)")
    plt.xlabel("Error (kW)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    err_path = os.path.join(RESULTS_DIR, "error_hist.png")
    plt.savefig(err_path, dpi=150)
    print("Saved error histogram to:", err_path)
    plt.close()

    # Feature importance (LightGBM)
    try:
        importances = point_model.feature_importances_
        fi = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False)
        print("\nLightGBM feature importances:\n", fi.to_string(index=False))
        fi.to_csv(os.path.join(RESULTS_DIR, "feature_importances.csv"), index=False)
    except Exception as e:
        print("Could not extract feature importances:", e)

    print("\nPipeline finished successfully.")
    print("Results saved to:", RESULTS_DIR)
    print("Models saved to:", MODELS_DIR)

if __name__ == "__main__":
    main()