"""
Test-Set Predictor
==================
Applies the same preprocessing + feature engineering pipeline to Test_set.csv
and generates fare predictions using the best trained model.

The transformations here mirror preprocessor.py and feature_engineering.py
exactly, but operate on data with no Price column and align the resulting
feature matrix to the training column set before predicting.
"""

import json
import logging

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    DATA_DIR, FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, TEST_FILE
)

logger = logging.getLogger(__name__)

# ── constants (must match preprocessor.py and feature_engineering.py) ─────────

_STOPS_MAP = {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}
_PREMIUM_AIRLINES = {"Jet Airways", "Vistara"}
_ECONOMY_AIRLINES = {"Air India", "Multiple carriers"}


def _parse_duration(raw: str) -> float:
    s = str(raw).strip()
    hours, minutes = 0, 0
    if "h" in s:
        parts = s.split("h")
        hours = int(parts[0].strip()) if parts[0].strip() else 0
        if "m" in parts[1]:
            minutes = int(parts[1].strip().replace("m", "").strip() or 0)
    elif "m" in s:
        minutes = int(s.replace("m", "").strip())
    return hours + minutes / 60.0


def _hour_to_period(hour: int) -> int:
    if 6 <= hour <= 11:
        return 1
    if 12 <= hour <= 17:
        return 2
    if 18 <= hour <= 20:
        return 3
    return 0


def _airline_tier(airline: str) -> int:
    if airline in _PREMIUM_AIRLINES:
        return 2
    if airline in _ECONOMY_AIRLINES:
        return 1
    return 0


def _preprocess_test(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and parse the test set (same logic as preprocessor.preprocess)."""
    df = df.copy()

    n_before = len(df)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    if n_before - len(df):
        logger.info(f"Removed {n_before - len(df)} null/duplicate rows from test set")

    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
    df["Journey_Day"]      = df["Date_of_Journey"].dt.day
    df["Journey_Month"]    = df["Date_of_Journey"].dt.month
    df["Journey_Weekdays"] = df["Date_of_Journey"].dt.weekday

    df["Destination"] = df["Destination"].replace("New Delhi", "Delhi")
    df["Airline"] = np.where(df["Airline"] == "Vistara Premium economy",           "Vistara",           df["Airline"])
    df["Airline"] = np.where(df["Airline"] == "Jet Airways Business",              "Jet Airways",       df["Airline"])
    df["Airline"] = np.where(df["Airline"] == "Multiple carriers Premium economy", "Multiple carriers", df["Airline"])

    df["Dep_Hours"]      = pd.DatetimeIndex(df["Dep_Time"]).hour
    df["Dep_Min"]        = pd.DatetimeIndex(df["Dep_Time"]).minute
    df["Arrival_Time"]   = df["Arrival_Time"].str[:5]
    df["Arrival_Hours"]  = pd.DatetimeIndex(df["Arrival_Time"]).hour
    df["Arrival_Min"]    = pd.DatetimeIndex(df["Arrival_Time"]).minute

    df["Total_Stops"] = df["Total_Stops"].map(_STOPS_MAP).fillna(0).astype(int)

    df.drop(columns=["Date_of_Journey", "Route", "Dep_Time", "Arrival_Time",
                     "Additional_Info"], inplace=True)
    return df


def _engineer_test(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering (same logic as feature_engineering.engineer_features)."""
    df = df.copy()

    df["Duration_Hours"] = df["Duration"].apply(_parse_duration)
    df.drop(columns=["Duration"], inplace=True)

    for col, cycle in [("Dep_Hours", 24), ("Arrival_Hours", 24)]:
        prefix = "Dep" if "Dep" in col else "Arr"
        df[f"{prefix}_Hour_Sin"] = np.sin(2 * np.pi * df[col] / cycle)
        df[f"{prefix}_Hour_Cos"] = np.cos(2 * np.pi * df[col] / cycle)

    df["Month_Sin"]   = np.sin(2 * np.pi * df["Journey_Month"]    / 12)
    df["Month_Cos"]   = np.cos(2 * np.pi * df["Journey_Month"]    / 12)
    df["Weekday_Sin"] = np.sin(2 * np.pi * df["Journey_Weekdays"] / 7)
    df["Weekday_Cos"] = np.cos(2 * np.pi * df["Journey_Weekdays"] / 7)

    df["Dep_Period"] = df["Dep_Hours"].apply(_hour_to_period)
    df["Arr_Period"] = df["Arrival_Hours"].apply(_hour_to_period)
    df["Is_Weekend"] = (df["Journey_Weekdays"] >= 5).astype(int)

    df["Route"] = (
        df["Source"].str.replace(" ", "_")
        + "_to_"
        + df["Destination"].str.replace(" ", "_")
    )
    df["Airline_Tier"] = df["Airline"].apply(_airline_tier)

    cat_cols = ["Airline", "Source", "Destination", "Route"]
    existing_cats = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, prefix=existing_cats, columns=existing_cats, dtype=int, drop_first=False)

    return df


def _align_columns(test_df: pd.DataFrame, train_cols: list) -> pd.DataFrame:
    """
    Ensure test_df has exactly the same columns as the training feature set.
    Missing columns (unseen OHE categories) are filled with 0.
    Extra columns (categories absent from training) are dropped.
    """
    for col in train_cols:
        if col not in test_df.columns:
            test_df[col] = 0
    return test_df[train_cols]


def _plot_predictions(predictions: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Test Set Fare Predictions", fontsize=15)

    # Distribution of predicted fares
    axes[0].hist(predictions["Predicted_Price"], bins=40, edgecolor="white")
    axes[0].set_title("Predicted Price Distribution")
    axes[0].set_xlabel("Price (INR)")
    axes[0].set_ylabel("Count")

    # Predicted price by airline
    airline_means = (
        predictions.groupby("Airline")["Predicted_Price"]
        .median()
        .sort_values(ascending=False)
    )
    axes[1].barh(airline_means.index, airline_means.values)
    axes[1].set_title("Median Predicted Price by Airline")
    axes[1].set_xlabel("Price (INR)")

    # Predicted price by month
    month_means = (
        predictions.groupby("Journey_Month")["Predicted_Price"]
        .mean()
        .reset_index()
    )
    axes[2].plot(month_means["Journey_Month"], month_means["Predicted_Price"],
                 marker="o", linewidth=2)
    axes[2].set_title("Mean Predicted Price by Month")
    axes[2].set_xlabel("Month")
    axes[2].set_ylabel("Price (INR)")
    axes[2].set_xticks(month_means["Journey_Month"])

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "20_test_predictions.png", bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved: outputs/figures/20_test_predictions.png")


def predict_test_set(train_feature_cols: list,
                     retrain_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Load Test_set.csv, apply the full pipeline, predict fares with best_model.pkl,
    and save results to outputs/processed/test_predictions.csv.

    Parameters
    ----------
    train_feature_cols : list
        Ordered list of feature column names from the training enriched_model_df
        (excluding Price). Used to align the test feature matrix.
    retrain_df : pd.DataFrame, optional
        If provided (should be the full enriched_model_df including Price), the
        best model is retrained on ALL labelled data before predicting. This gives
        better predictions than using a model trained on only 80% of the data.

    Returns
    -------
    pd.DataFrame  with original identifiers + Predicted_Price column.
    """
    # ── load raw test data ────────────────────────────────────────────────────
    raw = pd.read_csv(TEST_FILE)
    logger.info(f"Test set loaded: {raw.shape[0]} rows")

    # ── preprocess ────────────────────────────────────────────────────────────
    processed = _preprocess_test(raw)

    # Identifiers are extracted from processed (Airline/Source/Destination survive
    # preprocessing; reconstruct journey date from parsed day/month columns)
    identifiers = processed[["Airline", "Source", "Destination", "Total_Stops",
                              "Journey_Day", "Journey_Month"]].copy()
    identifiers["Date_of_Journey"] = (
        "2019-"
        + identifiers["Journey_Month"].astype(str).str.zfill(2)
        + "-"
        + identifiers["Journey_Day"].astype(str).str.zfill(2)
    )
    identifiers = identifiers[["Airline", "Date_of_Journey", "Source",
                                "Destination", "Total_Stops"]]

    # Keep Journey_Month for the plot (before dropping it via alignment)
    journey_months = processed["Journey_Month"].values

    # ── feature engineering ───────────────────────────────────────────────────
    engineered = _engineer_test(processed)

    # ── align to training columns ─────────────────────────────────────────────
    X_test = _align_columns(engineered, train_feature_cols)
    logger.info(f"Feature matrix aligned: {X_test.shape[1]} columns")

    # ── load best model, optionally retrain on full data ─────────────────────
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    logger.info(f"Loaded model: {type(model).__name__}")

    if retrain_df is not None:
        X_full = retrain_df.drop("Price", axis=1)[train_feature_cols]
        y_full = retrain_df["Price"]
        model.fit(X_full, y_full)
        joblib.dump(model, MODELS_DIR / "best_model_full.pkl")
        logger.info(f"Retrained on full data ({len(X_full):,} rows) -> saved best_model_full.pkl")

    predictions = model.predict(X_test)

    # ── assemble results ──────────────────────────────────────────────────────
    result = identifiers.copy()
    result["Journey_Month"]    = journey_months
    result["Predicted_Price"]  = predictions.round(2)

    result.to_csv(PROCESSED_DIR / "test_predictions.csv", index=False)
    logger.info(f"Saved: outputs/processed/test_predictions.csv  ({len(result)} predictions)")

    # ── summary stats ─────────────────────────────────────────────────────────
    logger.info(
        f"\n  Prediction summary:"
        f"\n    Count  : {len(result):,}"
        f"\n    Min    : Rs. {predictions.min():,.0f}"
        f"\n    Max    : Rs. {predictions.max():,.0f}"
        f"\n    Mean   : Rs. {predictions.mean():,.0f}"
        f"\n    Median : Rs. {np.median(predictions):,.0f}"
        f"\n    Std    : Rs. {predictions.std():,.0f}"
    )

    # ── save params used (for reproducibility) ────────────────────────────────
    params_path = REPORTS_DIR / "05_best_params.json"
    if params_path.exists():
        best_params = json.loads(params_path.read_text(encoding="utf-8"))
        logger.info(f"Model params: {best_params}")

    _plot_predictions(result)

    return result
