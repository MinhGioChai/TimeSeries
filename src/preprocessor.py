import logging

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR

logger = logging.getLogger(__name__)

_STOPS_MAP = {
    "non-stop": 0,
    "1 stop": 1,
    "2 stops": 2,
    "3 stops": 3,
    "4 stops": 4,
}

# Columns that are either redundant or consumed by feature_engineering.py
_DROP_BEFORE_MODEL = ["Date_of_Journey", "Route", "Dep_Time", "Arrival_Time", "Additional_Info"]
# Duration is kept in eda_df for feature_engineering.py to parse correctly; excluded from model_df
_DURATION_COL = "Duration"


def preprocess(data: pd.DataFrame):
    df = data.copy()

    # ── clean ─────────────────────────────────────────────────────────────────
    n_before = len(df)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"Removed {n_before - len(df)} rows (nulls/duplicates). Remaining: {len(df)}")

    # ── date features ─────────────────────────────────────────────────────────
    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
    df["Journey_Day"]      = df["Date_of_Journey"].dt.day
    df["Journey_Month"]    = df["Date_of_Journey"].dt.month
    df["Journey_Year"]     = df["Date_of_Journey"].dt.year   # kept for FE to decide
    df["Journey_Weekdays"] = df["Date_of_Journey"].dt.weekday
    logger.info("Extracted: Journey_Day, Journey_Month, Journey_Year, Journey_Weekdays")

    # ── normalise airline / destination labels ────────────────────────────────
    df["Destination"] = df["Destination"].replace("New Delhi", "Delhi")
    df["Airline"] = np.where(df["Airline"] == "Vistara Premium economy",             "Vistara",           df["Airline"])
    df["Airline"] = np.where(df["Airline"] == "Jet Airways Business",                "Jet Airways",       df["Airline"])
    df["Airline"] = np.where(df["Airline"] == "Multiple carriers Premium economy",   "Multiple carriers", df["Airline"])

    # ── departure time ────────────────────────────────────────────────────────
    df["Dep_Hours"] = pd.DatetimeIndex(df["Dep_Time"]).hour
    df["Dep_Min"]   = pd.DatetimeIndex(df["Dep_Time"]).minute

    # ── arrival time (strip date suffix before parsing) ───────────────────────
    df["Arrival_Time"]  = df["Arrival_Time"].str[:5]
    df["Arrival_Hours"] = pd.DatetimeIndex(df["Arrival_Time"]).hour
    df["Arrival_Min"]   = pd.DatetimeIndex(df["Arrival_Time"]).minute

    # ── stops → ordinal ───────────────────────────────────────────────────────
    df["Total_Stops"] = df["Total_Stops"].map(_STOPS_MAP).fillna(0).astype(int)

    # ── drop raw columns (keep Duration for feature_engineering.py) ───────────
    df.drop(columns=_DROP_BEFORE_MODEL, inplace=True)
    logger.info(f"Dropped raw columns: {_DROP_BEFORE_MODEL}")
    logger.info("Note: Duration kept in eda_df for feature engineering step")

    # eda_df: human-readable, has Duration + Journey_Year for FE
    eda_df = df.copy()

    # model_df (basic): exclude Duration (string) and Journey_Year (constant)
    # Feature engineering step will produce the final enriched model_df
    _basic_drop = [_DURATION_COL, "Journey_Year"]
    df_model = df.drop(columns=[c for c in _basic_drop if c in df.columns])
    model_df = pd.get_dummies(
        df_model,
        prefix=["Airline", "Source", "Destination"],
        columns=["Airline", "Source", "Destination"],
        dtype=int,
        drop_first=False,
    )

    logger.info(f"eda_df  : {eda_df.shape}  (with Duration + Year, for FE)")
    logger.info(f"model_df: {model_df.shape}  (basic one-hot, use after FE for final model)")

    eda_df.to_csv(PROCESSED_DIR / "eda_df.csv", index=False)
    model_df.to_csv(PROCESSED_DIR / "model_df.csv", index=False)
    logger.info("Saved: outputs/processed/eda_df.csv")
    logger.info("Saved: outputs/processed/model_df.csv")

    return eda_df, model_df
