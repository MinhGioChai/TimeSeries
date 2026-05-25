"""
Professional Feature Engineering  —  Flight Fare Dataset
==========================================================

Problems in the original notebook (Prediction.ipynb):
  1. Duration_Total_Hour is wrong: replaces 'h' with '1', so "2h 50m" becomes
     eval("21+50/60")=21.83 instead of 2.83. Every duration value is ~10x too
     large, corrupting one of the most important features.
  2. Journey_Year is constant (2019 only) — zero variance, pure noise for the model.
  3. Dep_Hours / Arrival_Hours are raw integers.  Hour 23 and hour 0 are
     numerically distant but physically adjacent.  A tree model can learn this,
     but cyclical (sin/cos) encoding makes it explicit and cheaper to learn.
  4. No airline tier information.  Jet Airways / Vistara charge very differently
     from IndiGo / SpiceJet even on the same route.
  5. Source and Destination are encoded separately.  "Delhi -> Cochin" is a
     specific route with its own price dynamics, not just "Delhi" + "Cochin".

New features added
------------------
  Duration_Hours        Correctly parsed flight duration in decimal hours.
  Dep_Period            Categorical time-of-day bin for departure  (0-3).
  Arr_Period            Same for arrival.
  Dep_Hour_Sin/Cos      Cyclical encoding of departure hour (24-h cycle).
  Arr_Hour_Sin/Cos      Cyclical encoding of arrival hour.
  Month_Sin/Cos         Cyclical encoding of journey month (12-month cycle).
  Weekday_Sin/Cos       Cyclical encoding of day of week (7-day cycle).
  Is_Weekend            1 if Saturday or Sunday, else 0.
  Airline_Tier          0 = Budget, 1 = Economy/Full-service, 2 = Premium.
  Route                 Combined "Source_to_Destination" string (5 categories),
                        then one-hot encoded.

Removed features
----------------
  Duration_Total_Hour   Replaced by correctly computed Duration_Hours.
  Journey_Year          Constant column (all 2019); zero predictive value.
"""

import logging

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

_PREMIUM_AIRLINES = {"Jet Airways", "Vistara"}
_ECONOMY_AIRLINES = {"Air India", "Multiple carriers"}
# Budget: everything else (IndiGo, SpiceJet, GoAir, Air Asia, Trujet)


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_duration(raw: str) -> float:
    """
    Correctly convert a duration string to decimal hours.
      "2h 50m"  ->  2 + 50/60  =  2.833
      "19h"     ->  19.0
      "45m"     ->   0.75
    """
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
    """0=Night (21-05), 1=Morning (06-11), 2=Afternoon (12-17), 3=Evening (18-20)."""
    if 6 <= hour <= 11:
        return 1
    if 12 <= hour <= 17:
        return 2
    if 18 <= hour <= 20:
        return 3
    return 0   # Night


def _airline_tier(airline: str) -> int:
    if airline in _PREMIUM_AIRLINES:
        return 2
    if airline in _ECONOMY_AIRLINES:
        return 1
    return 0   # Budget


# ── main function ─────────────────────────────────────────────────────────────

def engineer_features(eda_df: pd.DataFrame):
    """
    Takes the output of preprocess() (which retains the raw Duration string
    and Journey_Year) and returns:
      enriched_eda_df  — human-readable with all new features, for EDA
      enriched_model_df — one-hot encoded, ready for modelling
    """
    df = eda_df.copy()

    # ── 1. Fix Duration ───────────────────────────────────────────────────────
    if "Duration" in df.columns:
        df["Duration_Hours"] = df["Duration"].apply(_parse_duration)
        df.drop(columns=["Duration"], inplace=True)
        logger.info("Duration_Hours: correctly parsed (e.g. '2h 50m' -> 2.83 h)")
    else:
        logger.warning("'Duration' column not found — skipping Duration_Hours")

    # Remove the broken legacy column if it survived
    if "Duration_Total_Hour" in df.columns:
        df.drop(columns=["Duration_Total_Hour"], inplace=True)

    # ── 2. Drop constant Journey_Year ─────────────────────────────────────────
    if "Journey_Year" in df.columns:
        df.drop(columns=["Journey_Year"], inplace=True)
        logger.info("Journey_Year dropped (constant = 2019, zero variance)")

    # ── 3. Cyclical hour encoding ─────────────────────────────────────────────
    for col, cycle in [("Dep_Hours", 24), ("Arrival_Hours", 24)]:
        prefix = "Dep" if "Dep" in col else "Arr"
        df[f"{prefix}_Hour_Sin"] = np.sin(2 * np.pi * df[col] / cycle)
        df[f"{prefix}_Hour_Cos"] = np.cos(2 * np.pi * df[col] / cycle)
    logger.info("Added cyclical encoding: Dep_Hour_Sin/Cos, Arr_Hour_Sin/Cos")

    # ── 4. Cyclical month and weekday encoding ────────────────────────────────
    df["Month_Sin"]   = np.sin(2 * np.pi * df["Journey_Month"]    / 12)
    df["Month_Cos"]   = np.cos(2 * np.pi * df["Journey_Month"]    / 12)
    df["Weekday_Sin"] = np.sin(2 * np.pi * df["Journey_Weekdays"] / 7)
    df["Weekday_Cos"] = np.cos(2 * np.pi * df["Journey_Weekdays"] / 7)
    logger.info("Added cyclical encoding: Month_Sin/Cos, Weekday_Sin/Cos")

    # ── 5. Departure / arrival time-of-day period ─────────────────────────────
    df["Dep_Period"] = df["Dep_Hours"].apply(_hour_to_period)
    df["Arr_Period"] = df["Arrival_Hours"].apply(_hour_to_period)
    logger.info("Added Dep_Period and Arr_Period  (0=Night 1=Morning 2=Afternoon 3=Evening)")

    # ── 6. Weekend flag ───────────────────────────────────────────────────────
    df["Is_Weekend"] = (df["Journey_Weekdays"] >= 5).astype(int)
    logger.info("Added Is_Weekend")

    # ── 7. Route (combined Source + Destination) ──────────────────────────────
    df["Route"] = (
        df["Source"].str.replace(" ", "_")
        + "_to_"
        + df["Destination"].str.replace(" ", "_")
    )
    n_routes = df["Route"].nunique()
    logger.info(f"Added Route feature  ({n_routes} unique routes)")

    # ── 8. Airline tier ───────────────────────────────────────────────────────
    df["Airline_Tier"] = df["Airline"].apply(_airline_tier)
    logger.info("Added Airline_Tier  (0=Budget  1=Economy/Full-service  2=Premium)")

    # ── save enriched eda (categoricals still as strings) ─────────────────────
    enriched_eda = df.copy()
    enriched_eda.to_csv(PROCESSED_DIR / "enriched_eda_df.csv", index=False)
    logger.info("Saved: outputs/processed/enriched_eda_df.csv")

    # ── one-hot encode for modelling ──────────────────────────────────────────
    n_input_cols = eda_df.shape[1] - 1  # pre-FE feature count (excludes Price)

    cat_cols = ["Airline", "Source", "Destination", "Route"]
    existing_cats = [c for c in cat_cols if c in df.columns]

    enriched_model_df = pd.get_dummies(
        df,
        prefix=existing_cats,
        columns=existing_cats,
        dtype=int,
        drop_first=False,
    )

    logger.info(
        f"\n  Features before FE : {n_input_cols}  (eda_df, excludes Price)"
        f"\n  Features after FE  : {enriched_model_df.shape[1] - 1}  (excludes Price)"
        f"\n  Breakdown of additions:"
        f"\n    + Duration_Hours             (replaces broken Duration_Total_Hour)"
        f"\n    + Dep/Arr_Period             (2 ordinal time-of-day bins)"
        f"\n    + Dep/Arr_Hour_Sin/Cos       (4 cyclical hour features)"
        f"\n    + Month/Weekday_Sin/Cos      (4 cyclical calendar features)"
        f"\n    + Is_Weekend                 (binary)"
        f"\n    + Airline_Tier               (ordinal)"
        f"\n    + Route_*                    ({n_routes} route dummies)"
        f"\n    - Journey_Year               (dropped: constant)"
        f"\n    - Duration_Total_Hour        (dropped: replaced)"
    )

    enriched_model_df.to_csv(PROCESSED_DIR / "enriched_model_df.csv", index=False)
    logger.info("Saved: outputs/processed/enriched_model_df.csv")

    return enriched_eda, enriched_model_df
