"""
Flight Fare Forecaster
======================
Generates price forecasts for a specific airline/route for any future month.

How it works
------------
1. Build synthetic flight records for every day of the target month
   (one flight per day with the specified airline/route/stops/hour).
2. Apply the exact same feature engineering transformations used in training
   (cyclical encoding, periods, route OHE, airline tier, etc.).
3. Align the feature matrix to the 45 training columns.
4. Predict with best_model_full.pkl — trained on ALL of Data_Train.csv.
5. Overlay the SARIMAX weekly trend (if provided) to show the macro drift.
6. Output a DataFrame of daily predicted fares + a figure.

Why it works for July 2019 (unseen month)
------------------------------------------
The cyclical Month_Sin/Cos encoding places July (month=7) naturally on
the 12-month circle. July sin/cos values are interpolated between June
and August, so the model generalises without extrapolating a raw integer.
"""

import calendar
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, INR_TO_USD, MODELS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

# ── constants (must match feature_engineering.py) ─────────────────────────────

_PREMIUM_AIRLINES = {"Jet Airways", "Vistara"}
_ECONOMY_AIRLINES = {"Air India", "Multiple carriers"}


def _airline_tier(airline: str) -> int:
    if airline in _PREMIUM_AIRLINES:
        return 2
    if airline in _ECONOMY_AIRLINES:
        return 1
    return 0


def _hour_to_period(hour: int) -> int:
    if 6 <= hour <= 11:
        return 1
    if 12 <= hour <= 17:
        return 2
    if 18 <= hour <= 20:
        return 3
    return 0


# ── helpers ───────────────────────────────────────────────────────────────────

def _typical_duration(airline: str, source: str, destination: str) -> float:
    """Return median Duration_Hours for this airline/route from training data."""
    try:
        eda = pd.read_csv(PROCESSED_DIR / "enriched_eda_df.csv")
    except FileNotFoundError:
        return 5.0
    mask = (eda["Airline"] == airline) & (eda["Source"] == source) & (eda["Destination"] == destination)
    subset = eda[mask]
    if len(subset) == 0:
        mask = (eda["Source"] == source) & (eda["Destination"] == destination)
        subset = eda[mask]
    return float(subset["Duration_Hours"].median()) if len(subset) > 0 else 5.0


def _typical_stops(airline: str, source: str, destination: str) -> int:
    """Return most common Total_Stops for this airline/route from training data."""
    try:
        eda = pd.read_csv(PROCESSED_DIR / "enriched_eda_df.csv")
    except FileNotFoundError:
        return 1
    mask = (eda["Airline"] == airline) & (eda["Source"] == source) & (eda["Destination"] == destination)
    subset = eda[mask]
    if len(subset) == 0:
        mask = (eda["Source"] == source) & (eda["Destination"] == destination)
        subset = eda[mask]
    if len(subset) == 0:
        return 1
    mode_vals = subset["Total_Stops"].mode()
    return int(mode_vals.iloc[0]) if len(mode_vals) > 0 else 1


def _build_synthetic_flights(
    airline: str,
    source: str,
    destination: str,
    total_stops: int,
    dep_hour: int,
    duration_hours: float,
    forecast_year: int,
    forecast_month: int,
) -> pd.DataFrame:
    """One synthetic flight per day of the forecast month."""
    n_days = calendar.monthrange(forecast_year, forecast_month)[1]
    arr_hour = int(dep_hour + duration_hours) % 24
    arr_min  = int((duration_hours % 1) * 60)

    records = []
    for day in range(1, n_days + 1):
        dt = pd.Timestamp(year=forecast_year, month=forecast_month, day=day)
        records.append({
            "Airline":          airline,
            "Source":           source,
            "Destination":      destination,
            "Total_Stops":      total_stops,
            "Journey_Day":      day,
            "Journey_Month":    forecast_month,
            "Journey_Weekdays": dt.weekday(),
            "Dep_Hours":        dep_hour,
            "Dep_Min":          0,
            "Arrival_Hours":    arr_hour,
            "Arrival_Min":      arr_min,
            "Duration_Hours":   duration_hours,
        })
    return pd.DataFrame(records)


def _apply_fe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all FE transformations to an already-parsed synthetic DataFrame."""
    df = df.copy()

    # Cyclical hour encoding
    df["Dep_Hour_Sin"] = np.sin(2 * np.pi * df["Dep_Hours"] / 24)
    df["Dep_Hour_Cos"] = np.cos(2 * np.pi * df["Dep_Hours"] / 24)
    df["Arr_Hour_Sin"] = np.sin(2 * np.pi * df["Arrival_Hours"] / 24)
    df["Arr_Hour_Cos"] = np.cos(2 * np.pi * df["Arrival_Hours"] / 24)

    # Cyclical calendar encoding
    df["Month_Sin"]   = np.sin(2 * np.pi * df["Journey_Month"]    / 12)
    df["Month_Cos"]   = np.cos(2 * np.pi * df["Journey_Month"]    / 12)
    df["Weekday_Sin"] = np.sin(2 * np.pi * df["Journey_Weekdays"] / 7)
    df["Weekday_Cos"] = np.cos(2 * np.pi * df["Journey_Weekdays"] / 7)

    # Time-of-day periods
    df["Dep_Period"] = df["Dep_Hours"].apply(_hour_to_period)
    df["Arr_Period"] = df["Arrival_Hours"].apply(_hour_to_period)

    # Weekend flag
    df["Is_Weekend"] = (df["Journey_Weekdays"] >= 5).astype(int)

    # Route string + airline tier
    df["Route"] = (
        df["Source"].str.replace(" ", "_")
        + "_to_"
        + df["Destination"].str.replace(" ", "_")
    )
    df["Airline_Tier"] = df["Airline"].apply(_airline_tier)

    # One-hot encode categoricals
    cat_cols = ["Airline", "Source", "Destination", "Route"]
    existing = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, prefix=existing, columns=existing, dtype=int, drop_first=False)

    return df


def _align(df: pd.DataFrame, train_cols: list) -> pd.DataFrame:
    for col in train_cols:
        if col not in df.columns:
            df[col] = 0
    return df[train_cols]


def _sarimax_weekly_trend(sarimax_result, forecast_month: int, forecast_year: int) -> pd.Series:
    """
    Extrapolate the fitted SARIMAX weekly model forward to cover the forecast month.
    Returns a Series mapping each day to the SARIMAX weekly mean price estimate.
    """
    try:
        n_weeks  = 8   # forecast far enough to cover the full month
        forecast = sarimax_result.forecast(steps=n_weeks)

        # Build a weekly index starting from the first Monday on/after July 1
        start = pd.Timestamp(f"{forecast_year}-{forecast_month:02d}-01")
        weekly_dates = pd.date_range(start=start, periods=n_weeks, freq="W-MON")

        weekly = pd.Series(forecast.values, index=weekly_dates)

        # Map each calendar day to the nearest weekly value
        n_days   = calendar.monthrange(forecast_year, forecast_month)[1]
        all_days = pd.date_range(
            start=pd.Timestamp(f"{forecast_year}-{forecast_month:02d}-01"),
            periods=n_days, freq="D"
        )
        daily = weekly.reindex(all_days, method="nearest")
        return daily
    except Exception:
        return None


# ── main public function ───────────────────────────────────────────────────────

def forecast_route(
    airline: str,
    source: str,
    destination: str,
    train_feature_cols: list,
    total_stops: int = None,
    dep_hour: int = 9,
    forecast_year: int = 2019,
    forecast_month: int = 7,
    sarimax_result=None,
    csv_name: str = None,
) -> pd.DataFrame:
    """
    Forecast daily fares for a specific airline + route for an entire month.

    Parameters
    ----------
    airline            : e.g. "Jet Airways"
    source             : e.g. "Delhi"
    destination        : e.g. "Cochin"
    train_feature_cols : list of feature column names from enriched_model_df
    total_stops        : defaults to most common stops for this route/airline
    dep_hour           : departure hour (0-23), default 9 (9am)
    forecast_year      : default 2019
    forecast_month     : default 7 (July)
    sarimax_result     : fitted statsmodels SARIMAX result object (optional)
    csv_name           : if given, saves a CSV to outputs/processed/<csv_name>.csv

    Returns
    -------
    pd.DataFrame  with columns: Date, Weekday, Is_Weekend, Predicted_Price[, SARIMAX_Trend]
    """
    month_name = calendar.month_name[forecast_month]
    logger.info(f"Forecasting: {airline}  {source} -> {destination}  {month_name} {forecast_year}")

    # ── defaults from training data ───────────────────────────────────────────
    if total_stops is None:
        total_stops = _typical_stops(airline, source, destination)
    duration_h = _typical_duration(airline, source, destination)
    logger.info(f"  Using: stops={total_stops}  duration={duration_h:.2f}h  dep_hour={dep_hour:02d}:00")

    # ── build + transform synthetic flights ───────────────────────────────────
    synthetic = _build_synthetic_flights(
        airline, source, destination, total_stops,
        dep_hour, duration_h, forecast_year, forecast_month,
    )
    engineered = _apply_fe(synthetic)
    X = _align(engineered, train_feature_cols)

    # ── predict ───────────────────────────────────────────────────────────────
    model_path = MODELS_DIR / "best_model_full.pkl"
    if not model_path.exists():
        model_path = MODELS_DIR / "best_model.pkl"
    model = joblib.load(model_path)
    preds = model.predict(X)

    # ── assemble result DataFrame ─────────────────────────────────────────────
    n_days = calendar.monthrange(forecast_year, forecast_month)[1]
    dates  = pd.date_range(
        start=f"{forecast_year}-{forecast_month:02d}-01", periods=n_days, freq="D"
    )
    result = pd.DataFrame({
        "Date":            dates,
        "Weekday":         [d.strftime("%a") for d in dates],
        "Is_Weekend":      [1 if d.weekday() >= 5 else 0 for d in dates],
        "Predicted_Price": preds.round(0).astype(int),
    })

    # ── SARIMAX weekly trend overlay ──────────────────────────────────────────
    if sarimax_result is not None:
        trend = _sarimax_weekly_trend(sarimax_result, forecast_month, forecast_year)
        if trend is not None:
            result["SARIMAX_Trend"] = trend.values.round(0).astype(int)
            logger.info("  SARIMAX weekly trend overlay added")

    logger.info(
        f"\n  {month_name} {forecast_year} fare forecast  ({airline}  {source} -> {destination})"
        f"\n    Min  : Rs. {result['Predicted_Price'].min():,}"
        f"\n    Max  : Rs. {result['Predicted_Price'].max():,}"
        f"\n    Mean : Rs. {result['Predicted_Price'].mean():,.0f}"
        f"\n    Weekday avg   : Rs. {result[result['Is_Weekend']==0]['Predicted_Price'].mean():,.0f}"
        f"\n    Weekend avg   : Rs. {result[result['Is_Weekend']==1]['Predicted_Price'].mean():,.0f}"
    )

    if csv_name:
        out = result.copy()
        out.insert(0, "Airline", airline)
        out.insert(1, "Source", source)
        out.insert(2, "Destination", destination)
        out["Predicted_Price_USD"] = (out["Predicted_Price"] * INR_TO_USD).round(2)
        out.to_csv(PROCESSED_DIR / f"{csv_name}.csv", index=False)
        logger.info(f"Saved: outputs/processed/{csv_name}.csv")

    return result


def plot_forecast(
    result: pd.DataFrame,
    airline: str,
    source: str,
    destination: str,
    forecast_month: int = 7,
    forecast_year: int = 2019,
    rmse: float = 1900.0,
    fig_name: str = "21_route_forecast",
):
    """Plot daily fare forecast with uncertainty band and optional SARIMAX trend."""
    month_name = calendar.month_name[forecast_month]
    has_sarimax = "SARIMAX_Trend" in result.columns

    fig, axes = plt.subplots(1 + int(has_sarimax), 1,
                             figsize=(14, 5 + 4 * int(has_sarimax)),
                             sharex=not has_sarimax)
    ax = axes if not has_sarimax else axes[0]

    prices = result["Predicted_Price"].values
    dates  = result["Date"]

    # Confidence band (±1 RMSE from training)
    ax.fill_between(dates, prices - rmse, prices + rmse,
                    alpha=0.15, color="steelblue", label=f"±1 RMSE (Rs. {rmse:,.0f})")
    ax.plot(dates, prices, color="steelblue", linewidth=2, marker="o",
            markersize=4, label="Predicted fare")

    # Highlight weekends
    for _, row in result[result["Is_Weekend"] == 1].iterrows():
        ax.axvspan(row["Date"] - pd.Timedelta(hours=12),
                   row["Date"] + pd.Timedelta(hours=12),
                   alpha=0.08, color="orange")

    ax.set_title(
        f"{airline}  |  {source} -> {destination}  |  {month_name} {forecast_year}",
        fontsize=13
    )
    ax.set_ylabel("Fare (INR)")
    ax.legend(loc="upper left")
    ax.tick_params(axis="x", rotation=30)

    # SARIMAX panel
    if has_sarimax:
        axes[1].plot(dates, result["SARIMAX_Trend"], color="darkorange",
                     linewidth=2, linestyle="--", marker="s", markersize=4,
                     label="SARIMAX weekly trend")
        axes[1].set_title("SARIMAX Weekly Price Trend (macro forecast)", fontsize=12)
        axes[1].set_ylabel("Weekly mean fare (INR)")
        axes[1].legend(loc="upper left")
        axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = FIGURES_DIR / f"{fig_name}.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Saved: outputs/figures/{fig_name}.png")


def forecast_airline_comparison(
    source: str,
    destination: str,
    train_feature_cols: list,
    airlines: list = None,
    dep_hour: int = 9,
    forecast_year: int = 2019,
    forecast_month: int = 7,
    fig_name: str = "22_airline_comparison_forecast",
):
    """
    Compare July fare forecasts across multiple airlines on the same route.
    If airlines is None, uses all airlines that served this route in training.
    """
    if airlines is None:
        try:
            eda = pd.read_csv(PROCESSED_DIR / "enriched_eda_df.csv")
            mask = (eda["Source"] == source) & (eda["Destination"] == destination)
            airlines = sorted(eda[mask]["Airline"].unique().tolist())
        except FileNotFoundError:
            airlines = ["IndiGo", "Jet Airways", "Air India"]

    month_name = calendar.month_name[forecast_month]
    logger.info(f"Airline comparison forecast: {source} -> {destination}  {month_name} {forecast_year}")
    logger.info(f"  Airlines: {airlines}")

    model_path = MODELS_DIR / "best_model_full.pkl"
    if not model_path.exists():
        model_path = MODELS_DIR / "best_model.pkl"
    model = joblib.load(model_path)

    n_days = calendar.monthrange(forecast_year, forecast_month)[1]
    dates  = pd.date_range(start=f"{forecast_year}-{forecast_month:02d}-01",
                           periods=n_days, freq="D")

    all_forecasts = {}
    for airline in airlines:
        stops    = _typical_stops(airline, source, destination)
        duration = _typical_duration(airline, source, destination)
        synthetic  = _build_synthetic_flights(
            airline, source, destination, stops,
            dep_hour, duration, forecast_year, forecast_month,
        )
        X = _align(_apply_fe(synthetic), train_feature_cols)
        preds = model.predict(X).round(0).astype(int)
        all_forecasts[airline] = preds
        logger.info(f"  {airline:<30}  mean Rs. {preds.mean():,.0f}  (stops={stops})")

    # Plot
    plt.figure(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(airlines)))
    for (airline, preds), color in zip(all_forecasts.items(), colors):
        plt.plot(dates, preds, linewidth=2, marker="o", markersize=3,
                 label=airline, color=color)

    plt.title(f"Fare Forecast Comparison: {source} -> {destination}  |  {month_name} {forecast_year}",
              fontsize=13)
    plt.ylabel("Predicted Fare (INR)")
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tick_params(axis="x", rotation=30)
    plt.tight_layout()

    path = FIGURES_DIR / f"{fig_name}.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Saved: outputs/figures/{fig_name}.png")

    # Save wide-format CSV (one column per airline)
    df_wide = pd.DataFrame(all_forecasts, index=dates)
    df_wide.index.name = "Date"
    df_wide.to_csv(PROCESSED_DIR / "forecast_airline_comparison.csv")
    logger.info("Saved: outputs/processed/forecast_airline_comparison.csv")

    # Save long-format CSV (one row per airline-day, easier for Excel/reports)
    df_long = df_wide.reset_index().melt(id_vars="Date", var_name="Airline",
                                         value_name="Predicted_Price_INR")
    df_long["Source"]              = source
    df_long["Destination"]         = destination
    df_long["Weekday"]             = pd.to_datetime(df_long["Date"]).dt.strftime("%a")
    df_long["Is_Weekend"]          = (pd.to_datetime(df_long["Date"]).dt.weekday >= 5).astype(int)
    df_long["Predicted_Price_USD"] = (df_long["Predicted_Price_INR"] * INR_TO_USD).round(2)
    df_long = df_long[["Date", "Weekday", "Is_Weekend", "Airline",
                        "Source", "Destination",
                        "Predicted_Price_INR", "Predicted_Price_USD"]]
    df_long = df_long.sort_values(["Airline", "Date"]).reset_index(drop=True)
    df_long.to_csv(PROCESSED_DIR / "forecast_airline_comparison_long.csv", index=False)
    logger.info("Saved: outputs/processed/forecast_airline_comparison_long.csv")

    return df_wide
