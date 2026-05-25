"""
Statistical Time-Series Extensions  (Flight Fare Final Project)
================================================================

Three additions that give the project a complete statistical treatment:

1. Temporal Train/Test Split
   Train on March-May flights, test on June.  Correct TS evaluation:
   no data leakage from future dates.

2. SARIMAX on Weekly Aggregated Prices
   17-18 weekly observations; runs AIC-based order selection, fits
   SARIMAX, plots forecast + residual diagnostics.  Acts as the
   *statistical baseline* to compare against the ML model.

3. Hybrid Model  (LightGBM + SARIMAX weekly-trend feature)
   Enriches each flight record with the SARIMAX in-sample estimate
   for its journey week, then retrains LightGBM.  Demonstrates how
   statistical and ML methods can complement each other.

4. Residual Diagnostics on the ML Model
   Ljung-Box + ACF/PACF on ML residuals sorted by time.
   White-noise result is a *positive finding*: the ML model
   fully captures the temporal structure.
"""

import json
import logging
import warnings

import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import product as iproduct

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from src.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Temporal train / test split
# ─────────────────────────────────────────────────────────────────────────────

def temporal_train_test_split(model_df: pd.DataFrame):
    """
    Train on March-May (Journey_Month < 6), test on June (Journey_Month == 6).
    Simulates predicting future fares from past data — the correct TS protocol.
    """
    train = model_df[model_df["Journey_Month"] < 6].copy()
    test  = model_df[model_df["Journey_Month"] == 6].copy()

    x_train = train.drop("Price", axis=1)
    y_train = train["Price"]
    x_test  = test.drop("Price", axis=1)
    y_test  = test["Price"]

    logger.info(f"Train (Mar-May): {len(train):,} flights")
    logger.info(f"Test  (Jun)    : {len(test):,} flights  ({100*len(test)/len(model_df):.1f}% of data)")

    x_train.to_csv(PROCESSED_DIR / "temporal_X_train.csv", index=False)
    x_test.to_csv( PROCESSED_DIR / "temporal_X_test.csv",  index=False)
    y_train.to_csv(PROCESSED_DIR / "temporal_y_train.csv", index=False, header=True)
    y_test.to_csv( PROCESSED_DIR / "temporal_y_test.csv",  index=False, header=True)
    logger.info("Saved temporal splits to outputs/processed/temporal_*.csv")

    return x_train, x_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 2. SARIMAX on weekly aggregated prices
# ─────────────────────────────────────────────────────────────────────────────

def _build_weekly_series(raw_df: pd.DataFrame) -> pd.Series:
    df = raw_df.dropna(subset=["Date_of_Journey", "Price"]).copy()
    df["date"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
    df["week"] = df["date"].dt.to_period("W").dt.start_time
    weekly = df.groupby("week")["Price"].mean().sort_index()
    weekly.index = pd.DatetimeIndex(weekly.index)
    return weekly


def _aic_order_search(series: pd.Series) -> tuple:
    """Grid search over small ARIMA(p,d,q) orders to minimise AIC."""
    best_aic, best_order = np.inf, (1, 1, 0)
    for p, d, q in iproduct(range(3), range(2), range(3)):
        if p + q == 0:
            continue
        try:
            res = SARIMAX(
                series, order=(p, d, q), trend="n",
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
            if res.aic < best_aic:
                best_aic, best_order = res.aic, (p, d, q)
        except Exception:
            pass
    return best_order, best_aic


def run_sarimax_analysis(raw_df: pd.DataFrame):
    """
    Fit SARIMAX on weekly aggregated prices; return fitted result and series.
    """
    weekly = _build_weekly_series(raw_df)
    n = len(weekly)
    logger.info(f"Weekly series: {n} obs  "
                f"({weekly.index[0].date()} to {weekly.index[-1].date()})")

    # Stationarity
    adf_stat, adf_p = adfuller(weekly, autolag="AIC")[:2]
    logger.info(f"ADF: stat={adf_stat:.3f}  p={adf_p:.4f}  "
                f"-> {'stationary' if adf_p < 0.05 else 'non-stationary'}")

    # Train first 75% of weeks, test remainder
    split = max(int(n * 0.75), n - 4)
    train_s = weekly.iloc[:split]
    test_s  = weekly.iloc[split:]
    logger.info(f"SARIMAX train: {len(train_s)} weeks  |  test: {len(test_s)} weeks")

    # Model selection
    order, best_aic = _aic_order_search(train_s)
    logger.info(f"Best ARIMA order by AIC: ARIMA{order}  (AIC={best_aic:.1f})")

    # Fit on training weeks
    result = SARIMAX(
        train_s, order=order, trend="n",
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    logger.info(f"Fitted  AIC={result.aic:.1f}  BIC={result.bic:.1f}")

    in_sample = result.fittedvalues

    # Forecast test weeks
    forecast_vals = result.forecast(steps=len(test_s))

    test_r2, test_mae = np.nan, np.nan
    if len(test_s) > 1:
        test_r2  = r2_score(test_s, forecast_vals)
        test_mae = mean_absolute_error(test_s, forecast_vals)
        logger.info(f"SARIMAX forecast R2={test_r2:.4f}  MAE={test_mae:,.0f} INR")

    # Plots
    _plot_sarimax(weekly, train_s, test_s, in_sample, forecast_vals)
    _plot_weekly_acf_pacf(weekly)

    # Summary report
    lines = [
        "SARIMAX WEEKLY PRICE ANALYSIS",
        "==============================",
        f"Weekly observations : {n}",
        f"Date range          : {weekly.index[0].date()} to {weekly.index[-1].date()}",
        f"ADF statistic       : {adf_stat:.4f}  (p={adf_p:.6f})",
        f"Stationarity        : {'Yes' if adf_p < 0.05 else 'No — differencing applied'}",
        f"Best ARIMA order    : ARIMA{order}  (AIC={best_aic:.1f})",
        f"Fitted AIC          : {result.aic:.1f}",
        f"Fitted BIC          : {result.bic:.1f}",
        f"Forecast R2         : {test_r2:.4f}" if not np.isnan(test_r2) else "Forecast R2: n/a",
        f"Forecast MAE        : {test_mae:,.0f} INR" if not np.isnan(test_mae) else "Forecast MAE: n/a",
        "",
        "NOTE: Only 17-18 weekly observations — insufficient for reliable annual",
        "seasonality estimation.  Use this as a trend/smoothing baseline only.",
    ]
    (REPORTS_DIR / "07_sarimax_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved: outputs/reports/07_sarimax_summary.txt")

    sarimax_metrics = {"r2": round(test_r2, 4) if not np.isnan(test_r2) else None,
                       "mae": round(test_mae, 2) if not np.isnan(test_mae) else None}
    return result, weekly, in_sample, sarimax_metrics


def _plot_sarimax(weekly, train_s, test_s, in_sample, forecast):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle("SARIMAX on Weekly Mean Flight Price", fontsize=14)

    ax = axes[0]
    ax.plot(weekly.index, weekly.values, "o-", label="Actual weekly mean",
            linewidth=1.5, color="steelblue")
    ax.plot(in_sample.index, in_sample.values, "--", label="SARIMAX in-sample fit",
            linewidth=1.8, color="orange")
    if len(test_s) > 0:
        ax.plot(test_s.index, forecast.values, "s--", label="SARIMAX forecast",
                linewidth=1.8, color="crimson", markersize=6)
    ax.axvline(train_s.index[-1], color="grey", linestyle=":", alpha=0.8,
               label="Train / Test split")
    ax.set_ylabel("Mean Price (INR)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    axes[1].stem(in_sample.index, in_sample.values - train_s.values,
                 linefmt="purple", markerfmt="D", basefmt="k-")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("SARIMAX Residuals (INR)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "15_sarimax_forecast.png", bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved: outputs/figures/15_sarimax_forecast.png")


def _plot_weekly_acf_pacf(weekly: pd.Series):
    lags = min(10, len(weekly) // 2 - 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("ACF / PACF of Weekly Mean Price (ARIMA order identification)", fontsize=13)
    plot_acf( weekly, lags=lags, ax=axes[0])
    plot_pacf(weekly, lags=lags, ax=axes[1], method="ywm")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "16_weekly_acf_pacf.png", bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved: outputs/figures/16_weekly_acf_pacf.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ML residual diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def run_residual_diagnostics(x_test: pd.DataFrame, y_test: pd.Series, model) -> bool:
    """
    Ljung-Box + ACF/PACF on ML residuals sorted by journey date.
    Returns True if residuals are white noise (no autocorrelation).
    """
    y_pred = model.predict(x_test)
    tmp = x_test.copy()
    tmp["residual"] = y_test.values - y_pred
    tmp_sorted = tmp.sort_values(["Journey_Month", "Journey_Day"])
    res_sorted = tmp_sorted["residual"].reset_index(drop=True)

    lags_to_test = [5, 10, 20]
    lb = acorr_ljungbox(res_sorted, lags=lags_to_test, return_df=True)

    logger.info("Ljung-Box test on ML residuals (time-ordered):")
    for lag, row in lb.iterrows():
        conclusion = "white noise" if row["lb_pvalue"] > 0.05 else "AUTOCORRELATED"
        logger.info(f"  lag={lag:>2}  stat={row['lb_stat']:6.2f}  "
                    f"p={row['lb_pvalue']:.4f}  -> {conclusion}")

    white_noise = bool((lb["lb_pvalue"] > 0.05).all())
    if white_noise:
        logger.info("  Result: residuals are white noise — ML fully captures temporal structure")
    else:
        logger.info("  Result: residual autocorrelation found — ARIMA on residuals may help")

    lb.to_csv(REPORTS_DIR / "08_residual_ljungbox.csv")
    logger.info("Saved: outputs/reports/08_residual_ljungbox.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("ML Residual Diagnostics  (time-ordered)", fontsize=14)

    plot_acf( res_sorted, lags=20, ax=axes[0])
    axes[0].set_title("ACF of Residuals")

    plot_pacf(res_sorted, lags=20, ax=axes[1], method="ywm")
    axes[1].set_title("PACF of Residuals")

    axes[2].hist(tmp["residual"], bins=50, edgecolor="white", color="steelblue")
    axes[2].axvline(0, color="red", linewidth=1.5)
    axes[2].set_xlabel("Residual (INR)")
    axes[2].set_title("Residual Distribution")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "17_residual_diagnostics.png", bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved: outputs/figures/17_residual_diagnostics.png")

    return white_noise


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hybrid model  (LightGBM + SARIMAX weekly-trend feature)
# ─────────────────────────────────────────────────────────────────────────────

def run_hybrid_model(
    x_train: pd.DataFrame,
    x_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
    sarimax_result,
    weekly_series: pd.Series,
):
    """
    Adds the SARIMAX in-sample fitted weekly price as one extra feature for
    each flight.  Trains LightGBM with and without this feature and compares.
    """
    logger.info("Building SARIMAX weekly-trend feature for each flight...")
    fitted = sarimax_result.fittedvalues  # weekly fitted values
    fallback = weekly_series.mean()

    def _week_start(row):
        try:
            # Journey_Year is dropped by feature engineering (constant = 2019)
            d = pd.Timestamp(2019, int(row["Journey_Month"]), int(row["Journey_Day"]))
            return d - pd.Timedelta(days=d.weekday())
        except Exception:
            return None

    def _enrich(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        week_vals = X.apply(
            lambda r: fitted.get(_week_start(r), weekly_series.get(_week_start(r), fallback)),
            axis=1,
        )
        X["sarimax_weekly_price"] = week_vals.values
        return X

    x_train_h = _enrich(x_train)
    x_test_h  = _enrich(x_test)

    lgbm_params = dict(n_estimators=300, learning_rate=0.05,
                       max_depth=7, random_state=RANDOM_STATE, verbose=-1)

    # Baseline (no SARIMAX feature)
    base = LGBMRegressor(**lgbm_params).fit(x_train, y_train)
    yb   = base.predict(x_test)
    r2_b = r2_score(y_test, yb)
    mae_b = mean_absolute_error(y_test, yb)

    # Hybrid
    hybrid = LGBMRegressor(**lgbm_params).fit(x_train_h, y_train)
    yh    = hybrid.predict(x_test_h)
    r2_h  = r2_score(y_test, yh)
    mae_h = mean_absolute_error(y_test, yh)

    logger.info(f"LightGBM baseline        R2={r2_b:.4f}  MAE={mae_b:,.0f} INR")
    logger.info(f"LightGBM + SARIMAX feat  R2={r2_h:.4f}  MAE={mae_h:,.0f} INR")
    logger.info(f"Delta                    dR2={r2_h-r2_b:+.4f}  dMAE={mae_b-mae_h:+,.0f} INR")

    joblib.dump(hybrid, MODELS_DIR / "hybrid_lgbm_sarimax.pkl")
    logger.info("Saved: outputs/models/hybrid_lgbm_sarimax.pkl")

    _plot_hybrid_scatter(y_test, yb, yh, r2_b, mae_b, r2_h, mae_h)

    lines = [
        "HYBRID MODEL COMPARISON",
        "=======================",
        "",
        f"LightGBM baseline         R2={r2_b:.4f}  MAE={mae_b:,.0f} INR",
        f"LightGBM + SARIMAX feat   R2={r2_h:.4f}  MAE={mae_h:,.0f} INR",
        f"Improvement               dR2={r2_h-r2_b:+.4f}  dMAE={mae_b-mae_h:+,.0f} INR",
        "",
        "Extra feature: SARIMAX in-sample weekly fitted price for each flight's",
        "journey week.  Captures macro price level that shifts week-by-week.",
    ]
    (REPORTS_DIR / "09_hybrid_comparison.txt").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved: outputs/reports/09_hybrid_comparison.txt")

    return hybrid, {"r2": r2_h, "mae": mae_h, "r2_base": r2_b, "mae_base": mae_b}


def _plot_hybrid_scatter(y_test, yb, yh, r2_b, mae_b, r2_h, mae_h):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Predicted vs Actual: Baseline LightGBM  vs  Hybrid LightGBM + SARIMAX", fontsize=13)

    for ax, y_pred, r2, mae, label, color in [
        (axes[0], yb, r2_b, mae_b, "Baseline LightGBM",         "steelblue"),
        (axes[1], yh, r2_h, mae_h, "Hybrid: LightGBM + SARIMAX", "darkgreen"),
    ]:
        ax.scatter(y_test, y_pred, alpha=0.25, s=7, color=color)
        lo = min(y_test.min(), y_pred.min())
        hi = max(y_test.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlabel("Actual Price (INR)")
        ax.set_ylabel("Predicted Price (INR)")
        ax.set_title(f"{label}\nR2={r2:.4f}   MAE={mae:,.0f} INR")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "18_hybrid_scatter.png", bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved: outputs/figures/18_hybrid_scatter.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Comparison summary: all models side by side
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison_summary(comparison_data: dict):
    """
    Bar chart comparing SARIMAX, ML (random split), ML (temporal split),
    and Hybrid across R2 and MAE.
    """
    names  = list(comparison_data.keys())
    r2s    = [comparison_data[n]["r2"]  for n in names]
    maes   = [comparison_data[n]["mae"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Full Model Comparison: Statistical vs ML vs Hybrid", fontsize=14)

    colors = ["#e07b54", "#4c8fbd", "#5a9e6f", "#8e6bbf"][:len(names)]

    for ax, vals, label, ascending in [
        (axes[0], r2s,  "R2 Score (higher = better)", False),
        (axes[1], maes, "MAE  INR (lower = better)",  True),
    ]:
        order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=(not ascending))
        sorted_names = [names[i] for i in order]
        sorted_vals  = [vals[i]  for i in order]
        sorted_colors = [colors[i % len(colors)] for i in order]
        bars = ax.barh(sorted_names, sorted_vals, color=sorted_colors)
        ax.bar_label(bars, fmt="%.4f" if "R2" in label else "%.0f", padding=3)
        ax.set_title(label)
        ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "19_full_model_comparison.png", bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved: outputs/figures/19_full_model_comparison.png")
