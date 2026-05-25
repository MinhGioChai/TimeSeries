#!/usr/bin/env python
"""
Flight Fare Prediction — ML Pipeline
======================================
Runs every stage end-to-end and saves all artefacts under outputs/.

Usage
-----
    python pipeline.py                              # full run, defaults
    python pipeline.py --n-trials 20               # faster Optuna search
    python pipeline.py --forecast-month 8          # forecast August instead of July
    python pipeline.py --forecast-month 8 --forecast-year 2019 --forecast-hour 7

Optional arguments
------------------
  --n-trials N        Optuna trials per model (default: N_TRIALS from config.py)
  --forecast-month M  Month to forecast, 1-12  (default: 7 = July)
  --forecast-year  Y  Year  to forecast        (default: 2019)
  --forecast-hour  H  Departure hour for synthetic flights, 0-23 (default: 9)

Pipeline steps
--------------
  1  Load Data
  2  Preprocess               (clean, parse dates/times/stops, one-hot encode)
  3  EDA                      (7 figures, on cleaned pre-FE data)
  4  Feature Engineering      (duration fix, cyclical encoding, route, tier)
  5  Random Train/Test Split  (80/20, enriched features)
  6  Feature Importance       (ExtraTreesRegressor)
  7  Baseline Model Comparison
  8  Time-Series Suitability  (stationarity, ADF, TS model verdict)
  9  Statistical Extensions   (final project additions):
       9a  Temporal split  — train Mar-May, test Jun
       9b  SARIMAX         — weekly aggregated prices, AIC order search
       9c  Residual diag.  — Ljung-Box + ACF/PACF on ML residuals
       9d  Hybrid model    — LightGBM enriched with SARIMAX weekly trend
  10 Hyperparameter Tuning    (Optuna, 3 models)
  11 Test Set Prediction      (forecast fares for Test_set.csv)
  12 Route Forecasting        (daily fare forecasts per airline/route, configurable month)

Output layout
-------------
outputs/
  processed/   eda_df.csv  model_df.csv
               enriched_eda_df.csv  enriched_model_df.csv
               X/y_train.csv  X/y_test.csv
               temporal_X/y_train.csv  temporal_X/y_test.csv
               test_predictions.csv
               forecast_jetairways_del_cok.csv
               forecast_indigo_blr_del.csv
               forecast_airline_comparison.csv
               forecast_airline_comparison_long.csv
  figures/     25 PNG plots numbered 01-25
  models/      *_tuned.pkl  best_model.pkl  best_model_full.pkl
               hybrid_lgbm_sarimax.pkl
  reports/     01_data_summary.txt
               02_feature_importance.csv
               03_baseline_comparison.csv
               05_tuning_results.csv
               05_best_params.json
               06_ts_model_verdict.txt
               07_sarimax_summary.txt
               08_residual_ljungbox.csv
               09_hybrid_comparison.txt
               model_evaluation_summary.csv
"""

import argparse
import calendar
import logging
import sys
import time

import pandas as pd

from src.config import MODELS_TO_TUNE, N_TRIALS, REPORTS_DIR, INR_TO_USD, setup_dirs
from src.data_loader import load_data
from src.eda import run_eda
from src.feature_engineering import engineer_features
from src.feature_importance import analyze_features
from src.forecaster import forecast_route, forecast_airline_comparison, plot_forecast
from src.model_evaluator import (
    compare_models, train_test_split_data,
    plot_actual_vs_predicted, plot_temporal_comparison,
)
from src.predictor import predict_test_set
from src.preprocessor import preprocess
from src.statistical_models import (
    temporal_train_test_split,
    run_sarimax_analysis,
    run_residual_diagnostics,
    run_hybrid_model,
    plot_model_comparison_summary,
)
from src.time_series_analysis import run_ts_analysis
from src.tuner import tune_models
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Flight Fare Prediction — ML + Statistical Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--n-trials", type=int, default=None, metavar="N",
        help=f"Optuna trials per model (default: {N_TRIALS} from config.py)",
    )
    p.add_argument(
        "--forecast-month", type=int, default=7, metavar="M",
        help="Month to forecast, 1-12  (7 = July, the month after training ends)",
    )
    p.add_argument(
        "--forecast-year", type=int, default=2019, metavar="Y",
        help="Year to forecast",
    )
    p.add_argument(
        "--forecast-hour", type=int, default=9, metavar="H",
        help="Departure hour for synthetic forecast flights, 0-23",
    )
    args = p.parse_args()
    if not 1 <= args.forecast_month <= 12:
        p.error(f"--forecast-month must be 1-12, got {args.forecast_month}")
    if not 0 <= args.forecast_hour <= 23:
        p.error(f"--forecast-hour must be 0-23, got {args.forecast_hour}")
    return args


def _banner(step, title: str):
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  STEP {step}  |  {title}")
    logger.info("=" * 60)


def _save_evaluation_summary(comparison_data: dict, tuned_results: dict,
                              r2_temp: float, mae_temp: float,
                              sarimax_metrics: dict) -> None:
    """Write outputs/reports/model_evaluation_summary.csv with every model's metrics."""
    rows = []

    # 1. Baseline models from the saved CSV (5 sklearn/XGB/LGBM baselines)
    # CSV has index col "model", then train_r2, test_r2, adj_r2, mae, rmse
    baseline_path = REPORTS_DIR / "03_baseline_comparison.csv"
    if baseline_path.exists():
        bl = pd.read_csv(baseline_path)
        for _, r in bl.iterrows():
            rows.append({
                "Category":  "Baseline",
                "Model":     r["model"],
                "Split":     "Random 80/20",
                "R2":        round(float(r["test_r2"]), 4),
                "MAE_INR":   round(float(r["mae"]), 2),
                "MAE_USD":   round(float(r["mae"]) * INR_TO_USD, 2),
                "RMSE_INR":  round(float(r["rmse"]), 2),
                "RMSE_USD":  round(float(r["rmse"]) * INR_TO_USD, 2),
            })

    # 2. LightGBM on temporal split (in-memory)
    rows.append({
        "Category":  "ML – Temporal",
        "Model":     "LightGBM",
        "Split":     "Temporal Mar-May / Jun",
        "R2":        round(r2_temp, 4),
        "MAE_INR":   round(mae_temp, 2),
        "MAE_USD":   round(mae_temp * INR_TO_USD, 2),
        "RMSE_INR":  None,
        "RMSE_USD":  None,
    })

    # 3. SARIMAX (statistical baseline)
    rows.append({
        "Category":  "Statistical",
        "Model":     "SARIMAX",
        "Split":     "Temporal Mar-May / Jun (weekly agg.)",
        "R2":        sarimax_metrics.get("r2"),
        "MAE_INR":   sarimax_metrics.get("mae"),
        "MAE_USD":   round(sarimax_metrics["mae"] * INR_TO_USD, 2) if sarimax_metrics.get("mae") else None,
        "RMSE_INR":  None,
        "RMSE_USD":  None,
    })

    # 4. Hybrid LightGBM + SARIMAX (in-memory)
    hybrid = comparison_data.get("Hybrid LightGBM+SARIMAX", {})
    rows.append({
        "Category":  "ML – Hybrid",
        "Model":     "LightGBM + SARIMAX",
        "Split":     "Temporal Mar-May / Jun",
        "R2":        round(hybrid.get("r2", float("nan")), 4),
        "MAE_INR":   round(hybrid.get("mae", float("nan")), 2),
        "MAE_USD":   round(hybrid.get("mae", float("nan")) * INR_TO_USD, 2),
        "RMSE_INR":  None,
        "RMSE_USD":  None,
    })

    # 5. Tuned models from the saved CSV
    # CSV columns: model, cv_r2, test_r2, mae, rmse
    tuning_path = REPORTS_DIR / "05_tuning_results.csv"
    if tuning_path.exists():
        tu = pd.read_csv(tuning_path)
        for _, r in tu.iterrows():
            rows.append({
                "Category":  "Tuned",
                "Model":     r["model"],
                "Split":     "Random 80/20",
                "R2":        round(float(r["test_r2"]), 4),
                "MAE_INR":   round(float(r["mae"]), 2),
                "MAE_USD":   round(float(r["mae"]) * INR_TO_USD, 2),
                "RMSE_INR":  round(float(r["rmse"]), 2),
                "RMSE_USD":  round(float(r["rmse"]) * INR_TO_USD, 2),
            })

    summary = pd.DataFrame(rows)
    out_path = REPORTS_DIR / "model_evaluation_summary.csv"
    summary.to_csv(out_path, index=False)
    logger.info(f"Saved: outputs/reports/model_evaluation_summary.csv  ({len(summary)} models)")


def run():
    args = _parse_args()
    n_trials = args.n_trials if args.n_trials is not None else N_TRIALS
    fm = args.forecast_month
    fy = args.forecast_year
    fh = args.forecast_hour
    month_name = calendar.month_name[fm]

    t0 = time.time()
    logger.info("")
    logger.info("=" * 60)
    logger.info("  FLIGHT FARE PREDICTION -- ML + STATISTICAL PIPELINE")
    logger.info(f"  Optuna trials : {n_trials}  |  Models : {MODELS_TO_TUNE}")
    logger.info(f"  Forecast      : {month_name} {fy}  (dep hour {fh:02d}:00)")
    logger.info("=" * 60)

    setup_dirs()

    # ── core ML pipeline ──────────────────────────────────────────────────────
    _banner(1, "Load Data")
    df = load_data()

    _banner(2, "Preprocess  (clean / parse / one-hot encode)")
    eda_df, model_df = preprocess(df)

    _banner(3, "Exploratory Data Analysis  (cleaned, pre-FE)")
    run_eda(eda_df)

    _banner(4, "Feature Engineering  (duration / cyclical / route / tier)")
    enriched_eda_df, enriched_model_df = engineer_features(eda_df)

    _banner(5, "Random Train / Test Split  (80 / 20, enriched)")
    x_train_r, x_test_r, y_train_r, y_test_r = train_test_split_data(enriched_model_df)

    _banner(6, "Feature Importance Analysis")
    analyze_features(x_train_r, y_train_r)

    _banner(7, "Baseline Model Comparison  (random split, enriched)")
    compare_models(x_train_r, x_test_r, y_train_r, y_test_r)

    _banner(8, "Time-Series Model Suitability")
    run_ts_analysis(df)

    # ── statistical extensions (final project) ────────────────────────────────
    _banner("9a", "Temporal Train / Test Split  (Mar-May -> Jun, enriched)")
    x_train_t, x_test_t, y_train_t, y_test_t = temporal_train_test_split(enriched_model_df)

    # LightGBM on temporal split — for temporal baseline + hybrid model
    lgbm_temp = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                               max_depth=7, random_state=42, verbose=-1)
    lgbm_temp.fit(x_train_t, y_train_t)
    yp_temp = lgbm_temp.predict(x_test_t)
    r2_temp  = r2_score(y_test_t, yp_temp)
    mae_temp = mean_absolute_error(y_test_t, yp_temp)
    logger.info(f"LightGBM on temporal split -> R2={r2_temp:.4f}  MAE={mae_temp:,.0f} INR")
    plot_temporal_comparison(x_test_t, y_test_t, yp_temp,
                             "24_temporal_actual_vs_predicted")

    _banner("9b", "SARIMAX on Weekly Aggregated Prices")
    sarimax_result, weekly_series, _, sarimax_metrics = run_sarimax_analysis(df)

    # LightGBM trained on the random split — correct model for residual diagnostics
    # (residuals must come from the same split the model was trained on)
    lgbm_random = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                 max_depth=7, random_state=42, verbose=-1)
    lgbm_random.fit(x_train_r, y_train_r)

    _banner("9c", "ML Residual Diagnostics  (Ljung-Box + ACF/PACF)")
    run_residual_diagnostics(x_test_r, y_test_r, lgbm_random)

    _banner("9d", "Hybrid Model  (LightGBM + SARIMAX weekly-trend feature)")
    _, hybrid_metrics = run_hybrid_model(
        x_train_t, x_test_t, y_train_t, y_test_t,
        sarimax_result, weekly_series,
    )

    # ── hyperparameter tuning ─────────────────────────────────────────────────
    _banner(10, f"Hyperparameter Tuning  (Optuna, {n_trials} trials)")
    tuned_results = tune_models(x_train_r, x_test_r, y_train_r, y_test_r, n_trials=n_trials)

    # ── full model comparison summary ─────────────────────────────────────────
    best_tuned_name = max(tuned_results, key=lambda k: tuned_results[k]["r2"])
    best_tuned = tuned_results[best_tuned_name]

    yp_random = lgbm_random.predict(x_test_r)
    comparison_data = {
        "LightGBM (random split)":   {"r2": r2_score(y_test_r, yp_random),
                                       "mae": mean_absolute_error(y_test_r, yp_random)},
        "LightGBM (temporal split)": {"r2": r2_temp,  "mae": mae_temp},
        "Hybrid LightGBM+SARIMAX":   {"r2": hybrid_metrics["r2"], "mae": hybrid_metrics["mae"]},
        f"Tuned {best_tuned_name}":  {"r2": best_tuned["r2"],     "mae": best_tuned["mae"]},
    }
    plot_model_comparison_summary(comparison_data)
    _save_evaluation_summary(comparison_data, tuned_results, r2_temp, mae_temp, sarimax_metrics)

    best_preds_r = tuned_results[best_tuned_name]["model"].predict(x_test_r)
    plot_actual_vs_predicted(y_test_r, best_preds_r,
                             f"Tuned {best_tuned_name} (random split)",
                             "25_actual_vs_predicted_tuned")

    # ── test set prediction — retrain best model on ALL training data first ───
    # Model selection used 80/20 split; final predictions use a model
    # trained on 100% of Data_Train so no labelled rows are left out.
    train_feature_cols = list(enriched_model_df.drop("Price", axis=1).columns)
    _banner(11, "Test Set Prediction  (retrain on full data -> forecast Test_set.csv)")
    predict_test_set(train_feature_cols, retrain_df=enriched_model_df)

    # ── route forecasting ────────────────────────────────────────────────────
    best_rmse = best_tuned.get("rmse", 1900.0)
    _banner(12, f"Route Forecasting  ({month_name} {fy} daily fares, dep {fh:02d}:00)")

    # Fig 21: Jet Airways Delhi -> Cochin with SARIMAX trend overlay
    jw_forecast = forecast_route(
        airline="Jet Airways", source="Delhi", destination="Cochin",
        train_feature_cols=train_feature_cols,
        dep_hour=fh, forecast_month=fm, forecast_year=fy,
        sarimax_result=sarimax_result,
        csv_name="forecast_jetairways_del_cok",
    )
    plot_forecast(jw_forecast, "Jet Airways", "Delhi", "Cochin",
                  rmse=best_rmse, forecast_month=fm, forecast_year=fy,
                  fig_name="21_forecast_jetairways_del_cok")

    # Fig 22: IndiGo Banglore -> Delhi (budget route)
    ig_forecast = forecast_route(
        airline="IndiGo", source="Banglore", destination="Delhi",
        train_feature_cols=train_feature_cols,
        dep_hour=fh, forecast_month=fm, forecast_year=fy,
        sarimax_result=sarimax_result,
        csv_name="forecast_indigo_blr_del",
    )
    plot_forecast(ig_forecast, "IndiGo", "Banglore", "Delhi",
                  rmse=best_rmse, forecast_month=fm, forecast_year=fy,
                  fig_name="22_forecast_indigo_blr_del")

    # Fig 23: All airlines on Delhi -> Cochin side-by-side
    forecast_airline_comparison(
        source="Delhi", destination="Cochin",
        train_feature_cols=train_feature_cols,
        dep_hour=fh, forecast_month=fm, forecast_year=fy,
        fig_name="23_forecast_comparison_del_cok",
    )

    # ── final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  PIPELINE COMPLETE  ({elapsed:.1f}s)")
    logger.info("")
    logger.info("  Model                       R2       MAE (INR)")
    logger.info("  " + "-" * 44)
    for name, m in comparison_data.items():
        logger.info(f"  {name:<28}  {m['r2']:.4f}   {m['mae']:,.0f}")
    logger.info("")
    logger.info("  outputs/")
    logger.info("  +-- processed/   enriched CSVs + splits + predictions + forecasts")
    logger.info("  +-- figures/     25 PNG plots")
    logger.info("  +-- models/      tuned .pkl + best_model_full.pkl + hybrid .pkl")
    logger.info("  +-- reports/     10 text/CSV files + TS verdict + model_evaluation_summary.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
