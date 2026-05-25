import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.config import FIGURES_DIR, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR, TEST_SIZE

logger = logging.getLogger(__name__)


def train_test_split_data(model_df: pd.DataFrame):
    X = model_df.drop("Price", axis=1)
    y = model_df["Price"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Train: {len(x_train)} rows  |  Test: {len(x_test)} rows  |  Features: {X.shape[1]}")

    # Save splits so downstream work (inference, error analysis) can reload them
    x_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    x_test.to_csv(PROCESSED_DIR  / "X_test.csv",  index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False, header=True)
    y_test.to_csv(PROCESSED_DIR  / "y_test.csv",  index=False, header=True)
    logger.info("Saved: outputs/processed/X_train.csv  X_test.csv  y_train.csv  y_test.csv")

    return x_train, x_test, y_train, y_test


def _score_model(model, x_train, x_test, y_train, y_test) -> dict:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    n, p = len(x_test), x_test.shape[1]
    adj_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1)) if n > p + 1 else np.nan
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {
        "train_r2": round(model.score(x_train, y_train), 4),
        "test_r2":  round(r2, 4),
        "adj_r2":   round(adj_r2, 4),
        "mae":      round(mae, 2),
        "rmse":     round(rmse, 2),
    }


def compare_models(x_train, x_test, y_train, y_test) -> pd.DataFrame:
    candidates = {
        "LinearRegression":     LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "RandomForestRegressor": RandomForestRegressor(random_state=RANDOM_STATE),
        "XGBRegressor":          XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
        "LGBMRegressor":         LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
    }

    rows = []
    for name, model in candidates.items():
        logger.info(f"  {name} ...")
        metrics = _score_model(model, x_train, x_test, y_train, y_test)
        rows.append({"model": name, **metrics})
        logger.info(
            f"    train_R²={metrics['train_r2']:.4f}  test_R²={metrics['test_r2']:.4f}"
            f"  MAE={metrics['mae']:,.0f}  RMSE={metrics['rmse']:,.0f}"
        )

    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(REPORTS_DIR / "03_baseline_comparison.csv")
    logger.info("Saved: outputs/reports/03_baseline_comparison.csv")

    _plot_comparison(df, "10_baseline_model_comparison", "Baseline Model Comparison")

    best = df["test_r2"].idxmax()
    logger.info(f"Best baseline: {best}  (test R²={df.loc[best,'test_r2']:.4f})")
    return df


def plot_actual_vs_predicted(y_actual, y_pred, model_name: str, fig_name: str):
    """Scatter (actual vs predicted) + residuals histogram for a trained model."""
    y_actual  = np.array(y_actual)
    y_pred    = np.array(y_pred)
    residuals = y_pred - y_actual
    r2  = r2_score(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Actual vs Predicted — {model_name}", fontsize=14)

    # Scatter: actual on x, predicted on y, perfect-prediction diagonal
    ax = axes[0]
    ax.scatter(y_actual, y_pred, alpha=0.25, s=10, color="steelblue")
    lims = [min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Price (INR)")
    ax.set_ylabel("Predicted Price (INR)")
    ax.set_title(f"R² = {r2:.4f}  |  MAE = Rs. {mae:,.0f}")
    ax.legend()

    # Residuals histogram
    ax = axes[1]
    ax.hist(residuals, bins=50, color="steelblue", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero error")
    ax.set_xlabel("Residual (Predicted − Actual) INR")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{fig_name}.png", bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Saved: outputs/figures/{fig_name}.png")


def plot_temporal_comparison(x_test, y_actual, y_pred, fig_name: str):
    """Daily mean actual vs predicted for the temporal test split (June 2019)."""
    y_actual = np.array(y_actual)
    y_pred   = np.array(y_pred)

    tmp = x_test[["Journey_Day"]].copy()
    tmp["Actual"]    = y_actual
    tmp["Predicted"] = y_pred
    by_day = tmp.groupby("Journey_Day")[["Actual", "Predicted"]].mean()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(by_day.index, by_day["Actual"],    "o-",  color="steelblue",
            linewidth=2, markersize=5, label="Actual (daily mean)")
    ax.plot(by_day.index, by_day["Predicted"], "s--", color="darkorange",
            linewidth=2, markersize=5, label="Predicted (daily mean)")
    ax.fill_between(by_day.index, by_day["Actual"], by_day["Predicted"],
                    alpha=0.12, color="gray")
    ax.set_xlabel("Day of June 2019")
    ax.set_ylabel("Mean Fare (INR)")
    ax.set_title("Temporal Split — Actual vs Predicted Daily Mean Fare (June 2019)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{fig_name}.png", bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Saved: outputs/figures/{fig_name}.png")


def _plot_comparison(df: pd.DataFrame, filename: str, title: str):
    metrics = [
        ("test_r2", "R² (higher = better)"),
        ("mae",     "MAE (lower = better)"),
        ("rmse",    "RMSE (lower = better)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title, fontsize=15)

    for ax, (col, label) in zip(axes, metrics):
        ascending = col != "test_r2"
        vals = df[col].sort_values(ascending=ascending)
        sns.barplot(x=vals.values, y=vals.index, hue=vals.index, palette="magma", legend=False, ax=ax)
        ax.set_title(label)
        ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{filename}.png", bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Saved: outputs/figures/{filename}.png")
