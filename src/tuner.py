import json
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.config import (
    FIGURES_DIR,
    MODELS_DIR,
    MODELS_TO_TUNE,
    N_TRIALS,
    RANDOM_STATE,
    REPORTS_DIR,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


class RegressorOptunaTuner:
    def __init__(self, model_name: str, x_train, y_train, x_test, y_test, n_trials: int = N_TRIALS):
        self.model_name = model_name
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.n_trials = n_trials
        self.best_model = None
        self.best_params: dict = {}
        self.best_cv_score: float = 0.0

    # ── parameter spaces ─────────────────────────────────────────────────────

    def _build_model(self, trial: optuna.Trial):
        name = self.model_name

        if name == "RandomForestRegressor":
            p = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "random_state": RANDOM_STATE,
            }
            return RandomForestRegressor(**p)

        if name == "XGBRegressor":
            p = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 0.5),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 0.1, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                "random_state": RANDOM_STATE,
                "verbosity": 0,
            }
            return XGBRegressor(**p)

        if name == "LGBMRegressor":
            p = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                "random_state": RANDOM_STATE,
                "verbose": -1,
            }
            return LGBMRegressor(**p)

        raise ValueError(f"Unsupported model: {name}")

    # ── Optuna objective ──────────────────────────────────────────────────────

    def _objective(self, trial: optuna.Trial) -> float:
        model = self._build_model(trial)
        scores = cross_val_score(model, self.x_train, self.y_train, cv=5, scoring="r2")
        return float(scores.mean())

    # ── public API ────────────────────────────────────────────────────────────

    def tune(self) -> "RegressorOptunaTuner":
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params = study.best_trial.params
        self.best_cv_score = study.best_value

        # Rebuild best model and train on full training set
        if self.model_name == "RandomForestRegressor":
            self.best_model = RandomForestRegressor(**self.best_params, random_state=RANDOM_STATE)
        elif self.model_name == "XGBRegressor":
            self.best_model = XGBRegressor(**self.best_params, random_state=RANDOM_STATE, verbosity=0)
        elif self.model_name == "LGBMRegressor":
            self.best_model = LGBMRegressor(**self.best_params, random_state=RANDOM_STATE, verbose=-1)

        self.best_model.fit(self.x_train, self.y_train)
        return self

    def evaluate(self) -> dict:
        y_pred = self.best_model.predict(self.x_test)
        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        return {"r2": round(r2, 4), "mae": round(mae, 2), "rmse": round(rmse, 2)}


# ── pipeline function ─────────────────────────────────────────────────────────

def tune_models(x_train, x_test, y_train, y_test, n_trials: int = N_TRIALS) -> dict:
    results: dict = {}

    for model_name in MODELS_TO_TUNE:
        logger.info(f"  {model_name}  ({n_trials} Optuna trials) ...")
        tuner = RegressorOptunaTuner(model_name, x_train, y_train, x_test, y_test, n_trials=n_trials)
        tuner.tune()
        metrics = tuner.evaluate()

        results[model_name] = {
            "model": tuner.best_model,
            "params": tuner.best_params,
            "cv_r2": round(tuner.best_cv_score, 4),
            **metrics,
        }

        logger.info(
            f"    CV R²={tuner.best_cv_score:.4f}  |  "
            f"Test R²={metrics['r2']:.4f}  MAE={metrics['mae']:,.0f}  RMSE={metrics['rmse']:,.0f}"
        )

        # Save individual model
        joblib.dump(tuner.best_model, MODELS_DIR / f"{model_name}_tuned.pkl")
        logger.info(f"    Saved: outputs/models/{model_name}_tuned.pkl")

    _save_reports(results)
    _plot_tuned_comparison(results)

    best_name = max(results, key=lambda k: results[k]["r2"])
    logger.info(f"Best tuned model: {best_name}  (test R²={results[best_name]['r2']:.4f})")
    joblib.dump(results[best_name]["model"], MODELS_DIR / "best_model.pkl")
    logger.info("Saved: outputs/models/best_model.pkl")

    return results


def _save_reports(results: dict):
    rows = []
    params_out = {}
    for name, res in results.items():
        row = {"model": name, "cv_r2": res["cv_r2"], "test_r2": res["r2"], "mae": res["mae"], "rmse": res["rmse"]}
        rows.append(row)
        params_out[name] = res["params"]

    pd.DataFrame(rows).set_index("model").to_csv(REPORTS_DIR / "05_tuning_results.csv")
    logger.info("Saved: outputs/reports/05_tuning_results.csv")

    with open(REPORTS_DIR / "05_best_params.json", "w") as f:
        json.dump(params_out, f, indent=2)
    logger.info("Saved: outputs/reports/05_best_params.json")


def _plot_tuned_comparison(results: dict):
    names = list(results.keys())
    data = {
        "R² Score": [results[n]["r2"] for n in names],
        "MAE": [results[n]["mae"] for n in names],
        "RMSE": [results[n]["rmse"] for n in names],
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Tuned Model Comparison", fontsize=15)

    for ax, (metric, vals) in zip(axes, data.items()):
        ascending = metric != "R² Score"
        df = pd.DataFrame({"model": names, "value": vals}).sort_values("value", ascending=ascending)
        sns.barplot(x="value", y="model", data=df, hue="model", palette="crest", legend=False, ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "11_tuned_model_comparison.png", bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved: outputs/figures/11_tuned_model_comparison.png")
