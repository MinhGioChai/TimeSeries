import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor

from src.config import FIGURES_DIR, RANDOM_STATE, REPORTS_DIR

logger = logging.getLogger(__name__)


def analyze_features(x_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    logger.info("Fitting ExtraTreesRegressor to rank features...")
    model = ExtraTreesRegressor(random_state=RANDOM_STATE)
    model.fit(x_train, y_train)

    rank = (
        pd.DataFrame({"feature": x_train.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    rank["cumulative_pct"] = (rank["importance"].cumsum() * 100).round(2)

    logger.info("Top 10 features:")
    for _, row in rank.head(10).iterrows():
        logger.info(f"  {row['feature']:<35} {row['importance']:.4f}   (cum {row['cumulative_pct']:.1f}%)")

    rank.to_csv(REPORTS_DIR / "02_feature_importance.csv", index=False)
    logger.info("Saved: outputs/reports/02_feature_importance.csv")

    # Bar plot – all features
    plt.figure(figsize=(10, 9))
    sns.barplot(x="importance", y="feature", data=rank, hue="feature", palette="viridis", legend=False)
    plt.title("Feature Importance (ExtraTreesRegressor)", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08_feature_importance.png", bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved: outputs/figures/08_feature_importance.png")

    # Cumulative importance curve
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(rank) + 1), rank["cumulative_pct"], marker="o", linewidth=2)
    plt.axhline(90, color="red", linestyle="--", label="90% threshold")
    plt.xlabel("Number of features")
    plt.ylabel("Cumulative importance (%)")
    plt.title("Cumulative Feature Importance", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_cumulative_feature_importance.png", bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved: outputs/figures/09_cumulative_feature_importance.png")

    return rank
