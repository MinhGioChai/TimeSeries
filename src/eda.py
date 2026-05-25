import logging

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.config import FIGURES_DIR

logger = logging.getLogger(__name__)

sns.set_theme()


# ── helpers ──────────────────────────────────────────────────────────────────

def _save(name: str):
    path = FIGURES_DIR / f"{name}.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Saved: outputs/figures/{name}.png")


# ── individual plots ──────────────────────────────────────────────────────────

def _price_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Flight Price Distribution", fontsize=15)

    sns.histplot(df["Price"], kde=True, bins=40, ax=axes[0])
    axes[0].set_title("Histogram + KDE")
    axes[0].set_xlabel("Price (INR)")

    sns.boxplot(x=df["Price"], ax=axes[1])
    axes[1].set_title("Box Plot")
    axes[1].set_xlabel("Price (INR)")

    plt.tight_layout()
    _save("01_price_distribution")


def _categorical_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Categorical Feature Distributions", fontsize=15)

    for ax, col in zip(axes.flatten(), ["Airline", "Source", "Destination", "Total_Stops"]):
        if df[col].dtype == object:
            order = df[col].value_counts().index
            sns.countplot(data=df, y=col, order=order, ax=ax)
        else:
            sns.countplot(data=df, x=col, ax=ax)
        ax.set_title(col)

    plt.tight_layout()
    _save("02_categorical_distributions")


def _categorical_vs_price(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle("Categorical Features vs Price", fontsize=15)

    for ax, col in zip(axes, ["Airline", "Source", "Destination"]):
        order = df.groupby(col)["Price"].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x=col, y="Price", hue=col, order=order, palette="husl", legend=False, ax=ax)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
        ax.set_title(f"{col} vs Price")

    plt.tight_layout()
    _save("03_categorical_vs_price")


def _stops_vs_price(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Total Stops vs Price", fontsize=15)

    sns.countplot(data=df, x="Total_Stops", ax=axes[0])
    axes[0].set_title("Stop Count Distribution")

    sns.boxplot(data=df, x="Total_Stops", y="Price", ax=axes[1])
    axes[1].set_title("Price by Stop Count")

    plt.tight_layout()
    _save("04_stops_vs_price")


def _numerical_distributions(df: pd.DataFrame):
    num_cols = [c for c in df.select_dtypes(include="number").columns if c != "Price"]
    n = len(num_cols)
    rows = (n + 1) // 2

    fig, axes = plt.subplots(rows, 2, figsize=(14, rows * 4))
    fig.suptitle("Numerical Feature Distributions", fontsize=15)
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(col)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    _save("05_numerical_distributions")


def _correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include="number")
    plt.figure(figsize=(12, 9))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Correlation Heatmap", fontsize=15)
    plt.tight_layout()
    _save("06_correlation_heatmap")


def _monthly_price_trend(df: pd.DataFrame):
    monthly = df.groupby("Journey_Month")["Price"].mean().reset_index()
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=monthly, x="Journey_Month", y="Price", marker="o")
    plt.title("Average Price by Journey Month", fontsize=14)
    plt.xlabel("Month")
    plt.ylabel("Mean Price (INR)")
    plt.xticks(monthly["Journey_Month"])
    plt.tight_layout()
    _save("07_monthly_price_trend")


# ── public entry point ────────────────────────────────────────────────────────

def run_eda(eda_df: pd.DataFrame):
    logger.info("Generating EDA figures...")

    _price_distribution(eda_df)
    _categorical_distributions(eda_df)
    _categorical_vs_price(eda_df)
    _stops_vs_price(eda_df)
    _numerical_distributions(eda_df)
    _correlation_heatmap(eda_df)
    _monthly_price_trend(eda_df)

    logger.info("All EDA figures saved to outputs/figures/")
