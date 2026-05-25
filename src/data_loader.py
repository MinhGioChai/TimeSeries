import logging
import pandas as pd

from src.config import REPORTS_DIR, TRAIN_FILE

logger = logging.getLogger(__name__)


def load_data(path=None) -> pd.DataFrame:
    if path is None:
        path = TRAIN_FILE
    path = str(path)

    logger.info(f"Source: {path}")
    df = pd.read_csv(path)

    logger.info(f"Shape: {df.shape}  ({df.shape[0]} rows x {df.shape[1]} cols)")
    logger.info(f"Columns: {list(df.columns)}")

    missing = df.isna().sum()
    if missing.any():
        logger.info(f"Missing values:\n{missing[missing > 0].to_string()}")
    else:
        logger.info("No missing values")

    logger.info(f"Duplicate rows: {df.duplicated().sum()}")

    _save_data_summary(df)
    return df


def _save_data_summary(df: pd.DataFrame):
    lines = [
        f"Shape: {df.shape}",
        "",
        "Dtypes:",
        df.dtypes.to_string(),
        "",
        "Missing values:",
        df.isna().sum().to_string(),
        "",
        "Numeric summary:",
        df.describe().to_string(),
        "",
        "Categorical summary:",
        df.describe(include="object").to_string(),
    ]
    (REPORTS_DIR / "01_data_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved: outputs/reports/01_data_summary.txt")
