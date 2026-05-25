from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

# ── output sub-folders ────────────────────────────────────────────────────────
PROCESSED_DIR = OUTPUT_DIR / "processed"   # cleaned DataFrames + train/test splits
FIGURES_DIR   = OUTPUT_DIR / "figures"     # all PNG plots
MODELS_DIR    = OUTPUT_DIR / "models"      # serialised model files (.pkl)
REPORTS_DIR   = OUTPUT_DIR / "reports"     # metrics CSVs, JSON params, text verdicts

TRAIN_FILE = DATA_DIR / "Data_Train.csv"
TEST_FILE  = DATA_DIR / "Test_set.csv"

RANDOM_STATE = 42
TEST_SIZE    = 0.2
N_TRIALS     = 50   # Optuna trials per model; raise for better tuning

# 2019 average exchange rate (RBI reference): 1 USD = 70.42 INR
INR_TO_USD = 1 / 70.42

MODELS_TO_TUNE = ["RandomForestRegressor", "XGBRegressor", "LGBMRegressor"]


def setup_dirs():
    for d in [PROCESSED_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
