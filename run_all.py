"""Runs the whole project from raw data to final results.

Usage:  python run_all.py
"""
import config
from src.utils import set_seed, ensure_dirs


def main() -> None:
    set_seed(config.SEED)
    ensure_dirs()

    print("1. Load MT5 data            -> src/load.py      (Step 2)")
    print("2. Detect setups            -> src/detector.py  (Step 3)")
    print("3. Label trades             -> src/labels.py    (Step 4)")
    print("4. Build features           -> src/features.py  (Step 5)")
    print("5. Split by date            -> src/split.py     (Step 6)")
    print("6. Baselines + ML models    -> src/baselines.py, src/models.py (Steps 7-8)")
    print("7. Evaluate, backtest, plot -> src/evaluate.py, src/backtest.py, src/plots.py (Steps 9-10)")
    print("Done.")


if __name__ == "__main__":
    min()