"""Runs the whole project from raw data to final results.

Usage:  python run_all.py
"""
import numpy as np
import pandas as pd

import config
from src.utils import set_seed, ensure_dirs
from src.load import load_all
from src.swings import add_swings
from src.htf import add_levels, add_sweeps
from src.detector import find_all, summary
from src.models import run_models, baselines, trading_stats
from src import evaluate as ev


def main() -> None:
    set_seed(config.SEED)
    ensure_dirs()

    # 1. load data
    data = load_all()
    for sym, frames in data.items():
        for name, df in frames.items():
            print(f"{sym:7} {name:4} {len(df):>8} bars   {df.index.min()}  ->  {df.index.max()}")

    # 2. swings + previous day/week levels + sweeps for every symbol
    for sym, tf in data.items():
        m = add_swings(tf["m15"])
        m = add_sweeps(add_levels(m, tf["d1"], tf["w1"]))
        tf["m15"] = m
        days = m.groupby("day")[["sell_bias", "buy_bias"]].max().sum()
        print(sym, "sell days:", int(days["sell_bias"]), " buy days:", int(days["buy_bias"]))

    # 3. find + label setups (main strategy = all rules on)
    setups = find_all(data)
    setups.to_csv(config.RESULTS / "setups.csv", index=False)
    print("\nE1 strategy alone, full period:", summary(setups))

    # rule experiments (strategy only, no ML)
    variants = {"E3 no weekly (PWH/PWL) rule": find_all(data, require_pwh=False),
                "E4 no session rule": find_all(data, require_session=False),
                "E5 target = previous week": find_all(data, target="PWL")}
    for name, s in variants.items():
        print(f"{name}, full period:", summary(s))

    # 4. ML filter models (meta-labelling)
    out = run_models(setups)
    test = out["test"]
    test_start = test["entry_time"].min()
    print(f"\nsplit: train {len(out['train'])}, val {len(out['val'])}, test {len(test)} (test from {test_start.date()})")
    for name, m in out["models"].items():
        aucs = [r["AUC"] for r in m["runs"]]
        print(f"{name:12} params {m['params']}  val AUC {m['val_auc']:.3f}  test AUC {np.mean(aucs):.3f}")

    # best model = highest validation AUC (never chosen using the test set)
    best = max(out["models"], key=lambda n: out["models"][n]["val_auc"])
    runs = out["models"][best]["runs"]
    take = runs[0]["p_test"] >= runs[0]["threshold"]
    n_taken = int(take.sum())
    print(f"best model: {best}, threshold {runs[0]['threshold']}, takes {n_taken} of {len(test)} test trades")

    # 5. baselines + trading table (test period)
    base, rand_avg = baselines(test, n_taken)
    ml = {k.replace("trade_", ""): np.mean([r[k] for r in runs]) for k in runs[0] if k.startswith("trade_")}
    p_value = (rand_avg >= ml["avg_R"]).mean()
    print(f"ML filter avg R {ml['avg_R']:.3f} vs random filter {rand_avg.mean():.3f}  (p = {p_value:.3f})")

    rows = {"E1 strategy, all rules (= B1)": base["B1 take every trade"],
            f"E2 ML filter ({best})": ml,
            "B2 random filter (avg of 1000)": base["B2 random filter (avg of 1000)"],
            "B3 majority class (no trades)": base["B3 majority class (no trades)"],
            "B4 rule: target >= 3R": base["B4 rule: target >= 3R"]}
    for name, s in variants.items():
        rows[name] = trading_stats(s.loc[s["entry_time"] >= test_start, "r_net"])

    # 6. tables
    print("\n", ev.table_settings(out).to_string())
    print("\n", ev.table_classification(out).to_string())
    print("\n", ev.table_trading(rows).to_string())

    # 7. figures
    ex = ev.fig_example(data, setups, config.FIGURES / "fig1_example_setup.pdf")
    print("\nfigure 1 example:", ex["symbol"], ex["direction"], ex["entry_time"])
    ev.fig_equity(test, take, rand_avg, n_taken, config.FIGURES / "fig2_equity.pdf")
    ev.fig_roc_importance(out, best, config.FIGURES / "fig3_roc_importance.pdf")
    print("Done. Tables in results/, figures in figures/")


if __name__ == "__main__":
    main()