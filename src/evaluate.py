# results tables (csv + latex), the experiments and the 3 figures for the report
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # save figures to files, don't open windows
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.inspection import permutation_importance
import config
from src.models import make_X, trading_stats


def save_table(df, name, caption):
    # save as csv (to look at) and as a latex table (to paste into the report)
    df.to_csv(config.RESULTS / f"{name}.csv")
    cols = "l" + "r" * len(df.columns)
    lines = ["\\begin{table}[h]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{tab:{name}}}",
             "\\small", f"\\begin{{tabular}}{{{cols}}}", "\\hline",
             " & ".join([""] + [str(c).replace("_", "\\_") for c in df.columns]) + " \\\\", "\\hline"]
    for idx, row in df.iterrows():
        cells = [str(idx).replace("_", "\\_").replace(">=", "$\\geq$")] + [str(v).replace("_", "\\_") for v in row]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    (config.RESULTS / f"{name}.tex").write_text("\n".join(lines))


def fmt(mean, sd=None, d=3):
    if pd.isna(mean):
        return "-"
    return f"{mean:.{d}f}" if sd is None or pd.isna(sd) or sd < 0.0005 else f"{mean:.{d}f} $\\pm$ {sd:.{d}f}"


def table_settings(out):
    rows = {"Symbols": ", ".join(config.SYMBOLS),
            "Period": f"{config.START_DATE} to {config.END_DATE}",
            "Swing bars each side": config.SWING_BARS,
            "Session window (min)": config.SESSION_WINDOW_MIN,
            "Stop buffer (x ATR)": config.ATR_BUFFER,
            "Minimum target (R)": config.MIN_R,
            "Timeout (days)": config.TIMEOUT_DAYS,
            "Train / val / test trades": f"{len(out['train'])} / {len(out['val'])} / {len(out['test'])}"}
    for name, m in out["models"].items():
        rows[f"{name} (best)"] = ", ".join(f"{k}={v}" for k, v in m["params"].items())
    df = pd.DataFrame({"Value": rows})
    save_table(df, "table1_settings", "Strategy settings, data split and tuned hyperparameters.")
    return df


def table_classification(out):
    rows = {}
    for name, m in out["models"].items():
        runs = pd.DataFrame([{k: v for k, v in r.items() if k not in ("p_test", "model")} for r in m["runs"]])
        rows[name] = {"val AUC": fmt(m["val_auc"]),
                      **{k: fmt(runs[k].mean(), runs[k].std()) for k in
                         ["AUC", "precision", "recall", "MCC", "bal_acc", "brier"]}}
    rows["Coin flip"] = {"val AUC": "0.500", "AUC": "0.500", "precision": fmt(out["test"]["y"].mean()),
                         "recall": "0.500", "MCC": "0.000", "bal_acc": "0.500", "brier": "-"}
    df = pd.DataFrame(rows).T
    save_table(df, "table2_classification",
               "Test-set classification results (mean $\\pm$ SD over 5 seeds). AUC: higher is better, 0.5 = no skill.")
    return df


def table_trading(rows):
    df = pd.DataFrame(rows).T
    for c in ["pct_positive", "avg_R", "profit_factor"]:
        df[c] = df[c].map(lambda v: fmt(v))
    for c in ["total_R", "max_dd_R"]:
        df[c] = df[c].map(lambda v: fmt(v, d=1))
    df["trades"] = df["trades"].astype(int)
    save_table(df, "table3_trading",
               "Trading results on the test period after costs, in R (1R = the amount risked). pct\\_positive = share of trades closing with a profit.")
    return df


def fig_example(data, setups, path):
    # one setup drawn as 15 min candles with the levels, entry, stop and target
    wins = setups[setups["y"] == 1]
    s = wins.iloc[len(wins) // 2]          # a winning trade from the middle of the data
    m = data[s["symbol"]]["m15"]
    w = m.loc[s["entry_time"] - pd.Timedelta(hours=10): s["exit_time"] + pd.Timedelta(hours=2)]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(w))
    up = w["close"] >= w["open"]
    ax.vlines(x, w["low"], w["high"], color="black", linewidth=0.6)
    ax.bar(x[up], (w["close"] - w["open"])[up], bottom=w["open"][up], color="white", edgecolor="black", width=0.7)
    ax.bar(x[~up], (w["open"] - w["close"])[~up], bottom=w["close"][~up], color="black", width=0.7)
    e = w.index.get_loc(s["entry_time"])
    lvl = "pdh" if s["direction"] == "sell" else "pdl"
    ax.axhline(w[lvl].iloc[e], color="tab:blue", ls="--", lw=1, label=lvl.upper())
    ax.hlines(s["stop"], e, len(w) - 1, color="tab:red", lw=1.2, label="stop")
    ax.hlines(s["target"], e, len(w) - 1, color="tab:green", lw=1.2, label="target")
    ax.scatter([e], [s["entry"]], color="orange", zorder=5, label=f"entry ({s['direction']})")
    ticks = x[::max(1, len(x) // 8)]
    ax.set_xticks(ticks, [w.index[i].strftime("%d %b %H:%M") for i in ticks], rotation=30, fontsize=7)
    ax.set_ylabel("price")
    ax.set_title(f"{s['symbol']} example setup (server time)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return s


def fig_equity(test, take, rand_avg, n_taken, path):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    t = test["entry_time"].values
    ax.plot(t, test["r_net"].cumsum(), label="B1 take every trade")
    ax.plot(t[take], test["r_net"][take].cumsum(), label="ML filter")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("cumulative R (after costs)")
    ax.set_title(f"Test period equity (random filter avg R = {rand_avg.mean():.3f})")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_roc_importance(out, best, path):
    test = out["test"]
    y = test["y"].values
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    for name, m in out["models"].items():
        p = m["runs"][0]["p_test"]
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, label=f"{name} (AUC {m['runs'][0]['AUC']:.2f})")
    a1.plot([0, 1], [0, 1], color="grey", ls="--", label="coin flip")
    a1.set_xlabel("false positive rate"); a1.set_ylabel("true positive rate")
    a1.set_title("ROC curves (test)"); a1.legend(fontsize=7)

    model = out["models"][best]["runs"][0]["model"]
    X = make_X(test)
    imp = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=20, random_state=config.SEED)
    order = np.argsort(imp.importances_mean)[-10:]
    a2.barh(X.columns[order], imp.importances_mean[order])
    a2.set_title(f"{best}: top features (permutation)"); a2.set_xlabel("drop in AUC")
    a2.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)