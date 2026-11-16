# features, date split, baselines and the 3 ML filter models (meta-labelling)
import itertools
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             matthews_corrcoef, balanced_accuracy_score, brier_score_loss)
import config

NUMERIC = ["r_target", "risk_atr", "sweep_day_atr", "sweep_week_atr", "bars_since_bias",
           "hour_uk", "weekday", "atr_pct", "day_range_atr", "prev_range_atr",
           "trend_dir", "ret_4h_dir"]
CATEGORIES = {"session": ["asia", "london", "newyork", "none"],
              "symbol": config.SYMBOLS,
              "direction": ["buy", "sell"]}


def make_X(df):
    # numbers stay as they are, text columns become 0/1 columns (one-hot)
    X = df[NUMERIC].astype(float).copy()
    for col, cats in CATEGORIES.items():
        for c in cats:
            X[f"{col}_{c}"] = (df[col] == c).astype(float)
    return X


def split_by_date(df):
    # oldest 60% = train, next 20% = validation, newest 20% = test (never shuffled)
    df = df.sort_values("entry_time").reset_index(drop=True)
    n = len(df)
    i1 = int(n * config.TRAIN_FRAC)
    i2 = int(n * (config.TRAIN_FRAC + config.VAL_FRAC))
    train, val, test = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
    # purge: drop trades still open when the next part starts (no overlap = no leakage)
    train = train[train["exit_time"] < val["entry_time"].min()]
    val = val[val["exit_time"] < test["entry_time"].min()]
    return train, val, test


# each model + the settings we try (small grid search, picked on validation AUC)
GRIDS = {
    "LogReg": {"C": [0.01, 0.1, 1, 10]},
    "RandomForest": {"max_depth": [3, 5, None], "min_samples_leaf": [5, 20]},
    "GradBoost": {"learning_rate": [0.03, 0.1], "max_depth": [2, 3]},
}


def build(name, params, seed):
    if name == "LogReg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=params["C"], class_weight="balanced", max_iter=2000))
    if name == "RandomForest":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                    random_state=seed, **params))
    # gradient boosted trees (same family as XGBoost), handles missing values itself
    return HistGradientBoostingClassifier(max_iter=200, class_weight="balanced",
                                          random_state=seed, **params)


def tune(name, Xtr, ytr, Xva, yva):
    # try every combination in the grid, keep the one with the best validation AUC
    keys = list(GRIDS[name])
    best, best_auc = None, -1
    for values in itertools.product(*GRIDS[name].values()):
        params = dict(zip(keys, values))
        model = build(name, params, config.SEED).fit(Xtr, ytr)
        auc = roc_auc_score(yva, model.predict_proba(Xva)[:, 1])
        if auc > best_auc:
            best, best_auc = params, auc
    return best, best_auc


def choose_threshold(p, trades):
    # on validation: which cut-off gives the best average R while still taking >= 30% of trades
    best_t, best_r = 0.5, -np.inf
    for t in np.arange(0.30, 0.71, 0.05):
        take = p >= t
        if take.mean() < 0.30:
            continue
        r = trades["r_net"][take].mean()
        if r > best_r:
            best_t, best_r = round(t, 2), r
    return best_t


def trading_stats(r):
    # r = net R of each trade taken, in time order
    r = np.asarray(r, dtype=float)
    if len(r) == 0:
        return {"trades": 0, "pct_positive": np.nan, "avg_R": np.nan, "total_R": 0.0,
                "max_dd_R": 0.0, "profit_factor": np.nan}
    equity = np.cumsum(r)
    drawdown = np.maximum.accumulate(np.concatenate([[0], equity]))[1:] - equity
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {"trades": len(r), "pct_positive": (r > 0).mean(), "avg_R": r.mean(),
            "total_R": r.sum(), "max_dd_R": drawdown.max(),
            "profit_factor": gains / losses if losses > 0 else np.nan}


def class_stats(y, p, t):
    pred = (p >= t).astype(int)
    return {"AUC": roc_auc_score(y, p), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "MCC": matthews_corrcoef(y, pred),
            "bal_acc": balanced_accuracy_score(y, pred), "brier": brier_score_loss(y, p)}


def run_models(setups):
    train, val, test = split_by_date(setups)
    Xtr, Xva, Xte = make_X(train), make_X(val), make_X(test)
    ytr, yva, yte = train["y"].values, val["y"].values, test["y"].values

    out = {"train": train, "val": val, "test": test, "models": {}}
    for name in GRIDS:
        params, val_auc = tune(name, Xtr, ytr, Xva, yva)
        runs = []
        for seed in range(config.N_SEEDS):            # 5 seeds -> mean +- sd
            model = build(name, params, seed).fit(Xtr, ytr)
            t = choose_threshold(model.predict_proba(Xva)[:, 1], val)
            p = model.predict_proba(Xte)[:, 1]
            take = p >= t
            runs.append({"seed": seed, "threshold": t, "p_test": p, "model": model,
                         **class_stats(yte, p, t),
                         **{f"trade_{k}": v for k, v in trading_stats(test["r_net"][take]).items()}})
        out["models"][name] = {"params": params, "val_auc": val_auc, "runs": runs}
    return out


def baselines(test, n_taken, seed=config.SEED, repeats=1000):
    # simple rules the ML filter has to beat (all on the test set)
    rng = np.random.default_rng(seed)
    r = test["r_net"].values
    res = {"B1 take every trade": trading_stats(r)}

    # B2: random filter taking the same number of trades as the ML model (1000 tries)
    rand = [np.sort(rng.choice(len(r), n_taken, replace=False)) for _ in range(repeats)]
    rand_avg = np.array([r[i].mean() for i in rand])
    b2 = trading_stats(r[rand[0]])
    b2["avg_R"], b2["total_R"] = rand_avg.mean(), rand_avg.mean() * n_taken
    res["B2 random filter (avg of 1000)"] = b2

    # B3: majority class = always predict "loss" -> takes no trades
    res["B3 majority class (no trades)"] = trading_stats([])

    # B4: hand-made rule, only trades with target >= 3R
    res["B4 rule: target >= 3R"] = trading_stats(test.loc[test["r_target"] >= 3, "r_net"])
    return res, rand_avg