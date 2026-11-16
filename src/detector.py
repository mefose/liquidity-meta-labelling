# finds the trade setups (the strategy rules) and labels each one win / loss
import numpy as np
import pandas as pd
import config


def add_atr(m, period=config.ATR_PERIOD):
    # average true range = how much price normally moves in one 15 min bar
    prev_close = m["close"].shift(1)
    tr = pd.concat([m["high"] - m["low"],
                    (m["high"] - prev_close).abs(),
                    (m["low"] - prev_close).abs()], axis=1).max(axis=1)
    m = m.copy()
    m["atr"] = tr.rolling(period).mean()
    return m


def session_of(m):
    # name of the session window each bar is in ("" if none)
    names = pd.Series("", index=m.index)
    for name, (start, tz) in config.SESSION_OPENS.items():
        local = m["time_uk"].dt.tz_convert(tz)
        mins = local.dt.hour * 60 + local.dt.minute
        h, mi = start.split(":")
        open_min = int(h) * 60 + int(mi)
        inside = (mins >= open_min) & (mins < open_min + config.SESSION_WINDOW_MIN)
        names[inside] = name
    return names


def label_trade(m, pos, direction, entry, stop, target, risk):
    # walk forward from the bar after entry: did price hit the target or the stop first?
    end_time = m.index[pos] + pd.Timedelta(days=config.TIMEOUT_DAYS)
    end_pos = m.index.searchsorted(end_time)
    highs = m["high"].values[pos + 1:end_pos]
    lows = m["low"].values[pos + 1:end_pos]
    if len(highs) == 0:
        return None   # no data left after this trade (end of the dataset)

    if direction == "sell":
        stop_hits = np.where(highs >= stop)[0]
        tgt_hits = np.where(lows <= target)[0]
    else:
        stop_hits = np.where(lows <= stop)[0]
        tgt_hits = np.where(highs >= target)[0]

    first_stop = stop_hits[0] if len(stop_hits) else np.inf
    first_tgt = tgt_hits[0] if len(tgt_hits) else np.inf

    if first_tgt < first_stop:                  # target first = win
        win, r = 1, abs(entry - target) / risk
        exit_pos = pos + 1 + int(first_tgt)
    elif first_stop < np.inf:                   # stop first (or same bar) = loss
        win, r = 0, -1.0
        exit_pos = pos + 1 + int(first_stop)
    else:                                       # neither: close at the timeout
        exit_pos = end_pos - 1
        last = m["close"].values[exit_pos]
        move = (entry - last) if direction == "sell" else (last - entry)
        win, r = 0, move / risk
    return win, r, m.index[exit_pos]


def find_setups(m, symbol, require_pwh=config.REQUIRE_PWH,
                require_session=config.REQUIRE_SESSION, target=config.TARGET):
    m = add_atr(m)
    m["session"] = session_of(m)

    # bias: previous day level swept (+ previous week level if required)
    if require_pwh:
        sell = m["pdh_swept"] & m["pwh_swept"]
        buy = m["pdl_swept"] & m["pwl_swept"]
    else:
        sell = m["pdh_swept"]
        buy = m["pdl_swept"]
    m["sell"] = sell & ~buy
    m["buy"] = buy & ~sell

    # extra info used as features later
    m["day_high"] = m["high"].groupby(m["day"]).cummax()
    m["day_low"] = m["low"].groupby(m["day"]).cummin()
    m["week_high"] = m["high"].groupby(m["week"]).cummax()
    m["week_low"] = m["low"].groupby(m["week"]).cummin()
    m["trend"] = m["close"] / m["close"].rolling(480).mean() - 1   # vs ~5 day average
    m["ret_4h"] = m["close"] / m["close"].shift(16) - 1

    low_col, high_col = ("pdl", "pdh") if target == "PDL" else ("pwl", "pwh")
    rows = []

    for day, g in m.groupby("day"):
        biased = g[g["sell"] | g["buy"]]
        if biased.empty:
            continue
        first = biased.index[0]
        direction = "sell" if g.at[first, "sell"] else "buy"
        after = g[g.index > first]

        # entry = first 15 min close that breaks structure in the bias direction
        if direction == "sell":
            bos = after["sell"] & (after["close"] < after["last_sl"])
        else:
            bos = after["buy"] & (after["close"] > after["last_sh"])
        if require_session:
            bos = bos & (after["session"] != "")
        if not bos.any():
            continue

        t = bos.index[bos.values][0]
        row = g.loc[t]
        if pd.isna(row["atr"]) or pd.isna(row[low_col]):
            continue
        entry = row["close"]
        buf = config.ATR_BUFFER * row["atr"]

        if direction == "sell":
            stop = row["day_high"] + buf          # above the sweep high
            tgt = row[low_col]                    # previous day (or week) low
        else:
            stop = row["day_low"] - buf
            tgt = row[high_col]
        risk = abs(entry - stop)
        reward = (entry - tgt) if direction == "sell" else (tgt - entry)
        if risk <= 0 or reward <= 0 or reward / risk < config.MIN_R:
            continue   # target too close (or already passed): skip this day

        pos = m.index.get_loc(t)
        result = label_trade(m, pos, direction, entry, stop, tgt, risk)
        if result is None:
            continue
        win, r_gross, exit_time = result

        # trading cost (spread + slippage) turned into R
        spread_pts = max(row["spread"], config.MIN_SPREAD_POINTS[symbol])
        cost = spread_pts * config.POINT[symbol] * (1 + config.SLIPPAGE_SPREADS)
        sign = -1 if direction == "sell" else 1
        a = row["atr"]

        rows.append({
            "symbol": symbol, "direction": direction, "entry_time": t, "exit_time": exit_time,
            "session": row["session"] or "none", "entry": entry, "stop": stop, "target": tgt,
            "y": win, "r_gross": r_gross, "r_net": r_gross - cost / risk,
            # ---- features (only use info known at the entry bar) ----
            "r_target": reward / risk,
            "risk_atr": risk / a,
            "sweep_day_atr": ((row["day_high"] - row["pdh"]) if direction == "sell" else (row["pdl"] - row["day_low"])) / a,
            "sweep_week_atr": ((row["week_high"] - row["pwh"]) if direction == "sell" else (row["pwl"] - row["week_low"])) / a,
            "bars_since_bias": len(g.loc[first:t]) - 1,
            "hour_uk": row["time_uk"].hour + row["time_uk"].minute / 60,
            "weekday": day.dayofweek,
            "atr_pct": a / entry * 100,
            "day_range_atr": (row["day_high"] - row["day_low"]) / a,
            "prev_range_atr": (row["pdh"] - row["pdl"]) / a,
            "trend_dir": sign * row["trend"] * 100,
            "ret_4h_dir": sign * row["ret_4h"] * 100,
        })

    return pd.DataFrame(rows)


def find_all(data, **rules):
    # run the detector on every symbol and stack the results
    out = [find_setups(tf["m15"], sym, **rules) for sym, tf in data.items()]
    return pd.concat(out, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def summary(setups):
    # quick stats for the strategy on its own
    return {"trades": len(setups),
            "win_rate": round(setups["y"].mean(), 3),
            "avg_R": round(setups["r_net"].mean(), 3),
            "total_R": round(setups["r_net"].sum(), 1)}