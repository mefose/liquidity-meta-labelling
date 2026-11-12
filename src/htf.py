# previous day / week highs and lows + checking if they got swept
import pandas as pd


def add_levels(m15, d1, w1):
    df = m15.copy()
    df["day"] = df.index.normalize()
    # week starts on sunday (same as mt5)
    df["week"] = df["day"] - pd.to_timedelta((df.index.dayofweek + 1) % 7, unit="D")

    # yesterday's high and low, and last week's high and low
    prev_day = d1[["high", "low"]].shift(1).rename(columns={"high": "pdh", "low": "pdl"})
    prev_week = w1[["high", "low"]].shift(1).rename(columns={"high": "pwh", "low": "pwl"})

    # stick them onto every 15 min bar
    df = df.join(prev_day, on="day")
    df = df.join(prev_week, on="week")
    return df


def swept(df, level, group, side):
    # the bar at :45 is the last 15 min bar of the hour, so its close = the 1h close
    hour_close = pd.Series(df.index.minute == 45, index=df.index)

    if side == "high":
        # has price gone above the level yet today / this week
        went_past = (df["high"] > df[level]).astype(int).groupby(df[group]).cummax().astype(bool)
        # 1h candle closed back below it
        back_in = hour_close & went_past & (df["close"] < df[level])
    else:
        went_past = (df["low"] < df[level]).astype(int).groupby(df[group]).cummax().astype(bool)
        back_in = hour_close & went_past & (df["close"] > df[level])

    # once swept, stays swept for the rest of the day / week
    return back_in.astype(int).groupby(df[group]).cummax().astype(bool)


def add_sweeps(df):
    df = df.copy()
    df["pdh_swept"] = swept(df, "pdh", "day", "high")
    df["pdl_swept"] = swept(df, "pdl", "day", "low")
    df["pwh_swept"] = swept(df, "pwh", "week", "high")
    df["pwl_swept"] = swept(df, "pwl", "week", "low")

    sells = df["pdh_swept"] & df["pwh_swept"]
    buys = df["pdl_swept"] & df["pwl_swept"]
    df["sell_bias"] = sells & ~buys   # if both sides got swept theres no clear bias
    df["buy_bias"] = buys & ~sells
    return df