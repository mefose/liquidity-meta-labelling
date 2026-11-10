# finds swing highs and lows on the 15 min chart
import pandas as pd
import config


def add_swings(m15, n=config.SWING_BARS):
    df = m15.copy()
    size = 2 * n + 1   # 2 bars each side + the bar itself = 5

    # swing high = high is the biggest of the 5 bars around it
    df["is_sh"] = df["high"] == df["high"].rolling(size, center=True).max()
    # swing low = low is the smallest of the 5 bars around it
    df["is_sl"] = df["low"] == df["low"].rolling(size, center=True).min()

    # we only know its a swing 2 bars later, so move the price down 2 rows
    df["sh_conf"] = df["high"].where(df["is_sh"]).shift(n)
    df["sl_conf"] = df["low"].where(df["is_sl"]).shift(n)

    # every bar remembers the last confirmed swing high / low
    df["last_sh"] = df["sh_conf"].ffill()
    df["last_sl"] = df["sl_conf"].ffill()

    return df