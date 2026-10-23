"""Load MT5 M15 exports and build 1H, daily and weekly candles."""
import pandas as pd

import config

# How to combine small candles into a bigger one
OHLC_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "spread": "mean"}



def _detect_encoding(path) -> str:
    """MT5 sometimes saves CSVs as UTF-16. Check the file's first two bytes."""
    with open(path, "rb") as f:
        start = f.read(2)
    return "utf-16" if start in (b"\xff\xfe", b"\xfe\xff") else "utf-8"


def load_m15(symbol: str) -> pd.DataFrame:
    """Read data/raw/<symbol>_M15.csv into a clean, time-indexed table (server time)."""
    path = config.DATA_RAW / f"{symbol}_M15.csv"
    df = pd.read_csv(path, sep="\t", encoding=_detect_encoding(path))

    # "<OPEN>" -> "open"
    df.columns = [c.strip().strip("<>").lower() for c in df.columns]

    # Join date + time into one timestamp, e.g. "2012.01.02 00:15:00"
    df["time_server"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S")
    df = df.set_index("time_server").sort_index()

    # Remove duplicate timestamps and impossible bars
    df = df[~df.index.duplicated(keep="first")]
    df = df[["open", "high", "low", "close", "spread"]].astype(float)
    df = df[df["high"] >= df["low"]]

    # Keep only the study period (START_DATE included, END_DATE excluded)
    df = df[(df.index >= config.START_DATE) & (df.index < config.END_DATE)]
    return df


def add_uk_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'time_uk' column. Index stays in broker server time (for daily/weekly candles)."""
    ny_naive = df.index - pd.Timedelta(hours=config.SERVER_MINUS_NY_HOURS)
    ny = ny_naive.tz_localize(config.NY_TZ, ambiguous="NaT", nonexistent="shift_forward")
    df = df.copy()
    df["time_uk"] = ny.tz_convert(config.UK_TZ)
    return df[df["time_uk"].notna()]


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Build bigger candles from 15m bars. rule: '1h', '1D' or 'W-SUN'.
    Weekly candles are labelled by the Sunday they start on, like MT5."""
    if rule.startswith("W"):
        grouped = df[list(OHLC_AGG)].resample(rule, label="left", closed="left")
    else:
        grouped = df[list(OHLC_AGG)].resample(rule)
    out = grouped.agg(OHLC_AGG)
    return out.dropna(subset=["open"])