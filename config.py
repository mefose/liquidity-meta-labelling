"""All project settings in one place. Change values here, never inside src/."""
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# ---------- Data ----------
DATA_SOURCE = "Vantage Markets MT5 (demo server), M15 bid bars"
SYMBOLS = ["XAUUSD", "EURUSD", "GBPUSD", "NAS100"]   # must match data/raw/<SYMBOL>_M15.csv
START_DATE = "2019-01-01"     # first year with complete M15 data for all 4 symbols
END_DATE = "2026-10-01"       # fixed end so results never change

# ---------- Time zones ----------
UK_TZ = "Europe/London"
NY_TZ = "America/New_York"
SERVER_MINUS_NY_HOURS = 7     # server = UK + 2h on 23 Oct 2026 -> broker uses New York 5pm close

# ---------- Strategy rules ----------
SWING_BARS = 2            # rule 14: bars each side to confirm a swing
ATR_PERIOD = 14           # ATR length on 15m
ATR_BUFFER = 0.1          # rule 10: stop = trap high + 0.1 x ATR
MIN_R = 2.0               # rule 11: skip if target < 2R away
TIMEOUT_DAYS = 3          # rule 12: close open trades after 3 days
MAX_TRADES_PER_DAY = 1    # per instrument

# ---------- Session opens (rule 7) ----------
SESSION_OPENS = {
    "asia":    ("00:00", UK_TZ),
    "london":  ("08:00", UK_TZ),
    "newyork": ("09:30", NY_TZ),
}
SESSION_WINDOW_MIN = 90   # trap must happen within 90 min of an open

# ---------- Trading costs ----------
SLIPPAGE_SPREADS = 0.5    # extra cost on top of the bar's spread (0.5 = half a spread)

# ---------- Experiment switches ----------
REQUIRE_PWH = True        # E3: set False to test without the weekly sweep
REQUIRE_SESSION = True    # E4: set False to test without the session rule
TARGET = "PDL"            # E5: "PDL" (previous day) or "PWL" (previous week)

# ---------- Data split ----------
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2            # test = the remaining 0.2
PURGE_DAYS = 3            # gap between train/val/test so trades don't overlap

# ---------- Models ----------
SEED = 42
N_SEEDS = 5
OPTUNA_TRIALS = 50



# ---------- Trading costs per symbol ----------
# size of one MT5 "point" (the unit the <SPREAD> column is in)
POINT = {"XAUUSD": 0.01, "EURUSD": 0.00001, "GBPUSD": 0.00001, "NAS100": 0.01}
# some old bars have spread 0, so use at least this many points
MIN_SPREAD_POINTS = {"XAUUSD": 15, "EURUSD": 8, "GBPUSD": 12, "NAS100": 100}