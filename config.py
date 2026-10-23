"""All project settings in one place. Change values here, never inside src/."""
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# ---------- Instruments (use your MT5 symbol names) ----------
SYMBOLS = ["XAUUSD", "EURUSD", "GBPUSD", "US100"]

# ---------- Time zones ----------
UK_TZ = "Europe/London"
NY_TZ = "America/New_York"
SERVER_TZ = None          # fill in after checking MT5 in Step 2

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

# ---------- Experiment switches ----------
REQUIRE_PWH = True        # E3: set False to test without the weekly sweep
REQUIRE_SESSION = True    # E4: set False to test without the session rule
TARGET = "PDL"            # E5: "PDL" or "PWL"

# ---------- Data split ----------
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2            # test = the remaining 0.2

# ---------- Models ----------
SEED = 42
N_SEEDS = 5
OPTUNA_TRIALS = 50