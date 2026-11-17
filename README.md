# Liquidity-Sweep Meta-Labelling (COMP1818 Coursework)

Investigates whether machine learning can filter the trade signals of a rule-based
liquidity-sweep strategy (previous day/week high-low sweeps + 15-minute break of structure),
using meta-labelling.

## Data
- Source: Vantage Markets, MetaTrader 5, 15-minute bid bars
- Instruments: XAUUSD, EURUSD, GBPUSD, NAS100
- Period: January 2019 – September 2026
- Files: `data/raw/<SYMBOL>_M15.csv`

## How to run
```bash
conda env create -f environment.yml
conda activate liqml
python run_all.py
```
Alternatively: `pip install -r requirements.txt` then `python run_all.py`. Takes about 1 minute.

## Outputs
- `results/` – detected setups (`setups.csv`) and Tables 1–3 (`.csv` and `.tex`)
- `figures/` – Figures 1–3 (PDF)

## Code
| File | What it does |
|---|---|
| `config.py` | All settings (symbols, dates, strategy rules, split, seeds) |
| `run_all.py` | Runs the whole pipeline from raw data to tables and figures |
| `src/load.py` | Loads MT5 data, builds 1H / daily / weekly candles, adds UK time |
| `src/swings.py` | Confirmed 15-minute swing highs and lows |
| `src/htf.py` | Previous day/week highs and lows and sweep detection |
| `src/detector.py` | Strategy rules (entry, stop, target) and trade labelling |
| `src/models.py` | Features, date split, baselines, ML models |
| `src/evaluate.py` | Results tables and figures |
| `src/utils.py` | Random seed and folder helpers |
