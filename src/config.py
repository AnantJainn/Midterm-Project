"""Shared paths and constants for the CSE558 midterm project (Elliptic++ Bitcoin fraud)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Elliptic++ Dataset"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = ROOT / "report" / "figs"

for d in (PROCESSED, RESULTS, FIGS):
    d.mkdir(parents=True, exist_ok=True)

# Elliptic++ class codes (README of git-disl/EllipticPlusPlus)
CLASS_NAMES = {1: "illicit", 2: "licit", 3: "unknown"}

# The 17 transaction features that Elliptic++ added with real names
# (the other 165 are Elliptic's anonymised Local_/Aggregate_ features)
TX_NAMED = [
    "in_txs_degree", "out_txs_degree", "total_BTC", "fees", "size",
    "num_input_addresses", "num_output_addresses",
    "in_BTC_min", "in_BTC_max", "in_BTC_mean", "in_BTC_median", "in_BTC_total",
    "out_BTC_min", "out_BTC_max", "out_BTC_mean", "out_BTC_median", "out_BTC_total",
]

SAT_PER_BTC = 100_000_000
SHUTDOWN_STEP = 43  # dark-market shutdown reported by Weber et al. (2019)
