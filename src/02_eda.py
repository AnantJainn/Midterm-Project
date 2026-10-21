"""Step 2: exploratory analysis. Answers the basic questions (Section 4) and
draws the EDA figures into report/figs/.

Outputs
  report/figs/fig_*.pdf
  results/eda_summary.json
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGS, PROCESSED, RESULTS, SHUTDOWN_STEP

LICIT, ILLICIT = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150,
})

tx = pd.read_parquet(PROCESSED / "transactions.parquet")
w = pd.read_parquet(PROCESSED / "wallets.parquet")
lab = tx[tx["label"] != "unknown"]
wl = w[w["label"] != "unknown"]
out = {}


def save(fig, name):
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGS / name, bbox_inches="tight")
    plt.close(fig)


def grouped_bars(ax, cats, ill, lic):
    x = np.arange(len(cats))
    for off, vals, col, name in ((-0.2, lic, LICIT, "licit"), (0.2, ill, ILLICIT, "illicit")):
        ax.bar(x + off, vals, width=0.38, color=col, label=name, edgecolor="white", linewidth=1)
        for xi, v in zip(x + off, vals):
            if v >= 0.05:
                ax.text(xi, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(x, cats)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(loc="upper right")


# --- Q1: how rare is fraud, and how does it move over time?
share = lab.groupby("time_step")["is_illicit"].mean()
n_step = lab.groupby("time_step").size()
out["illicit_share_labelled"] = float(lab["is_illicit"].mean())
out["labelled_share_of_all"] = float(len(lab) / len(tx))
out["illicit_share_by_step"] = share.round(4).to_dict()
out["labelled_per_step_min_max"] = [int(n_step.min()), int(n_step.max())]
out["peak_step"] = int(share.idxmax())
out["peak_share"] = float(share.max())

fig, ax = plt.subplots(figsize=(3.4, 1.9))
ax.axvspan(35.5, 42.5, color=GRID, alpha=0.6, lw=0)
ax.axvspan(42.5, 49.5, color=ILLICIT, alpha=0.10, lw=0)
ax.plot(share.index, share.values, color=ILLICIT, lw=1.6, marker="o", ms=2.5)
ax.axvline(SHUTDOWN_STEP - 0.5, color=INK2, lw=0.8, ls="--")
ax.text(SHUTDOWN_STEP - 0.2, 0.33, "dark-market\nshutdown (43)", fontsize=6.5, color=INK2, va="top")
ax.set_xlabel("Time step (about two weeks each)")
ax.set_ylabel("Illicit share of\nlabelled transactions")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlim(0.5, 49.5)
save(fig, "fig_timeline.pdf")

# --- Q2: do illicit transactions move more money?
med = lab.groupby("label")[["total_BTC", "fees", "size", "fee_rate"]].median()
out["median_by_class"] = med.to_dict()

# --- EDA: shape of a transaction (how many addresses receive the money)
k = lab["num_output_addresses"].clip(upper=5)
dist = pd.crosstab(lab["label"], k, normalize="index")
cats = ["1", "2", "3", "4", "5+"]
out["output_dist"] = dist.round(4).to_dict(orient="index")
out["single_input_share"] = lab.groupby("label")["single_input"].mean().to_dict()
fig, ax = plt.subplots(figsize=(3.4, 1.8))
grouped_bars(ax, cats, dist.loc["illicit"].values, dist.loc["licit"].values)
ax.set_xlabel("Number of output addresses (receivers)")
ax.set_ylabel("Share of transactions")
ax.set_ylim(0, 0.9)
save(fig, "fig_outputs.pdf")

# --- EDA: fee rate (satoshi per byte), log scale
pos = lab[lab["fee_rate"] > 0]
bins = np.linspace(0, 4.4, 45)
fig, ax = plt.subplots(figsize=(3.4, 1.8))
for name, col in (("licit", LICIT), ("illicit", ILLICIT)):
    v = pos.loc[pos["label"] == name, "log10_fee_rate"]
    ax.hist(v, bins=bins, density=True, histtype="step", lw=1.6, color=col, label=f"{name} (median {10**v.median():.0f})")
ax.set_xticks([0, 1, 2, 3, 4], ["1", "10", "100", "1,000", "10,000"])
ax.set_xlabel("Fee rate (satoshi per byte, log scale)")
ax.set_ylabel("Density")
ax.legend(loc="upper left", fontsize=6.5)
save(fig, "fig_feerate.pdf")
out["fee_rate_median_by_class"] = pos.groupby("label")["fee_rate"].median().to_dict()

# --- EDA: wallets - what does a fraud-linked wallet do with its money?
order = ["receive_only", "pass_through", "send_only"]
wdist = pd.crosstab(wl["label"], wl["pattern"], normalize="index")[order]
out["wallet_pattern_dist"] = wdist.round(4).to_dict(orient="index")
out["wallet_total_txs_median"] = wl.groupby("label")["total_txs"].median().to_dict()
fig, ax = plt.subplots(figsize=(3.4, 1.8))
grouped_bars(ax, ["receive only", "receive and send\n(pass-through)", "send only"],
             wdist.loc["illicit"].values, wdist.loc["licit"].values)
ax.set_ylabel("Share of wallets")
ax.set_ylim(0, 0.85)
save(fig, "fig_wallets.pdf")

# --- money trail: share illicit among transactions that spend an illicit output
mt = lab.groupby("receives_from_illicit")["is_illicit"].agg(["mean", "size"])
out["illicit_share_by_receives_from_illicit"] = mt.to_dict(orient="index")

with open(RESULTS / "eda_summary.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "illicit_share_by_step"}, indent=2, default=float))
