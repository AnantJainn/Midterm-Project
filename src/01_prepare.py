"""Step 1: load the raw Elliptic++ files, clean them, encode them, and record
before/after statistics for Sections 1-3 of the report.

Outputs
  data/processed/transactions.parquet   one row per transaction
  data/processed/wallets.parquet        one row per wallet address
  results/prep_summary.json             every number quoted in Sections 1-3
"""
import json

import numpy as np
import pandas as pd

from config import CLASS_NAMES, PROCESSED, RAW, RESULTS, SAT_PER_BTC, TX_NAMED

summary = {}


def iqr_upper_fence(s):
    q1, q3 = s.quantile([0.25, 0.75])
    return q3 + 1.5 * (q3 - q1)


# --------------------------------------------------------------- transactions
tx = pd.read_csv(RAW / "txs_features.csv")
tx_cls = pd.read_csv(RAW / "txs_classes.csv")
edges = pd.read_csv(RAW / "txs_edgelist.csv")
tx = tx.merge(tx_cls, on="txId", how="left", validate="one_to_one")

missing_mask = tx[TX_NAMED].isna()
rate_raw = tx["fees"] * SAT_PER_BTC / tx["size"]
summary["tx_before"] = {
    "rows": len(tx),
    "cols": tx.shape[1],
    "cells": int(tx.shape[0] * tx.shape[1]),
    "class_counts": tx["class"].map(CLASS_NAMES).value_counts().to_dict(),
    "rows_missing_named": int(missing_mask.any(axis=1).sum()),
    "rows_missing_all_named": int(missing_mask.all(axis=1).sum()),
    "missing_by_class": tx.loc[missing_mask.any(axis=1), "class"].map(CLASS_NAMES).value_counts().to_dict(),
    "missing_cells_total": int(tx.isna().sum().sum()),
    "duplicate_rows": int(tx.duplicated().sum()),
    "duplicate_txId": int(tx["txId"].duplicated().sum()),
    "zero_fee": int((tx["fees"] == 0).sum()),
    "fee_rate_above_iqr_fence": int((rate_raw > iqr_upper_fence(rate_raw.dropna())).sum()),
    "fee_rate_max": float(rate_raw.max()),
    "fee_rate_median": float(rate_raw.median()),
    "total_BTC_above_iqr_fence": int((tx["total_BTC"] > iqr_upper_fence(tx["total_BTC"].dropna())).sum()),
    "label_dtype": str(tx_cls["class"].dtype),
}

# Bitcoin accounting check: inputs = outputs + fee, for every row with data
ok = tx[TX_NAMED].notna().all(axis=1)
summary["tx_before"]["accounting_identity_holds"] = float(
    np.isclose(tx.loc[ok, "in_BTC_total"], tx.loc[ok, "out_BTC_total"] + tx.loc[ok, "fees"], atol=1e-6).mean()
)

# The 17 named features are missing together (structural), so the row is dropped
tx = tx[ok].copy()
tx["label"] = tx["class"].map(CLASS_NAMES)

# Derived features
tx["fee_rate"] = tx["fees"] * SAT_PER_BTC / tx["size"]                # satoshi per byte
tx["log10_fee_rate"] = np.where(tx["fee_rate"] > 0, np.log10(tx["fee_rate"].where(tx["fee_rate"] > 0)), np.nan)
tx["single_output"] = (tx["num_output_addresses"] == 1).astype(int)
tx["single_input"] = (tx["num_input_addresses"] == 1).astype(int)
tx["fee_rate_outlier"] = (tx["fee_rate"] > iqr_upper_fence(tx["fee_rate"])).astype(int)

# Money-trail feature from the graph: does this transaction spend an output of
# an illicit transaction? (edge txId1 -> txId2 means money flows from 1 to 2;
# labels of all 203,769 rows are used, including those dropped above)
cls_all = tx_cls.set_index("txId")["class"]
src_cls = edges["txId1"].map(cls_all)
receives_from_illicit = set(edges.loc[src_cls == 1, "txId2"])
tx["receives_from_illicit"] = tx["txId"].isin(receives_from_illicit).astype(int)
tx["n_parents"] = tx["txId"].map(edges.groupby("txId2").size()).fillna(0).astype(int)

# One-hot encoding of the label
tx = pd.concat([tx, pd.get_dummies(tx["label"], prefix="is", dtype=int)], axis=1)
tx = tx.rename(columns={"Time step": "time_step"})

summary["tx_after"] = {
    "rows": len(tx),
    "cols": tx.shape[1],
    "cells": int(tx.shape[0] * tx.shape[1]),
    "class_counts": tx["label"].value_counts().to_dict(),
    "missing_named": int(tx[TX_NAMED].isna().sum().sum()),
    "zero_fee_rows_excluded_from_log": int((tx["fee_rate"] == 0).sum()),
    "fee_rate_outliers_flagged": int(tx["fee_rate_outlier"].sum()),
    "added_columns": ["label", "fee_rate", "log10_fee_rate", "single_output", "single_input",
                      "fee_rate_outlier", "receives_from_illicit", "n_parents",
                      "is_illicit", "is_licit", "is_unknown"],
}
summary["edges"] = {"rows": len(edges), "receives_from_illicit_tx": int(tx["receives_from_illicit"].sum())}
tx.to_parquet(PROCESSED / "transactions.parquet", index=False)

# --------------------------------------------------------------------- wallets
w = pd.read_csv(RAW / "wallets_features.csv")
w_cls = pd.read_csv(RAW / "wallets_classes.csv")
feat = [c for c in w.columns if c not in ("address", "Time step")]
summary["w_before"] = {
    "rows": len(w),
    "cols": w.shape[1] + 1,  # + class, which ships in a separate file
    "cells": int(w.shape[0] * (w.shape[1] + 1)),
    "unique_addresses": int(w["address"].nunique()),
    "exact_duplicate_rows": int(w.duplicated().sum()),
    "missing_cells": int(w.isna().sum().sum()),
    "class_counts": w_cls["class"].map(CLASS_NAMES).value_counts().to_dict(),
}

w = w.drop_duplicates()
summary["w_after_dedup_rows"] = len(w)
summary["w_addresses_in_several_steps"] = int((w.groupby("address").size() > 1).sum())

# Every feature is a whole-life summary of the address: it does not change from
# one time step to the next, so the remaining repeats are the same wallet
# counted again. Check that, then keep one row per address.
varying = w.groupby("address")[feat].nunique().max()
assert (varying <= 1).all(), varying[varying > 1]
first_step = w.groupby("address")["Time step"].min()
w = w.drop_duplicates("address").drop(columns="Time step")
w["first_time_step"] = w["address"].map(first_step)
w = w.merge(w_cls, on="address", how="left", validate="one_to_one")
w = w.rename(columns={"num_txs_as receiver": "num_txs_as_receiver"})
w["label"] = w["class"].map(CLASS_NAMES)

sent = w["num_txs_as_sender"] > 0
recv = w["num_txs_as_receiver"] > 0
w["pattern"] = np.select([recv & ~sent, sent & ~recv, sent & recv], ["receive_only", "send_only", "pass_through"], "none")
w = pd.concat([w, pd.get_dummies(w["label"], prefix="is", dtype=int),
               pd.get_dummies(w["pattern"], prefix="pat", dtype=int)], axis=1)

summary["w_after"] = {
    "rows": len(w),
    "cols": w.shape[1],
    "cells": int(w.shape[0] * w.shape[1]),
    "class_counts": w["label"].value_counts().to_dict(),
    "pattern_counts": w["pattern"].value_counts().to_dict(),
    "btc_transacted_total_above_iqr_fence": int(
        (w["btc_transacted_total"] > iqr_upper_fence(w["btc_transacted_total"])).sum()),
}
w.to_parquet(PROCESSED / "wallets.parquet", index=False)

with open(RESULTS / "prep_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=int)
print(json.dumps(summary, indent=2, default=int))
