"""Step 3: the five hypothesis tests of Section 5, each with the tests we
considered as alternatives.

Hypotheses, sides, alpha = 0.05 and the smallest differences that matter (MID)
are fixed in the TESTS dictionary below, before any statistic is computed.

Output: results/tests.json (and a readable print-out)
"""
import itertools
import json

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable, Table2x2
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from config import PROCESSED, RESULTS, SHUTDOWN_STEP

ALPHA = 0.05
RNG = np.random.default_rng(558)
N_BOOT = 2000

TESTS = {
    "T1": {"H0": "p_ill - p_lic <= 0", "H1": "p_ill - p_lic > 0", "side": "one-sided (right)",
           "test": "two-proportion z-test", "MID": "10 percentage points"},
    "T2": {"H0": "wallet activity pattern is independent of class",
           "H1": "pattern depends on class", "side": "chi-square (upper tail)",
           "test": "Pearson chi-square test of independence (2x3)",
           "MID": "Cramer's V >= 0.10 and a 10-point gap in the pass-through share"},
    "T3": {"H0": "sigma_ill >= sigma_lic (log10 fee rate)", "H1": "sigma_ill < sigma_lic",
           "side": "one-sided (left)", "test": "Brown-Forsythe test", "MID": "SD ratio <= 0.80"},
    "T4": {"H0": "common odds ratio OR_MH <= 1", "H1": "OR_MH > 1", "side": "one-sided (right)",
           "test": "Cochran-Mantel-Haenszel test, stratified by time step", "MID": "OR >= 2"},
    "T5": {"H0": "mu_after - mu_before >= 0", "H1": "mu_after - mu_before < 0",
           "side": "one-sided (left)", "test": "Welch t-test on per-step illicit shares",
           "MID": "5 percentage points"},
}

tx = pd.read_parquet(PROCESSED / "transactions.parquet")
w = pd.read_parquet(PROCESSED / "wallets.parquet")
lab = tx[tx["label"] != "unknown"].copy()
res = {}


def welch_ci(a, b, level=0.95):
    """Welch t-test pieces for mean(a) - mean(b): difference, df, CI."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    d = a.mean() - b.mean()
    h = stats.t.ppf(0.5 + level / 2, df) * np.sqrt(va + vb)
    return d, df, (d - h, d + h)


# ======================= T1: does illicit money go to a single receiver?
ill = lab[lab["label"] == "illicit"]["single_output"]
lic = lab[lab["label"] == "licit"]["single_output"]
count, nobs = np.array([ill.sum(), lic.sum()]), np.array([len(ill), len(lic)])
z, p = proportions_ztest(count, nobs, alternative="larger")
lo, hi = confint_proportions_2indep(count[0], nobs[0], count[1], nobs[1], method="newcomb")
table = np.array([[count[0], nobs[0] - count[0]], [count[1], nobs[1] - count[1]]])
chi2, p_chi, _, _ = stats.chi2_contingency(table, correction=False)
or_f, p_f = stats.fisher_exact(table, alternative="greater")
t_w = stats.ttest_ind(lab.loc[lab.label == "illicit", "num_output_addresses"],
                      lab.loc[lab.label == "licit", "num_output_addresses"], equal_var=False)
res["T1"] = {**TESTS["T1"], "n_ill": int(nobs[0]), "n_lic": int(nobs[1]),
             "p_ill": count[0] / nobs[0], "p_lic": count[1] / nobs[1],
             "estimate": count[0] / nobs[0] - count[1] / nobs[1], "ci95": [lo, hi],
             "stat": z, "df": None, "p": p, "reject": bool(p < ALPHA),
             "alt": {"pearson_chi2": {"stat": chi2, "df": 1, "p_two_sided": p_chi, "z_squared": z ** 2},
                     "fisher_exact": {"odds_ratio": or_f, "p_one_sided": p_f},
                     "welch_t_on_num_outputs": {"t": t_w.statistic, "df": float(t_w.df), "p": t_w.pvalue,
                                                "mean_ill": lab.loc[lab.label == "illicit", "num_output_addresses"].mean(),
                                                "mean_lic": lab.loc[lab.label == "licit", "num_output_addresses"].mean()}}}

# ======================= T2: wallets - pass-through behaviour
wl = w[w["label"] != "unknown"]
ct = pd.crosstab(wl["label"], wl["pattern"])[["receive_only", "pass_through", "send_only"]]
chi2, p, dof, exp = stats.chi2_contingency(ct.values, correction=False)
n = ct.values.sum()
V = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
# bootstrap CI for Cramer's V (resample wallets within each class)
codes = {k: wl.loc[wl.label == k, "pattern"].map({"receive_only": 0, "pass_through": 1, "send_only": 2}).values
         for k in ("illicit", "licit")}
boot_v = []
for _ in range(N_BOOT):
    rows = [np.bincount(RNG.choice(codes[k], len(codes[k])), minlength=3) for k in ("illicit", "licit")]
    c2 = stats.chi2_contingency(np.array(rows), correction=False)[0]
    boot_v.append(np.sqrt(c2 / n))
pt_i, pt_l = ct.loc["illicit", "pass_through"], ct.loc["licit", "pass_through"]
ni, nl = ct.loc["illicit"].sum(), ct.loc["licit"].sum()
lo, hi = confint_proportions_2indep(pt_i, ni, pt_l, nl, method="newcomb")
g, p_g, _, _ = stats.chi2_contingency(ct.values, correction=False, lambda_="log-likelihood")
z_pt, p_pt = proportions_ztest([pt_i, pt_l], [ni, nl], alternative="larger")
mw = stats.mannwhitneyu(wl.loc[wl.label == "illicit", "total_txs"], wl.loc[wl.label == "licit", "total_txs"],
                        alternative="two-sided")
std_resid = (ct.values - exp) / np.sqrt(exp)
res["T2"] = {**TESTS["T2"], "table": ct.to_dict(orient="index"),
             "row_shares": (ct.div(ct.sum(axis=1), axis=0)).round(4).to_dict(orient="index"),
             "stat": chi2, "df": int(dof), "p": p, "reject": bool(p < ALPHA),
             "cramers_v": V, "cramers_v_ci95": list(np.percentile(boot_v, [2.5, 97.5])),
             "estimate": pt_i / ni - pt_l / nl, "ci95": [lo, hi],
             "std_residuals": pd.DataFrame(std_resid, index=ct.index, columns=ct.columns).round(1).to_dict(orient="index"),
             "min_expected": float(exp.min()),
             "alt": {"g_test": {"G": g, "df": int(dof), "p": p_g},
                     "z_pass_through_only": {"z": z_pt, "p": p_pt},
                     "mann_whitney_total_txs": {"U": float(mw.statistic), "p": mw.pvalue}}}

# ======================= T3: are illicit fee rates less variable?
pos = lab[lab["fee_rate"] > 0]
a = pos.loc[pos.label == "illicit", "log10_fee_rate"].values
b = pos.loc[pos.label == "licit", "log10_fee_rate"].values
da, db = np.abs(a - np.median(a)), np.abs(b - np.median(b))
# Brown-Forsythe = pooled t-test on absolute deviations from group medians
# (with two groups F = t^2), which gives the one-sided p-value
t_bf = stats.ttest_ind(da, db, equal_var=True, alternative="less")
lev = stats.levene(a, b, center="median")
sd_ratio = a.std(ddof=1) / b.std(ddof=1)
boot_r = [RNG.choice(a, len(a)).std(ddof=1) / RNG.choice(b, len(b)).std(ddof=1) for _ in range(N_BOOT)]
f_stat = a.var(ddof=1) / b.var(ddof=1)
p_f = stats.f.cdf(f_stat, len(a) - 1, len(b) - 1)
bart = stats.bartlett(a, b)
flig = stats.fligner(a, b)
# within the same time step (controls for the fee market changing over time)
g_sd = pos.groupby(["time_step", "label"])["log10_fee_rate"].agg(["std", "size"]).unstack()
keep = (g_sd["size"].min(axis=1) >= 10)
sd_pairs = g_sd["std"][keep]
wil = stats.wilcoxon(sd_pairs["illicit"], sd_pairs["licit"], alternative="less")
fees_ill = pos.loc[pos.label == "illicit"]
fixed = fees_ill["fees"].round(8) == 0.0002
res["T3"] = {**TESTS["T3"], "n_ill": len(a), "n_lic": len(b), "excluded_zero_fee": int((lab.fee_rate == 0).sum()),
             "sd_ill": a.std(ddof=1), "sd_lic": b.std(ddof=1),
             "estimate": sd_ratio, "ci95": list(np.percentile(boot_r, [2.5, 97.5])),
             "stat": t_bf.statistic, "df": int(len(a) + len(b) - 2), "p": t_bf.pvalue,
             "F_equiv": lev.statistic, "levene_median_p_two_sided": lev.pvalue,
             "reject": bool(t_bf.pvalue < ALPHA),
             "fixed_fee_0.0002_share_ill": float(fixed.mean()),
             "fixed_fee_0.0002_share_lic": float((pos.loc[pos.label == "licit", "fees"].round(8) == 0.0002).mean()),
             "fixed_fee_0.0002_median_size_ill": float(fees_ill.loc[fixed, "size"].median()),
             "alt": {"F_test": {"F": f_stat, "df": [len(a) - 1, len(b) - 1], "p_one_sided": p_f},
                     "bartlett": {"stat": bart.statistic, "p": bart.pvalue},
                     "fligner_killeen": {"stat": flig.statistic, "p": flig.pvalue},
                     "normality_shapiro_ill_sample": stats.shapiro(RNG.choice(a, 5000)).pvalue,
                     "skew_ill": float(stats.skew(a)), "kurt_ill": float(stats.kurtosis(a)),
                     "skew_lic": float(stats.skew(b)), "kurt_lic": float(stats.kurtosis(b)),
                     "within_step_wilcoxon": {"n_steps": int(keep.sum()),
                                              "steps_ill_narrower": int((sd_pairs["illicit"] < sd_pairs["licit"]).sum()),
                                              "W": wil.statistic, "p_one_sided": wil.pvalue,
                                              "median_sd_ill": sd_pairs["illicit"].median(),
                                              "median_sd_lic": sd_pairs["licit"].median()}}}

# ======================= T4: does fraud follow the money trail?
strata, crude = [], np.zeros((2, 2))
for step, d in lab.groupby("time_step"):
    t = np.array([[((d.receives_from_illicit == 1) & (d.is_illicit == 1)).sum(),
                   ((d.receives_from_illicit == 1) & (d.is_illicit == 0)).sum()],
                  [((d.receives_from_illicit == 0) & (d.is_illicit == 1)).sum(),
                   ((d.receives_from_illicit == 0) & (d.is_illicit == 0)).sum()]], dtype=float)
    crude += t
    if t[0].sum() > 0 and t[:, 0].sum() > 0:  # strata without exposed rows or cases carry no information
        strata.append(t)
st = StratifiedTable(strata)
cmh = st.test_null_odds(correction=False)
z_cmh = np.sign(st.oddsratio_pooled - 1) * np.sqrt(cmh.statistic)
p_one = stats.norm.sf(z_cmh)
crude_t = Table2x2(crude)
chi2_c, p_c, _, _ = stats.chi2_contingency(crude, correction=False)
or_fish, p_fish = stats.fisher_exact(crude, alternative="greater")
or_k = np.array([(s[0, 0] + .5) * (s[1, 1] + .5) / ((s[0, 1] + .5) * (s[1, 0] + .5)) for s in strata])
bd_strata = [s for s in strata if (s > 0).all()]
bd = StratifiedTable(bd_strata).test_equal_odds(adjust=True)
res["T4"] = {**TESTS["T4"], "n": int(crude.sum()), "n_strata_used": len(strata), "crude_table": crude.tolist(),
             "risk_exposed": crude[0, 0] / crude[0].sum(), "risk_unexposed": crude[1, 0] / crude[1].sum(),
             "estimate": st.oddsratio_pooled, "ci95": list(st.oddsratio_pooled_confint()),
             "stat": cmh.statistic, "df": 1, "z": z_cmh, "p": p_one, "reject": bool(p_one < ALPHA),
             "alt": {"pearson_chi2_crude": {"stat": chi2_c, "df": 1, "p": p_c},
                     "crude_or": crude_t.oddsratio, "crude_or_ci95": list(crude_t.oddsratio_confint()),
                     "crude_rr": crude_t.riskratio, "crude_rr_ci95": list(crude_t.riskratio_confint()),
                     "fisher_exact_crude": {"odds_ratio": or_fish, "p_one_sided": p_fish},
                     "breslow_day": {"stat": bd.statistic, "p": bd.pvalue, "n_strata": len(bd_strata)},
                     "stratum_or_haldane": {"n_above_1": int((or_k > 1).sum()), "n_above_2": int((or_k >= 2).sum()),
                                            "median": float(np.median(or_k)),
                                            "q25_q75": list(np.percentile(or_k, [25, 75]))}}}

# ======================= T5: did the illicit share fall after the dark-market shutdown?
share = lab.groupby("time_step")["is_illicit"].mean()
before = share.loc[SHUTDOWN_STEP - 7:SHUTDOWN_STEP - 1].values
after = share.loc[SHUTDOWN_STEP:SHUTDOWN_STEP + 6].values
wt = stats.ttest_ind(after, before, equal_var=False, alternative="less")
d, df, ci = welch_ci(after, before)
st_t = stats.ttest_ind(after, before, equal_var=True, alternative="less")
mwu = stats.mannwhitneyu(after, before, alternative="less", method="exact")
allv = np.concatenate([after, before])
obs = after.mean() - before.mean()
perm = [allv[list(c)].mean() - np.delete(allv, list(c)).mean() for c in itertools.combinations(range(14), 7)]
p_perm = np.mean(np.array(perm) <= obs + 1e-12)
win_b = lab[lab.time_step.between(SHUTDOWN_STEP - 7, SHUTDOWN_STEP - 1)]
win_a = lab[lab.time_step.between(SHUTDOWN_STEP, SHUTDOWN_STEP + 6)]
z_pool, p_pool = proportions_ztest([win_a.is_illicit.sum(), win_b.is_illicit.sum()], [len(win_a), len(win_b)],
                                   alternative="smaller")
lev5 = stats.levene(after, before, center="median")
res["T5"] = {**TESTS["T5"], "steps_before": f"{SHUTDOWN_STEP - 7}-{SHUTDOWN_STEP - 1}",
             "steps_after": f"{SHUTDOWN_STEP}-{SHUTDOWN_STEP + 6}",
             "before": before.round(4).tolist(), "after": after.round(4).tolist(),
             "mean_before": before.mean(), "mean_after": after.mean(),
             "estimate": d, "ci95": list(ci), "stat": wt.statistic, "df": df, "p": wt.pvalue,
             "reject": bool(wt.pvalue < ALPHA),
             "alt": {"student_t": {"t": st_t.statistic, "df": 12, "p": st_t.pvalue},
                     "mann_whitney_exact": {"U": float(mwu.statistic), "p": mwu.pvalue},
                     "exact_permutation": {"n_splits": len(perm), "p": p_perm},
                     "pooled_two_proportion_z": {"z": z_pool, "p": p_pool, "n_after": len(win_a), "n_before": len(win_b)},
                     "levene_median": {"stat": lev5.statistic, "p": lev5.pvalue},
                     "shapiro_before": stats.shapiro(before).pvalue, "shapiro_after": stats.shapiro(after).pvalue}}


# p-values below ~1e-308 underflow to 0.0 in floating point; keep their log10
LN10 = np.log(10)
res["T1"]["log10_p"] = stats.norm.logsf(res["T1"]["stat"]) / LN10
res["T2"]["log10_p"] = -res["T2"]["stat"] / 2 / LN10   # chi-square with 2 d.f.: p = exp(-x/2)
res["T3"]["log10_p"] = stats.norm.logcdf(res["T3"]["stat"]) / LN10  # t with 45,995 d.f. is normal
res["T4"]["log10_p"] = stats.norm.logsf(res["T4"]["z"]) / LN10
res["T4"]["alt"]["breslow_day"]["log10_p"] = stats.chi2.logsf(bd.statistic, len(bd_strata) - 1) / LN10


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


with open(RESULTS / "tests.json", "w") as f:
    json.dump(clean(res), f, indent=2)
print(json.dumps(clean(res), indent=2))
