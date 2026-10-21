# Follow the money: how fraud-linked Bitcoin transactions and wallets behave differently

CSE558 Data Science (Monsoon 2026, IIIT-Delhi): midterm project, statistical inference with
hypothesis tests.

We take the **Elliptic++** dataset of real Bitcoin transactions and wallets, labelled
*illicit* (scams, ransomware, Ponzi schemes, dark markets), *licit* or *unknown*. We then test
four questions about how fraud-linked money behaves:

| Problem | Test | Result |
|---|---|---|
| P1: Is illicit money forwarded whole to a single receiver? | T1: two-proportion z-test | 64.9% vs 6.7% single-output, +58.2 points (95% CI 56.8 to 59.6) |
| P1: Do illicit wallets pass money straight through? | T2: chi-square test of independence (2×3) | 63.3% vs 11.5% pass-through, Cramér's V = 0.34 |
| P2: Do illicit fees carry a fixed-fee "fingerprint"? | T3: Brown–Forsythe test (one-sided) | SD of log fee rate 44% smaller (ratio 0.56, CI 0.54 to 0.57); holds in 41 of 45 time steps |
| P3: Does fraud follow the money trail? | T4: Cochran–Mantel–Haenszel test, stratified by time step | OR = 19.3 (CI 16.7 to 22.3) |
| P4: Did the illicit share fall after the dark-market shutdown (step 43)? | T5: Welch t-test on per-step shares | −5.4 points (CI −10.2 to −0.5), p = 0.016 |

The report is [report/report.pdf](report/report.pdf). Every test is also compared with the
alternatives we considered, which are listed in the report and computed in `src/03_tests.py`.

## Repository layout

```
src/
  config.py        paths and constants
  01_prepare.py    load, clean, encode; before/after statistics  -> data/processed/, results/prep_summary.json
  02_eda.py        basic questions and EDA figures                -> report/figs/, results/eda_summary.json
  03_tests.py      the five hypothesis tests and their alternatives -> results/tests.json
results/           every number quoted in the report (JSON)
report/            report.tex (course template), report.pdf, figs/
data/              not in git: raw/ (downloaded) and processed/ (made by step 1)
```

## How to reproduce

You need Python 3.10 or newer, about 2.5 GB of free disk space, and about 8 GB of RAM.

```bash
pip install -r requirements.txt

# 1. Download Elliptic++ (about 2.2 GB) from the authors' Google Drive folder,
#    linked at https://github.com/git-disl/EllipticPlusPlus
mkdir -p data/raw && cd data/raw
gdown --folder https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l
cd ../..
# this creates data/raw/Elliptic++ Dataset/*.csv

# 2. Run the pipeline (from inside src/)
cd src
python 01_prepare.py   # about 1-2 min
python 02_eda.py
python 03_tests.py     # about 1 min (bootstrap with a fixed seed of 558)

# 3. Build the report (pdfLaTeX twice, or Tectonic once)
cd ../report
tectonic report.tex    # or: pdflatex report.tex && pdflatex report.tex
```

The scripts are deterministic: the bootstrap and resampling use `numpy.random.default_rng(558)`.

## Data

* **Source:** Elliptic++ (Y. Elmougy and L. Liu, *Demystifying Fraudulent Transactions and
  Illicit Nodes in the Bitcoin Network for Financial Forensics*, KDD 2023). It extends the
  Elliptic dataset of M. Weber et al., KDD Workshop on Anomaly Detection in Finance, 2019.
  Downloaded on 21 October 2026.
* **Transactions:** 203,769 rows × 185 columns (182 features + `txId`, `Time step`, class).
  Only 17 features have real names; the 165 `Local_`/`Aggregate_` ones are anonymised by
  Elliptic.
* **Wallets:** 1,268,260 rows × 58 columns (822,942 unique addresses, 55 named features).
* **Labels:** 1 = illicit, 2 = licit, 3 = unknown. Tests use only labelled rows.

## Data preparation, in brief

* We joined the labels to the features, mapped them to names, and one-hot encoded them.
* *Unknown* rows are kept in the files but left out of the tests, because unknown is not the
  same as licit.
* We dropped 965 transactions that miss all 17 named features together (none of them illicit).
* We removed 347,569 exact duplicate wallet rows, and then collapsed the 61,487 wallets
  repeated across time steps to one row per address. The features are constant per address,
  and the code asserts this.
* Heavy-tailed fee rates are kept and flagged; tests use log10 fee rate.
* We derived these features:
  * `fee_rate` (sat/byte)
  * `single_output`, `single_input`
  * the wallet `pattern` (receive only / pass-through / send only)
  * `receives_from_illicit`, built from the transaction edge list
