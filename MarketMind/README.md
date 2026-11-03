# MarketMind

An honest test of one question: **can public historical data predict whether the Nepal Stock Exchange (NEPSE) index goes up or down on the next trading day, better than simple guessing?**

> **Educational research only. Not financial advice. Do not trade money based on this project.**

## Result in plain English
- Using about 2,300 daily sessions (2016-07 to 2026-10), a random forest model got **57.1% direction accuracy** on 644 unseen sessions (2024-01 to 2026-10). Always guessing "down" gets 52.5%. Guessing "same direction as yesterday" gets 55.9%.
- Its probability forecasts were modestly better than the base rate (Brier skill +4.2%) and better than a fair "yesterday's direction" rule (p ≈ 0.003, but that test was post-hoc and the test period was examined three times).
- Year by year the edge was inconsistent, close to zero in 2026. Models for predicting the size of the return showed no reliable gain.
- **Conclusion:** a small, fragile directional signal, with no evidence it is tradable. Costs, execution and slippage were not modelled.

Read `reports/research_report.md`, then `reports/limitations.md`.

- Please verify claims by running the code. Nothing here should be taken on trust.

## Data
`data/raw/NEPSE_INDEX.csv` is an unchanged copy from [binayabaral/nepal-market-data](https://github.com/binayabaral/nepal-market-data) (MIT licence), downloaded 2026-10-27. A secondary source; **not cross-checked** against nepalstock.com or the Mendeley dataset, which were unreachable during the work. Details in `data/raw/PROVENANCE.md` and `reports/data_audit.md`.

## Method (short)
13 features computed only from information available at each session's close; expanding-window walk-forward evaluation; hyperparameters chosen on 2021–2023 validation; final test run once on 2024–2026; baselines first; block-bootstrap confidence intervals with Holm correction for multiple comparisons.

## Run it
```
pip install -r requirements.txt
python -m pytest -q              # 14 tests
python -m src.run audit          # clean data, tables, figure
python -m src.run validate       # choose hyperparameters (test rows removed)
python -m src.run test           # frozen final evaluation (use sparingly)
python -m src.paired_test        # post-hoc comparison with a fair previous-direction baseline
python -m src.forward predict    # log a prediction for the next session
python -m src.forward score      # score earlier predictions after updating the data
```
The forward log (`results/forward_log.csv`) is the clean test going forward: about 300 scored sessions are needed before it means anything.

## What is NOT done
Second data source cross-check, literature review, trading simulation, news features, notebooks. See `reports/limitations.md`.

## Licence
Code: MIT (see `LICENSE`). Data keeps its own MIT licence from the source repository.
