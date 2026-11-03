# MarketMind: can public history predict the next NEPSE index move?
Status: executed on real data (see "What actually ran"). Claim labels: **[BG]** background, **[OBS]** observed dataset property, **[RES]** result of an executed experiment, **[INT]** interpretation, **[HYP]** hypothesis for future testing.

## Abstract
Using 2,309 daily sessions of the NEPSE index (2016-07 to 2026-10), I compared simple baselines with logistic regression, ridge regression, random forests and gradient boosting in an expanding-window walk-forward design. Hyperparameters were frozen on a 2021-2023 validation block; the final test (2024-01-01 to 2026-10-16, 644 sessions) was then run once. **[RES]** The random forest direction classifier beat the base-rate probability forecast on Brier score (skill +4.2%, block-bootstrap 95% CI for the loss difference excludes zero, Holm-adjusted p = 0.015 across 6 comparisons). The boosting classifier was weaker and not significant after correction; logistic regression showed no skill. No return-forecast model beat the historical-mean forecast with statistical support (OOS R² between -2.2% and +3.5%). **[INT]** The evidence is consistent with a small, fragile short-term directional signal, not with a usable trading edge: skill was near zero in 2026, one test period is a single historical draw, and the plain "tomorrow = today's direction" rule already achieved 55.9% accuracy. No trading simulation was run.

## 1. Question
Can models trained on public historical NEPSE data predict next-session index direction or return better than simple baselines on strictly unseen future data? Primary metrics were fixed in `config/experiment.yaml` before models were run: Brier skill vs the expanding training base rate (direction) and OOS R² vs the expanding historical mean (returns).

## 2. Background and literature
Not done. No literature review was performed, so no novelty claim is made and no papers are cited. **[HYP]** Thin trading and daily price limits on small exchanges may create short-term return dependence; this should be checked against published NEPSE studies before being asserted.

## 3. Data
**[OBS]** Source: `binayabaral/nepal-market-data` (MIT licence), file `NEPSE_INDEX.csv`, 6,721 rows 1997-07-20 to 2026-10-26, downloaded 2026-10-27; a secondary source (scraped from exchange data), not independently cross-checked. nepalstock.com and the Mendeley dataset were unreachable from my sandbox (network allow-list), so they were not verified. Details: `reports/data_audit.md`, `data/raw/PROVENANCE.md`.
Key properties: the 2020 closure (47-51 days) is a real gap and breaks the series; before mid-2016 OHLC fields are copies of the close and `per_change` is unreliable, so the primary sample starts 2016-07-01 (chosen for data quality, before any model was run). Festival closures (<= 12 days) are treated as ordinary consecutive sessions. Nothing was interpolated or removed for being extreme; 2 returns above 10% are flagged and retained.

## 4. Targets and features
r(t+1) = C(t+1)/C(t) - 1 using the next recorded session in the same segment. Direction = 1 if r(t+1) > 0, else 0 (zero returns count as "not up": 1 of 2,309 rows). Sessions with no valid next session (end of a segment) have no target. 13 features (lagged returns, 5/10-day momentum, 10/20-day volatility, distance from 20-day mean, streak, intraday range, turnover ratio), all computed from data up to the close of session t; dictionary in `src/features.py`. A unit test shows features at t are unchanged if all later data is deleted or altered. Calendar effects were not tested.

## 5. Design
Dev = 2016-07-03..2023-12-31 (1,665 rows). Validation = walk-forward over 2021-01-01..2023-12-31 (708 rows), initial training on earlier rows, refit every 21 sessions, expanding window. Test = 2024-01-01..2026-10-16 (644 rows), same scheme with frozen hyperparameters and refit every 21 sessions; the training rows for a block are only those before the block, so their targets were known. Scaling is inside the model pipeline. A canary test confirms the harness would expose a leaking feature. Grid size: 14 validation configurations (all logged in `results/experiment_log.csv`).

## 6. Results (test period, n = 644, identical dates for every row) **[RES]**
### Direction (test)
| model                   |   accuracy |   balanced_acc |   precision |   recall |   brier |   brier_skill |   share_pred_up |
|:------------------------|-----------:|---------------:|------------:|---------:|--------:|--------------:|----------------:|
| BASE majority (train)   |     0.5248 |         0.5    |      0      |   0      |  0.4752 |       -0.9031 |          0      |
| BASE previous-direction |     0.559  |         0.5579 |      0.5359 |   0.5359 |  0.441  |       -0.7662 |          0.4752 |
| BASE base-rate prob     |     0.5248 |         0.5    |      0      |   0      |  0.2497 |        0      |          0      |
| logistic_clf            |     0.5233 |         0.5128 |      0.4973 |   0.3007 |  0.2518 |       -0.0086 |          0.2873 |
| rf_clf                  |     0.5714 |         0.5646 |      0.5647 |   0.4281 |  0.2392 |        0.0421 |          0.3602 |
| hgb_clf                 |     0.5792 |         0.572  |      0.5771 |   0.4281 |  0.2441 |        0.0224 |          0.3525 |

### Returns (test)
| model          |     mae |    rmse |   oos_r2_vs_histmean |   mean_error |
|:---------------|--------:|--------:|---------------------:|-------------:|
| BASE zero      | 0.00822 | 0.01191 |              9e-05   |      0.00045 |
| BASE hist-mean | 0.00824 | 0.01191 |              0       |      0.00026 |
| ridge_reg      | 0.00839 | 0.01204 |             -0.02215 |      0.00023 |
| rf_reg         | 0.00818 | 0.01176 |              0.02547 |      0.0002  |
| hgb_reg        | 0.00815 | 0.0117  |              0.0346  |      0.00025 |

### Block-bootstrap vs baseline (loss difference; negative = model better)
| comparison                         |   mean_diff |    ci_lo |    ci_hi |   p_not_better |   p_holm |
|:-----------------------------------|------------:|---------:|---------:|---------------:|---------:|
| logistic_clf vs base-rate (Brier)  |     0.00216 | -0.00321 |  0.00742 |         0.7855 |   1      |
| rf_clf vs base-rate (Brier)        |    -0.01052 | -0.01736 | -0.00324 |         0.0025 |   0.015  |
| hgb_clf vs base-rate (Brier)       |    -0.00558 | -0.01304 |  0.00245 |         0.0775 |   0.3875 |
| ridge_reg vs hist-mean (sq. error) |     0       | -0       |  1e-05   |         0.845  |   1      |
| rf_reg vs hist-mean (sq. error)    |    -0       | -1e-05   |  0       |         0.13   |   0.52   |
| hgb_reg vs hist-mean (sq. error)   |    -0       | -1e-05   |  0       |         0.1565 |   0.52   |

Brier of the two hard-label baselines is not comparable to probability forecasts; use accuracy for them. Base-rate accuracy is 52.5% because 52.5% of test sessions were not up.

Bootstrap: circular block bootstrap (block 10, 2,000 resamples, seed fixed) of the mean loss difference; `p_not_better` is the share of resampled means >= 0; Holm correction over the 6 model-vs-baseline tests. It assumes dependence is short relative to 10 sessions, and it does not correct for the 14 validation configurations or for my design choices.

### Robustness (`results/tables/test_robustness.csv`)
RF Brier skill by year: 2024 +4.0%, 2025 +7.3%, 2026 +0.7%. HGB: +4.7%, +2.7%, -1.3%. Logistic: -2.7%, +1.6%, -1.6%. Excluding sessions with |r| >= 4% leaves RF at +4.2% (no extreme-day dependence). The classifiers predict "up" only 29-36% of the time while 47.5% of sessions were up, so predictions are far from constant, but they lean "down".

## 7. Audit questions
- Beat a trivial baseline consistently? Only RF, and only clearly in 2024-2025.
- Only one period? 2026 is near zero; sample per year is ~190-230 sessions, so single-year numbers are noisy.
- Predicting the majority class? No (balanced accuracy 56-57% for RF/HGB).
- Does a simpler model do as well? The previous-direction rule gets 55.9% accuracy vs 57.1% (RF) and 57.9% (HGB); I did not test those accuracy differences, and with n = 644 accuracy has a standard error of about 2 points, so they are indistinguishable. **[INT]** Much of the apparent skill may just be day-to-day direction persistence.
- Logistic regression, the "primary interpretable benchmark", had no skill, which suggests any signal is non-linear or interaction-based **[HYP]**.
- Data-quality artefacts? Not ruled out: single secondary source, and thin-trading/price-limit mechanics may create persistence that cannot be traded.
- Independent reproduction? Not done beyond rerunning my own code.

## 8. Conclusion
**Result: weak, partly supported evidence of small directional skill for a random-forest model in this test window; inconclusive about robustness; no evidence for return-magnitude forecasting.** This is not evidence of profitability. Costs, execution and slippage were not modelled.

## 9. Process disclosures
1. The gap rule was changed from 7 to 20 calendar days after the data audit and before any model ran (7 split at normal festival closures).
2. The test stage was run twice. Run 1 had a reporting bug (the historical-mean-return baseline used the direction base rate), which produced nonsense R² values. I fixed it and reran; classification results were identical, models and hyperparameters unchanged. Run 1 rows in the log are marked invalid. The test period was therefore seen twice, though no modelling choice depended on it.

## 10. Reproduce
```
pip install -r requirements.txt
python -m pytest -q            # 14 tests
python -m src.run audit        # clean data, tables, figure
python -m src.run validate     # select hyperparameters (test rows removed)
python -m src.run test         # frozen final evaluation
```
Data: place `NEPSE_INDEX.csv` from the GitHub repo in `data/raw/` (newer downloads contain newer rows and will change results; the exact file used is hashed in `data/raw/PROVENANCE.md`).

## 11. Post-hoc paired test (added later; the test period was looked at a third time) [RES]
Baseline: probability of "up" given whether yesterday was up, estimated on training rows only, same walk-forward schedule (`src/paired_test.py`, `results/tables/paired_test_*.csv`).
- Prev-direction (probabilistic): Brier skill +1.2% vs base rate (CI for loss difference includes zero, so not clearly better than the base rate).
- Random forest vs prev-direction: Brier difference -0.0074 (95% CI -0.0128 to -0.0021, p = 0.0025, uncorrected for multiple comparisons and post-hoc).
- Gradient boosting vs prev-direction: -0.0025 (CI -0.0089 to 0.0043), no evidence.
- Accuracy difference RF minus prev-direction: +1.2 points (CI about -2.5 to +5.0), indistinguishable.
[INT] The forest's probability forecasts are somewhat better calibrated/sharper than a one-feature persistence rule, but its hit-rate advantage is not demonstrated. Development-only attempts to improve it (extra features, boosting/averaging ensembles; 2019-2023 walk-forward) gave no gain over the current forest.
