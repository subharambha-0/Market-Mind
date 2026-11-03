# Data audit (all numbers are directly observed properties of the downloaded file)
- 6,721 rows, 1997-07-20 -> 2026-10-26; no duplicate dates; sorted; no non-positive closes; `traded_quantity` is empty for every row.
- Sessions per year ~225-245, except 2020 (184): the exchange was closed ~47-51 days (gaps 2020-05-12 and 2020-06-29) -> treated as a series break (no return across it).
- Ordinary festival closures (8-12 days) are kept as normal consecutive sessions.
- Quality changes over time: before ~2011 `per_change` is 0.00 on most days even when the close moves 5-10% (field unreliable); 2011-2015 closes are mostly integers and `per_change` often disagrees with the close-to-close return; OHLC are all equal to the close until 2016-06-16.
  From 2016-07 on, `per_change` matches the computed return and OHLC fields are real. => Primary sample starts 2016-07-01 (chosen from data quality, before looking at model results).
- 138 zero returns in the full file (2.1%); 1 inside the modelling table. Zero return -> label 0 ("not up").
- 2 returns above 10% in absolute value (flagged, kept; not removed). 22 rows with close outside [low, high] (flagged, kept; close is the only price used in targets).
- Modelling table: 2,309 rows (2016-07-03 -> 2026-10-16). Dev = 1,665 (to 2023-12-31); validation block 708; test 644.
- Not done: cross-check against a second source (Mendeley / NEPSE site unreachable from the sandbox).
Figure: results/figures/data_audit.png
