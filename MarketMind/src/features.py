"""Features for predicting session t+1. Every feature uses data up to and including the close of
session t only (information cutoff = close of t). Windows never cross a segment break."""
import numpy as np
import pandas as pd

FEATURE_DICTIONARY = {
    "ret_0": ("close_t/close_{t-1}-1", "latest return; short-term momentum/reversal"),
    "ret_1": ("return of t-1", "lagged return"),
    "ret_2": ("return of t-2", "lagged return"),
    "ret_3": ("return of t-3", "lagged return"),
    "ret_4": ("return of t-4", "lagged return"),
    "ret_5d": ("close_t/close_{t-5}-1", "one-week momentum"),
    "ret_10d": ("close_t/close_{t-10}-1", "two-week momentum"),
    "vol_10": ("std of last 10 returns", "recent volatility regime"),
    "vol_20": ("std of last 20 returns", "medium volatility regime"),
    "ma_gap_20": ("close_t/mean(close_{t-19..t})-1", "distance from trend"),
    "streak": ("signed run length of same-direction returns, capped at 5", "directional persistence"),
    "range_0": ("(high_t-low_t)/close_t", "intraday range as volatility proxy (needs real OHLC)"),
    "turnover_ratio": ("log(amount_t / mean(amount_{t-19..t}))", "unusual trading activity"),
}
FEATURES = list(FEATURE_DICTIONARY)


def _streak(sign: pd.Series, cap=5) -> pd.Series:
    out, run, prev = [], 0, 0
    for s in sign.fillna(0).to_numpy():
        run = run + s if (s != 0 and s == prev) else s
        run = max(-cap, min(cap, run))
        out.append(run)
        prev = s
    return pd.Series(out, index=sign.index)


def _per_segment(g: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=g.index)
    r = g["ret"]
    for k in range(5):
        f[f"ret_{k}"] = r.shift(k)
    f["ret_5d"] = g["close"] / g["close"].shift(5) - 1
    f["ret_10d"] = g["close"] / g["close"].shift(10) - 1
    f["vol_10"] = r.rolling(10).std()
    f["vol_20"] = r.rolling(20).std()
    f["ma_gap_20"] = g["close"] / g["close"].rolling(20).mean() - 1
    f["streak"] = _streak(np.sign(r))
    f["range_0"] = (g["high"] - g["low"]) / g["close"]
    amt = g["traded_amount"].where(g["traded_amount"] > 0)
    f["turnover_ratio"] = np.log(amt / amt.rolling(20).mean())
    return f


def build_features(df: pd.DataFrame, warmup: int = 50) -> pd.DataFrame:
    parts = [_per_segment(g) for _, g in df.groupby("segment")]
    f = pd.concat(parts).loc[df.index]
    pos = df.groupby("segment").cumcount()
    f[pos < warmup] = np.nan   # same warm-up everywhere so rows are comparable
    return f[FEATURES]
