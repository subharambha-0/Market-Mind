import numpy as np, pandas as pd
from src.features import build_features, FEATURES
from src.validation import validate_index


def synth(n=300, seed=1):
    rng = np.random.default_rng(seed)
    c = 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    d = pd.date_range("2018-01-01", periods=n, freq="D")
    return pd.DataFrame({"published_date": d.strftime("%Y-%m-%d"), "open": c, "high": c * 1.01, "low": c * 0.99,
                         "close": c, "per_change": 0.0, "traded_amount": rng.uniform(1e6, 2e6, n)})


def test_no_lookahead_features_at_t_unchanged_when_future_removed():
    clean, _ = validate_index(synth())
    full = build_features(clean, warmup=50)
    t = 200
    trunc = build_features(clean.iloc[: t + 1].copy(), warmup=50)
    assert np.allclose(full.iloc[t].to_numpy(), trunc.iloc[t].to_numpy(), equal_nan=True)


def test_future_shock_does_not_change_past_features():
    clean, _ = validate_index(synth())
    a = build_features(clean, warmup=50)
    c2 = clean.copy(); c2.loc[250:, "close"] *= 3; c2.loc[250:, "high"] *= 3; c2.loc[250:, "low"] *= 3
    b = build_features(c2.assign(ret=c2["close"].pct_change()), warmup=50)
    assert np.allclose(a.iloc[:250].to_numpy(), b.iloc[:250].to_numpy(), equal_nan=True)


def test_features_exclude_targets():
    assert not any(f.startswith("y_") for f in FEATURES)
