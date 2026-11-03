import numpy as np, pandas as pd
from src.evaluation import walk_forward, block_bootstrap_diff, holm
from src.features import FEATURES


def test_walk_forward_trains_only_on_past_rows():
    n = 400
    rng = np.random.default_rng(0)
    d = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    d["date"] = pd.date_range("2019-01-01", periods=n); d["y_dir"] = (rng.random(n) > .5).astype(float); d["y_ret"] = rng.normal(0, .01, n)
    out = walk_forward(d, FEATURES, "logistic", "clf", {"C": 1.0}, "2019-09-01", "2020-01-31", 21)
    first_test_row = out.index.min()
    assert out["n_train"].iloc[0] == first_test_row           # trained on rows strictly before
    assert (out["n_train"].to_numpy()[1:] >= out["n_train"].to_numpy()[:-1]).all()  # expanding


def test_leaky_feature_is_detected_by_the_harness():
    """Canary: if a feature secretly contains the target, the harness must show huge skill (so real runs are not leaking)."""
    n = 500; rng = np.random.default_rng(1)
    d = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    d["y_ret"] = rng.normal(0, .01, n); d["y_dir"] = (d["y_ret"] > 0).astype(float); d["ret_0"] = d["y_ret"]
    d["date"] = pd.date_range("2019-01-01", periods=n)
    out = walk_forward(d, FEATURES, "logistic", "clf", {"C": 1.0}, "2020-03-01", "2020-05-15", 21)
    assert ((out["pred"] > .5) == (out["y"] == 1)).mean() > 0.9


def test_bootstrap_and_holm_basic():
    x = np.random.default_rng(0).normal(size=300)
    r = block_bootstrap_diff(x - 1.0, x, block=10, n_boot=500)
    assert r["ci_hi"] < 0 and r["p_not_better"] < .01
    assert (holm(np.array([.01, .02, .5])) >= np.array([.01, .02, .5])).all()
