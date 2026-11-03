import numpy as np, pandas as pd, pytest
from src.validation import validate_index, DataValidationError


def make(n=30):
    d = pd.date_range("2020-01-01", periods=n, freq="D")
    c = 100 + np.arange(n, dtype=float)
    return pd.DataFrame({"published_date": d.strftime("%Y-%m-%d"), "open": c, "high": c + 1, "low": c - 1,
                         "close": c, "per_change": 0.0, "traded_amount": 1e6})


def test_missing_column_raises():
    with pytest.raises(DataValidationError):
        validate_index(make().drop(columns=["close"]))


def test_nonpositive_close_raises():
    df = make(); df.loc[3, "close"] = 0
    with pytest.raises(DataValidationError):
        validate_index(df)


def test_bad_date_raises():
    df = make(); df.loc[2, "published_date"] = "not-a-date"
    with pytest.raises(DataValidationError):
        validate_index(df)


def test_exact_duplicate_dropped_and_recorded():
    df = pd.concat([make(), make().iloc[[5]]])
    clean, rep = validate_index(df)
    assert len(clean) == 30 and rep["exclusions"][0]["n"] == 1


def test_conflicting_duplicate_raises():
    df = pd.concat([make(), make().iloc[[5]].assign(close=999.0)])
    with pytest.raises(DataValidationError):
        validate_index(df)


def test_gap_creates_segment_and_no_cross_gap_return():
    df = make(); df["published_date"] = pd.to_datetime(df["published_date"])
    df.loc[15:, "published_date"] += pd.Timedelta(days=40)
    clean, _ = validate_index(df.assign(published_date=df["published_date"].dt.strftime("%Y-%m-%d")))
    assert clean["segment"].nunique() == 2
    assert np.isnan(clean.loc[15, "ret"])
