"""Data validation for the NEPSE index. Fails loudly on essential problems; records everything else."""
import pandas as pd

REQUIRED = ["published_date", "open", "high", "low", "close", "per_change", "traded_amount"]


class DataValidationError(ValueError):
    pass


def validate_index(raw: pd.DataFrame, max_gap_days: int = 7, suspicious_abs_return: float = 0.10):
    """Return (clean_df, report). Nothing is filled, interpolated or winsorised."""
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    df = raw[REQUIRED].copy()
    report = {"rows_raw": len(df), "exclusions": []}

    df["date"] = pd.to_datetime(df["published_date"], errors="coerce")
    if df["date"].isna().any():
        raise DataValidationError(f"{int(df['date'].isna().sum())} unparsable dates")
    for c in ["open", "high", "low", "close", "per_change", "traded_amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df["close"].isna().any() or not (df["close"] > 0).all():
        raise DataValidationError("Non-numeric, missing or non-positive close values")

    # duplicates: identical rows are dropped (recorded); conflicting rows are an error
    dup_date = df["date"].duplicated(keep=False)
    if dup_date.any():
        sub = df[dup_date]
        if sub.drop_duplicates().groupby("date").size().max() > 1:
            raise DataValidationError("Duplicate dates with conflicting values")
        n = int(df.duplicated(subset=["date"]).sum())
        df = df.drop_duplicates(subset=["date"])
        report["exclusions"].append({"reason": "exact duplicate row", "n": n})

    df = df.sort_values("date").reset_index(drop=True)
    df["gap_days"] = df["date"].diff().dt.days
    df["segment"] = (df["gap_days"] > max_gap_days).cumsum()
    # return from the previous available session, only valid inside a segment
    ret = df["close"].pct_change()
    ret[df["segment"] != df["segment"].shift(1)] = float("nan")
    df["ret"] = ret
    df["suspicious_return"] = df["ret"].abs() > suspicious_abs_return
    df["ohlc_flat"] = (df["open"] == df["close"]) & (df["high"] == df["close"]) & (df["low"] == df["close"])
    df["ohlc_inconsistent"] = (df["high"] < df["low"]) | (df["close"] > df["high"]) | (df["close"] < df["low"])

    report.update(
        rows_clean=len(df),
        first_date=str(df["date"].min().date()),
        last_date=str(df["date"].max().date()),
        n_segments=int(df["segment"].nunique()),
        gaps_over_limit=df.loc[df["gap_days"] > max_gap_days, ["date", "gap_days"]].astype(str).to_dict("records"),
        n_suspicious_returns=int(df["suspicious_return"].sum()),
        n_zero_returns=int((df["ret"] == 0).sum()),
        n_flat_ohlc=int(df["ohlc_flat"].sum()),
        n_ohlc_inconsistent=int(df["ohlc_inconsistent"].sum()),
    )
    return df, report
