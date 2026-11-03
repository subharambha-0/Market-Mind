"""Targets. r_{t+1} = C_{t+1}/C_t - 1 using the NEXT RECORDED session in the same segment.
Direction = 1 if r_{t+1} > 0 else 0 (zero returns -> 0; counted and reported).
The last session of a segment (e.g. before the 2020 closure) has no valid target and is NaN."""
import numpy as np
import pandas as pd


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    nxt_close = out.groupby("segment")["close"].shift(-1)
    out["y_ret"] = nxt_close / out["close"] - 1
    out["y_dir"] = np.where(out["y_ret"].isna(), np.nan, (out["y_ret"] > 0).astype(float))
    return out
