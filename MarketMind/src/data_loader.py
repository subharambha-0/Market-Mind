import pandas as pd
from .validation import validate_index


def load_clean_index(path, max_gap_days=7, suspicious_abs_return=0.10):
    raw = pd.read_csv(path)
    return validate_index(raw, max_gap_days, suspicious_abs_return)
