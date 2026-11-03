"""Baselines. All fitted parameters come from TRAINING rows only."""
import numpy as np


def zero_return(n):
    return np.zeros(n)


def previous_direction(ret_0):
    """Predict that tomorrow goes the same way as today (ret_0 > 0 -> up)."""
    return (np.asarray(ret_0) > 0).astype(float)


def majority_direction(y_train, n):
    return np.full(n, float(np.mean(y_train) >= 0.5))


def base_rate_prob(y_train, n):
    return np.full(n, float(np.mean(y_train)))


def historical_mean_return(y_train, n):
    return np.full(n, float(np.mean(y_train)))
