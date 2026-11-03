import numpy as np, pandas as pd
from src.targets import add_targets


def frame(close, seg):
    return pd.DataFrame({"close": close, "segment": seg})


def test_target_definition_and_zero_return_is_zero_label():
    t = add_targets(frame([100., 101., 101., 99.], [0, 0, 0, 0]))
    assert np.allclose(t["y_ret"].iloc[:3], [0.01, 0.0, 99 / 101 - 1])
    assert list(t["y_dir"].iloc[:3]) == [1.0, 0.0, 0.0]
    assert np.isnan(t["y_ret"].iloc[3])


def test_no_target_across_segment_break():
    t = add_targets(frame([100., 110., 50., 55.], [0, 0, 1, 1]))
    assert np.isnan(t["y_ret"].iloc[1]) and np.isnan(t["y_dir"].iloc[1])
