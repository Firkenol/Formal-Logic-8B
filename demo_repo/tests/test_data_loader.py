import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.data_loader import clip_outliers, drop_missing, l2_normalize, min_max_scale


def test_l2_normalize_array():
    out = l2_normalize(np.array([3.0, 4.0]))
    assert np.allclose(out, [0.6, 0.8])


def test_l2_normalize_list():
    out = l2_normalize([3.0, 4.0])
    assert np.allclose(out, [0.6, 0.8])


def test_l2_normalize_zero():
    assert np.allclose(l2_normalize([0.0, 0.0]), [0.0, 0.0])


def test_drop_missing():
    assert np.allclose(drop_missing([1.0, float("nan"), 2.0]), [1.0, 2.0])


def test_min_max_scale_list():
    assert np.allclose(min_max_scale([2, 4, 6]), [0.0, 0.5, 1.0])


def test_clip_outliers():
    out = clip_outliers(list(range(101)), 10, 90)
    assert out.min() == 10 and out.max() == 90
