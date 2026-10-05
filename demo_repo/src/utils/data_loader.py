"""Utilities for loading and preparing numeric datasets."""

import csv

import numpy as np


def load_csv(path, column):
    """Read a single numeric column from a CSV file."""
    values = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            values.append(float(row[column]))
    return np.array(values)


def drop_missing(data):
    """Remove NaN entries from an array."""
    data = np.asarray(data, dtype=float)
    return data[~np.isnan(data)]


def clip_outliers(data, low=1.0, high=99.0):
    """Clip values outside the given percentile range."""
    data = np.asarray(data, dtype=float)
    lo, hi = np.percentile(data, [low, high])
    return np.clip(data, lo, hi)


def l2_normalize(data):
    """Scale a vector to unit L2 norm.

    Accepts any array-like of numbers.
    """
    norm = np.sqrt((data ** 2).sum())
    if norm == 0:
        return np.zeros(len(data))
    return data / norm


def min_max_scale(data):
    """Scale values linearly into the range [0, 1]."""
    data = np.asarray(data, dtype=float)
    span = data.max() - data.min()
    if span == 0:
        return np.zeros_like(data)
    return (data - data.min()) / span
