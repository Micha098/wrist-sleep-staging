"""Smoke tests on synthetic data: the inputs have the trained shapes and a night can be scored end to end."""
import numpy as np
import pandas as pd

from wristsleep.inputs import epoch_table, lgbm_features, network_minute_inputs, raw_epochs
from wristsleep.measures import night_measures

START = 1_714_782_600.0  # 2024-05-04 00:30 UTC
HOURS = 8


def synthetic():
    rng = np.random.default_rng(0)
    t = START - 3600 + 60 * np.arange((HOURS + 2) * 60)
    minutes = pd.DataFrame({'timestamp_unix': t * 1000, 'pulse_rate_bpm': 70 + rng.normal(0, 3, len(t)),
                            'prv_rmssd_ms': 60 + rng.normal(0, 5, len(t)), 'temperature_celsius': 34 + rng.normal(0, .2, len(t)),
                            'respiratory_rate_brpm': 16 + rng.normal(0, 1, len(t)), 'activity_counts': rng.poisson(1, len(t)),
                            'step_counts': 0, 'wearing_detection_percentage': 100})
    ts = START - 3600 + np.arange(int((HOURS + 1.1) * 3600 * 64)) / 64
    bvp = pd.DataFrame({'time': ts, 'bvp': np.sin(2 * np.pi * 1.1 * ts)})
    acc = pd.DataFrame({'time': ts, 'x': 0.0, 'y': -1.0, 'z': 0.0})
    return minutes, bvp, acc


def test_inputs_have_trained_shapes():
    minutes, bvp, acc = synthetic()
    E = epoch_table(minutes, START, START + HOURS * 3600)
    assert len(E) == HOURS * 120
    z = network_minute_inputs(E, np.zeros(9), np.ones(9))
    assert z.shape == (len(E), 19) and np.isfinite(z).all()
    r = raw_epochs(bvp, acc, START, len(E))
    assert r.shape == (len(E), 5, 960) and r[:, 4].max() == 0  # no missing raw data
    assert lgbm_features(E).shape[1] == 39


def test_night_measures():
    s = np.r_[[0] * 20, [1] * 100, [0] * 4, [2] * 100, [3] * 50, [0] * 10]
    m = night_measures(s, ['Wake', 'Light', 'N3', 'REM'])
    assert m['SOL_min'] == 10 and m['WASO_min'] == 2 and m['awakenings'] == 1 and m['TST_min'] == 125


def test_score_night_end_to_end():
    import os
    import pytest
    from wristsleep.score import WEIGHTS, Scorer
    if not os.path.exists(WEIGHTS / 'rawnet.pt'):
        pytest.skip('weights not present')
    minutes, bvp, acc = synthetic()
    ep, s = Scorer().score_night(minutes, START, START + HOURS * 3600, bvp=bvp, acc=acc)
    assert len(ep) == HOURS * 120 and set(ep.stage) <= {'Wake', 'Light', 'N3', 'REM'}
    assert 0 <= s['net_TST_min'] <= HOURS * 60
