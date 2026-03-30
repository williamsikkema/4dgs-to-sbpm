import numpy as np

from face_mc_gs.sync.coarse_sync import cross_correlation_lag


def test_recover_lag():
    rng = np.random.default_rng(0)
    n = 200
    base = np.cumsum(rng.standard_normal(n))
    lag_true = 15
    other = np.zeros_like(base)
    other[lag_true:] = base[:-lag_true]
    est, corr = cross_correlation_lag(base, other, max_lag=40)
    # other is delayed by lag_true → best lag is -lag_true under this correlation convention
    assert est == -lag_true or abs(est + lag_true) <= 1
