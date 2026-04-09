import numpy as np

from rfpop.core import (
    extract_changepoints_backtrack,
    gamma_builder_l2,
    rfpop_algorithm1_main,
)


def test_rfpop_l2_smoke_runs():
    y = np.array([0.0, 0.1, 0.2, 4.9, 5.0, 5.1], dtype=float)
    cp_tau, qt_vals, _ = rfpop_algorithm1_main(
        list(y), lambda y_t, t: gamma_builder_l2(y_t, t), beta=1.0
    )

    assert len(cp_tau) == len(y)
    assert len(qt_vals) == len(y)
    assert isinstance(extract_changepoints_backtrack(cp_tau), list)
