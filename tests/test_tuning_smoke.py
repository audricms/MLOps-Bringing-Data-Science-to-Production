import numpy as np

from rfpop.tuning import select_params_bic, select_params_elbow


def test_bic_and_elbow_selection_smoke():
    y = np.array([0.0, 0.1, 0.2, 3.0, 3.1, 3.2, 0.2, 0.1], dtype=float)

    bic = select_params_bic(y, loss="l2", beta_values=[0.5, 1.0, 2.0])
    elbow = select_params_elbow(y, loss="l2", beta_values=[0.5, 1.0, 2.0])

    assert bic["strategy"] == "bic"
    assert elbow["strategy"] == "elbow"
    assert "best" in bic
    assert "best" in elbow
