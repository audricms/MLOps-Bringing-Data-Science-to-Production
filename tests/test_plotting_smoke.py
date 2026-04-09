import matplotlib
import numpy as np
import pandas as pd

from rfpop.plotting import plot_segments_from_dataframe

matplotlib.use("Agg")


def test_plot_segments_from_dataframe_smoke():
    idx = pd.date_range("2020-01-01", periods=8, freq="D")
    values = np.array([0.0, 0.1, 0.2, 3.0, 3.1, 3.2, 0.2, 0.1], dtype=float)
    df = pd.DataFrame({"signal": values}, index=idx)

    fig, ax, changepoints = plot_segments_from_dataframe(df=df, column="signal", loss="l2")

    assert fig is not None
    assert ax is not None
    assert isinstance(changepoints, list)
