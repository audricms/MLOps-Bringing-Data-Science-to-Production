from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import get_segments_from_cp_tau, rfpop_algorithm1_main
from .tuning import compute_loss_bound_k, compute_penalty_beta
from .core import gamma_builder_biweight, gamma_builder_huber, gamma_builder_l2

LossName = Literal["huber", "biweight", "l2"]


def plot_segments(
    y: Sequence[float],
    loss: LossName = "biweight",
    beta: Optional[float] = None,
    k_value: Optional[float] = None,
    x: Optional[Sequence[object]] = None,
    show_points: bool = True,
    figsize: Tuple[int, int] = (12, 5),
):
    y_arr = np.asarray(y, dtype=float)

    if beta is None:
        beta = compute_penalty_beta(y_arr, loss)
    if loss == "huber" and k_value is None:
        k_value = compute_loss_bound_k(y_arr, "huber")
    if loss == "biweight" and k_value is None:
        k_value = compute_loss_bound_k(y_arr, "biweight")

    if loss == "huber":
        if k_value is None:
            raise ValueError("k_value is required for huber loss.")

        def gamma_builder(y_t: float, t: int):
            return gamma_builder_huber(y_t, k_value, t)

    elif loss == "biweight":
        if k_value is None:
            raise ValueError("k_value is required for biweight loss.")

        def gamma_builder(y_t: float, t: int):
            return gamma_builder_biweight(y_t, k_value, t)

    else:

        def gamma_builder(y_t: float, t: int):
            return gamma_builder_l2(y_t, t)

    cp_tau, _, _ = rfpop_algorithm1_main(list(y_arr), gamma_builder, float(beta))
    segments = get_segments_from_cp_tau(cp_tau, y_arr)

    x_values = np.arange(len(y_arr)) if x is None else np.asarray(x)

    fig, ax = plt.subplots(figsize=figsize)

    if show_points:
        ax.plot(x_values, y_arr, ".", markersize=2, label="series")
    else:
        ax.plot(x_values, y_arr, linewidth=1, label="series")

    for idx, (start, end, seg_mean) in enumerate(segments):
        x_start = x_values[start]
        x_end = x_values[end]
        ax.plot(
            [x_start, x_end],
            [seg_mean, seg_mean],
            color="black",
            linewidth=2,
            label="segment mean" if idx == 0 else "",
        )
        if end < len(y_arr) - 1:
            ax.axvline(x=x_end, color="red", linestyle="--", alpha=0.5)

    title_suffix = f"loss={loss}, beta={beta:.3g}"
    if loss in {"huber", "biweight"}:
        if k_value is None:
            raise ValueError("k_value is required for robust losses.")
        title_suffix += f", K={float(k_value):.3g}"

    ax.set_title(f"Detected changepoint segments ({title_suffix})")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    changepoints = sorted(set(cp_tau) - {0})
    return fig, ax, changepoints


def plot_segments_from_dataframe(
    df: pd.DataFrame,
    column: str,
    date_filter_start: Optional[str] = None,
    loss: LossName = "biweight",
    beta_scale: float = 1.0,
):
    series = df[column].dropna()
    if date_filter_start is not None:
        series = series[series.index > date_filter_start]

    beta = compute_penalty_beta(series.to_numpy(), loss) * beta_scale
    k_value = None
    if loss == "huber":
        k_value = compute_loss_bound_k(series.to_numpy(), "huber")
    if loss == "biweight":
        k_value = compute_loss_bound_k(series.to_numpy(), "biweight")

    return plot_segments(
        y=series.to_numpy(),
        x=series.index.to_numpy(),
        loss=loss,
        beta=beta,
        k_value=k_value,
    )
