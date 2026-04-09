from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, cast

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import norm
from statsmodels import robust

from .core import (
    extract_changepoints_backtrack,
    gamma_builder_biweight,
    gamma_builder_huber,
    gamma_builder_l1,
    gamma_builder_l2,
    rfpop_algorithm1_main,
)

LossName = Literal["huber", "biweight", "l2", "l1"]


def compute_penalty_beta(y: np.ndarray, loss: LossName) -> float:
    ys = pd.Series(y)
    sigma = robust.mad(ys.diff().dropna()) / np.sqrt(2)
    n = len(y)

    if loss == "l2":
        return float(2 * sigma**2 * np.log(n))
    if loss == "biweight":
        k_std = 3.0
        e_phi2, _ = integrate.quad(
            lambda z: (2 * z if abs(z) <= k_std else 0.0) ** 2 * norm.pdf(z),
            -np.inf,
            np.inf,
        )
        return float(2 * sigma**2 * np.log(n) * e_phi2)
    if loss == "huber":
        k_std = 1.345
        e_phi2, _ = integrate.quad(
            lambda z: (2 * z if abs(z) <= k_std else 2 * k_std * np.sign(z)) ** 2 * norm.pdf(z),
            -np.inf,
            np.inf,
        )
        return float(2 * sigma**2 * np.log(n) * e_phi2)

    return float(np.log(n))


def compute_loss_bound_k(y: np.ndarray, loss: Literal["huber", "biweight"]) -> float:
    ys = pd.Series(y)
    mad = robust.mad(ys.diff().dropna()) / np.sqrt(2)
    if loss == "biweight":
        return float(3 * mad)
    return float(1.345 * mad)


def select_params_bic(
    y: np.ndarray,
    loss: LossName = "biweight",
    beta_values: Optional[Iterable[float]] = None,
    k_values: Optional[Iterable[Optional[float]]] = None,
) -> Dict[str, Any]:
    y_arr = np.asarray(y, dtype=float)
    n = len(y_arr)

    beta_ref = compute_penalty_beta(y=y_arr, loss=loss if loss != "l1" else "l2")
    if loss == "huber":
        k_ref = compute_loss_bound_k(y=y_arr, loss="huber")
    elif loss == "biweight":
        k_ref = compute_loss_bound_k(y=y_arr, loss="biweight")
    else:
        k_ref = None

    beta_list: List[float]
    if beta_values is None:
        beta_list = [float(v) for v in beta_ref * np.array([0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])]
    else:
        beta_list = [float(v) for v in beta_values]

    k_list: List[Optional[float]]
    if k_ref is not None and k_values is None:
        k_list = [float(v) for v in k_ref * np.array([0.5, 0.75, 1.0, 1.5, 2.0])]
    elif k_values is not None:
        k_list = [None if v is None else float(v) for v in k_values]
    else:
        k_list = [None]

    results: List[Dict[str, Any]] = []

    for beta in beta_list:
        for k_value in k_list:
            cp_tau, qt_vals = _run_with_params(y_arr, loss, float(beta), k_value)
            changepoints = extract_changepoints_backtrack(cp_tau)
            n_cp = len(changepoints)
            bic = float(qt_vals[-1] + np.log(n) * n_cp)

            results.append(
                {
                    "beta": float(beta),
                    "k": None if k_value is None else float(k_value),
                    "changepoints": changepoints,
                    "n_changepoints": n_cp,
                    "qt_final": float(qt_vals[-1]),
                    "bic": bic,
                }
            )

    best = min(results, key=lambda r: float(r["bic"]))

    return {
        "strategy": "bic",
        "loss": loss,
        "best": best,
        "reference": {"beta": float(beta_ref), "k": None if k_ref is None else float(k_ref)},
        "all_results": results,
    }


def select_params_elbow(
    y: np.ndarray,
    loss: LossName = "biweight",
    beta_values: Optional[Iterable[float]] = None,
    k_values: Optional[Iterable[Optional[float]]] = None,
) -> Dict[str, Any]:
    y_arr = np.asarray(y, dtype=float)

    search = select_params_bic(
        y=y_arr,
        loss=loss,
        beta_values=beta_values,
        k_values=k_values,
    )

    all_results = cast(List[Dict[str, Any]], search["all_results"])
    results = sorted(all_results, key=lambda x: float(x["beta"]))
    n_cps = [r["n_changepoints"] for r in results]

    if not n_cps:
        raise ValueError("No valid RFPOP result was produced for elbow search.")

    max_ncp = max(n_cps)
    threshold = max_ncp / 2

    best = None
    for i in range(len(n_cps) - 1):
        if n_cps[i] > 0 and n_cps[i] <= threshold and n_cps[i + 1] == n_cps[i]:
            best = results[i]
            break
    if best is None:
        for i in range(len(n_cps) - 1):
            if n_cps[i] > 0 and n_cps[i + 1] == n_cps[i]:
                best = results[i]
                break
    if best is None:
        best = next((r for r in results if r["n_changepoints"] > 0), results[0])

    return {
        "strategy": "elbow",
        "loss": loss,
        "best": best,
        "reference": search["reference"],
        "all_results": results,
    }


def _run_with_params(
    y: np.ndarray,
    loss: LossName,
    beta: float,
    k_value: Optional[float],
) -> Tuple[List[int], List[float]]:
    y_list = list(y)

    if loss == "huber":
        if k_value is None:
            raise ValueError("k_value is required for huber loss.")

        def gamma_builder(y_t: float, t: int):
            return gamma_builder_huber(y=y_t, k_value=k_value, tau_for_new=t)

    elif loss == "biweight":
        if k_value is None:
            raise ValueError("k_value is required for biweight loss.")

        def gamma_builder(y_t: float, t: int):
            return gamma_builder_biweight(y=y_t, k_value=k_value, tau_for_new=t)

    elif loss == "l2":

        def gamma_builder(y_t: float, t: int):
            return gamma_builder_l2(y=y_t, tau_for_new=t)

    else:

        def gamma_builder(y_t: float, t: int):
            return gamma_builder_l1(y=y_t, tau_for_new=t)

    cp_tau, qt_vals, _ = rfpop_algorithm1_main(y=y_list, gamma_builder=gamma_builder, beta=beta)
    return cp_tau, qt_vals
