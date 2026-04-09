# pylint: disable=non-keyword-args

from __future__ import annotations

import math
from typing import Callable, List, Tuple

import numpy as np

QuadPiece = Tuple[float, float, float, float, float, int]
INF = 1e18


def rfpop_algorithm3_min_over_theta(qt_pieces: List[QuadPiece]) -> Tuple[float, int]:
    best_val = float("inf")
    best_tau = 0
    for a, b, a_coef, b_coef, c_coef, tau in qt_pieces:
        if abs(a_coef) < 1e-16:
            left = a + 1e-12
            right = b
            v_left = a_coef * left * left + b_coef * left + c_coef
            v_right = a_coef * right * right + b_coef * right + c_coef
            theta_star = left if v_left <= v_right else right
        else:
            theta_star = -b_coef / (2.0 * a_coef)
            if theta_star <= a:
                theta_star = a + 1e-12
            elif theta_star > b:
                theta_star = b

        val = a_coef * theta_star * theta_star + b_coef * theta_star + c_coef
        if val < best_val:
            best_val = val
            best_tau = tau

    return float(best_val), int(best_tau)


def rfpop_algorithm2_add_qstar_and_gamma(
    qstar_pieces: List[QuadPiece],
    gamma_pieces: List[QuadPiece],
) -> List[QuadPiece]:
    out: List[QuadPiece] = []
    i = 0
    j = 0

    while i < len(qstar_pieces) and j < len(gamma_pieces):
        pa, pb, p_a, p_b, p_c, p_tau = qstar_pieces[i]
        ga, gb, g_a, g_b, g_c, _ = gamma_pieces[j]

        a = max(pa, ga)
        b = min(pb, gb)

        out.append((a, b, p_a + g_a, p_b + g_b, p_c + g_c, p_tau))

        if abs(b - pb) < 1e-12:
            i += 1
        if abs(b - gb) < 1e-12:
            j += 1

        if (
            (i < len(qstar_pieces) and j < len(gamma_pieces))
            and a >= b - 1e-14
            and abs(b - qstar_pieces[i][1]) > 1e-12
            and abs(b - gamma_pieces[j][1]) > 1e-12
        ):
            break

    return _merge_neighboring_pieces(out)


def rfpop_algorithm4_prune_compare_to_constant(
    qt_pieces: List[QuadPiece],
    qt_val: float,
    beta: float,
    t_index_for_new: int,
) -> List[QuadPiece]:
    thr = qt_val + beta
    out: List[QuadPiece] = []

    for a, b, a_coef, b_coef, c_coef, tau in qt_pieces:
        roots: List[float] = []

        if abs(a_coef) < 1e-16:
            if abs(b_coef) > 1e-16:
                x = -(c_coef - thr) / b_coef
                if x + 1e-12 >= a and x - 1e-12 <= b:
                    roots.append(max(min(x, b), a))
        else:
            disc = b_coef * b_coef - 4.0 * a_coef * (c_coef - thr)
            if disc >= -1e-14:
                disc = max(disc, 0.0)
                sqrt_disc = math.sqrt(disc)
                x1 = (-b_coef - sqrt_disc) / (2.0 * a_coef)
                x2 = (-b_coef + sqrt_disc) / (2.0 * a_coef)
                for x in (x1, x2):
                    if x + 1e-12 >= a and x - 1e-12 <= b:
                        x_clamped = max(min(x, b), a)
                        if not any(abs(x_clamped - r) < 1e-9 for r in roots):
                            roots.append(x_clamped)

        roots.sort()
        breaks = [a] + roots + [b]

        for k in range(len(breaks) - 1):
            lo = breaks[k]
            hi = breaks[k + 1]
            mid = (lo + hi) / 2.0
            val_mid = a_coef * mid * mid + b_coef * mid + c_coef - thr

            if val_mid <= 1e-12:
                out.append((lo, hi, a_coef, b_coef, c_coef, tau))
            else:
                out.append((lo, hi, 0.0, 0.0, thr, t_index_for_new))

    return _merge_neighboring_pieces(out)


def rfpop_algorithm1_main(
    y: List[float],
    gamma_builder: Callable[[float, int], List[QuadPiece]],
    beta: float,
) -> Tuple[List[int], List[float], List[QuadPiece]]:
    y_arr = np.asarray(y, dtype=float)
    n = len(y_arr)

    lo = float(np.min(y_arr)) - 1.0
    hi = float(np.max(y_arr)) + 1.0

    qstar = [(lo, hi, 0.0, 0.0, 0.0, 0)]
    cp_tau = [0] * n
    qt_vals = [0.0] * n

    for t_idx in range(n):
        gamma_pcs = gamma_builder(y_t=float(y_arr[t_idx]), t=t_idx)
        qt_pcs = rfpop_algorithm2_add_qstar_and_gamma(qstar_pieces=qstar, gamma_pieces=gamma_pcs)

        qt_val, tau_t = rfpop_algorithm3_min_over_theta(qt_pcs)
        cp_tau[t_idx] = tau_t
        qt_vals[t_idx] = qt_val

        qstar = rfpop_algorithm4_prune_compare_to_constant(
            qt_pieces=qt_pcs, qt_val=qt_val, beta=beta, t_index_for_new=t_idx
        )

    return cp_tau, qt_vals, qstar


def gamma_builder_l2(y: float, tau_for_new: int) -> List[QuadPiece]:
    return [(-INF, INF, 1.0, -2.0 * y, y * y, tau_for_new)]


def gamma_builder_biweight(y: float, k_value: float, tau_for_new: int) -> List[QuadPiece]:
    const_c = float(k_value * k_value)
    return [
        (-INF, y - k_value, 0.0, 0.0, const_c, tau_for_new),
        (y - k_value, y + k_value, 1.0, -2.0 * y, y * y, tau_for_new),
        (y + k_value, INF, 0.0, 0.0, const_c, tau_for_new),
    ]


def gamma_builder_huber(y: float, k_value: float, tau_for_new: int) -> List[QuadPiece]:
    return [
        (
            -INF,
            y - k_value,
            0.0,
            -2.0 * k_value,
            2.0 * k_value * y - k_value * k_value,
            tau_for_new,
        ),
        (y - k_value, y + k_value, 1.0, -2.0 * y, y * y, tau_for_new),
        (y + k_value, INF, 0.0, 2.0 * k_value, -2.0 * k_value * y - k_value * k_value, tau_for_new),
    ]


def gamma_builder_l1(y: float, tau_for_new: int) -> List[QuadPiece]:
    return [
        (-INF, y, 0.0, -1.0, float(y), tau_for_new),
        (y, INF, 0.0, 1.0, float(-y), tau_for_new),
    ]


def extract_changepoints_backtrack(cp_tau: List[int]) -> List[int]:
    n = len(cp_tau)
    changepoints: List[int] = []
    t = n - 1

    while t > 0:
        tau = cp_tau[t]
        if tau > 0:
            changepoints.append(tau)
        t = tau

    changepoints.reverse()
    return changepoints


def get_segments_from_cp_tau(cp_tau: List[int], y: np.ndarray) -> List[Tuple[int, int, float]]:
    n = len(cp_tau)
    segments: List[Tuple[int, int, float]] = []
    t = n - 1

    while t > 0:
        t_prev = int(cp_tau[t])
        seg_mean = float(np.mean(y[t_prev : t + 1]))
        segments.append((t_prev, t, seg_mean))
        t = t_prev

    segments.reverse()
    return segments


def _merge_neighboring_pieces(pieces: List[QuadPiece]) -> List[QuadPiece]:
    if not pieces:
        return []

    merged: List[QuadPiece] = [pieces[0]]
    for piece in pieces[1:]:
        a, b, a_coef, b_coef, c_coef, tau = piece
        ma, mb, m_a, m_b, m_c, m_tau = merged[-1]

        if (
            abs(a_coef - m_a) < 1e-14
            and abs(b_coef - m_b) < 1e-14
            and abs(c_coef - m_c) < 1e-9
            and tau == m_tau
            and abs(a - mb) < 1e-9
        ):
            merged[-1] = (ma, b, m_a, m_b, m_c, m_tau)
        else:
            merged.append(piece)

    return merged
