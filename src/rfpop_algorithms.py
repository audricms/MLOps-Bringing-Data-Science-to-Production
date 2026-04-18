import math
from typing import Callable, List

import numpy as np

from src.utils import QuadPiece
from src.variables import EPS_BOUNDARY, EPS_COEFF, EPS_CONST, EPS_ZERO


def min_over_theta(Qt_pieces: List[QuadPiece]) -> tuple[float, int]:
    """Find the minimum value of Q_t(theta) over a piecewise quadratic function.

    Each element of ``Qt_pieces`` is a QuadPiece describing an interval (a, b]
    and a quadratic A*theta^2 + B*theta + C defined on that interval. This
    function computes the minimal value of the piecewise function and returns
    the value together with the tau index associated with the minimizing
    piece.

    Parameters
    ----------
    Qt_pieces:
        List of QuadPiece tuples (a, b, A, B, C, tau).

    Returns
    -------
    (best_val, best_tau)
        best_val (float):
            Minimal value over all pieces.
        best_tau (int):
            Tau associated with the piece attaining the minimum.
    """
    best_val = float("inf")
    best_tau = 0
    for a, b, A, B, C, tau in Qt_pieces:
        if abs(A) < EPS_ZERO:
            left = a + EPS_BOUNDARY
            right = b
            vleft = A * left * left + B * left + C
            vright = A * right * right + B * right + C
            theta_star = left if vleft <= vright else right
        else:
            theta_star = -B / (2.0 * A)
            if theta_star <= a:
                theta_star = a + EPS_BOUNDARY
            elif theta_star > b:
                theta_star = b
        val = A * theta_star * theta_star + B * theta_star + C
        if val < best_val:
            best_val = val
            best_tau = tau
    return float(best_val), int(best_tau)


def add_qstar_and_gamma(
    Qstar_pieces: List[QuadPiece],
    gamma_pieces: List[QuadPiece],
) -> List[QuadPiece]:
    """Add (pointwise) a Q* representation and a gamma representation.

    Both inputs represent piecewise quadratic functions as ordered lists of
    QuadPiece tuples. The function computes the piecewise sum Q*_t + gamma and
    returns a merged list of QuadPiece with adjacent identical quadratics
    coalesced.

    Parameters
    ----------
    Qstar_pieces, gamma_pieces:
        Lists of QuadPiece tuples (a, b, A, B, C, tau). Intervals are assumed
        to be ordered and cover the real line (or the domain of interest).

    Returns
    -------
    List[QuadPiece]
        The merged piecewise quadratic representation of the sum.
    """
    out: List[QuadPiece] = []
    i = 0
    j = 0
    while i < len(Qstar_pieces) and j < len(gamma_pieces):
        pa, pb, pA, pB, pC, p_tau = Qstar_pieces[i]
        ga, gb, gA, gB, gC, _ = gamma_pieces[j]

        a = max(pa, ga)
        b = min(pb, gb)

        newA = pA + gA
        newB = pB + gB
        newC = pC + gC
        out.append((a, b, newA, newB, newC, p_tau))
        if abs(b - pb) < EPS_BOUNDARY:
            i += 1
        if abs(b - gb) < EPS_BOUNDARY:
            j += 1
        if (
            (i < len(Qstar_pieces) and j < len(gamma_pieces))
            and a >= b - EPS_COEFF
            and abs(b - Qstar_pieces[i][1]) > EPS_BOUNDARY
            and abs(b - gamma_pieces[j][1]) > EPS_BOUNDARY
        ):
            break
    if not out:
        return []
    merged: List[QuadPiece] = [out[0]]
    for pc in out[1:]:
        a, b, A, B, C, tau = pc
        ma, mb, mA, mB, mC, mtau = merged[-1]
        if (
            abs(A - mA) < EPS_COEFF
            and abs(B - mB) < EPS_COEFF
            and abs(C - mC) < EPS_CONST
            and tau == mtau
            and abs(a - mb) < EPS_CONST
        ):
            merged[-1] = (ma, b, mA, mB, mC, mtau)
        else:
            merged.append(pc)
    return merged


def prune_compare_to_constant(
    Qt_pieces: List[QuadPiece],
    Qt_val: float,
    beta: float,
    t_index_for_new: int,
) -> List[QuadPiece]:
    """Prune Qt by comparing to a constant threshold and create Q*_{t+1}.

    For each interval of the piecewise quadratic Qt, compute where Qt(theta)
    <= threshold (thr = Qt_val + beta). Subsegments where Qt > thr are
    replaced by the constant thr and assigned a changepoint index
    ``t_index_for_new``. Adjacent equal pieces are merged before returning.

    Parameters
    ----------
    Qt_pieces : List[QuadPiece]
        Piecewise representation of Q_t.
    Qt_val : float
        Minimum value of Q_t (used to build threshold).
    beta : float
        Penalty parameter; threshold is Qt_val + beta.
    t_index_for_new : int
        Index to set as tau for segments replaced by the constant thr.

    Returns
    -------
    List[QuadPiece]
        The pruned/merged piecewise representation (Q*_{t+1}).
    """
    thr = Qt_val + beta
    out: List[QuadPiece] = []
    for a, b, A, B, C, tau in Qt_pieces:
        roots: List[float] = []
        if abs(A) < EPS_ZERO:
            if abs(B) > EPS_ZERO:
                x = -(C - thr) / B
                if x + EPS_BOUNDARY >= a and x - EPS_BOUNDARY <= b:
                    x_clamped = max(min(x, b), a)
                    roots.append(x_clamped)
        else:
            D = B * B - 4.0 * A * (C - thr)
            if D >= -EPS_COEFF:
                D = max(D, 0.0)
                sqrtD = math.sqrt(D)
                x1 = (-B - sqrtD) / (2.0 * A)
                x2 = (-B + sqrtD) / (2.0 * A)
                for x in (x1, x2):
                    if x + EPS_BOUNDARY >= a and x - EPS_BOUNDARY <= b:
                        x_clamped = max(min(x, b), a)
                        if not any(abs(x_clamped - r) < EPS_CONST for r in roots):
                            roots.append(x_clamped)
        roots.sort()
        breaks = [a] + roots + [b]
        for k in range(len(breaks) - 1):
            lo = breaks[k]
            hi = breaks[k + 1]
            mid = (lo + hi) / 2.0
            val_mid = A * mid * mid + B * mid + C - thr
            if val_mid <= EPS_BOUNDARY:
                out.append((lo, hi, A, B, C, tau))
            else:
                out.append((lo, hi, 0.0, 0.0, thr, t_index_for_new))
    if not out:
        return []
    merged: List[QuadPiece] = [out[0]]
    for pc in out[1:]:
        a, b, A, B, C, tau = pc
        ma, mb, mA, mB, mC, mtau = merged[-1]
        if (
            abs(A - mA) < EPS_COEFF
            and abs(B - mB) < EPS_COEFF
            and abs(C - mC) < EPS_CONST
            and tau == mtau
            and abs(a - mb) < EPS_CONST
        ):
            merged[-1] = (ma, b, mA, mB, mC, mtau)
        else:
            merged.append(pc)
    return merged


def rfpop_algorithm(
    y: List[float],
    gamma_builder: Callable[[float, int], List[QuadPiece]],
    beta: float,
) -> tuple[list[int], list[float], List[QuadPiece]]:
    """Run the RFPOP dynamic program on a sequence y with given loss builder.

    This function implements the top-level RFPOP loop. At each time t it:
      - builds gamma(y_t) via gamma_builder (returns a list of QuadPiece),
      - computes Qt = Q*_t + gamma(y_t),
      - finds the minimum of Qt over theta,
      - prunes Qt against the constant Qt_val + beta to obtain Q*_{t+1}.

    Parameters
    ----------
    y : List[float]
        Observations (time series) to process.
    gamma_builder : Callable
        Function taking (y_t: float, t: int) and returning List[QuadPiece]
        representing gamma(y_t, theta).
    beta : float
        Penalty parameter used when pruning to construct Q*_{t+1}.

    Returns
    -------
    (cp_tau, Qt_vals, Qstar)
        cp_tau : list[int]
            Backpointer array of changepoint indices per t
        Qt_vals : list[float]
            Minimal Qt value at each t
        Qstar : List[QuadPiece]
        final piecewise representation Q*_{n}
            Final piecewise representation Q*_{n}
    """
    y_arr = np.asarray(y, dtype=float)
    n = len(y_arr)
    lo = float(np.min(y_arr)) - 1.0
    hi = float(np.max(y_arr)) + 1.0
    Qstar = [(lo, hi, 0.0, 0.0, 0.0, 0)]
    cp_tau = [0] * n
    Qt_vals = [0.0] * n
    for t_idx in range(n):
        gamma_pcs = gamma_builder(y_t=float(y_arr[t_idx]), t=t_idx)
        Qt_pcs = add_qstar_and_gamma(Qstar_pieces=Qstar, gamma_pieces=gamma_pcs)
        Qt_val, tau_t = min_over_theta(Qt_pieces=Qt_pcs)
        cp_tau[t_idx] = tau_t
        Qt_vals[t_idx] = Qt_val
        Qstar = prune_compare_to_constant(
            Qt_pieces=Qt_pcs, Qt_val=Qt_val, beta=beta, t_index_for_new=t_idx
        )
    return cp_tau, Qt_vals, Qstar
