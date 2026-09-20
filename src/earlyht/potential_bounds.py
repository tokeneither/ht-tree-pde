"""Energy-error lower bounds retaining additive lower approximations of V.

The pointwise potential is represented in the same physical grid coordinates
as the target cut matrix. No separability of the full potential is assumed.
These are ordinary floating-point evaluations of exact-arithmetic bounds,
not interval-arithmetic certificates.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from scipy.linalg import cholesky

from .trees import CutProblem, rank_lower_bound


def additive_minorants(
    potential: np.ndarray,
) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
    """Yield four nonnegative additive minorants p[:,None]+q[None,:] <= V.

    Rows and columns refer to the physical-grid coordinates of one cut.
    The average candidate is feasible by convexity of the inequalities.
    Every output array is newly allocated; the input is never changed.
    """
    if not np.isrealobj(potential):
        raise ValueError("potential must be real")
    potential = np.asarray(potential, dtype=float)
    if potential.ndim != 2 or min(potential.shape) == 0:
        raise ValueError("potential must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(potential)) or np.any(potential < 0):
        raise ValueError("potential must be finite and nonnegative")
    m, n = potential.shape
    minimum = float(np.min(potential))
    yield np.full(m, minimum / 2), np.full(n, minimum / 2), "constant"

    row_first_p = np.min(potential, axis=1)
    row_first_q = np.min(potential - row_first_p[:, None], axis=0)
    yield row_first_p.copy(), row_first_q.copy(), "row_first"

    column_first_q = np.min(potential, axis=0)
    column_first_p = np.min(potential - column_first_q[None, :], axis=1)
    yield column_first_p.copy(), column_first_q.copy(), "column_first"

    yield (
        0.5 * row_first_p + 0.5 * column_first_p,
        0.5 * row_first_q + 0.5 * column_first_q,
        "balanced_average",
    )


def minorant_lower_bound(problem: CutProblem, rank: int) -> float:
    """Strengthen the base weighted-tail bound using nonnegative parts of V.

    For each minorant p_i+q_j <= V_ij, evaluate the standard separate-tail
    bound with left+diag(p), right+diag(q). The maximum with the base bound
    is still valid and cannot be weaker. No nonlinear fit is needed.
    """
    # This call also applies the existing public rank-validation convention.
    best = rank_lower_bound(problem, rank)
    if rank >= min(problem.matrix.shape) or not np.any(problem.potential):
        return best
    if not np.any(problem.matrix):
        return best
    rank = int(rank)
    for p, q, _ in additive_minorants(problem.potential):
        if not np.any(p) and not np.any(q):
            continue
        # Upper Cholesky factors satisfy R.T @ R = A. Right multiplication
        # must therefore use R.T, matching the energy identity exactly.
        left_root = cholesky(
            problem.left + np.diag(p), lower=False, check_finite=False
        )
        right_root = cholesky(
            problem.right + np.diag(q), lower=False, check_finite=False
        )
        left_singular = np.linalg.svd(left_root @ problem.matrix, compute_uv=False)
        right_singular = np.linalg.svd(
            problem.matrix @ right_root.T, compute_uv=False
        )
        squared = (
            np.sum(left_singular[rank:] ** 2)
            + np.sum(right_singular[rank:] ** 2)
        )
        best = max(best, float(np.sqrt(squared)))
    return best
