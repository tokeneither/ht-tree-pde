"""Scalar/product separation that survives coordinate spectral compression.

The model has ``A(X)[i,j] = (d[i] + d[j]) * X[i,j] / 2`` and
``F = diag(sqrt(d))`` with positive ``d``.  The exact rank-one gain and the
balanced product-metric bound are both one.  Removing a diagonal forcing
entry incurs exactly one unit of the commuting-projection tail allowance.

The scalar-center optimum is evaluated by a one-dimensional convex primal
problem and, independently, its simplex dual.  These routines use ordinary
floating point; agreement of their values is a numerical check, not an
interval-arithmetic proof.  Mathematical proofs belong in the manuscript.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class ScalarPrimal:
    value: float
    coercivity: float
    residual_threshold: float
    center_diagonal: np.ndarray


@dataclass(frozen=True)
class ScalarDual:
    value: float
    multiplier: float
    weights: np.ndarray


@dataclass(frozen=True)
class CompressionOptimum:
    value: float
    scalar_value: float
    tail_energy: float
    retained_indices: np.ndarray
    # Entry j corresponds to retaining the sorted suffix j,...,k-1;
    # the last entry is the empty projection, whose tail is k.
    suffix_totals: np.ndarray


def _positive_spectrum(diagonal):
    values = np.asarray(diagonal, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("diagonal must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("diagonal entries must be finite and positive")
    return values


def optimized_scalar_primal(diagonal):
    """Optimize over all scalar centers for the diagonal-forcing model.

    With m=min(d), t=z/sqrt(m), and q=d/m, the exact reduced problem is
    ``min_{t >= 1} t**2 + sum((sqrt(q)-t)_+**2/(q-1))``.  Terms with q=1
    impose t>=1 and have no quadratic center cost.  Diagonal sign averaging
    justifies restricting an unrestricted matrix center to its diagonal.
    """
    d = _positive_spectrum(diagonal)
    m = float(np.min(d))
    q = d / m
    nonminimal = d > m
    roots = np.sqrt(q[nonminimal])
    denominators = q[nonminimal] - 1.0

    def half_derivative(threshold):
        return float(threshold - np.sum(
            np.maximum(roots - threshold, 0.0) / denominators))

    if not roots.size or half_derivative(1.0) >= 0.0:
        threshold = 1.0
    else:
        threshold = float(brentq(half_derivative, 1.0, float(roots.max()),
                                 xtol=1e-13, rtol=1e-14))
    positive_residual = np.maximum(roots - threshold, 0.0)
    value = float(threshold ** 2
                  + np.sum(positive_residual ** 2 / denominators))
    center = np.zeros_like(d)
    center[nonminimal] = positive_residual / (denominators * np.sqrt(m))
    return ScalarPrimal(value, m, threshold * np.sqrt(m), center)


def optimized_scalar_dual(diagonal):
    """Independently solve the concave simplex allocation dual.

    Its objective is ``sum(q*p/(1+(q-1)*p))``, with p>=0 and sum(p)=1.
    The KKT multiplier is at least one because min(q)=1.  At multiplier
    one, unused probability mass goes to a minimal spectral entry.
    """
    d = _positive_spectrum(diagonal)
    q = d / float(np.min(d))
    nonminimal = q > 1.0
    q_high = q[nonminimal]

    def allocation(multiplier):
        return np.maximum(np.sqrt(q_high / multiplier) - 1.0, 0.0) / (
            q_high - 1.0)

    p = np.zeros_like(d)
    at_one = allocation(1.0)
    if float(at_one.sum()) <= 1.0:
        multiplier = 1.0
        p[nonminimal] = at_one
        p[int(np.argmin(d))] = max(0.0, 1.0 - float(at_one.sum()))
    else:
        multiplier = float(brentq(
            lambda value: float(allocation(value).sum()) - 1.0,
            1.0, float(q_high.max()), xtol=1e-13, rtol=1e-14))
        p[nonminimal] = allocation(multiplier)
    # Remove the final floating-point simplex residual.  This changes the
    # KKT point by rounding only and keeps the reported dual point feasible.
    p /= float(p.sum())
    value = float(np.sum(q * p / (1.0 + (q - 1.0) * p)))
    return ScalarDual(value, multiplier, p)


def compressed_scalar_bound(diagonal, retained_indices):
    """Scalar optimum plus exact omitted energy for an arbitrary subset."""
    d = _positive_spectrum(diagonal)
    raw = np.asarray(retained_indices)
    if raw.ndim != 1 or (raw.size and raw.dtype.kind not in "iu"):
        raise ValueError("retained_indices must be a one-dimensional integer array")
    retained = raw.astype(int)
    if (np.any(retained < 0) or np.any(retained >= len(d))
            or np.unique(retained).size != retained.size):
        raise ValueError("retained_indices must be distinct valid indices")
    scalar = optimized_scalar_primal(d[retained]).value if retained.size else 0.0
    return float(scalar + len(d) - retained.size)


def optimal_suffix_compression(diagonal):
    """Evaluate all suffixes, including the empty spectral projection.

    For F_ii=sqrt(d_i), including an additional retained diagonal entry
    increases a fixed-coercivity scalar optimum by at most one, while
    reducing the tail by one.  Thus a best coordinate projection can be
    chosen as a suffix of sorted d.  The benchmark separately checks this
    reduction by enumerating every subset for small dimensions.
    """
    d = _positive_spectrum(diagonal)
    order = np.argsort(d, kind="stable")
    sorted_d = d[order]
    totals = np.empty(len(d) + 1)
    scalar_values = np.empty(len(d) + 1)
    for removed in range(len(d)):
        scalar_values[removed] = optimized_scalar_primal(sorted_d[removed:]).value
        totals[removed] = removed + scalar_values[removed]
    totals[-1] = float(len(d))
    scalar_values[-1] = 0.0
    removed = int(np.argmin(totals))
    return CompressionOptimum(float(totals[removed]),
                              float(scalar_values[removed]), float(removed),
                              order[removed:], totals)


def polynomial_spectrum(size):
    """The explicit d_i=i^2 example, indexed from one."""
    if isinstance(size, bool) or int(size) != size or size < 1:
        raise ValueError("size must be a positive integer")
    return np.arange(1.0, int(size) + 1.0) ** 2


def compression_gap_lower_bound(size):
    """The manuscript lower bound k^(2/3)/4, for even k>=64."""
    polynomial_spectrum(size)
    if size < 64 or size % 2:
        raise ValueError("the stated theorem requires even size >= 64")
    return float(size ** (2.0 / 3.0) / 4.0)
