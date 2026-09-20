"""Gain intervals translated into an excess squared energy-error guarantee.

The exact-arithmetic bound is max(upper_bounds) - feasible_gain.  The extra
floating-point guard below is a decision safeguard, not a validated bound on
all accumulated roundoff errors.
"""
from __future__ import annotations

import math


def excess_energy_certificate(feasible_gain, upper_bounds, *, tolerance=None,
                              relative_guard=1e-9):
    """Report regret of the ACTUAL candidate whose gain was supplied.

    All candidate-family upper bounds must refer to the same full-space
    problem. In particular, add any spectral-tail allowance before calling.
    ``tolerance`` is an absolute excess *squared* energy error, not a relative
    error against the unknown optimal residual. ``None`` disables acceptance.
    Positive infinity is allowed for an as-yet-unbounded family.
    """
    gain = float(feasible_gain)
    upper = tuple(float(value) for value in upper_bounds)
    if not math.isfinite(gain) or gain < 0:
        raise ValueError("feasible_gain must be finite and nonnegative")
    if not upper or any(math.isnan(value) or value < 0 for value in upper):
        raise ValueError("nonnegative, nonempty family upper bounds are required")
    if not math.isfinite(relative_guard) or relative_guard < 0:
        raise ValueError("relative_guard must be finite and nonnegative")
    if tolerance is not None and (not math.isfinite(tolerance) or tolerance < 0):
        raise ValueError("tolerance must be finite and nonnegative")
    largest = max(upper)
    scale = max(1e-12, gain, *(value for value in upper if math.isfinite(value)))
    guard = float(relative_guard * scale)
    gap = largest - gain
    if gap < -guard:
        raise ArithmeticError("the candidate gain exceeds all family upper bounds")
    guarded_gap = max(0.0, gap) + guard
    return {
        "candidate_gain": gain,
        "maximum_family_upper": largest,
        "gain_gap": float(gap),
        "excess_energy_bound": float(guarded_gap),
        "excess_energy_tolerance": tolerance,
        "epsilon_certified": tolerance is not None and guarded_gap <= tolerance,
        "excess_energy_guard": guard,
    }
