#!/usr/bin/env python3
"""Select a rank-one cut family, then reuse its certificate for a nearby load.

Run after ``python -m pip install -e .``. No full PDE solution is computed.
The three families constrain one balanced matrix cut; leaf ranks are free.
"""
from __future__ import annotations

import numpy as np

from earlyht.direct_full import select_direct_full
from earlyht.parametric import NAMES, correlated_load_problem, fit_selected
from earlyht.reuse import (
    CertificateLibrary,
    feasible_direction_correlations,
    parameter_gram,
)
from earlyht.spectral_baselines import SeparableSpectralInverse


def show_candidate(problem, candidate, upper):
    """Use the actual tensor's gain, never a transported family lower bound."""
    gain = float(2 * np.sum(problem.rhs * candidate) - problem.energy(candidate))
    excess = max(0.0, max(upper) - gain)
    print(f"  Actual candidate gain: {gain:.6e}")
    print(f"  Excess squared energy error <= {excess:.6e}")


def main():
    # Analytic source shapes are prescribed directly; no target solution is used.
    problem, basis, _ = correlated_load_problem(n=6, seed=41)
    theta = np.array([1.0, 0.025, 0.035])
    problem.rhs[:] = np.tensordot(theta, basis, axes=1)
    fresh = select_direct_full(
        problem, rank=1, sweeps=(0, 2, 6), metric_policy="compact",
        dual_refinements=12, return_approximation=True,
    )
    lower = [fresh["bounds"][name]["lower"] for name in NAMES]
    upper = [fresh["bounds"][name]["upper"] for name in NAMES]
    candidate = fresh["approximation"]
    winner = NAMES.index(fresh["selected"])

    print("Four-dimensional reaction-diffusion PDE; grid 6 x 6 x 6 x 6")
    print("Three rank-one balanced-cut families; unrestricted leaf ranks")
    print("\nFresh selection: family gain intervals [lower, upper]")
    for name, lo, hi in zip(NAMES, lower, upper):
        print(f"  {name}: [{lo:.6e}, {hi:.6e}]")
    print(f"  Candidate family: {NAMES[winner]}")
    print(f"  Unique best family certified: {fresh['certified']}")
    show_candidate(problem, candidate, upper)

    # B = sum_i A_i + min(V) I <= A. Applying B^{-1} needs only
    # separable spectral transforms, not a solve with the complete PDE operator.
    inverse_minorant = SeparableSpectralInverse(problem, float(problem.potential.min()))
    gram = parameter_gram(basis, apply_inverse_minorant=inverse_minorant.apply)
    library = CertificateLibrary(gram, len(NAMES), homogeneous_scaling=True)
    direction = feasible_direction_correlations(
        basis, candidate, apply_operator=problem.apply,
    )
    library.add_reference(
        theta, lower, upper, winner=winner, direction_correlations=direction,
    )
    nearby = 1.15 * np.array([1.0, 0.028, 0.029])
    reused = library.query(nearby)
    print("\nNearby source: query the stored certificate")
    print(f"  Unique best family certified: {reused['certified']}")
    if reused["certified"]:
        index = reused["selected_index"]
        print(f"  Certified family: {NAMES[index]}")
        # Reusing a family certificate does not produce the new approximation.
        problem.rhs[:] = np.tensordot(nearby, basis, axes=1)
        initial = candidate if index == winner else None
        fitted, _ = fit_selected(problem, index, initial, rank=1, sweeps=6)
        show_candidate(problem, fitted, reused["upper"])
    else:
        print("  Abstained: compute a fresh certificate before claiming a winner.")

    print("\nExcess compares the returned tensor with the best tensor across")
    print("these three rank-constrained families; it is not total PDE error.")
    print("Bounds use floating-point arithmetic, not interval verification.")


if __name__ == "__main__":
    main()
