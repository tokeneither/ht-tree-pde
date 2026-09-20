#!/usr/bin/env python3
"""Check the proposed scalar-plus-tail gap; this is not a runtime benchmark.

The derivation justifying arbitrary centers and arbitrary retained spectral
sets is in paper/draft.pdf. This script evaluates
the resulting exact one-dimensional convex problems and checks all subsets
for small dimensions. No existing manuscript results are overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


def retained_scalar_bound(energies, kept):
    """Optimal centered rank-one scalar bound for diagonal forcing sqrt(d)."""
    indices = np.asarray(kept, dtype=int)
    if not len(indices):
        return 0.0, 0.0
    values = energies[indices]
    m = float(np.min(values))
    higher = values[values > m]
    forcing = np.sqrt(higher)
    lo, hi = np.sqrt(m), np.sqrt(float(np.max(values)))

    def derivative(t):
        return t / m - np.sum(np.maximum(forcing - t, 0) / (higher - m))

    threshold = lo if derivative(lo) >= 0 else brentq(derivative, lo, hi, xtol=1e-11)
    bound = threshold**2 / m + np.sum(np.maximum(forcing - threshold, 0)**2 / (higher - m))
    return float(bound), float(threshold)


def best_suffix(energies):
    k = len(energies)
    choices = []
    for first in range(k):
        bound, threshold = retained_scalar_bound(energies, np.arange(first, k))
        choices.append((first + bound, first, bound, threshold))
    total, first, bound, threshold = min(choices)
    assert total <= k + 1e-10
    return {
        "k": k,
        "first_retained_one_based": first + 1,
        "exact_tail_energy": first,
        "optimized_retained_scalar_bound": bound,
        "optimized_scalar_plus_tail": total,
        "center_residual_threshold": threshold,
        "exact_product_bound_and_rank_one_gain": 1.0,
        "normalized_by_k_power_two_thirds": total / k**(2/3),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=root / 'results/generated/gap/dirichlet.json')
    args = parser.parse_args()
    sizes = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
    polynomial = [best_suffix(np.arange(1, k+1, dtype=float)**2) for k in sizes]
    dirichlet = []
    for k in sizes:
        i = np.arange(1, k+1, dtype=float)
        eigenvalues = 4 * (k+1)**2 * np.sin(np.pi * i / (2 * (k+1)))**2
        dirichlet.append(best_suffix(2 * eigenvalues))

    subset_checks = []
    for k in range(2, 11):
        energies = np.arange(1, k+1, dtype=float)**2
        all_sets_best = float(k)
        for bitmask in range(1, 2**k):
            kept = [i for i in range(k) if bitmask & (1 << i)]
            scalar, _ = retained_scalar_bound(energies, kept)
            all_sets_best = min(all_sets_best, k-len(kept)+scalar)
        suffix_best = best_suffix(energies)["optimized_scalar_plus_tail"]
        discrepancy = abs(all_sets_best-suffix_best)
        assert discrepancy < 1e-9
        subset_checks.append({"k": k, "subsets_including_empty": 2**k,
                              "max_optimum_discrepancy": discrepancy})

    expected = {32: 13.12, 128: 36.64, 256: 60.38}
    for result in polynomial:
        if result["k"] in expected:
            assert abs(result["optimized_scalar_plus_tail"] - expected[result["k"]]) < 0.005
    report = {
        "description": "Audit of one optimized spectral projection + centered scalar bound + exact tail energy",
        "scope": "Finite diagonal examples; no performance or literature-priority claim",
        "polynomial_energy_i_squared": polynomial,
        "two_dimensional_dirichlet_energy_two_lambda_i": dirichlet,
        "exhaustive_subset_checks": subset_checks,
        "polynomial_asymptotic_constant_proved_in_note": 3 / 2**(2/3),
        "status": "passed",
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    for item in polynomial:
        print(f"k={item['k']:4d}  scalar+tail={item['optimized_scalar_plus_tail']:.8f}  "
              f"first={item['first_retained_one_based']:3d}  product=1")
    print(f"All-subset checks passed for k=2..10; saved {output}")


if __name__ == "__main__":
    main()
