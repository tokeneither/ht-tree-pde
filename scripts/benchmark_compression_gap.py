"""Reproduce the compression-resistant scalar/product separation.

Run from the repository root with ``PYTHONPATH=src python
scripts/benchmark_compression_gap.py``.  Results concern bound quality, not
solver runtime.  No full tensor is assembled for the large-size cases.
"""

import argparse
from itertools import combinations
import json
import os
from pathlib import Path
import platform

os.environ.setdefault("MPLCONFIGDIR", "/tmp/earlyht-compression-gap-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from earlyht.compression_gap import (
    compressed_scalar_bound,
    compression_gap_lower_bound,
    optimal_suffix_compression,
    optimized_scalar_dual,
    optimized_scalar_primal,
    polynomial_spectrum,
)


def enumerate_subset_check(size):
    d = polynomial_spectrum(size)
    all_subsets = [
        (compressed_scalar_bound(d, subset), list(subset))
        for count in range(size + 1)
        for subset in combinations(range(size), count)
    ]
    best, subset = min(all_subsets, key=lambda entry: entry[0])
    suffix = optimal_suffix_compression(d)
    return {"size": size, "subsets_evaluated": len(all_subsets),
            "all_subsets_minimum": best, "suffix_minimum": suffix.value,
            "difference": abs(best - suffix.value),
            "best_retained_indices_zero_based": subset}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--exhaustive-max", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/generated/gap"))
    parser.add_argument("--table", type=Path,
                        default=None, help="Optional LaTeX table destination")
    parser.add_argument("--figure", type=Path,
                        default=None, help="Defaults to OUTPUT/figure.png")
    args = parser.parse_args()
    if args.figure is None:
        args.figure = args.output / "figure.png"
    if args.exhaustive_max < 1 or args.exhaustive_max > 20:
        parser.error("--exhaustive-max must be between 1 and 20")
    rows = []
    suffix_curves = {}
    for size in args.sizes:
        d = polynomial_spectrum(size)
        full_primal = optimized_scalar_primal(d)
        full_dual = optimized_scalar_dual(d)
        compression = optimal_suffix_compression(d)
        retained = d[compression.retained_indices]
        reduced_dual = optimized_scalar_dual(retained) if retained.size else None
        scalar_dual_value = reduced_dual.value if reduced_dual is not None else 0.
        row = {
            "size": size,
            "product_bound_and_exact_gain": 1.,
            "uncompressed_scalar": full_primal.value,
            "uncompressed_dual": full_dual.value,
            "uncompressed_primal_dual_gap": abs(full_primal.value - full_dual.value),
            "best_compressed_scalar_plus_tail": compression.value,
            "retained_scalar": compression.scalar_value,
            "retained_scalar_dual": scalar_dual_value,
            "retained_primal_dual_gap": abs(compression.scalar_value - scalar_dual_value),
            "omitted_count": int(compression.tail_energy),
            "retained_count": int(retained.size),
            "theorem_lower_bound": compression_gap_lower_bound(size)
            if size >= 64 and size % 2 == 0 else None,
        }
        rows.append(row)
        suffix_curves[str(size)] = compression.suffix_totals.tolist()
        print(f"k={size:5d} scalar={full_primal.value:.8f} "
              f"compressed+tail={compression.value:.8f} "
              f"omitted={int(compression.tail_energy)}", flush=True)

    checks = [enumerate_subset_check(size)
              for size in range(1, args.exhaustive_max + 1)]
    largest_gap = max(max(row["uncompressed_primal_dual_gap"],
                          row["retained_primal_dual_gap"]) for row in rows)
    if largest_gap > 1e-8 or any(check["difference"] > 1e-10 for check in checks):
        raise RuntimeError("independent numerical validation failed")
    if any(row["theorem_lower_bound"] is not None and
           row["best_compressed_scalar_plus_tail"] < row["theorem_lower_bound"]
           for row in rows):
        raise RuntimeError("computed bound contradicts the stated theorem")

    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "model": {"operator_entry": "(i^2+j^2)/2", "forcing_diagonal": "i",
                  "indices_start_at": 1, "target_rank": 1,
                  "projection_class": "coordinate product spectral projections",
                  "scalar_metric": "best coercivity on the retained space",
                  "tail_energy": "exact: one per omitted diagonal forcing entry"},
        "arithmetic": "float64; numerical validation, not interval arithmetic",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "rows": rows, "all_subset_checks": checks,
        "maximum_reported_primal_dual_gap": largest_gap,
        "suffix_totals": suffix_curves,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    table = [r"% Generated by scripts/benchmark_compression_gap.py; do not edit.",
             r"\begin{tabular}{rrrrr}", r"\toprule",
             r"$k$ & Scalar & Best scalar + tail & Omitted & $k^{2/3}/4$ \\",
             r"\midrule"]
    for row in rows:
        theorem = (f"{row['theorem_lower_bound']:.2f}"
                   if row["theorem_lower_bound"] is not None else "--")
        table.append(f"{row['size']} & {row['uncompressed_scalar']:.2f} & "
                     f"{row['best_compressed_scalar_plus_tail']:.2f} & "
                     f"{row['omitted_count']} & {theorem} " + r"\\")
    table.extend([r"\bottomrule", r"\end{tabular}"])
    if args.table is not None:
        args.table.parent.mkdir(parents=True, exist_ok=True)
        args.table.write_text("\n".join(table) + "\n")

    args.figure.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42}):
        figure, ax = plt.subplots(figsize=(6.6, 4.1), constrained_layout=True)
        sizes = np.array([row["size"] for row in rows])
        ax.loglog(sizes, [row["uncompressed_scalar"] for row in rows],
                  "o-", label="Optimized scalar, full space", color="#777777")
        ax.loglog(sizes, [row["best_compressed_scalar_plus_tail"] for row in rows],
                  "s-", label="Best compressed scalar + exact tail", color="#155e9c")
        theorem_rows = [row for row in rows if row["theorem_lower_bound"] is not None]
        if theorem_rows:
            ax.loglog([row["size"] for row in theorem_rows],
                      [row["theorem_lower_bound"] for row in theorem_rows],
                      "--", label=r"Lower bound $k^{2/3}/4$", color="#ab572a")
        ax.axhline(1., color="#3b7955", linestyle=":", linewidth=2,
                   label="Product bound = exact rank-one gain")
        ax.set_xlabel(r"Number of diagonal forcing entries $k$")
        ax.set_ylabel("Upper bound / exact rank-one gain")
        ax.set_ylim(bottom=.8)
        ax.grid(True, which="major", alpha=.18)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
        figure.savefig(args.figure, dpi=180)
        plt.close(figure)
    print(f"Maximum primal/dual discrepancy: {largest_gap:.3g}")
    print(f"Wrote {args.output / 'results.json'} and {args.figure}")


if __name__ == "__main__":
    main()
