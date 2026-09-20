"""Genuine all-edge HT tree selection: fixed, reproducible dense experiments.

The certified method sees only (f, diagonal operator factors). The reference
baseline alone forms u=A^{-1}f. Timings include the complete decision procedure
after input construction. This is a small dense verification benchmark, not a
claim of a scalable HT implementation or a comparison with every HT algorithm.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy

from earlyht.ht_energy import fit_ht_energy
from earlyht.ht_primal import balanced_tree, ht_round, tree_label, edge_cuts, unfold, leaves
from earlyht.ht_search import all_balanced_trees, prescribed_ht_forcing, search_ht


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "generated" / "ht"
FIGURES = OUTPUT
RANK = 2
DIMENSION = 8
GROUPS = ("small", "band", "band_fresh", "control")


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def diagonal_sum(eigenvalues):
    shape = tuple(len(values) for values in eigenvalues)
    result = np.zeros(shape)
    for axis, values in enumerate(eigenvalues):
        broadcast_shape = [1] * len(shape)
        broadcast_shape[axis] = len(values)
        result += np.asarray(values).reshape(broadcast_shape)
    return result


def dirichlet_spectrum(n):
    return 4 * (n + 1) ** 2 * np.sin(np.pi * np.arange(1, n + 1) / (2 * (n + 1))) ** 2


def canonical(tree):
    if isinstance(tree, int):
        return tree
    from earlyht.ht_primal import leaves
    children = [canonical(child) for child in tree]
    return tuple(sorted(children, key=lambda child: tuple(sorted(leaves(child)))))


def build_case(group, n=None, seed=None):
    if group == "small":
        permutation = tuple(int(i) for i in np.random.default_rng(17000 + seed).permutation(DIMENSION))
        source_tree = balanced_tree(permutation)
        forcing = prescribed_ht_forcing((n,) * DIMENSION, source_tree, rank=RANK, seed=seed)
        eigenvalues = tuple(dirichlet_spectrum(n) for _ in range(DIMENSION))
        metadata = dict(id=f"n{n}_seed{seed}", group=group, n=n, seed=seed,
                        description="Random nested HT source, independently scrambled tree; full Dirichlet spectra",
                        axis_permutation=permutation, source_tree=tree_label(canonical(source_tree)))
    elif group in ("band", "band_fresh"):
        permutation = (tuple(range(DIMENSION)) if seed in (101, 103, 107) else
                       tuple(int(i) for i in np.random.default_rng(17000 + seed).permutation(DIMENSION)))
        low_tree = balanced_tree(permutation)
        high_order = tuple(permutation[i] for i in (0, 2, 4, 6, 1, 3, 5, 7))
        high_tree = balanced_tree(high_order)
        low = prescribed_ht_forcing((2,) * DIMENSION, low_tree, rank=RANK, seed=seed)
        high = prescribed_ht_forcing((2,) * DIMENSION, high_tree, rank=RANK, seed=seed + 1)
        forcing = np.zeros((4,) * DIMENSION)
        forcing[(slice(0, 2),) * DIMENSION] = low
        forcing[(slice(2, 4),) * DIMENSION] = 4 * high
        values = 4 * 65 ** 2 * np.sin(np.pi * np.array([1, 2, 11, 12]) / 130) ** 2
        eigenvalues = tuple(values.copy() for _ in range(DIMENSION))
        metadata = dict(id=f"band_seed{seed}", group=group, n=4, seed=seed,
                        description="Low-band HT source plus four times a differently structured high-band source",
                        axis_permutation=permutation, low_tree=tree_label(canonical(low_tree)),
                        high_tree=tree_label(canonical(high_tree)), physical_grid_n=64,
                        retained_one_based_modes=[1, 2, 11, 12], high_amplitude=4.0)
    elif group == "control":
        # One tensor-product eigenfunction is an exact rank-one solution in
        # every tree; a strict unique-winner certificate must abstain.
        forcing = np.zeros((2,) * DIMENSION)
        forcing[(0,) * DIMENSION] = 1.0
        eigenvalues = tuple(dirichlet_spectrum(2) for _ in range(DIMENSION))
        metadata = dict(id="all_trees_tie", group=group, n=2, seed=None,
                        description="One product eigenfunction: exact equal gain in all 315 trees")
    else:
        raise ValueError(group)
    metadata.update(shape=list(forcing.shape), input_entries=forcing.size, rank=RANK,
                    nonzero_entries=int(np.count_nonzero(forcing)))
    return metadata, forcing, eigenvalues


def case_specs(groups):
    cases = []
    if "small" in groups:
        cases += [("small", n, seed) for n in (2, 3) for seed in (31, 47, 59)]
    if "band" in groups:
        cases += [("band", None, seed) for seed in (101, 103, 107, 131, 137)]
    if "band_fresh" in groups:
        cases += [("band_fresh", None, seed) for seed in (151, 157, 163)]
    if "control" in groups:
        cases += [("control", None, None)]
    return cases


def scaled_gain(forcing, diagonal, candidate):
    energy = float(np.sum(diagonal * candidate ** 2))
    inner = float(np.sum(forcing * candidate))
    return inner ** 2 / energy if energy > 0 else 0.0


def observed_ranks(candidate, tree):
    """Post-timing independent feasibility diagnostic on the returned tensor."""
    ranks = {}
    for axes in edge_cuts(tree):
        matrix = unfold(candidate, axes)
        # Removing exactly zero rows/columns preserves all nonzero singular
        # values and is useful for the explicitly band-supported cases.
        rows = np.any(matrix != 0, axis=1)
        columns = np.any(matrix != 0, axis=0)
        singular = np.linalg.svd(matrix[np.ix_(rows, columns)], compute_uv=False)
        threshold = 1e-9 * singular[0] if singular.size else 0.0
        ranks[",".join(str(i) for i in axes)] = int(np.count_nonzero(singular > threshold))
    return ranks


def shared_rounder(tensor, rank):
    """Reuse exactly identical leaf/subtree SVDs across candidate trees.

    This reproduces the nested projections of ht_round; it does not approximate
    their singular values or exploit knowledge of the intended winning tree.
    Caches are fresh for every complete baseline invocation.
    """
    memo = {}
    stats = dict(subtree_svd_evaluations=0, subtree_cache_hits=0)

    def build(node):
        if node in memo:
            stats["subtree_cache_hits"] += 1
            return memo[node]
        axes = leaves(node)
        matrix = unfold(tensor, axes)
        if isinstance(node, int):
            vectors, singular, _ = np.linalg.svd(matrix, full_matrices=False)
            frame = vectors[:, :min(rank, len(singular))]
            frames, cores = {axes: frame}, {}
        else:
            left, left_frames, left_cores = build(node[0])
            right, right_frames, right_cores = build(node[1])
            product = np.kron(left, right)
            projected = product.T @ matrix
            vectors, singular, _ = np.linalg.svd(projected, full_matrices=False)
            kept = min(rank, len(singular))
            core = vectors[:, :kept]
            frame = product @ core
            frames = {**left_frames, **right_frames, axes: frame}
            cores = {**left_cores, **right_cores,
                     axes: core.reshape(left.shape[1], right.shape[1], kept)}
        stats["subtree_svd_evaluations"] += 1
        memo[node] = frame, frames, cores
        return memo[node]

    def round_tree(tree):
        left, left_frames, left_cores = build(tree[0])
        right, right_frames, right_cores = build(tree[1])
        order = leaves(tree)
        matrix = tensor.transpose(order).reshape(left.shape[0], right.shape[0])
        root_core = left.T @ matrix @ right
        approximation = ((left @ root_core @ right.T)
                         .reshape(tuple(tensor.shape[i] for i in order))
                         .transpose(np.argsort(order)).copy())
        return dict(tree=tree, frames={**left_frames, **right_frames},
                    cores={**left_cores, **right_cores, order: root_core},
                    approximation=approximation)

    return round_tree, stats


def exhaustive_baseline(forcing, eigenvalues, all_energy=False, shared_subtrees=False):
    """Full solution, all 315 nested roundings, then local energy refinement.

    In the inexpensive baseline only the best rounded candidate gets ALS2.
    The optional stronger, more expensive baseline gives ALS2 to every tree.
    Both are heuristics: neither certifies optimality over the HT families.
    """
    start = perf_counter()
    diagonal = diagonal_sum(eigenvalues)
    solution = forcing / diagonal
    solve_seconds = perf_counter() - start
    trees = all_balanced_trees(tuple(range(forcing.ndim)))
    gains = []
    best_gain = -np.inf
    best_index = None
    best_rounded = None
    best_fit = None
    round_tree, cache_stats = shared_rounder(solution, RANK) if shared_subtrees else (None, {})
    for index, tree in enumerate(trees):
        rounded = round_tree(tree) if shared_subtrees else ht_round(solution, tree, RANK, diagnostics=False)
        if all_energy:
            fit = fit_ht_energy(forcing, diagonal, rounded, max_sweeps=2, diagnostics=False)
            gain = fit["captured_gain"]
        else:
            fit = None
            gain = scaled_gain(forcing, diagonal, rounded["approximation"])
        gains.append(gain)
        if gain > best_gain:
            best_gain, best_index, best_rounded, best_fit = gain, index, rounded, fit
    if not all_energy:
        best_fit = fit_ht_energy(forcing, diagonal, best_rounded, max_sweeps=2, diagnostics=False)
    result = dict(elapsed=perf_counter() - start, solve_seconds=solve_seconds,
                  selected=tree_label(trees[best_index]), selected_index=best_index,
                  captured_gain=best_fit["captured_gain"], best_rounding_gain=best_gain,
                  exact_solution_energy=float(np.sum(forcing * solution)),
                  theoretical_coefficients=best_fit["theoretical_coefficients"],
                  tree_roundings=len(trees), tree_fits=len(trees) if all_energy else 1,
                  certified=False, full_solution_formed=True, all_candidate_gains=gains,
                  shared_subtrees=shared_subtrees, **cache_stats)
    result["observed_edge_ranks"] = observed_ranks(best_fit["approximation"], trees[best_index])
    assert max(result["observed_edge_ranks"].values()) <= RANK
    return result


def run_search(forcing, eigenvalues, control=False, joint_iterations=None, joint_schedule="late"):
    arguments = dict(rank=RANK, stages=(0,) if control else (0, 2, 5, 10),
                     fits_per_stage=1 if control else 8, als_sweeps=2, initial="product")
    if joint_iterations is not None:
        signature = inspect.signature(search_ht)
        if "joint_maxiter" not in signature.parameters:
            raise ValueError("search_ht does not expose joint_maxiter")
        arguments["joint_maxiter"] = joint_iterations
    if "joint_schedule" in inspect.signature(search_ht).parameters:
        arguments["joint_schedule"] = joint_schedule
    elif joint_schedule != "late":
        raise ValueError("search_ht does not expose joint_schedule")
    start = perf_counter()
    result = search_ht(forcing, eigenvalues, **arguments)
    result["wall_seconds"] = perf_counter() - start
    result["full_solution_formed"] = False
    result["arguments"] = arguments
    if control:
        assert not result["certified"], "Equal-gain control must not certify a strict winner"
    return result


def environment():
    sources = ["scripts/benchmark_ht_v3.py", "src/earlyht/ht_search.py", "src/earlyht/ht_bounds.py",
               "src/earlyht/ht_primal.py", "src/earlyht/ht_energy.py", "src/earlyht/ht_joint_dual.py"]
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.platform(), processor=platform.processor(),
                threads={key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in sources if (ROOT / name).exists()})


def summarize():
    paths = sorted(OUTPUT.glob("case_*.json"))
    if paths:
        rows = [json.loads(path.read_text()) for path in paths]
    elif (OUTPUT / "all_results.json.gz").exists():
        with gzip.open(OUTPUT / "all_results.json.gz", "rt") as source:
            rows = json.load(source)
    else:
        return
    def median(row, method, field):
        return float(np.median([run[method][field] for run in row["runs"]]))
    summary = []
    for row in rows:
        first = row["runs"][0]
        baseline_method = "solve_round_all" if "solve_round_all" in first else "solve_round_shared"
        new, baseline = first["certified_search"], first[baseline_method]
        decisions = {(run["certified_search"]["selected"], run["certified_search"]["certified"])
                     for run in row["runs"]}
        assert len(decisions) == 1, "Decision changed across repeated identical input"
        new_time = median(row, "certified_search", "wall_seconds")
        baseline_time = median(row, baseline_method, "elapsed")
        item = dict(id=row["case"]["id"], group=row["case"]["group"], repeats=len(row["runs"]),
                    certified=new["certified"], selected=new["selected"],
                    selected_index=new["selected_index"], baseline_selected=baseline["selected"],
                    baseline_method=baseline_method,
                    agreement=new["selected"] == baseline["selected"],
                    tree_fits=new["tree_fits"], proxy_operator_steps=new["proxy_operator_steps"],
                    distinct_trees_fitted=len({entry["tree_index"] for entry in new["trace"]}),
                    margin=new["margin"], median_search_seconds=new_time,
                    median_baseline_seconds=baseline_time,
                    baseline_over_search=baseline_time / new_time,
                    search_gain=max(new["lower"]), baseline_gain=baseline["captured_gain"],
                    exact_solution_energy=baseline["exact_solution_energy"],
                    joint_refinements=len(new.get("joint_refinements", [])),
                    max_baseline_observed_edge_rank=max(baseline["observed_edge_ranks"].values()))
        if "solve_als_all" in first:
            item["median_exhaustive_als_seconds"] = median(row, "solve_als_all", "elapsed")
            item["exhaustive_als_selected"] = first["solve_als_all"]["selected"]
            item["exhaustive_als_gain"] = first["solve_als_all"]["captured_gain"]
        if "solve_round_shared" in first and all("solve_round_shared" in run for run in row["runs"]):
            item["median_shared_baseline_seconds"] = median(row, "solve_round_shared", "elapsed")
            item["shared_baseline_over_search"] = item["median_shared_baseline_seconds"] / new_time
            item["shared_baseline_selected"] = first["solve_round_shared"]["selected"]
            item["shared_subtree_svds"] = first["solve_round_shared"]["subtree_svd_evaluations"]
        summary.append(item)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, default=plain) + "\n")
    packed = (json.dumps(rows, indent=2, default=plain) + "\n").encode()
    (OUTPUT / "all_results.json.gz").write_bytes(gzip.compress(packed, mtime=0))
    noncontrol = [row for row in summary if row["group"] != "control"]
    successes = sum(row["certified"] for row in noncontrol)
    strong_available = [row for row in noncontrol if "shared_baseline_over_search" in row]
    faster = sum(row["shared_baseline_over_search"] < 1 for row in strong_available)
    if faster == len(strong_available) and strong_available:
        timing_conclusion = ("**The strengthened exhaustive-rounding heuristic is faster on every non-tie case in this run.** "
                             "The demonstrated benefit of this general-HT extension is a global tree-family certificate "
                             "with few feasible HT fits, not a wall-time improvement over that heuristic. "
                             "Comparing only against naive repeated rounding would hide this result.")
    else:
        timing_conclusion = (f"The strengthened exhaustive-rounding heuristic is faster in {faster}/{len(strong_available)} "
                             "non-tie comparisons. Per-case timings are retained below; this finite study does not "
                             "establish universal runtime superiority. The certificate provides a stronger guarantee "
                             "than the heuristic's selected feasible approximation.")
    schedules = sorted({row["config"].get("joint_schedule", "late") for row in rows})
    zero_steps = sum(row["proxy_operator_steps"] == 0 for row in noncontrol)
    lines = ["# Genuine HT tree selection: dense eight-dimensional verification", "",
             f"{successes}/{len(noncontrol)} non-tie cases certified among all **315 balanced binary trees**; "
             "rank ≤ 2 on every nonroot edge, including singleton leaves. "
             "This is a finite balanced-tree family, not all possible binary topologies.", "",
             timing_conclusion, "",
             f"Joint-dual schedule: `{', '.join(schedules)}`. In {zero_steps}/{len(noncontrol)} non-tie cases "
             "the selector took zero Richardson steps. An early schedule tries joint cut bounds immediately "
             "when a few competitors remain, before fitting those competitors or refining a solution proxy.", "",
             "The source is constructed from random nested HT cores; no target solution is manufactured. "
             "The band cases mix different low- and high-frequency structures. "
             "Seeds 101, 103 and 107 are development cases; 131 and 137 were first used with the late schedule "
             "and were subsequently observed during early-schedule development. Seeds 151, 157 and 163, when "
             "present, were fixed before testing the early schedule and use the same axis-scrambling rule. "
             "All algorithms receive the same tensor and diagonal operator factors.", "",
             "The new method computes feasible energy gains, shared cut bounds, and optionally a small number "
             "of preconditioned Richardson steps. The **proxy steps** column makes those partial iterations explicit. "
             "It does not form the full solution. The baseline forms the exact spectral solution, performs nested "
             "HT rounding for all 315 trees, ranks them by optimally rescaled actual energy gain, then applies "
             "two energy-ALS sweeps only to the best candidate. The strengthened version shares identical "
             "leaf/subtree SVDs across all trees, with fresh caches for each call; it is the primary timing "
             "baseline whenever present. This baseline is a heuristic, not a global "
             "energy-optimal tree certificate.", "",
             "| Case | Certifies | HT fits | Proxy steps | New, s | Strong baseline, s | Baseline/new | Same tree |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        baseline_time = row.get("median_shared_baseline_seconds", row["median_baseline_seconds"])
        ratio = row.get("shared_baseline_over_search", row["baseline_over_search"])
        lines.append(f"| {row['id']} | {'yes' if row['certified'] else 'no'} | {row['tree_fits']} | "
                     f"{row['proxy_operator_steps']} | {row['median_search_seconds']:.4f} | "
                     f"{baseline_time:.4f} | {ratio:.2f}× | "
                     f"{'yes' if row['agreement'] else 'no'} |")
    lines += ["", "Times are medians of the number of repeats recorded per case in `summary.json`. "
              "Method order is shuffled deterministically within each repeat of the original comparison. "
              "For the original late-schedule data, the shared-subtree baseline was added in a separate pass. "
              "For the early-schedule data, both reported methods are shuffled within each repeat. "
              "Input generation is excluded; "
              "operator setup, cut SVDs, all tree fits and decision checks are included. "
              "Candidate-tree enumeration is shared and warmed before timings. Post-timing independent "
              "edge-rank diagnostics on the baseline output are excluded. NumPy runs with one BLAS thread. "
              "Raw per-repeat bounds, gains, decisions, source hashes and environment are in `all_results.json.gz`.", "",
              "The naive repeated-rounding validation is also retained in raw data (outside reported timing "
              "when `--baseline shared` is used); "
              "its times should not be used to claim an advantage over the stronger shared-subtree baseline. "
              "All 315 individual rounded gains from the two implementations are cross-checked for agreement.", "",
              "The exact spectral solve itself is inexpensive for this separable model. Any speedup here "
              "comes from reducing tree search and approximation work; it does not show an advantage over "
              "the spectral PDE solver alone. Dense storage is used and no continuum-discretization or "
              "large-dimension scalability guarantee is claimed.", "",
              "A certificate compares a feasible lower gain for the winner with valid upper gains for every "
              "competitor. Local energy ALS need not find the exact optimum for certification to be valid. "
              "Conversely, failure to certify is retained as an abstention. Bounds use floating-point "
              "arithmetic with tolerances, not interval arithmetic. The equal-gain control must abstain.", ""]
    for row in summary:
        if "median_exhaustive_als_seconds" in row:
            lines += [f"For `{row['id']}`, the additional baseline with two energy-ALS sweeps for **all 315 trees** "
                      f"takes {row['median_exhaustive_als_seconds']:.4f} s (median), compared with "
                      f"{row['median_search_seconds']:.4f} s for the certified method. "
                      f"Its selected tree is `{row['exhaustive_als_selected']}`; the certified method selects "
                      f"`{row['selected']}`. This local-optimization enumeration is still not a global certificate.", ""]
    output_argument = str(OUTPUT.relative_to(ROOT)) if OUTPUT.is_relative_to(ROOT) else str(OUTPUT)
    reproduction_argument = ("results/generated/ht" if OUTPUT == ROOT / "results/ht"
                             else output_argument)
    reproduced_groups = " ".join(group for group in GROUPS if any(row["group"] == group for row in summary))
    baseline_argument = "shared" if all("solve_round_all" not in row["runs"][0] for row in rows) else "all"
    extra = " --exhaustive-energy" if any("median_exhaustive_als_seconds" in row for row in summary) else ""
    lines += ["Reproduce from the project root:", "", "```sh",
              f"PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/benchmark_ht_v3.py --repeats 3 --output {reproduction_argument} --joint-schedule {schedules[0]} --groups {reproduced_groups} --baseline {baseline_argument}{extra}",
              "```", "", "To regenerate the report and plots from raw files: "
              f"`PYTHONPATH=src python scripts/benchmark_ht_v3.py --report-only --output {output_argument}`.", ""]
    (OUTPUT / "README.md").write_text("\n".join(lines))
    plot(summary)


def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    shown = [row for row in summary if row["group"] != "control"]
    x = np.arange(len(shown))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(x - .18, [row["median_search_seconds"] for row in shown], .36,
                label="Certified screening", color="#286f8e")
    axes[0].bar(x + .18, [row.get("median_shared_baseline_seconds", row["median_baseline_seconds"]) for row in shown], .36,
                label="Solve + 315 shared HT roundings + ALS", color="#d89c49")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Median elapsed time (seconds, log scale)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Complete tree-selection cost")
    axes[1].bar(x, [row["tree_fits"] for row in shown], color=["#286f8e" if row["certified"] else "#be635c" for row in shown])
    axes[1].set_ylabel("HT energy fits evaluated")
    axes[1].set_title("HT fit calls while screening 315 trees\nRed = no unique-winner certificate")
    for axis in axes:
        axis.set_xticks(x, [row["id"].replace("_seed", " / ") for row in shown], rotation=65, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(FIGURES / "figure.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    global OUTPUT, FIGURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=["small", "band", "control"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--exhaustive-energy", action="store_true",
                        help="Also give ALS2 to all 315 trees for n2_seed31")
    parser.add_argument("--joint-iterations", type=int, default=40)
    parser.add_argument("--joint-schedule", choices=("early", "late"), default="early")
    parser.add_argument("--output", type=Path, default=Path("results/generated/ht"))
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--baseline", choices=("all", "shared"), default="shared",
                        help="Shared measures only the cached baseline and validates naive rounding once outside reported timing")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--strengthen-existing", action="store_true",
                        help="Add a fresh shared-subtree rounding baseline to saved cases")
    args = parser.parse_args()
    OUTPUT = args.output if args.output.is_absolute() else ROOT / args.output
    FIGURES = ((args.figures if args.figures.is_absolute() else ROOT / args.figures)
               if args.figures is not None else OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        summarize()
        return
    if args.strengthen_existing:
        strengthened_environment = environment()
        for path in sorted(OUTPUT.glob("case_*.json")):
            item = json.loads(path.read_text())
            metadata = item["case"]
            if metadata["group"] not in args.groups:
                continue
            _, forcing, eigenvalues = build_case(metadata["group"], n=metadata["n"], seed=metadata["seed"])
            for run in item["runs"]:
                if args.skip_existing and "solve_round_shared" in run:
                    continue
                result = exhaustive_baseline(forcing, eigenvalues, shared_subtrees=True)
                old = np.asarray(run["solve_round_all"]["all_candidate_gains"])
                new = np.asarray(result["all_candidate_gains"])
                assert np.allclose(old, new, rtol=2e-11, atol=1e-14)
                result["maximum_naive_gain_discrepancy"] = float(np.max(np.abs(old - new)))
                result["added_utc"] = datetime.now(timezone.utc).isoformat()
                result["environment"] = strengthened_environment
                run["solve_round_shared"] = result
                search = run["certified_search"]
                violation = float(np.max(new - np.asarray(search["upper"])))
                assert violation <= 1e-9 * result["exact_solution_energy"]
                result["maximum_upper_bound_violation"] = violation
                print(metadata["id"], run["repeat"], "solve_round_shared", f"{result['elapsed']:.6f}s",
                      result["selected"], f"svds={result['subtree_svd_evaluations']}", flush=True)
                path.write_text(json.dumps(item, indent=2, default=plain) + "\n")
        summarize()
        return
    if args.repeats < 1:
        parser.error("repeats must be positive")
    trees = all_balanced_trees(tuple(range(DIMENSION)))
    assert len(trees) == 315
    rng = np.random.default_rng(20260920)
    started = datetime.now(timezone.utc).isoformat()
    run_environment = environment()
    config = dict(started_utc=started, groups=args.groups, repeats=args.repeats,
                  exhaustive_energy=args.exhaustive_energy, joint_iterations=args.joint_iterations,
                  joint_schedule=args.joint_schedule, baseline=args.baseline,
                  dimension=DIMENSION, rank=RANK, tree_count=len(trees),
                  initial="product", stages=[0, 2, 5, 10], fits_per_stage=8, als_sweeps=2,
                  method_order_seed=20260920, environment=run_environment)
    config_name = "config_" + "_".join(args.groups) + ".json"
    (OUTPUT / config_name).write_text(json.dumps(config, indent=2, default=plain) + "\n")
    # Warm numerical libraries with a different small source. Shared tree
    # enumeration is intentionally outside every algorithm's measured time.
    _, warm_f, warm_a = build_case("small", n=2, seed=197)
    run_search(warm_f, warm_a, joint_iterations=args.joint_iterations, joint_schedule=args.joint_schedule)
    exhaustive_baseline(warm_f, warm_a, shared_subtrees=True)
    for group, n, seed in case_specs(args.groups):
        metadata, forcing, eigenvalues = build_case(group, n=n, seed=seed)
        destination = OUTPUT / f"case_{metadata['id']}.json"
        if args.skip_existing and destination.exists():
            continue
        item = dict(case=metadata, config=config, runs=[])
        if args.baseline == "shared":
            item["naive_validation"] = exhaustive_baseline(forcing, eigenvalues)
            item["naive_validation"]["excluded_from_reported_timing"] = True
        for repeat in range(args.repeats):
            methods = (["certified_search", "solve_round_shared"] if args.baseline == "shared"
                       else ["certified_search", "solve_round_all", "solve_round_shared"])
            if args.exhaustive_energy and metadata["id"] == "n2_seed31":
                methods.append("solve_als_all")
            rng.shuffle(methods)
            run = dict(repeat=repeat, method_order=methods.copy(), started_utc=datetime.now(timezone.utc).isoformat())
            for method in methods:
                method_started = datetime.now(timezone.utc).isoformat()
                if method == "certified_search":
                    result = run_search(forcing, eigenvalues, control=group == "control",
                                        joint_iterations=args.joint_iterations, joint_schedule=args.joint_schedule)
                else:
                    result = exhaustive_baseline(forcing, eigenvalues, all_energy=method == "solve_als_all",
                                                 shared_subtrees=method == "solve_round_shared")
                run[method] = result
                result["started_utc"] = method_started
                result["finished_utc"] = datetime.now(timezone.utc).isoformat()
                print(metadata["id"], repeat, method,
                      f"{result.get('wall_seconds', result['elapsed']):.6f}s",
                      result["selected"], result["certified"],
                      f"fits={result['tree_fits']}", flush=True)
            # A certified winner's feasible gain need not dominate a different
            # local fit of the *same* family. For any different selected tree,
            # however, its feasible gain cannot exceed the winner's lower bound.
            search = run["certified_search"]
            if "solve_round_shared" in run:
                naive_result = run["solve_round_all"] if "solve_round_all" in run else item["naive_validation"]
                naive = np.asarray(naive_result["all_candidate_gains"])
                shared = np.asarray(run["solve_round_shared"]["all_candidate_gains"])
                assert np.allclose(naive, shared, rtol=2e-11, atol=1e-14)
                run["solve_round_shared"]["maximum_naive_gain_discrepancy"] = float(np.max(np.abs(naive - shared)))
            for name in methods:
                if name == "certified_search":
                    continue
                baseline = run[name]
                # Every explicitly constructed competitor is a feasible HT
                # tensor, so its achieved gain must lie below that tree's
                # independently derived upper bound, including pruned trees.
                violation = float(np.max(np.asarray(baseline["all_candidate_gains"]) -
                                         np.asarray(search["upper"])))
                assert violation <= 1e-9 * baseline["exact_solution_energy"]
                baseline["maximum_upper_bound_violation"] = violation
                if search["certified"] and search["selected"] != baseline["selected"]:
                    assert baseline["captured_gain"] <= max(search["lower"]) + 1e-9 * baseline["exact_solution_energy"]
            item["runs"].append(run)
            destination.write_text(json.dumps(item, indent=2, default=plain) + "\n")
        summarize()
    summarize()


if __name__ == "__main__":
    main()
