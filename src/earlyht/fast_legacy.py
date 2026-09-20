"""Efficient implementation of the original proxy/error tree criterion.

Weighted SVD tails are evaluated from Gram-matrix leading eigenvalues rather
than complete singular spectra. A conservative allowance is subtracted from
the squared lower bound to address cancellation and floating-point spectral
error. This remains a numerical bound, not an interval-arithmetic proof.
The optional ARPACK backend affects feasible upper candidates only: every
lower bound uses deterministic indexed Gram eigenvalues.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from time import perf_counter

import numpy as np
from scipy.linalg import cholesky, eigh
from scipy.sparse.linalg import ArpackNoConvergence, svds

from .pde import BALANCED_CUTS, matricize
from .potential_bounds import additive_minorants
from .selection import compare_bounds, cut_name
from .trees import CutProblem, fit_cut


def gram_tail_squared(matrix, rank):
    """Numerically conservative squared SVD tail from indexed Gram spectra.

    ``1e-10 + 32 eps max(shape)`` times the Frobenius energy is subtracted.
    This explicitly sacrifices tiny-tail resolution; it avoids inventing a
    positive lower bound from cancellation at nearly exact low rank. LAPACK
    supplies the largest indexed eigenvalues of the computed Gram matrix.
    """
    matrix = np.asarray(matrix, dtype=float)
    dimension = min(matrix.shape)
    energy = float(np.sum(matrix * matrix))
    if rank >= dimension or energy == 0:
        return 0.0
    if rank == 0:
        leading = 0.0
    else:
        gram = matrix @ matrix.T if matrix.shape[0] <= matrix.shape[1] else matrix.T @ matrix
        values = eigh(gram, subset_by_index=(dimension - rank, dimension - 1),
                      eigvals_only=True, check_finite=False, driver="evr")
        leading = float(np.sum(np.maximum(values, 0.0)))
    allowance = (1e-10 + 32 * np.finfo(float).eps * max(matrix.shape)) * energy
    return max(0.0, energy - leading - allowance)


def truncated_candidate(matrix, rank, backend="gram"):
    """Feasible rank-r projection without materializing a complete SVD.

    The result's energy error is always evaluated separately in the full PDE
    metric. ARPACK convergence or an exact largest-subspace proof is therefore
    unnecessary for validity of this upper bound. Its residual is diagnostic.
    """
    matrix = np.asarray(matrix, dtype=float)
    m, n = matrix.shape
    rank = min(rank, m, n)
    if rank == 0 or not np.any(matrix):
        return np.zeros_like(matrix), 0, dict(backend="zero", residual=0.0)
    if rank == min(m, n):
        return matrix.copy(), rank, dict(backend="full_rank", residual=0.0)
    if backend == "svds" and min(m, n) >= 32:
        try:
            u, singular, vh = svds(matrix, k=rank, which="LM", tol=1e-10,
                                   random_state=2026)
            candidate = (u * singular) @ vh
            residual = float(np.sqrt(np.sum((matrix @ vh.T - u * singular) ** 2)
                                      + np.sum((matrix.T @ u - vh.T * singular) ** 2)))
            return candidate, rank, dict(backend="svds", residual=residual)
        except ArpackNoConvergence:
            # Returning a feasible Gram candidate preserves the same contract.
            pass
    if min(m, n) < 32:
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        candidate = (u[:, :rank] * singular[:rank]) @ vh[:rank]
        threshold = np.finfo(float).eps * max(m, n) * singular[0]
        observed = int(np.sum(singular[:rank] > threshold))
        return candidate, observed, dict(backend="small_full_svd", residual=0.0)
    use_left = m <= n
    gram = matrix @ matrix.T if use_left else matrix.T @ matrix
    dimension = min(m, n)
    values, vectors = eigh(gram, subset_by_index=(dimension - rank, dimension - 1),
                           check_finite=False, driver="evr")
    candidate = vectors @ (vectors.T @ matrix) if use_left else (matrix @ vectors) @ vectors.T
    residual = float(np.linalg.norm(gram @ vectors - vectors * values))
    threshold = (np.finfo(float).eps * max(m, n)) ** 2 * max(float(values[-1]), 0.0)
    observed = int(np.sum(values > threshold))
    return candidate, observed, dict(backend="gram", residual=residual)


@dataclass
class LegacyCache:
    problem: object
    potential_bound: str
    cuts: dict
    setup_seconds: float


def prepare_fast_legacy(problem, potential_bound="minorants"):
    """Cache operator/Cholesky data reusable across proxies for one PDE.

    The caller must count ``setup_seconds`` when using this cache and must not
    mutate the problem's operators or potential after preparing it.
    """
    start = perf_counter()
    if len(problem.shape) != 4:
        raise ValueError("This selector supports four-variable balanced trees.")
    if potential_bound not in ("discard", "minorants"):
        raise ValueError("potential_bound must be discard or minorants.")
    cuts = {}
    for index, cut in enumerate(BALANCED_CUTS):
        left, right, potential = problem.cut_operators(cut)
        factors = [(cholesky(left, lower=False, check_finite=False),
                    cholesky(right, lower=False, check_finite=False))]
        if potential_bound == "minorants" and np.any(potential):
            for p, q, _ in additive_minorants(potential):
                if not np.any(p) and not np.any(q):
                    continue
                factors.append((cholesky(left + np.diag(p), lower=False, check_finite=False),
                                cholesky(right + np.diag(q), lower=False, check_finite=False)))
        cuts[cut_name(cut)] = dict(cut=cut, index=index, left=left, right=right,
                                   potential=potential, factors=factors)
    return LegacyCache(problem, potential_bound, cuts, perf_counter() - start)


def select_tree_fast(
    problem, proxy, delta, rank, *, starts=2, max_sweeps=25, seed=0,
    strategy="screened", potential_bound="minorants", backend="gram", cache=None,
):
    """Same weighted-tail/feasible-error separation as the original selector.

    ``elapsed`` includes cache setup when no cache is supplied. With an
    explicit reusable cache it covers this call only; setup time is reported
    separately through ``cache_setup_seconds`` and ``setup_included``.
    """
    start = perf_counter()
    if strategy not in ("screened", "als"):
        raise ValueError("strategy must be screened or als.")
    if backend not in ("gram", "svds"):
        raise ValueError("backend must be gram or svds.")
    if isinstance(rank, bool) or not isinstance(rank, Integral) or rank < 0:
        raise ValueError("rank must be a nonnegative integer.")
    if not np.isfinite(delta) or delta < 0:
        raise ValueError("delta must be finite and nonnegative.")
    proxy = np.asarray(proxy, dtype=float)
    if proxy.shape != problem.shape or not np.all(np.isfinite(proxy)):
        raise ValueError("The proxy must be a finite tensor on the PDE grid.")
    setup_included = cache is None
    if cache is None:
        cache = prepare_fast_legacy(problem, potential_bound)
    elif cache.problem is not problem or cache.potential_bound != potential_bound:
        raise ValueError("Cache must belong to this problem and potential-bound policy.")
    bounds, matrices = {}, {}
    for name, state in cache.cuts.items():
        matrix = matricize(proxy, state["cut"])
        matrices[name] = matrix
        actual_rank = min(rank, min(matrix.shape))
        lower_squared = max(
            gram_tail_squared(left_root @ matrix, actual_rank)
            + gram_tail_squared(matrix @ right_root.T, actual_rank)
            for left_root, right_root in state["factors"]
        )
        lower = float(np.sqrt(lower_squared))
        if strategy == "als":
            cp = CutProblem(matrix, state["left"], state["right"], state["potential"])
            fit = fit_cut(cp, actual_rank, starts=starts, max_sweeps=max_sweeps,
                          seed=seed + state["index"])
            bounds[name] = dict(lower=max(lower, float(fit["lower"])),
                                upper=float(fit["upper"]), sweeps=int(fit["sweeps"]),
                                rank_observed=int(fit["rank_observed"]),
                                history=[float(value) for value in fit["history"]], source="als")
        else:
            candidate, observed, diagnostics = truncated_candidate(matrix, actual_rank, backend)
            error = matrix - candidate
            action = state["left"] @ error + error @ state["right"] + state["potential"] * error
            upper = float(np.sqrt(max(0.0, np.sum(error * action))))
            bounds[name] = dict(lower=lower, upper=upper, sweeps=0,
                                rank_observed=observed, history=[upper],
                                source="fast_svd", spectral_diagnostics=diagnostics)
    decision = compare_bounds(bounds, delta)
    refinements = 0
    if strategy == "screened" and not decision["certified"]:
        eligible = [name for name in bounds
                    if min(bounds[other]["lower"] for other in bounds if other != name)
                    - bounds[name]["lower"] - 2 * delta > decision["numerical_guard"]]
        if len(eligible) > 1:
            raise AssertionError("At most one lower-bound interval can be separated.")
        for name in eligible:
            state = cache.cuts[name]
            cp = CutProblem(matrices[name], state["left"], state["right"], state["potential"])
            fit = fit_cut(cp, rank, starts=starts, max_sweeps=max_sweeps,
                          seed=seed + state["index"])
            if fit["upper"] <= bounds[name]["upper"]:
                bounds[name] = dict(lower=max(bounds[name]["lower"], float(fit["lower"])),
                                    upper=float(fit["upper"]), sweeps=int(fit["sweeps"]),
                                    rank_observed=int(fit["rank_observed"]),
                                    history=[float(value) for value in fit["history"]],
                                    source="als_refinement")
            refinements += 1
        decision = compare_bounds(bounds, delta)
    return dict(**decision, bounds=bounds, strategy=strategy, potential_bound=potential_bound,
                backend=backend, lower_backend="indexed_gram_eigenvalues",
                squared_tail_relative_allowance=1e-10,
                als_refinements=refinements, cache_setup_seconds=cache.setup_seconds,
                setup_included=setup_included, elapsed=perf_counter() - start)
