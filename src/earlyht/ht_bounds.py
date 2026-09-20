"""Reference-free gain intervals for arbitrary binary hierarchical Tucker trees.

Inputs are small dense coefficient tensors in orthonormal axis eigenbases.
The PDE operator is a positive Kronecker sum plus a constant potential.
Every tree edge supplies a rank-constrained matrix relaxation. Their upper
bounds can share zero-center SVDs across trees, while a feasible HT tensor
supplies the lower bound in the original PDE energy.
"""

from __future__ import annotations

from numbers import Integral
from time import perf_counter

import numpy as np

from .direct_selection import product_metrics


def _leaves(tree):
    if isinstance(tree, Integral) and not isinstance(tree, bool):
        return (int(tree),)
    if not isinstance(tree, (tuple, list)) or len(tree) != 2:
        raise ValueError("A dimension tree is a binary tuple with integer leaves.")
    return _leaves(tree[0]) + _leaves(tree[1])


def _canonical_cut(cut, ndim):
    cut = tuple(cut)
    if (not cut or len(cut) >= ndim or len(set(cut)) != len(cut)
            or any(isinstance(i, bool) or not isinstance(i, Integral)
                   or i < 0 or i >= ndim for i in cut)):
        raise ValueError("A cut must be a nonempty proper subset of the tensor axes.")
    cut = tuple(sorted(cut))
    other = tuple(i for i in range(ndim) if i not in cut)
    return min((cut, other), key=lambda item: (len(item), item))


def canonical_tree_cuts(tree, ndim):
    """Return unique nonroot edge cuts, identifying a cut with its complement.

    Singleton edges are included: they constrain the leaf ranks when the
    requested rank is smaller than an axis size. All axis labels must occur
    exactly once. The ordering is deterministic and independent of child order.
    """
    axes = _leaves(tree)
    if ndim < 2 or sorted(axes) != list(range(ndim)):
        raise ValueError("The tree must contain every tensor axis exactly once.")
    cuts = set()

    def visit(node, root=False):
        if not root:
            cuts.add(_canonical_cut(_leaves(node), ndim))
        if not isinstance(node, Integral):
            visit(node[0])
            visit(node[1])

    visit(tree, root=True)
    return tuple(sorted(cuts, key=lambda item: (len(item), item)))


def _unfold(tensor, cut):
    other = tuple(i for i in range(tensor.ndim) if i not in cut)
    row_count = int(np.prod([tensor.shape[i] for i in cut]))
    return np.transpose(tensor, cut + other).reshape(row_count, -1)


def _nonzero_singular_values(matrix):
    """Discard exactly zero rows/columns, preserving every nonzero singular value.

    Spectral band inputs can have very sparse support. No tolerance, spectral
    truncation, or assumption about the unknown solution is used here.
    """
    rows = np.any(matrix != 0, axis=1)
    columns = np.any(matrix != 0, axis=0)
    if not rows.any() or not columns.any():
        return np.empty(0)
    if not rows.all() or not columns.all():
        matrix = matrix[np.ix_(rows, columns)]
    return np.linalg.svd(matrix, compute_uv=False)


def _axis_sum(eigenvalues, axes):
    shape = tuple(len(eigenvalues[i]) for i in axes)
    result = np.zeros(shape)
    for local, axis in enumerate(axes):
        index = [1] * len(axes)
        index[local] = shape[local]
        result += eigenvalues[axis].reshape(index)
    return result.ravel()


class CutBoundCache:
    """Cache cut spectra for one immutable forcing/operator/rank combination.

    Reuse this object across candidate trees. Stored arrays are private copies,
    so later mutation of a caller's input cannot silently invalidate a bound.
    The counter records actual bound SVDs, excluding feasibility checks.
    """

    def __init__(self, forcing, eigenvalues, rank, potential=0., metric_policy="geometric",
                 center=None):
        self.forcing = np.array(forcing, dtype=float, copy=True)
        self.eigenvalues = tuple(np.array(values, dtype=float, copy=True)
                                 for values in eigenvalues)
        if (self.forcing.ndim < 2 or not np.all(np.isfinite(self.forcing))
                or len(self.eigenvalues) != self.forcing.ndim):
            raise ValueError("A finite tensor and one spectrum per axis are required.")
        for n, values in zip(self.forcing.shape, self.eigenvalues):
            if (values.shape != (n,) or not np.all(np.isfinite(values))
                    or np.any(values <= 0)):
                raise ValueError("Axis spectra must have the matching size and be positive.")
        if isinstance(rank, bool) or not isinstance(rank, Integral) or rank < 1:
            raise ValueError("rank must be a positive integer.")
        self.rank = int(rank)
        self.potential = float(potential)
        if not np.isfinite(self.potential) or self.potential < 0:
            raise ValueError("The constant potential must be finite and nonnegative.")
        if metric_policy not in ("geometric", "compact", "full", "scalar"):
            raise ValueError("Unknown product metric policy.")
        self.metric_policy = metric_policy
        self.diagonal = (_axis_sum(self.eigenvalues, tuple(range(self.forcing.ndim)))
                         .reshape(self.forcing.shape) + self.potential)
        self.forcing.setflags(write=False)
        self.diagonal.setflags(write=False)
        for values in self.eigenvalues:
            values.setflags(write=False)
        self._states = {}
        self._zero_bounds = {}
        self._shared_centers = []
        self._shared_bounds = {}
        self.svd_evaluations = 0
        self.zero_cache_hits = 0
        self.shared_cache_hits = 0
        if center is not None:
            self.add_center(center)

    def add_center(self, center):
        """Add an unrestricted shared center; cuts are refined lazily on demand.

        This permits a single cheap polynomial-in-operator proxy to tighten all
        candidate trees. Previously computed upper bounds are always retained.
        A new center cannot invalidate or increase a cached cut upper bound.
        """
        center = np.array(center, dtype=float, copy=True)
        if center.shape != self.forcing.shape or not np.all(np.isfinite(center)):
            raise ValueError("The shared center must be finite with the forcing shape.")
        center.setflags(write=False)
        self._shared_centers.append(center)

    def upper_for_cut(self, cut):
        """Best cached upper bound over zero and all globally shared centers."""
        cut = _canonical_cut(cut, self.forcing.ndim)
        if cut in self._shared_bounds:
            evaluated, best = self._shared_bounds[cut]
            if evaluated == len(self._shared_centers):
                self.shared_cache_hits += 1
                return dict(best)
        else:
            evaluated = 0
            zero = self.bound(cut)
            best = dict(zero, center="zero", zero_upper=zero["upper"])
        for index in range(evaluated, len(self._shared_centers)):
            result = self.bound(cut, self._shared_centers[index])
            if result["upper"] < best["upper"]:
                best = dict(result, center=f"shared_{index}", zero_upper=best["zero_upper"])
        self._shared_bounds[cut] = len(self._shared_centers), best
        return dict(best)

    def compatible(self, forcing, eigenvalues, rank, potential, metric_policy):
        return (rank == self.rank and float(potential) == self.potential
                and metric_policy == self.metric_policy
                and np.array_equal(forcing, self.forcing)
                and len(eigenvalues) == len(self.eigenvalues)
                and all(np.array_equal(a, b) for a, b in zip(eigenvalues, self.eigenvalues)))

    def _state(self, cut):
        cut = _canonical_cut(cut, self.forcing.ndim)
        if cut in self._states:
            return cut, self._states[cut]
        other = tuple(i for i in range(self.forcing.ndim) if i not in cut)
        a = _axis_sum(self.eigenvalues, cut) + self.potential / 2
        b = _axis_sum(self.eigenvalues, other) + self.potential / 2
        diagonal = a[:, None] + b[None, :]
        metrics = []
        candidates = ([(np.full_like(a, float(a.min() + b.min())), np.ones_like(b),
                        "scalar")] if self.metric_policy == "scalar"
                      else product_metrics(a, b))
        for left, right, name in candidates:
            if self.metric_policy == "geometric" and name != "balanced_geometric":
                continue
            if self.metric_policy == "compact" and name not in (
                    "balanced_geometric", "power_0.25", "power_0.75"):
                continue
            metric = left[:, None] * right[None, :]
            if (not np.all(np.isfinite(metric)) or np.any(metric <= 0)
                    or np.max(metric - diagonal) > 1e-10 * max(1., float(diagonal.max()))):
                raise ValueError("A product metric must be positive and below the operator.")
            metrics.append((name, np.sqrt(metric), diagonal - metric))
        state = dict(forcing=_unfold(self.forcing, cut), metrics=metrics)
        self._states[cut] = state
        return cut, state

    def bound(self, cut, center=None):
        """Return the best product-metric upper bound at a specified center.

        The center is an unrestricted full tensor. It need not have any low
        ranks. ``None`` means the zero center and enables cross-tree caching.
        """
        cut = _canonical_cut(cut, self.forcing.ndim)
        if center is None and cut in self._zero_bounds:
            self.zero_cache_hits += 1
            return dict(self._zero_bounds[cut])
        cut, state = self._state(cut)
        if center is not None:
            center = np.asarray(center, dtype=float)
            if center.shape != self.forcing.shape or not np.all(np.isfinite(center)):
                raise ValueError("The center must be a finite tensor with the forcing shape.")
            center_matrix = _unfold(center, cut)
        else:
            center_matrix = np.zeros_like(state["forcing"])
        best = dict(upper=float("inf"), metric=None)
        for name, root_metric, remainder in state["metrics"]:
            shifted = (state["forcing"] - remainder * center_matrix) / root_metric
            singular = _nonzero_singular_values(shifted)
            upper = float(np.sum(singular[:self.rank] ** 2)
                          + np.sum(remainder * center_matrix ** 2))
            self.svd_evaluations += 1
            if upper < best["upper"]:
                best = dict(upper=upper, metric=name)
        if center is None:
            self._zero_bounds[cut] = dict(best)
        return best


def ht_gain_bounds(forcing, eigenvalues, tree, rank, candidate=None, *, potential=0.,
                   cache=None, center=None, centered_cuts="all", metric_policy="geometric",
                   rank_tolerance=1e-9):
    """Bound the optimal captured energy of a genuine HT rank-constrained family.

    The lower bound requires a candidate feasible at EVERY edge, not only the
    root split. The function checks all active edge ranks. Passing a root-only
    low-rank matrix that violates a lower edge raises ``ValueError``.

    Each edge yields an upper bound because the HT family is a subset of that
    edge's matrix-rank family. Zero-center bounds are always retained. A center
    defaults to the optimally scaled candidate; an explicit unrestricted center
    is permitted. ``centered_cuts`` selects all edges, just the root split, or
    the given number of most restrictive zero-center cuts (zero disables it).

    The result bounds the discrete family's optimum, independent of candidate
    optimizer convergence. Floating-point rank checks/inequalities are not
    interval-arithmetic proofs. Dense materialization limits this prototype to
    small retained coefficient spaces; no full-grid PDE solve is performed.
    """
    start = perf_counter()
    if isinstance(rank, bool) or not isinstance(rank, Integral) or rank < 1:
        raise ValueError("rank must be a positive integer.")
    forcing = np.asarray(forcing, dtype=float)
    eigenvalues = tuple(eigenvalues)
    if cache is None:
        cache = CutBoundCache(forcing, eigenvalues, rank, potential, metric_policy)
    elif not cache.compatible(forcing, eigenvalues, rank, potential, metric_policy):
        raise ValueError("The cache is bound to a different forcing/operator/rank/policy.")
    all_cuts = canonical_tree_cuts(tree, forcing.ndim)
    # A full leaf rank imposes no restriction and its cut is not needed to
    # relax the intersection. For d=8, m=r=2 this leaves 63 possible balanced
    # cuts globally (28 pairs and 35 complementary quartets), not 71.
    cuts = tuple(cut for cut in all_cuts
                 if min(_unfold(forcing, cut).shape) > rank)
    if not cuts:
        cuts = (_canonical_cut(_leaves(tree[0]), forcing.ndim),)
    if not np.isfinite(rank_tolerance) or rank_tolerance <= 0:
        raise ValueError("rank_tolerance must be finite and positive.")
    lower, rank_tails, scaled_candidate = 0., {}, None
    if candidate is not None:
        candidate = np.asarray(candidate, dtype=float)
        if candidate.shape != forcing.shape or not np.all(np.isfinite(candidate)):
            raise ValueError("The candidate must be a finite tensor with the forcing shape.")
        norm = float(np.linalg.norm(candidate))
        for cut in all_cuts:
            matrix = _unfold(candidate, cut)
            if min(matrix.shape) <= rank:
                relative_tail = 0.
            else:
                singular = _nonzero_singular_values(matrix)
                relative_tail = float(np.linalg.norm(singular[rank:])) / max(norm, 1e-300)
            rank_tails[cut] = relative_tail
            if relative_tail > rank_tolerance:
                raise ValueError(f"The candidate violates the HT rank at edge {cut}.")
        pairing = float(np.sum(forcing * candidate))
        energy = float(np.sum(cache.diagonal * candidate ** 2))
        if energy > 0:
            lower = pairing ** 2 / energy
            scaled_candidate = candidate * (pairing / energy)
        else:
            scaled_candidate = np.zeros_like(forcing)
    selected_center = scaled_candidate if center is None else np.asarray(center, dtype=float)
    if selected_center is not None and (selected_center.shape != forcing.shape
                                       or not np.all(np.isfinite(selected_center))):
        raise ValueError("The center must be a finite tensor with the forcing shape.")
    if (centered_cuts not in ("all", "root")
            and (isinstance(centered_cuts, bool) or not isinstance(centered_cuts, Integral)
                 or centered_cuts < 0)):
        raise ValueError("centered_cuts must be 'all', 'root', or a nonnegative integer.")
    before = cache.svd_evaluations
    old_hits = cache.zero_cache_hits
    old_shared_hits = cache.shared_cache_hits
    edge_bounds = {cut: cache.upper_for_cut(cut) for cut in cuts}
    zero_upper = min(bound["zero_upper"] for bound in edge_bounds.values())
    if centered_cuts == "all":
        refine = cuts
    elif centered_cuts == "root":
        refine = (_canonical_cut(_leaves(tree[0]), forcing.ndim),)
        if refine[0] not in edge_bounds:
            edge_bounds[refine[0]] = cache.upper_for_cut(refine[0])
    else:
        refine = tuple(sorted(cuts, key=lambda cut: edge_bounds[cut]["upper"]))[:centered_cuts]
    if selected_center is not None:
        for cut in refine:
            result = cache.bound(cut, selected_center)
            if result["upper"] < edge_bounds[cut]["upper"]:
                edge_bounds[cut] = dict(result, center="provided",
                                        zero_upper=edge_bounds[cut]["zero_upper"])
    best_cut = min(edge_bounds, key=lambda cut: edge_bounds[cut]["upper"])
    upper = edge_bounds[best_cut]["upper"]
    guard = 1e-9 * max(1e-12, upper, lower)
    if lower > upper + guard:
        raise ArithmeticError("A feasible gain exceeds its upper bound beyond the numerical guard.")
    return dict(lower=lower, upper=upper, zero_center_upper=zero_upper,
                best_cut=list(best_cut), rank=int(rank), tree_edges=len(all_cuts),
                bounded_edges=len(edge_bounds),
                max_relative_rank_tail=max(rank_tails.values(), default=0.),
                edge_bounds=[dict(cut=list(cut), **bound) for cut, bound in edge_bounds.items()],
                bound_svd_evaluations=cache.svd_evaluations - before,
                zero_cache_hits=cache.zero_cache_hits - old_hits,
                shared_cache_hits=cache.shared_cache_hits - old_shared_hits,
                numerical_guard=guard, elapsed=perf_counter() - start,
                full_pde_iterations=0,
                scope="all edge ranks of a binary HT tree; discrete separable SPD operator")
