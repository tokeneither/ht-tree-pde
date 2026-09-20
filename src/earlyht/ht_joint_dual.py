"""Two-edge allocation duals coupling two hierarchical Tucker rank constraints.

Every objective evaluation is an upper bound. Convex optimization improves
the bound but optimizer success is never required for its validity. Inputs
are small coefficient tensors for a separable positive diagonal PDE operator.
"""

from __future__ import annotations

from numbers import Integral
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from .ht_bounds import CutBoundCache, _canonical_cut


class JointCutDual:
    """Convex forcing-allocation dual for two distinct edge rank constraints.

    Write F1=alpha*F+D, F2=(1-alpha)*F-D and A_i=alpha_i*A.
    Each term uses its own edge-separable geometric metric B_i <= A_i
    and a common unrestricted center Y. The sum of the two centered matrix
    upper bounds bounds the gain of tensors satisfying BOTH edge ranks.

    ``support='union'`` restricts only the allocation D to the union of exact
    nonzeros of F and Y. It does not project a feasible HT tensor onto that
    support. SVDs retain every nonzero row/column of each shifted forcing;
    interior zero entries remain in the matrices and may be filled by their
    low-rank maximizers. This distinction is necessary for validity.
    """

    def __init__(self, forcing, eigenvalues, cuts, rank, center=None, *, potential=0.,
                 alpha=.5, support="union"):
        operator = CutBoundCache(forcing, eigenvalues, rank, potential)
        potential = operator.potential
        self.shape = operator.forcing.shape
        self.rank = operator.rank
        self.cuts = tuple(_canonical_cut(cut, len(self.shape)) for cut in cuts)
        if len(self.cuts) != 2 or self.cuts[0] == self.cuts[1]:
            raise ValueError("Exactly two distinct cuts are required.")
        alpha = float(alpha)
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one.")
        self.alpha = alpha
        forcing = operator.forcing
        center = np.zeros_like(forcing) if center is None else np.asarray(center, dtype=float)
        if center.shape != forcing.shape or not np.all(np.isfinite(center)):
            raise ValueError("The center must be finite with the forcing shape.")
        if support == "union":
            mask = (forcing != 0) | (center != 0)
        elif support == "full":
            mask = np.ones_like(forcing, dtype=bool)
        else:
            raise ValueError("support must be 'union' or 'full'.")
        self.support_indices = np.flatnonzero(mask.ravel())
        self.variables = len(self.support_indices)
        coordinates = np.unravel_index(self.support_indices, self.shape)
        self.terms = []
        self.svd_evaluations = 0
        f = forcing.ravel()[self.support_indices]
        y = center.ravel()[self.support_indices]
        diagonal = operator.diagonal.ravel()[self.support_indices]
        for cut, weight, sign in zip(self.cuts, (alpha, 1 - alpha), (1., -1.)):
            other = tuple(i for i in range(len(self.shape)) if i not in cut)
            row_ids = np.ravel_multi_index(tuple(coordinates[i] for i in cut),
                                          tuple(self.shape[i] for i in cut))
            col_ids = np.ravel_multi_index(tuple(coordinates[i] for i in other),
                                          tuple(self.shape[i] for i in other))
            rows, row_map = np.unique(row_ids, return_inverse=True)
            cols, col_map = np.unique(col_ids, return_inverse=True)
            a = np.full(self.variables, potential / 2)
            b = np.full(self.variables, potential / 2)
            for i in cut:
                a += operator.eigenvalues[i][coordinates[i]]
            for i in other:
                b += operator.eigenvalues[i][coordinates[i]]
            min_a = potential / 2 + sum(float(operator.eigenvalues[i].min()) for i in cut)
            min_b = potential / 2 + sum(float(operator.eigenvalues[i].min()) for i in other)
            shift = (min_b - min_a) / 2
            metric = weight * 2 * np.sqrt(a + shift) * np.sqrt(b - shift)
            remainder = weight * diagonal - metric
            if self.variables and np.min(remainder) < -1e-10 * max(1., float(diagonal.max())):
                raise ArithmeticError("The geometric metric is not an operator minorant.")
            self.terms.append(dict(
                shape=(len(rows), len(cols)), rows=row_map, cols=col_map,
                root_metric=np.sqrt(metric), shifted=weight * f - remainder * y,
                constant=float(np.sum(remainder * y ** 2)), sign=sign,
            ))

    def value_gradient(self, allocation):
        """Return a valid upper bound and a Danskin gradient/subgradient.

        At a singular-value tie across the rank cutoff, this chosen derivative
        is a subgradient. L-BFGS is used as a practical convex minimizer; a
        nonsmooth stop never invalidates an already evaluated upper bound.
        """
        allocation = np.asarray(allocation, dtype=float)
        if allocation.shape != (self.variables,) or not np.all(np.isfinite(allocation)):
            raise ValueError("The allocation must be a finite vector of the declared support size.")
        value = 0.
        gradient = np.zeros_like(allocation)
        for term in self.terms:
            if not self.variables:
                continue
            weighted = np.zeros(term["shape"])
            weighted[term["rows"], term["cols"]] = (
                term["shifted"] + term["sign"] * allocation) / term["root_metric"]
            left, singular, right = np.linalg.svd(weighted, full_matrices=False)
            retained = min(self.rank, len(singular))
            truncated = (left[:, :retained] * singular[:retained]) @ right[:retained]
            candidate_support = (truncated[term["rows"], term["cols"]]
                                 / term["root_metric"])
            value += term["constant"] + float(np.sum(singular[:retained] ** 2))
            gradient += 2 * term["sign"] * candidate_support
            self.svd_evaluations += 1
        return float(value), gradient


def optimize_joint_bound(forcing, eigenvalues, cuts, rank, center=None, *, potential=0.,
                         alpha=.5, support="union", maxiter=80, target=None,
                         initial_allocation=None, return_allocation=False):
    """Retain the smallest valid two-edge upper bound evaluated by L-BFGS.

    ``target`` is an optional feasible competing-tree gain. Stop immediately
    after a bound excludes this family, with a relative numerical guard.
    This does not imply global convergence of the convex dual optimization.
    The result must still be intersected with independent single-edge bounds.
    """
    start = perf_counter()
    if isinstance(maxiter, bool) or not isinstance(maxiter, Integral) or maxiter < 0:
        raise ValueError("maxiter must be a nonnegative integer.")
    if target is not None and (not np.isfinite(target) or target < 0):
        raise ValueError("target must be finite and nonnegative.")
    dual = JointCutDual(forcing, eigenvalues, cuts, rank, center, potential=potential,
                       alpha=alpha, support=support)
    allocation = (np.zeros(dual.variables) if initial_allocation is None
                  else np.asarray(initial_allocation, dtype=float))
    best_upper = float("inf")
    best_allocation = allocation.copy()
    trace = []
    evaluations = 0
    iterations = 0
    initial_upper = None

    class _Excluded(Exception):
        pass

    def objective(point):
        nonlocal best_upper, best_allocation, evaluations, initial_upper
        upper, gradient = dual.value_gradient(point)
        evaluations += 1
        if initial_upper is None:
            initial_upper = upper
        if upper < best_upper:
            best_upper = upper
            best_allocation = point.copy()
            trace.append(dict(evaluation=evaluations, upper=upper))
        guard = 1e-9 * max(1e-12, upper, target or 0.)
        if target is not None and upper < target - guard:
            raise _Excluded
        return upper, gradient

    def callback(_):
        nonlocal iterations
        iterations += 1

    success = False
    status = "initial evaluation only"
    try:
        objective(allocation)
        if maxiter and dual.variables:
            result = minimize(objective, allocation, jac=True, method="L-BFGS-B",
                              callback=callback,
                              options=dict(maxiter=int(maxiter), gtol=1e-11,
                                           ftol=1e-14, maxls=40, maxcor=15))
            success, status = bool(result.success), str(result.message)
    except _Excluded:
        status = "valid bound excludes this family"
    guard = 1e-9 * max(1e-12, best_upper, target or 0.)
    result = dict(upper=best_upper, initial_upper=initial_upper,
                  excluded=bool(target is not None and best_upper < target - guard),
                  numerical_guard=guard, evaluations=evaluations, iterations=iterations,
                  svd_evaluations=dual.svd_evaluations,
                  allocation_variables=dual.variables,
                  active_matrix_shapes=[list(term["shape"]) for term in dual.terms],
                  cuts=[list(cut) for cut in dual.cuts], rank=int(rank), alpha=alpha,
                  optimizer_success=success, status=status, trace=trace,
                  elapsed=perf_counter() - start, full_pde_iterations=0)
    if return_allocation:
        result["allocation"] = best_allocation
        result["support_indices"] = dual.support_indices.copy()
    return result
