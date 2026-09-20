"""Full-grid variational tree selection using centered product-metric bounds.

Unlike a proxy-error certificate, this module compares attainable gains
directly from the forcing and PDE operator. It never computes a full-grid PDE
solution. Full-grid tensors and cut eigendecompositions are still materialized;
this implementation does not claim high-dimensional compressed complexity.
"""

from __future__ import annotations

from numbers import Integral
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from .direct_selection import choose_from_gains, product_metrics
from .pde import BALANCED_CUTS, matricize, unmatricize
from .potential_bounds import additive_minorants
from .selection import cut_name
from .trees import solve_cut


def _apply(state, matrix):
    return state["left"] @ matrix + matrix @ state["right"] + state["potential"] * matrix


def _scaled_feasible(state, matrix):
    pairing = float(np.sum(state["forcing"] * matrix))
    energy = float(np.sum(matrix * _apply(state, matrix)))
    if energy <= 0:
        return 0.0, np.zeros_like(matrix)
    return pairing * pairing / energy, matrix * (pairing / energy)


def _spectral_minorants(left, right, potential, include_minorants, metric_policy):
    """Cache product metrics in eigenbases of additive operator minorants."""
    groups = []
    potential_is_constant = bool(np.all(potential == potential.flat[0]))
    preferred_basis = ("balanced_average" if include_minorants and not potential_is_constant
                       else "constant")
    for p, q, label in additive_minorants(potential):
        if label != "constant" and (not include_minorants or potential_is_constant):
            continue
        # The balanced minorant is enough for an inexpensive second basis.
        if label not in ("constant", "balanced_average"):
            continue
        if metric_policy != "full" and label != preferred_basis:
            continue
        a, qr = np.linalg.eigh(left + np.diag(p))
        b, qc = np.linalg.eigh(right + np.diag(q))
        if min(a[0], b[0]) <= 0:
            raise ValueError("The additive minorant must have positive axis spectra.")
        metrics = []
        candidates = ([(np.full_like(a,float(a.min()+b.min())),np.ones_like(b),'scalar')]
                      if metric_policy == 'scalar' else product_metrics(a,b))
        for metric_left, metric_right, metric_name in candidates:
            allowed = ({'scalar'} if metric_policy == 'scalar' else
                       {"balanced_geometric"} if metric_policy == "geometric" else {
                "balanced_geometric", "power_0.25", "power_0.75"})
            if metric_policy != "full" and metric_name not in allowed:
                continue
            metric = metric_left[:, None] * metric_right[None, :]
            if np.any(metric <= 0) or not np.all(np.isfinite(metric)):
                raise ValueError("Product metrics must be finite and positive.")
            error = float(np.max(metric - a[:, None] - b[None, :]))
            if error > 1e-10 * max(1.0, float(metric.max())):
                raise ValueError("Product metric exceeds its additive minorant.")
            metrics.append((metric, np.sqrt(metric), metric_name))
        groups.append(dict(qr=qr, qc=qc, metrics=metrics, label=label))
    return groups


@dataclass
class DirectOperatorCache:
    """Reusable fixed-operator setup; changing the forcing is permitted.

    Operators/potential must remain fixed. Centers and forcing transforms are
    fresh on every call, so sharing this cache does not share primal answers.
    """
    problem: object
    potential_minorants: bool
    metric_policy: str
    states: dict
    setup_seconds: float


def prepare_direct_operator(problem, *, potential_minorants=True, metric_policy="full"):
    """Prepare the operator-only eigensystems for repeated right-hand sides."""
    if len(problem.shape) != 4:
        raise ValueError("The prototype requires four tensor modes.")
    if metric_policy not in ("full", "geometric", "compact", "scalar"):
        raise ValueError("Unknown metric policy.")
    start = perf_counter()
    states = {}
    for cut in BALANCED_CUTS:
        left, right, potential = problem.cut_operators(cut)
        states[cut_name(cut)] = dict(
            cut=cut, left=left, right=right, potential=potential,
            groups=_spectral_minorants(left, right, potential, potential_minorants, metric_policy))
    return DirectOperatorCache(problem, bool(potential_minorants), metric_policy,
                               states, perf_counter()-start)


def select_direct_full(
    problem,
    rank=1,
    sweeps=(0, 2, 6, 16),
    potential_minorants=True,
    dual_refinements=0,
    metric_policy="full",
    cache=None,
    return_approximation=False,
):
    """Compare three balanced full-leaf families in the original PDE energy.

    For every product metric B <= A and any center Y, C=A-B is positive
    semidefinite. Young's inequality gives the captured-energy upper bound

        <Y,CY> + sum_{j<=r} sigma_j(B^(-1/2) (F-CY))**2,

    where B^(-1/2) acts by rank-preserving row/column scaling in its product
    eigenbasis. Weighted truncated SVDs give feasible candidates; their gains
    are evaluated and optimally scaled in the ORIGINAL energy, including the
    entire nonseparable potential. ALS only improves feasible lower bounds.

    ``dual_refinements`` optionally sets the L-BFGS iteration budget for each
    remaining competitor's best product metric. Its center is unrestricted
    in rank; every evaluated objective remains a valid upper bound, including
    when the optimizer stops at a singular-value tie or fails to converge.
    This extra work uses full cut matrices and can be expensive.

    ``metric_policy='geometric'`` uses one geometric-mean product in the
    balanced potential-minorant basis (the constant basis when appropriate).
    ``'compact'`` uses three powers in that basis; ``'full'`` retains all
    eight products in both bases. The cheaper choices preserve validity but
    can produce weaker bounds. They are announced policies, not oracle tuning.

    Each nonzero entry in ``sweeps`` is the additional sweep budget for that
    stage. No global optimality of an ALS result is assumed. A family is
    skipped only when its retained valid upper bound is below a competitor's
    feasible gain. Bounds are floating-point evaluations, not interval proofs.

    An optional ``prepare_direct_operator`` cache reuses fixed-operator setup.
    Its cost is reported separately and must be counted in repeated-query
    benchmarks. ``return_approximation`` exposes the actual winning feasible
    tensor, rescaled to the reported lower gain; default records stay compact.
    """
    start = perf_counter()
    if len(problem.shape) != 4:
        raise ValueError("The prototype requires four tensor modes.")
    max_rank = min(min(int(np.prod([problem.shape[i] for i in cut])),
                       int(np.prod([problem.shape[i] for i in range(4) if i not in cut])))
                   for cut in BALANCED_CUTS)
    if isinstance(rank, bool) or not isinstance(rank, Integral) or not 1 <= rank <= max_rank:
        raise ValueError("rank must be a positive feasible integer for every cut.")
    sweeps = tuple(sweeps)
    if not sweeps or any(isinstance(stage, bool) or not isinstance(stage, Integral)
                         or stage < 0 for stage in sweeps):
        raise ValueError("sweeps must be a nonempty sequence of nonnegative integers.")
    if (isinstance(dual_refinements, bool) or not isinstance(dual_refinements, Integral)
            or dual_refinements < 0):
        raise ValueError("dual_refinements must be a nonnegative integer.")
    if metric_policy not in ("full", "geometric", "compact", "scalar"):
        raise ValueError("metric_policy must be full, geometric, compact, or scalar.")
    states, bounds, trace = {}, {}, []
    counters = dict(metric_evaluations=0, als_calls=0, als_sweeps=0, pruned_evaluations=0,
                    dual_calls=0, dual_iterations=0, dual_svd_evaluations=0)
    setup_included = cache is None
    if cache is None:
        cache = prepare_direct_operator(problem, potential_minorants=potential_minorants,
                                        metric_policy=metric_policy)
    elif (cache.problem is not problem or cache.potential_minorants != bool(potential_minorants)
          or cache.metric_policy != metric_policy):
        raise ValueError("Cache must belong to this fixed problem and metric policy.")
    for cut in BALANCED_CUTS:
        prepared = cache.states[cut_name(cut)]
        left, right, potential = (prepared[key] for key in ("left", "right", "potential"))
        forcing = matricize(problem.rhs, cut)
        groups = [dict(group) for group in prepared["groups"]]
        for group in groups:
            group["forcing"] = group["qr"].T @ forcing @ group["qc"]
        name = cut_name(cut)
        states[name] = dict(left=left, right=right, potential=potential,
                            forcing=forcing, center=np.zeros_like(forcing), groups=groups)
        bounds[name] = dict(lower=0.0, upper=float("inf"), upper_source=None)
    setup_seconds = perf_counter() - start

    for stage_index, stage in enumerate(sweeps):
        incumbent = max(entry["lower"] for entry in bounds.values())
        for name, state in states.items():
            if bounds[name]["upper"] < incumbent:
                counters["pruned_evaluations"] += 1
                continue
            center = state["center"]
            if stage:
                fitted = solve_cut(state["left"], state["right"], state["potential"],
                                   state["forcing"], rank, initial=center, starts=1,
                                   max_sweeps=int(stage), seed=2026)
                counters["als_calls"] += 1
                counters["als_sweeps"] += fitted["sweeps"]
                gain, candidate = _scaled_feasible(state, fitted["approximation"])
                if gain >= bounds[name]["lower"]:
                    bounds[name]["lower"] = gain
                    center = candidate
            center_action = _apply(state, center)
            best_gain, best_candidate = _scaled_feasible(state, center)
            for group_index, group in enumerate(state["groups"]):
                qr, qc = group["qr"], group["qc"]
                center_hat = qr.T @ center @ qc
                action_hat = qr.T @ center_action @ qc
                for metric_index, (metric, root_metric, metric_name) in enumerate(group["metrics"]):
                    remainder_action = action_hat - metric * center_hat
                    constant = float(np.sum(center_hat * remainder_action))
                    shifted = (group["forcing"] - remainder_action) / root_metric
                    u, singular, vh = np.linalg.svd(shifted, full_matrices=False)
                    upper = float(constant + np.sum(singular[:rank] ** 2))
                    candidate_hat = ((u[:, :rank] * singular[:rank]) @ vh[:rank]) / root_metric
                    candidate = qr @ candidate_hat @ qc.T
                    gain, candidate = _scaled_feasible(state, candidate)
                    counters["metric_evaluations"] += 1
                    if upper < bounds[name]["upper"]:
                        bounds[name]["upper"] = upper
                        bounds[name]["upper_source"] = f"{group['label']}:{metric_name}"
                        state["best_metric"] = (group_index, metric_index)
                        state["best_dual_center"] = center_hat.copy()
                    if gain > best_gain:
                        best_gain, best_candidate = gain, candidate
            state["center"] = best_candidate
            bounds[name]["lower"] = max(bounds[name]["lower"], best_gain)
            incumbent = max(incumbent, best_gain)
        decision = choose_from_gains(bounds)
        trace.append(dict(stage=int(stage), stage_index=stage_index, **decision,
                          elapsed=perf_counter() - start,
                          bounds={key: dict(value) for key, value in bounds.items()},
                          **counters))
        if decision["certified"]:
            break

    dual_status = {}
    if dual_refinements and not decision["certified"]:
        # Upper bounds of the current leader do not affect its certificate.
        # Refine threatening alternatives in descending order of their bounds.
        order = sorted(states, key=lambda name: bounds[name]["upper"], reverse=True)
        for name in order:
            decision = choose_from_gains(bounds)
            if decision["certified"]:
                break
            if name == decision["selected"]:
                continue
            threshold = max(entry["lower"] for key, entry in bounds.items() if key != name)
            if bounds[name]["upper"] < threshold - decision["numerical_guard"]:
                continue
            state = states[name]
            group_index, metric_index = state["best_metric"]
            group = state["groups"][group_index]
            metric, root_metric, metric_name = group["metrics"][metric_index]
            qr, qc = group["qr"], group["qc"]
            shape = state["forcing"].shape
            local_iterations = 0

            def remainder_action(matrix):
                physical = qr @ matrix @ qc.T
                return qr.T @ _apply(state, physical) @ qc - metric * matrix

            class _FamilyExcluded(Exception):
                pass

            def dual_objective(flat):
                center_hat = flat.reshape(shape)
                cy = remainder_action(center_hat)
                weighted = (group["forcing"] - cy) / root_metric
                u, singular, vh = np.linalg.svd(weighted, full_matrices=False)
                candidate_hat = ((u[:, :rank] * singular[:rank]) @ vh[:rank]) / root_metric
                upper = float(np.sum(center_hat * cy) + np.sum(singular[:rank] ** 2))
                counters["metric_evaluations"] += 1
                counters["dual_svd_evaluations"] += 1
                if upper < bounds[name]["upper"]:
                    bounds[name]["upper"] = upper
                    bounds[name]["upper_source"] = f"dual:{group['label']}:{metric_name}"
                    state["best_dual_center"] = center_hat.copy()
                # A dual SVD also offers a new primal candidate, but never
                # impose its rank on the center being optimized.
                gain, candidate = _scaled_feasible(state, qr @ candidate_hat @ qc.T)
                if gain > bounds[name]["lower"]:
                    bounds[name]["lower"] = gain
                    state["center"] = candidate
                current = choose_from_gains(bounds)
                competitor_gain = max(entry["lower"] for key, entry in bounds.items()
                                      if key != name)
                if bounds[name]["upper"] < competitor_gain - current["numerical_guard"]:
                    raise _FamilyExcluded
                gradient = 2 * remainder_action(center_hat - candidate_hat)
                return upper, gradient.ravel()

            def dual_callback(_):
                nonlocal local_iterations
                local_iterations += 1
                counters["dual_iterations"] += 1

            counters["dual_calls"] += 1
            try:
                fitted = minimize(
                    dual_objective, state["best_dual_center"].ravel(), jac=True,
                    method="L-BFGS-B", callback=dual_callback,
                    options=dict(maxiter=int(dual_refinements), gtol=1e-10,
                                 ftol=1e-13, maxls=40),
                )
                dual_status[name] = dict(status=str(fitted.message),
                                         optimizer_success=bool(fitted.success),
                                         iterations=local_iterations)
            except _FamilyExcluded:
                dual_status[name] = dict(status="upper bound excludes this family",
                                         optimizer_success=False,
                                         iterations=local_iterations)
            decision = choose_from_gains(bounds)
            trace.append(dict(stage="dual", family=name, **decision,
                              elapsed=perf_counter() - start,
                              bounds={key: dict(value) for key, value in bounds.items()},
                              **counters))
        decision = choose_from_gains(bounds)
    result = dict(**decision, bounds=bounds, trace=trace, rank=int(rank),
                setup_seconds=setup_seconds, elapsed=perf_counter() - start,
                cache_setup_seconds=cache.setup_seconds, setup_included=setup_included,
                full_pde_iterations=0, potential_minorants=bool(potential_minorants),
                dual_refinements=int(dual_refinements), dual_status=dual_status,
                metric_policy=metric_policy,
                **counters,
                scope="three balanced full-leaf families; discrete PDE; floating-point bounds")
    if return_approximation:
        winner = decision["selected"]
        gain, candidate = _scaled_feasible(states[winner], states[winner]["center"])
        if not np.isclose(gain, bounds[winner]["lower"], rtol=1e-9, atol=1e-14):
            raise ArithmeticError("Retained candidate does not attain the reported lower gain.")
        result["approximation"] = unmatricize(candidate, problem.shape, cache.states[winner]["cut"])
    return result
