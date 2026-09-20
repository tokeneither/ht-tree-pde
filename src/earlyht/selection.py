"""Tree selection only sees a proxy and its certified error bound.

This module has no reference-solution argument or reference-solver import.
All bounds are absolute energy norms for one fixed finite-dimensional PDE.
"""
from __future__ import annotations

from time import perf_counter
import math
import numpy as np

from .pde import BALANCED_CUTS, matricize
from .trees import CutProblem, fit_cut, rank_lower_bound
from .potential_bounds import minorant_lower_bound


def cut_name(cut):
    other = tuple(i for i in range(4) if i not in cut)
    return ''.join(str(i + 1) for i in cut) + '|' + ''.join(str(i + 1) for i in other)


def compare_bounds(bounds, delta):
    """Certify the min-upper-bound candidate, with a floating-point guard.

    The theorem is deterministic in exact arithmetic. The guard reduces
    floating-point false positives but is not interval arithmetic certification.
    """
    if delta < 0 or not math.isfinite(delta):
        raise ValueError('delta must be finite and nonnegative')
    names = list(bounds)
    if not names:
        raise ValueError('at least one candidate is required')
    for entry in bounds.values():
        if not (0 <= entry['lower'] <= entry['upper'] + 1e-9*max(1., entry['upper'])):
            raise ValueError('invalid lower/upper bounds')
    winner = min(names, key=lambda name: (bounds[name]['upper'], name))
    upper = bounds[winner]['upper']
    alternatives = [bounds[name]['lower'] for name in names if name != winner]
    margin = min(alternatives) - upper - 2*delta if alternatives else float('inf')
    scale = max(1., delta, *(entry['upper'] for entry in bounds.values()))
    guard = 1e-9 * scale
    error_upper = upper + delta
    error_lower = max(0., min(entry['lower'] for entry in bounds.values()) - delta)
    return dict(selected=winner, certified=bool(margin > guard), margin=float(margin),
                numerical_guard=guard, selected_error_upper=error_upper,
                best_error_lower=error_lower,
                additive_regret_bound=max(0., error_upper-error_lower))


def select_tree(problem, proxy, delta, rank, *, starts=2, max_sweeps=25, seed=0,
                strategy='als', potential_bound='discard'):
    if len(problem.shape) != 4:
        raise ValueError('this experiment supports four-variable balanced trees')
    start = perf_counter()
    if strategy not in ('als','screened'):
        raise ValueError('strategy must be als or screened')
    if potential_bound not in ('discard','minorants'):
        raise ValueError('potential_bound must be discard or minorants')
    bounds = {}
    cut_problems = {}
    for index, cut in enumerate(BALANCED_CUTS):
        left, right, potential = problem.cut_operators(cut)
        cp = CutProblem(matricize(proxy, cut), left, right, potential)
        name = cut_name(cut)
        cut_problems[name] = (cp,index)
        if strategy == 'als':
            fit = fit_cut(cp, rank, starts=starts, max_sweeps=max_sweeps, seed=seed+index)
            bounds[name] = dict(
                lower=float(fit['lower']), upper=float(fit['upper']),
                sweeps=int(fit['sweeps']), rank_observed=int(fit['rank_observed']),
                history=[float(x) for x in fit['history']],source='als')
        else:
            u,s,vh = np.linalg.svd(cp.matrix,full_matrices=False)
            approx = (u[:,:rank]*s[:rank]) @ vh[:rank]
            upper = float(np.sqrt(max(0.,cp.energy(cp.matrix-approx))))
            bounds[name] = dict(lower=rank_lower_bound(cp,rank),upper=upper,sweeps=0,
                                rank_observed=int(np.linalg.matrix_rank(approx)),
                                history=[upper],source='svd')
        if potential_bound == 'minorants':
            bounds[name]['lower'] = minorant_lower_bound(cp,rank)
    decision = compare_bounds(bounds, delta)
    refinements = 0
    if strategy == 'screened' and not decision['certified']:
        eligible = [name for name in bounds
            if min(bounds[other]['lower'] for other in bounds if other!=name)
               -bounds[name]['lower']-2*delta > decision['numerical_guard']]
        if len(eligible)>1:
            raise AssertionError('at most one lower-bound interval can be separated')
        for name in eligible:
            cp,index = cut_problems[name]
            fit = fit_cut(cp,rank,starts=starts,max_sweeps=max_sweeps,seed=seed+index)
            # fit_cut contains the truncated-SVD initialization as a feasible
            # candidate, so it cannot intentionally worsen this upper bound.
            if fit['upper'] <= bounds[name]['upper']:
                bounds[name] = dict(lower=max(bounds[name]['lower'],float(fit['lower'])),upper=float(fit['upper']),
                    sweeps=int(fit['sweeps']),rank_observed=int(fit['rank_observed']),
                    history=[float(x) for x in fit['history']],source='als_refinement')
            refinements += 1
        decision = compare_bounds(bounds,delta)
    return dict(**decision, bounds=bounds, strategy=strategy, potential_bound=potential_bound,
                als_refinements=refinements,
                elapsed=perf_counter()-start)
