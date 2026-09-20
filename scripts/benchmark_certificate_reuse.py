#!/usr/bin/env python3
"""Matched offline/online study of parameter-region certificate reuse.

No method sees evaluation-oracle results. Fixed-operator competitors may solve
all affine forcing directions offline. The same reuse transport is attached
to direct and solve-first certificates. Heuristics reuse trees and candidates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from earlyht.direct_full import prepare_direct_operator, select_direct_full
from earlyht.parametric import (NAMES, correlated_load_problem, parameter_sequence,
    AffineSolutionBaseline, all_cut_rounding, fit_selected)
from earlyht.reuse import (CertificateLibrary, parameter_gram,
                           feasible_direction_correlations)
from earlyht.spectral_baselines import SeparableSpectralInverse


ROOT = Path(__file__).resolve().parents[1]
METHODS = ('direct_fresh', 'direct_reuse', 'offline_certificate_fresh',
           'offline_certificate_reuse', 'offline_round_all', 'offline_warm_tree')


def run_sequence(problem, basis, parameters, method, *, use_scaling=True):
    start = perf_counter()
    direct = method.startswith('direct')
    reused = method.endswith('reuse')
    heuristic = method in ('offline_round_all','offline_warm_tree')
    operator_cache = prepare_direct_operator(problem,metric_policy='compact') if direct else None
    offline = None if direct else AffineSolutionBaseline(problem,basis)
    if not direct and not heuristic:
        offline.prepare_certificate()
    library = None
    if reused:
        inverse_minorant = SeparableSpectralInverse(problem,float(np.min(problem.potential)))
        gram = parameter_gram(basis,apply_inverse_minorant=inverse_minorant.apply)
        library = CertificateLibrary(gram,len(NAMES),homogeneous_scaling=use_scaling)
    setup = perf_counter()-start
    records = []
    candidate_cache = {}
    previous_index = None
    previous_refresh_error = None
    for query_index,theta in enumerate(parameters):
        query_start = perf_counter()
        used_reuse = False
        selected_initial = None
        decision = library.query(theta) if library is not None else None
        assembled = False
        refreshed = False
        if decision is not None and decision['certified']:
            index = decision['selected_index']
            used_reuse = True
            lower,upper = decision['lower'],decision['upper']
            certified = True
        else:
            problem.rhs[:] = np.tensordot(theta,basis,axes=1)
            assembled = True
            if direct:
                fresh = select_direct_full(problem,rank=1,sweeps=(0,2,6),
                    metric_policy='compact',dual_refinements=12,
                    cache=operator_cache,return_approximation=True)
                index = NAMES.index(fresh['selected'])
                lower = [fresh['bounds'][name]['lower'] for name in NAMES]
                upper = [fresh['bounds'][name]['upper'] for name in NAMES]
                selected_initial = fresh['approximation']
                certified = fresh['certified']
            elif not heuristic:
                fresh = offline.certificate(theta)
                index = fresh['selected_index']
                lower,upper = fresh['lower'],fresh['upper']
                selected_initial = fresh['solution']
                certified = fresh['certified']
            else:
                # A cheap energy monitor is available from the offline affine
                # Gram matrix, so a heuristic can trigger structural refreshes.
                refreshed = method=='offline_round_all' or previous_index is None or query_index%16==0
                index = previous_index
                lower,upper = [0.]*len(NAMES),[None]*len(NAMES)
                certified = False
                if refreshed:
                    index,selected_initial,_ = all_cut_rounding(problem,offline.solution(theta))
        selection_seconds = perf_counter()-query_start
        fit_start = perf_counter()
        deferred_selection_seconds = 0.
        if not assembled:
            problem.rhs[:] = np.tensordot(theta,basis,axes=1)
        if selected_initial is None:
            selected_initial = candidate_cache.get(index)
        candidate,sweeps = fit_selected(problem,index,selected_initial,sweeps=6)
        gain = float(np.sum(problem.rhs*candidate))
        if heuristic:
            energy_lower,energy_upper,_ = offline.energy_interval(theta)
            error = max(0.,1-gain/max(energy_upper,1e-300))
            if (not refreshed and previous_refresh_error is not None
                    and error > 1.10*max(previous_refresh_error,1e-8)):
                # Count the failed warm fit, complete refresh and replacement fit.
                refresh_start = perf_counter()
                index,selected_initial,_ = all_cut_rounding(problem,offline.solution(theta))
                deferred_selection_seconds += perf_counter()-refresh_start
                candidate,extra = fit_selected(problem,index,selected_initial,sweeps=6)
                sweeps += extra
                gain = float(np.sum(problem.rhs*candidate))
                error = max(0.,1-gain/max(energy_upper,1e-300))
                refreshed = True
            if refreshed:
                previous_refresh_error = error
            lower[index] = gain
        fit_seconds = perf_counter()-fit_start-deferred_selection_seconds
        selection_seconds += deferred_selection_seconds
        candidate_cache[index] = candidate
        previous_index = index
        update_start = perf_counter()
        if not heuristic:
            lower[index] = max(lower[index],gain)
            scale = max(1e-12,*upper)
            if gain>upper[index]+1e-8*scale:
                raise ArithmeticError('Actual candidate gain violates its family upper bound')
            if library is not None and not used_reuse:
                correlations = feasible_direction_correlations(basis,candidate,apply_operator=problem.apply)
                library.add_reference(theta,lower,upper,winner=index,direction_correlations=correlations)
                combined = library.query(theta)
                if combined['certified'] and combined['selected_index']==index:
                    lower,upper = combined['lower'],combined['upper']
                    certified = True
        update_seconds = perf_counter()-update_start
        # Retained tensors are tested by their actual newly rescaled gain. The
        # transported family lower need not be attained by this tensor.
        actual_gain = float(2*np.sum(problem.rhs*candidate)-problem.energy(candidate))
        assert np.isclose(gain,actual_gain,rtol=1e-9,atol=1e-14)
        records.append(dict(query=query_index,theta=theta.tolist(),selected_index=int(index),
            selected=NAMES[index],certified=bool(certified),reuse_hit=used_reuse,
            references=len(library.references) if library else 0,tree_refresh=refreshed,
            candidate_gain=actual_gain,lower=lower,upper=upper,
            excess_energy_upper=max(0.,max(upper)-actual_gain) if not heuristic else None,
            fit_sweeps=sweeps,selection_seconds=selection_seconds+update_seconds,
            fit_seconds=fit_seconds,total_seconds=perf_counter()-query_start))
    total = perf_counter()-start
    return dict(method=method,setup_seconds=setup,total_seconds=total,
        selection_total_seconds=setup+sum(r['selection_seconds'] for r in records),
        fit_total_seconds=sum(r['fit_seconds'] for r in records),
        certified=sum(r['certified'] for r in records),
        reuse_hits=sum(r['reuse_hit'] for r in records),
        seed_selections=len(records)-sum(r['reuse_hit'] for r in records) if not heuristic else None,
        references=len(library.references) if library else 0,
        tree_refreshes=sum(r['tree_refresh'] for r in records),
        offline_basis_solves=len(basis) if offline else 0,
        offline_basis_iterations=offline.iterations if offline else [],
        full_solution_formed=not direct,records=records)


def validate_case(problem,basis,parameters,runs):
    """Independent post-timing energy diagnostics and cross-method checks."""
    oracle = AffineSolutionBaseline(problem,basis,rtol=1e-12)
    summary = {key:dict(max_relative_energy_error=0.,mean_relative_energy_error=0.,
        certified_disagreements=0,max_excess_over_best_observed_relative=0.) for key in runs}
    violations = []
    per_query = []
    for q,theta in enumerate(parameters):
        elo,ehi,delta = oracle.energy_interval(theta)
        energy = .5*(elo+ehi)
        first = {method:values[0]['records'][q] for method,values in runs.items()}
        # Validate repeat consistency before using the common deterministic
        # result for cross-method quality checks. Timings may vary; decisions,
        # gains and gain intervals must not silently change across repetitions.
        for method,values in runs.items():
            reference = first[method]
            for repetition in values[1:]:
                current = repetition['records'][q]
                for key in ('selected_index','certified','reuse_hit','tree_refresh'):
                    if current[key] != reference[key]:
                        violations.append(dict(query=q,method=method,reason=f'repeated {key} differs'))
                for key in ('candidate_gain','lower','upper'):
                    if key=='upper' and reference[key][0] is None:
                        continue
                    if not np.allclose(current[key],reference[key],rtol=1e-11,atol=1e-16):
                        violations.append(dict(query=q,method=method,reason=f'repeated {key} differs'))
        winners = {r['selected_index'] for r in first.values() if r['certified']}
        if len(winners)>1:
            violations.append(dict(query=q,reason='contradictory strict certificates'))
        best = max(r['candidate_gain'] for r in first.values())
        for method,record in first.items():
            relative_error = max(0.,1-record['candidate_gain']/energy)
            entry=summary[method]
            entry['max_relative_energy_error']=max(entry['max_relative_energy_error'],relative_error)
            entry['mean_relative_energy_error']+=relative_error/len(parameters)
            entry['max_excess_over_best_observed_relative']=max(
                entry['max_excess_over_best_observed_relative'],(best-record['candidate_gain'])/energy)
            if winners and record['selected_index'] not in winners:
                entry['certified_disagreements']+=1
            if record['candidate_gain']>ehi+1e-8*energy:
                violations.append(dict(query=q,method=method,reason='gain exceeds full solution energy'))
            if record['upper'][0] is not None:
                for other in first.values():
                    if other['candidate_gain']>record['upper'][other['selected_index']]+1e-8*energy:
                        violations.append(dict(query=q,method=method,reason='upper excludes independent feasible candidate'))
        per_query.append(dict(query=q,solution_energy_lower=elo,solution_energy_upper=ehi,
            residual_radius=delta,certified_winner=next(iter(winners)) if len(winners)==1 else None,
            best_observed_gain=best))
    if violations:
        raise ArithmeticError(f'Validation failed: {violations[:4]}')
    return dict(methods=summary,queries=per_query,violations=violations,
                oracle_used_only_after_timing=True)


def summarize(runs, validation):
    result={}
    for method,values in runs.items():
        for field in ('certified','reuse_hits','seed_selections','tree_refreshes'):
            assert len({v[field] for v in values})==1,(method,field)
        reference=values[0]
        result[method]={key:median(v[key] for v in values) for key in (
            'setup_seconds','total_seconds','selection_total_seconds','fit_total_seconds')}
        result[method].update({key:reference[key] for key in (
            'certified','reuse_hits','seed_selections','references','tree_refreshes','offline_basis_solves')})
        result[method].update(validation['methods'][method])
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--n',type=int,default=8)
    parser.add_argument('--seeds',type=int,nargs='+',default=[41,67])
    parser.add_argument('--groups',nargs='+',default=['local','stable','transition'])
    parser.add_argument('--queries',type=int,default=64)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--methods',nargs='+',default=list(METHODS))
    parser.add_argument('--output',type=Path,default=ROOT/'results/generated/reuse')
    parser.add_argument('--no-scaling',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    files=['src/earlyht/parametric.py','src/earlyht/reuse.py','src/earlyht/direct_full.py',
           'src/earlyht/fast_legacy.py','scripts/benchmark_certificate_reuse.py']
    config=dict(date=datetime.now(timezone.utc).isoformat(),n=args.n,seeds=args.seeds,
        groups=args.groups,queries=args.queries,repeats=args.repeats,methods=args.methods,
        homogeneous_scaling=not args.no_scaling,source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
            platform=platform.platform(),OPENBLAS_NUM_THREADS=os.getenv('OPENBLAS_NUM_THREADS')),
        protocol='all setup and per-query fit costs included; input generator and post-timing oracle excluded')
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    all_results=[]
    for seed in args.seeds:
        for group in args.groups:
            problem,basis,metadata=correlated_load_problem(args.n,seed)
            parameters=parameter_sequence(group,args.queries,seed=10000+seed)
            runs={method:[] for method in args.methods}
            for repeat in range(args.repeats):
                order=list(args.methods)
                np.random.default_rng(20260920+repeat).shuffle(order)
                for method in order:
                    run=run_sequence(problem,basis,parameters,method,use_scaling=not args.no_scaling)
                    runs[method].append(run)
                    print(f'n={args.n} seed={seed} {group} repeat={repeat+1} {method}: '
                          f'{run["total_seconds"]:.3f}s cert={run["certified"]}/{len(parameters)} '
                          f'reuse={run["reuse_hits"]}',flush=True)
            validation=validate_case(problem,basis,parameters,runs)
            case=dict(id=f'n{args.n}_s{seed}_{group}',metadata=metadata,group=group,
                      parameters=parameters.tolist(),runs=runs,validation=validation,
                      summary=summarize(runs,validation))
            all_results.append(case)
            (args.output/'all_results.json').write_text(json.dumps(all_results,indent=2)+'\n')
    print('Completed and independently validated all sequences.',flush=True)


if __name__=='__main__':
    main()
