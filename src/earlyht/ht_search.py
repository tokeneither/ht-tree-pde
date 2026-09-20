"""Certified screening among complete balanced HT trees in a small dense space."""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from time import perf_counter
import numpy as np

from .ht_primal import ht_round, tree_label, leaves
from .ht_energy import fit_ht_energy
from .ht_bounds import CutBoundCache, canonical_tree_cuts, ht_gain_bounds
from .certificates import excess_energy_certificate


@lru_cache(None)
def all_balanced_trees(axes):
    """Canonical unordered-child trees with equally sized child leaf sets."""
    axes=tuple(sorted(axes))
    if len(axes)==1:
        return (axes[0],)
    if len(axes)%2:
        raise ValueError('the exhaustive benchmark uses powers of two dimensions')
    result=[]
    for rest in combinations(axes[1:],len(axes)//2-1):
        first=(axes[0],)+rest
        second=tuple(i for i in axes if i not in first)
        for left in all_balanced_trees(first):
            for right in all_balanced_trees(second):
                result.append((left,right))
    return tuple(result)


def prescribed_ht_forcing(shape,tree,rank=2,seed=31):
    """Random nested source coefficients; no target PDE solution is constructed."""
    rng=np.random.default_rng(seed)
    def frame(node):
        if isinstance(node,int):
            return np.linalg.qr(rng.standard_normal((shape[node],rank)),mode='reduced')[0]
        left,right=frame(node[0]),frame(node[1])
        product=np.kron(left,right)
        transfer=np.linalg.qr(rng.standard_normal((product.shape[1],rank)),mode='reduced')[0]
        return product@transfer
    left,right=frame(tree[0]),frame(tree[1])
    # Equal root singular values avoid an accidentally almost rank-one source.
    root=np.eye(rank)/np.sqrt(rank)
    order=leaves(tree)
    forcing=(left@root@right.T).reshape(tuple(shape[i] for i in order)).transpose(np.argsort(order))
    return forcing/np.linalg.norm(forcing)


def _refine_survivors(forcing,eigenvalues,rank,trees,cuts,cache,lower,upper,center,
                      maxiter,max_survivors):
    """Couple two edges only after individual bounds leave a small shortlist."""
    from .ht_joint_dual import optimize_joint_bound
    winner=int(np.argmax(lower))
    active=[i for i in range(len(trees)) if i!=winner and upper[i]>=lower[winner]]
    records=[]
    if not maxiter or not 0<len(active)<=max_survivors:
        return records
    for index in sorted(active,key=lambda i:-upper[i]):
        root=tuple(sorted(leaves(trees[index][0])))
        complement=tuple(i for i in range(forcing.ndim) if i not in root)
        root=min((root,complement),key=lambda cut:(len(cut),cut))
        others=sorted((cut for cut in cuts[index] if cut!=root),
                      key=lambda cut:cache.upper_for_cut(cut)['upper'])[:2]
        for other in others:
            for alpha in (.5,.8,.95):
                joint=optimize_joint_bound(forcing,eigenvalues,(root,other),rank,
                    center=center,alpha=alpha,maxiter=maxiter,target=float(lower[winner]))
                upper[index]=min(upper[index],joint['upper'])
                records.append(dict(tree_index=index,**joint))
                if joint['excluded']:
                    break
            if upper[index]<lower[winner]-1e-9*max(1e-12,lower[winner]):
                break
    return records


def search_ht(forcing,eigenvalues,rank=2,stages=(0,2,5,10),fits_per_stage=8,als_sweeps=2,
              initial='product',joint_maxiter=40,joint_max_survivors=6,
              joint_schedule='early',excess_energy_tolerance=None,
              return_approximation=False):
    """Eliminate trees with valid gain bounds and fit only surviving families.

    The optional refinement couples a root cut to up to two other
    restrictive cuts. Its fixed allocation fractions were chosen on pilot
    cases; fresh-seed experiments evaluate this same announced policy. The
    early policy attempts it once per proxy stage as soon as the shortlist
    is small, before spending the remaining budget on additional HT fits.
    The late policy retains the original end-of-search refinement ablation.

    An optional absolute ``excess_energy_tolerance`` allows stopping when
    the actual best candidate has squared energy error at most this much
    above the global optimum over all prescribed families. This can succeed
    on a tie without certifying a unique family. ``certified`` retains its
    strict-family meaning; ``epsilon_certified`` reports the separate test.
    The best rescaled tensor is returned only if ``return_approximation`` is
    true, so the default benchmark records remain JSON serializable. The
    excess-energy decision uses a floating-point guard, not interval arithmetic.
    """
    start=perf_counter()
    if joint_schedule not in ('early','late'):
        raise ValueError('joint_schedule must be early or late')
    if excess_energy_tolerance is not None and (
            not np.isfinite(excess_energy_tolerance) or excess_energy_tolerance < 0):
        raise ValueError('excess_energy_tolerance must be finite and nonnegative')
    trees=all_balanced_trees(tuple(range(forcing.ndim)))
    cache=CutBoundCache(forcing,eigenvalues,rank)
    if initial not in ('product','forcing'):
        raise ValueError('initial must be product or forcing')
    target=forcing.copy()
    inverse_product=np.ones_like(forcing)
    if initial=='product':
        # Weighted AM-GM: d product_i (A_i+s_i I)^(1/d) <= sum_i A_i,
        # with shifted minima equal and sum_i s_i=0. Its inverse acts modewise
        # and preserves every HT edge rank, unlike the full PDE inverse.
        mean_minimum=sum(float(np.min(a)) for a in eigenvalues)/len(eigenvalues)
        inverse_product/=len(eigenvalues)
        for axis,values in enumerate(eigenvalues):
            factor=(np.asarray(values)+mean_minimum-float(np.min(values)))**(-1/len(eigenvalues))
            shape=[1]*forcing.ndim;shape[axis]=len(values)
            inverse_product*=factor.reshape(shape)
        target=forcing*inverse_product
        cache.add_center(target)
    cuts=[]
    for tree in trees:
        all_cuts=canonical_tree_cuts(tree,forcing.ndim)
        cuts.append(tuple(cut for cut in all_cuts
                          if min(int(np.prod([forcing.shape[i] for i in cut])),
                                 forcing.size//int(np.prod([forcing.shape[i] for i in cut])))>rank))
        if not cuts[-1]:
            # Even an unconstrained family needs a finite gain bound.
            cuts[-1]=(all_cuts[0],)
    lower=np.zeros(len(trees));upper=np.full(len(trees),float('inf'))
    proxy=target.copy() if initial=='product' else np.zeros_like(forcing)
    if initial=='product':
        # Diagonal spectral action preserves exact forcing support. A support-
        # restricted step length is valid here; no continuum support claim.
        active=np.abs(forcing)>0
        spectrum=(cache.diagonal*inverse_product)[active]
        omega=2/(float(spectrum.min())+float(spectrum.max())) if spectrum.size else 1.
    else:
        omega=2/(float(cache.diagonal.min())+float(cache.diagonal.max()))
    steps=0;fits=0;trace=[];joint_records=[];joint_stages=set()
    best_approximation=None

    def finish(reason):
        winner=int(np.argmax(lower))
        competing=max((upper[j] for j in range(len(trees)) if j!=winner),
                      default=-float('inf'))
        margin=float(lower[winner]-competing)
        guard=1e-9*max(1e-12,float(np.max(upper)))
        quality=excess_energy_certificate(lower[winner],upper,
                                         tolerance=excess_energy_tolerance)
        result=dict(certified=margin>guard,selected=tree_label(trees[winner]),
            selected_index=winner,margin=margin,termination_reason=reason,
            trees=len(trees),tree_fits=fits,proxy_operator_steps=steps,
            bound_svds=cache.svd_evaluations+sum(item['svd_evaluations'] for item in joint_records),
            elapsed=perf_counter()-start,trace=trace,lower=lower.tolist(),upper=upper.tolist(),
            input_entries=forcing.size,rank=rank,joint_refinements=joint_records,
            joint_schedule=joint_schedule,**quality)
        if return_approximation:
            result['approximation']=(best_approximation.copy() if best_approximation is not None
                                     else np.zeros_like(forcing))
        return result
    for stage in stages:
        while steps<stage:
            proxy += omega*inverse_product*(forcing-cache.diagonal*proxy)
            steps+=1
        if stage:
            cache.add_center(proxy)
        for i,edges in enumerate(cuts):
            if upper[i]>=lower.max():
                upper[i]=min(upper[i],*(cache.upper_for_cut(cut)['upper'] for cut in edges))
        visited=set()
        for _ in range(fits_per_stage):
            available=[i for i in range(len(trees)) if i not in visited and upper[i]>=lower.max()]
            if not available:
                break
            index=max(available,key=lambda i:(upper[i],-i))
            visited.add(index)
            rounded=ht_round(proxy if stage else target,trees[index],rank,diagnostics=False)
            fit=fit_ht_energy(forcing,cache.diagonal,rounded,max_sweeps=als_sweeps,diagnostics=False)
            bound=ht_gain_bounds(forcing,eigenvalues,trees[index],rank,fit['approximation'],
                                 cache=cache,centered_cuts=1)
            best_gain=float(np.max(lower))
            if return_approximation and (bound['lower']>best_gain or
                    (bound['lower']==best_gain and index<=int(np.argmax(lower)))):
                candidate=fit['approximation']
                energy=float(np.sum(cache.diagonal*candidate**2))
                pairing=float(np.sum(forcing*candidate))
                best_approximation=candidate*(pairing/energy) if energy>0 else np.zeros_like(forcing)
            lower[index]=max(lower[index],bound['lower'])
            upper[index]=min(upper[index],bound['upper'])
            fits+=1
            winner=int(np.argmax(lower))
            # Check an allowed error budget before paying for joint refinement.
            if excess_energy_tolerance is not None:
                quality=excess_energy_certificate(lower[winner],upper,
                                                 tolerance=excess_energy_tolerance)
                if quality['epsilon_certified']:
                    trace.append(dict(stage=stage,tree_index=index,fit_gain=bound['lower'],
                        winner=winner,excess_energy_bound=quality['excess_energy_bound'],
                        elapsed=perf_counter()-start))
                    return finish('excess_energy')
            if joint_schedule=='early' and stage not in joint_stages:
                refinements=_refine_survivors(forcing,eigenvalues,rank,trees,cuts,cache,
                    lower,upper,proxy,joint_maxiter,joint_max_survivors)
                if refinements:
                    joint_stages.add(stage)
                    joint_records.extend(refinements)
            competing=max((upper[j] for j in range(len(trees)) if j!=winner),
                          default=-float('inf'))
            margin=float(lower[winner]-competing)
            guard=1e-9*max(1e-12,float(np.max(upper)))
            trace.append(dict(stage=stage,tree_index=index,fit_gain=bound['lower'],winner=winner,
                              margin=margin,pruned=int(np.sum(upper<lower[winner]-guard)),
                              elapsed=perf_counter()-start))
            if margin>guard:
                return finish('strict_family')
            if excess_energy_tolerance is not None:
                quality=excess_energy_certificate(lower[winner],upper,
                                                 tolerance=excess_energy_tolerance)
                if quality['epsilon_certified']:
                    return finish('excess_energy')
    winner=int(np.argmax(lower))
    if joint_schedule=='late':
        joint_records.extend(_refine_survivors(forcing,eigenvalues,rank,trees,cuts,cache,
            lower,upper,proxy,joint_maxiter,joint_max_survivors))
    result=finish('budget')
    if result['certified']:
        result['termination_reason']='strict_family'
    elif result['epsilon_certified']:
        result['termination_reason']='excess_energy'
    return result
