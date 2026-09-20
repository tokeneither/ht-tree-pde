"""Reference-free tree comparison using variational gains and spectral tails.

An experimental second-generation selector. Every feasible primal value is a
lower bound on captured energy; product-metric conjugates give upper bounds.
For V >= min(V), a separable operator minorant also controls omitted modes.
No reference solution or full-grid PDE iteration enters this module.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np

from .pde import BALANCED_CUTS, matricize, _mode_product
from .selection import cut_name
from .trees import solve_cut
from .spectral_baselines import SeparableSpectralInverse


def spectral_data(problem, modes):
    if not 1 <= modes <= min(problem.shape):
        raise ValueError('modes must lie between 1 and the smallest grid dimension')
    vmin = float(np.min(problem.potential))
    spectral = SeparableSpectralInverse(problem, vmin)
    eigenvalues = spectral.eigenvalues
    eigenvectors = []
    for n, vectors in zip(problem.shape, spectral.eigenvectors):
        if vectors is None:
            frequencies = np.arange(1,n+1)
            vectors = np.sqrt(2/(n+1))*np.sin(np.pi*frequencies[:,None]*frequencies[None,:]/(n+1))
        eigenvectors.append(vectors)
    forcing = spectral.forward(problem.rhs)
    denominator = spectral.denominator
    retained = tuple(slice(0, modes) for _ in problem.shape)
    reduced = forcing[retained].copy()
    # Exact spectral tail for the separable minorant, accumulated without
    # subtracting two close large energies. No solution tensor is constructed.
    tail_density = forcing**2/denominator
    tail_density[retained] = 0.
    tail = float(np.sum(tail_density))
    return dict(forcing=reduced, eigenvalues=eigenvalues, eigenvectors=eigenvectors,
                tail=tail, modes=modes, vmin=vmin)


def product_metrics(a, b):
    """Positive separable diagonal metrics below a_i+b_j, in spectral space."""
    diagonal = a[:, None]+b[None, :]
    shift = (float(b.min())-float(a.min()))/2
    yield 2*np.sqrt(a+shift), np.sqrt(b-shift), 'balanced_geometric'
    for theta in (0., .25, .5, .75, 1.):
        left, right = a**theta, b**(1-theta)
        scale = float(np.min(diagonal/(left[:, None]*right[None, :])))
        yield left*scale, right, f'power_{theta:g}'
    for shift in (float(b.min()), float(np.sqrt(b.min()*b.max())), float(b.max())):
        left = a+shift
        right = np.min(diagonal/left[:, None], axis=0)
        yield left, right, f'row_shift_{shift:g}'


def metric_bound(forcing, a, b, left, right, rank, center=None):
    """Centered Fenchel upper bound, and a feasible weighted-SVD candidate."""
    metric = left[:, None]*right[None, :]
    remainder = a[:, None]+b[None, :]-metric
    if remainder.min() < -1e-10*max(1., float(metric.max())):
        raise ValueError('product metric is not an operator minorant')
    # Rounding-sized negative entries do not get silently turned into a
    # different operator. The final comparison includes a numerical guard.
    center = np.zeros_like(forcing) if center is None else center
    shifted = forcing-remainder*center
    weighted = shifted/np.sqrt(metric)
    u, s, vh = np.linalg.svd(weighted, full_matrices=False)
    upper = float(np.sum(s[:rank]**2)+np.sum(remainder*center**2))
    candidate = ((u[:, :rank]*s[:rank]) @ vh[:rank])/np.sqrt(metric)
    return upper, candidate


def scaled_gain(forcing, diagonal, candidate):
    pairing = float(np.sum(forcing*candidate))
    energy = float(np.sum(diagonal*candidate**2))
    if energy <= 0:
        return 0., np.zeros_like(candidate)
    scale = pairing/energy
    return pairing**2/energy, scale*candidate


def full_feasible_gain(problem, data, cut, candidate, rank):
    """Evaluate a reduced spectral candidate in the ORIGINAL full PDE energy."""
    modes = data['modes']
    other = tuple(i for i in range(4) if i not in cut)
    qr = np.kron(*(data['eigenvectors'][axis][:, :modes] for axis in cut))
    qc = np.kron(*(data['eigenvectors'][axis][:, :modes] for axis in other))
    # Candidate has low matrix rank: transform its factors instead of a dense
    # full tensor-product interpolation matrix.
    u, s, vh = np.linalg.svd(candidate, full_matrices=False)
    if not len(s) or s[0] == 0:
        return 0.
    active = np.arange(rank)
    candidate = (u[:,active]*s[active]) @ vh[active]
    physical = ((qr @ u[:, active])*s[active]) @ (qc @ vh[active].T).T
    a = (data['eigenvalues'][cut[0]][:modes,None]+data['eigenvalues'][cut[1]][None,:modes]).ravel()
    b = (data['eigenvalues'][other[0]][:modes,None]+data['eigenvalues'][other[1]][None,:modes]).ravel()
    energy = float(np.sum((a[:,None]+b[None,:])*candidate**2)
                   + np.sum(matricize(problem.potential,cut)*physical**2))
    pairing = float(np.sum(matricize(data['forcing'],cut)*candidate))
    return pairing**2/energy if energy > 0 else 0.


def choose_from_gains(bounds):
    winner = max(bounds, key=lambda key: (bounds[key]['lower'], key))
    lower = bounds[winner]['lower']
    alternatives = [entry['upper'] for key, entry in bounds.items() if key != winner]
    margin = lower-max(alternatives)
    guard = 1e-9*max(1e-12, *(entry['upper'] for entry in bounds.values()))
    return dict(selected=winner, certified=bool(margin > guard), margin=margin,
                numerical_guard=guard,
                squared_error_regret_bound=max(0., max(entry['upper'] for entry in bounds.values())-lower))


def select_direct(problem, rank=1, modes=4, sweeps=(0, 2, 6, 16)):
    """Try spectral conjugate bounds, then centered bounds from small ALS fits."""
    if len(problem.shape) != 4:
        raise ValueError('this prototype compares three balanced four-mode trees')
    if not 1 <= rank <= modes**2:
        raise ValueError('invalid separation rank')
    start = perf_counter()
    data = spectral_data(problem, modes)
    setup_seconds = perf_counter()-start
    states, bounds, trace = {}, {}, []
    for cut in BALANCED_CUTS:
        name = cut_name(cut)
        other = tuple(i for i in range(4) if i not in cut)
        a = (data['eigenvalues'][cut[0]][:modes, None]+data['eigenvalues'][cut[1]][None, :modes]).ravel()+data['vmin']/2
        b = (data['eigenvalues'][other[0]][:modes, None]+data['eigenvalues'][other[1]][None, :modes]).ravel()+data['vmin']/2
        f = matricize(data['forcing'], cut)
        states[name] = dict(cut=cut, a=a, b=b, f=f, center=np.zeros_like(f),
                            metrics=list(product_metrics(a,b)))
        bounds[name] = dict(lower=0., upper=float('inf'), reduced_upper=float('inf'))

    constant_potential = np.all(problem.potential == data['vmin'])
    for stage in sweeps:
        incumbent = max(entry['lower'] for entry in bounds.values())
        for name, state in states.items():
            if bounds[name]['upper'] < incumbent:
                continue
            a,b,f = state['a'],state['b'],state['f']
            diagonal = a[:, None]+b[None, :]
            if stage:
                fitted = solve_cut(np.diag(a), np.diag(b), np.zeros_like(f), f,
                                   rank, initial=state['center'], starts=1, max_sweeps=stage,
                                   seed=2026)
                state['center'] = fitted['approximation']
            best_reduced, best_candidate = scaled_gain(f,diagonal,state['center'])
            upper = bounds[name]['reduced_upper']
            for left, right, _ in state['metrics']:
                bound, candidate = metric_bound(f,a,b,left,right,rank,state['center'])
                upper = min(upper,bound)
                gain, candidate = scaled_gain(f,diagonal,candidate)
                if gain > best_reduced:
                    best_reduced,best_candidate = gain,candidate
            state['center'] = best_candidate
            lower = best_reduced if constant_potential else full_feasible_gain(problem,data,state['cut'],best_candidate,rank)
            bounds[name] = dict(lower=max(bounds[name]['lower'],lower),
                                upper=upper+data['tail'], reduced_upper=upper)
            incumbent = max(incumbent,lower)
        decision = choose_from_gains(bounds)
        trace.append(dict(stage=stage,**decision,bounds={key:dict(value) for key,value in bounds.items()}))
        if decision['certified']:
            break
    return dict(**decision,bounds=bounds,trace=trace,rank=rank,modes=modes,
                spectral_tail_bound=data['tail'],setup_seconds=setup_seconds,
                elapsed=perf_counter()-start,full_pde_iterations=0,
                scope='three balanced full-leaf families, fixed discrete PDE; floating-point bounds')
