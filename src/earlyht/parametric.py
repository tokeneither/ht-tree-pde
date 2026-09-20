"""Fixed-operator affine PDE inputs and strong offline/online comparisons.

The source generator specifies smooth correlated loads, never target solutions.
The offline baseline solves each forcing basis vector once. It is deliberately
allowed to exploit the complete affine structure, just like certificate reuse.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from .pde import PDEProblem, BALANCED_CUTS, _dirichlet_laplacian, matricize, unmatricize
from .spectral_baselines import SeparableSpectralInverse
from .fast_legacy import prepare_fast_legacy, select_tree_fast, truncated_candidate
from .selection import cut_name
from .trees import solve_cut


NAMES = tuple(cut_name(cut) for cut in BALANCED_CUTS)


def correlated_load_problem(n=8, seed=41, potential_strength=35.):
    """Four-coordinate reaction--diffusion with three affine source channels.

    Coordinate pair interactions model correlated loads; this is an analytic
    PDE benchmark, not a measured engineering application. The continuum
    source shapes and potential are fixed by seed as n varies.
    """
    rng = np.random.default_rng(seed)
    grid = np.arange(1, n+1)/(n+1)
    x = np.meshgrid(grid, grid, grid, grid, indexing='ij', sparse=True)
    permutation = tuple(int(i) for i in rng.permutation(4))
    beta = float(rng.uniform(8., 12.))
    envelope = np.ones((n,)*4)
    for coordinate in x:
        envelope *= np.sin(np.pi*coordinate)
    basis = []
    # The factor h^2 turns Euclidean grid norms into a common quadrature
    # scaling while preserving the underlying continuum source across grids.
    for pairing in ((0,1,2,3), (0,2,1,3), (0,3,1,2)):
        a,b,c,d = (permutation[i] for i in pairing)
        basis.append(envelope*np.exp(-beta*((x[a]-x[b])**2+(x[c]-x[d])**2))/(n+1)**2)
    basis = np.asarray(basis)
    p = permutation
    potential = np.broadcast_to(5 + potential_strength*(
        1+.25*np.sin(2*np.pi*(x[p[0]]+x[p[2]]))
         +.25*np.cos(2*np.pi*(x[p[1]]-x[p[3]])))**2, (n,)*4).copy()
    diffusion = np.empty(4)
    diffusion[list(permutation)] = (1.,1.2,.8,1.1)
    problem = PDEProblem(f'correlated_n{n}_s{seed}', (n,)*4,
        tuple(a*_dirichlet_laplacian(n) for a in diffusion), potential, basis[0])
    metadata = dict(n=n, seed=seed, beta=beta, permutation=permutation,
        diffusion=diffusion.tolist(), potential_strength=potential_strength,
        source_basis_count=len(basis), rank=1, candidates=list(NAMES),
        description='Fixed reaction-diffusion operator and smooth affine correlated source channels')
    return problem, basis, metadata


def parameter_sequence(kind, count=64, seed=503):
    """Announced source-only sequences with stable regions and transitions."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.,1.,count)
    amplitude = 1.+.4*np.sin(2*np.pi*t)
    if kind == 'local':
        theta = np.column_stack((np.ones(count), .025+.01*np.sin(4*np.pi*t),
                                 .025+.01*np.cos(6*np.pi*t)))
    elif kind == 'stable':
        theta = np.column_stack((np.ones(count), .15+.10*np.sin(4*np.pi*t),
                                 .10+.07*np.cos(6*np.pi*t)))
    elif kind == 'transition':
        theta = np.column_stack((1.-t,t,.03*np.ones(count)))
    elif kind == 'scattered':
        theta = rng.dirichlet([.6,.6,.6],size=count)
    elif kind == 'amplitude':
        theta = np.tile([1.,.03,.03],(count,1))
    else:
        raise ValueError('Unknown sequence kind')
    return theta*amplitude[:,None]


class AffineSolutionBaseline:
    """Three shared-preconditioner basis solves and exact affine reconstruction.

    Basis solves stop at rtol=1e-11. Their independently recomputed residuals
    give a rigorous-in-exact-arithmetic residual radius for every combination.
    Numerical guards are not interval proofs. All preparation is timed.
    """
    def __init__(self, problem, basis, rtol=1e-11):
        start = perf_counter()
        self.problem, self.basis = problem, np.asarray(basis)
        self.preconditioner = SeparableSpectralInverse(problem, float(np.mean(problem.potential)))
        operator = problem.as_linear_operator()
        size = problem.rhs.size
        pre = LinearOperator((size,size), matvec=lambda x:self.preconditioner.apply(x.reshape(problem.shape)).ravel(),dtype=float)
        solutions, residuals, actions, iterations = [], [], [], []
        for source in self.basis:
            counter = [0]
            def callback(_):
                counter[0] += 1
            vector, status = cg(operator,source.ravel(),M=pre,rtol=rtol,atol=0.,
                                maxiter=2000,callback=callback)
            solution = vector.reshape(problem.shape)
            action = problem.apply(solution)
            residual = source-action
            if status or np.linalg.norm(residual)>rtol*np.linalg.norm(source)*1.1:
                raise ArithmeticError('Offline basis solution did not meet its true residual target')
            solutions.append(solution);residuals.append(residual);actions.append(action)
            iterations.append(counter[0])
        self.solutions = np.asarray(solutions)
        self.residuals = np.asarray(residuals)
        flat = self.solutions.reshape(len(basis),-1)
        self.energy_gram = flat@np.asarray(actions).reshape(len(basis),-1).T
        self.energy_gram = (self.energy_gram+self.energy_gram.T)/2
        self.iterations = iterations
        self.setup_seconds = perf_counter()-start
        self.legacy_cache = None

    def solution(self, theta):
        return np.tensordot(theta,self.solutions,axes=1)

    def energy_interval(self, theta):
        # Taking the norm of the actual residual combination avoids trusting
        # cancellation-sensitive tiny Gram eigenvalues.
        residual = np.tensordot(theta,self.residuals,axes=1)
        delta = float(np.linalg.norm(residual)/np.sqrt(self.problem.coercivity_lower))
        root_energy = np.sqrt(max(0.,float(theta@self.energy_gram@theta)))
        return max(root_energy-delta,0.)**2, (root_energy+delta)**2, delta

    def prepare_certificate(self):
        start = perf_counter()
        self.legacy_cache = prepare_fast_legacy(self.problem,'minorants')
        return perf_counter()-start

    def certificate(self, theta, rank=1):
        if self.legacy_cache is None:
            raise ValueError('Prepare and count certificate setup first')
        solution = self.solution(theta)
        energy_lower,energy_upper,delta = self.energy_interval(theta)
        result = select_tree_fast(self.problem,solution,delta,rank,
            strategy='screened',potential_bound='minorants',backend='svds',
            starts=1,max_sweeps=6,cache=self.legacy_cache)
        lower,upper = [],[]
        for name in NAMES:
            distance = result['bounds'][name]
            lower.append(max(0.,energy_lower-(distance['upper']+delta)**2))
            upper.append(max(0.,energy_upper-max(0.,distance['lower']-delta)**2))
        index = int(np.argmax(lower))
        margin = lower[index]-max(value for j,value in enumerate(upper) if j!=index)
        guard = 1e-9*max(1e-12,*upper)
        return dict(lower=lower,upper=upper,selected_index=index,certified=bool(margin>guard),
                    margin=margin,residual_radius=delta,solution=solution,
                    original_distance_certified=result['certified'])


def fit_selected(problem, candidate_index, initial=None, *, rank=1, sweeps=6):
    cut = BALANCED_CUTS[candidate_index]
    left,right,potential = problem.cut_operators(cut)
    fitted = solve_cut(left,right,potential,matricize(problem.rhs,cut),rank,
        initial=None if initial is None else matricize(initial,cut),
        starts=1,max_sweeps=sweeps,tol=1e-10,seed=2026)
    candidate = unmatricize(fitted['approximation'],problem.shape,cut)
    return rescale_candidate(problem,candidate),fitted['sweeps']


def rescale_candidate(problem, candidate):
    energy = problem.energy(candidate)
    pairing = float(np.sum(problem.rhs*candidate))
    return candidate*(pairing/energy) if energy>0 else np.zeros_like(candidate)


def all_cut_rounding(problem, solution, rank=1):
    candidates,gains = [],[]
    for cut in BALANCED_CUTS:
        matrix,_,_ = truncated_candidate(matricize(solution,cut),rank,'svds')
        candidate = rescale_candidate(problem,unmatricize(matrix,problem.shape,cut))
        candidates.append(candidate)
        gains.append(float(np.sum(problem.rhs*candidate)))
    index = int(np.argmax(gains))
    return index,candidates[index],gains
