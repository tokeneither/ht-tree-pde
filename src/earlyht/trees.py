"""Certified lower bounds and feasible ALS upper bounds for a matrix cut.

The upper bound is the error of an actual low-rank matrix. Alternating least
squares does not establish the globally optimal low-rank approximation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky


def _real_array(value: np.ndarray, name: str) -> np.ndarray:
    if not np.isrealobj(value):
        raise ValueError(f"{name} must be real")
    result = np.array(value, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


@dataclass
class CutProblem:
    """Target W and SPD energy A(X) = Ar X + X Ac + potential * X.

    ``energy`` returns a squared norm. Public approximation bounds are norms,
    without normalization by the target norm. Arrays are copied on creation.
    """

    matrix: np.ndarray
    left: np.ndarray
    right: np.ndarray
    potential: np.ndarray
    _left_root: np.ndarray = field(init=False, repr=False)
    _right_root: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("matrix", "left", "right", "potential"):
            setattr(self, name, _real_array(getattr(self, name), name))
        if self.matrix.ndim != 2 or min(self.matrix.shape) == 0:
            raise ValueError("matrix must be a nonempty two-dimensional array")
        m, n = self.matrix.shape
        if self.left.shape != (m, m) or self.right.shape != (n, n):
            raise ValueError("left/right dimensions must match the target matrix")
        if self.potential.shape != (m, n) or np.any(self.potential < 0):
            raise ValueError("potential must match matrix and be nonnegative")
        for name in ("left", "right"):
            value = getattr(self, name)
            if not np.allclose(value, value.T, rtol=1e-12, atol=1e-12):
                raise ValueError(f"{name} must be symmetric positive definite")
            # Store the symmetric matrix actually used by the Cholesky factor.
            value = (value + value.T) * 0.5
            setattr(self, name, value)
            try:
                root = cholesky(value, lower=False, check_finite=False)
            except np.linalg.LinAlgError as exc:
                raise ValueError(f"{name} must be symmetric positive definite") from exc
            setattr(self, f"_{name}_root", root)

    def operator(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != self.matrix.shape:
            raise ValueError("operator input must have the target matrix shape")
        return self.left @ matrix + matrix @ self.right + self.potential * matrix

    def energy(self, matrix: np.ndarray) -> float:
        matrix = np.asarray(matrix, dtype=float)
        return float(np.sum(matrix * self.operator(matrix)))


def _rank(rank: int, shape: tuple[int, int]) -> int:
    if isinstance(rank, bool) or not isinstance(rank, Integral) or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    return min(int(rank), min(shape))


def rank_lower_bound(problem: CutProblem, rank: int) -> float:
    """Lower bound on the best rank-at-most-r error in the energy norm.

    Each transformed approximation still has rank at most r. Eckart--Young
    therefore bounds the left and right energy contributions separately.
    Their sum remains a lower bound when the nonnegative potential is omitted.
    """
    rank = _rank(rank, problem.matrix.shape)
    if rank == min(problem.matrix.shape):
        return 0.0
    left_singular = np.linalg.svd(problem._left_root @ problem.matrix,
                                  compute_uv=False)
    right_singular = np.linalg.svd(problem.matrix @ problem._right_root.T,
                                   compute_uv=False)
    squared = np.sum(left_singular[rank:] ** 2) + np.sum(right_singular[rank:] ** 2)
    return float(np.sqrt(squared))


def _solve_factor(left: np.ndarray, right: np.ndarray,
                  potential: np.ndarray, target_action: np.ndarray,
                  frame: np.ndarray) -> np.ndarray:
    """Minimize error of L frame.T for an orthonormal column frame.

    C-order vectorization yields left (x) I_r + I_m (x) frame.T right
    frame, plus one r-by-r potential block for each row.
    """
    m, rank = left.shape[0], frame.shape[1]
    projected_right = frame.T @ right @ frame
    normal = np.kron(left, np.eye(rank)) + np.kron(np.eye(m), projected_right)
    blocks = np.einsum("ij,jk,jl->ikl", potential, frame, frame, optimize=True)
    for i in range(m):
        rows = slice(i * rank, (i + 1) * rank)
        normal[rows, rows] += blocks[i]
    rhs = (target_action @ frame).ravel(order="C")
    factor = cho_factor(normal, lower=True, check_finite=False)
    return cho_solve(factor, rhs, check_finite=False).reshape(m, rank)


def _als_sweep(problem: CutProblem, rhs: np.ndarray,
               rfactor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two exact conditional minimizations, ending with an orthonormal R."""
    lfactor = _solve_factor(problem.left, problem.right, problem.potential,
                            rhs, rfactor)
    lfactor, triangular = np.linalg.qr(lfactor, mode="reduced")
    # The same product can be represented with R @ triangular.T, so the
    # following conditional minimization contains the previous approximation.
    rfactor = _solve_factor(problem.right, problem.left, problem.potential.T,
                            rhs.T, lfactor)
    rfactor, triangular = np.linalg.qr(rfactor, mode="reduced")
    return lfactor @ triangular.T, rfactor


def _validate_controls(starts: int, max_sweeps: int, tol: float) -> None:
    for name, value, minimum in (("starts", starts, 1),
                                 ("max_sweeps", max_sweeps, 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")


def fit_cut(problem: CutProblem, rank: int, starts: int = 2,
            max_sweeps: int = 25, tol: float = 1e-10,
            seed: int = 0) -> dict:
    """Fit a feasible low-rank approximation using exact alternating solves.

    The first start is a truncated SVD; further starts use random orthonormal
    right frames. The best *actual* approximation is retained. ``history``
    contains energy-error norms at initialization and after accepted complete
    sweeps for that start. ``sweeps`` counts those accepted sweeps.
    No global optimality is claimed for the ALS result.
    """
    rank = _rank(rank, problem.matrix.shape)
    _validate_controls(starts, max_sweeps, tol)
    target = problem.matrix
    lower = rank_lower_bound(problem, rank)
    target_energy = max(problem.energy(target), 0.0)

    def result(approximation: np.ndarray, history: list[float],
               stationarity: float = 0.0) -> dict:
        error_squared = max(problem.energy(target - approximation), 0.0)
        return dict(lower=lower, upper=float(np.sqrt(error_squared)),
                    approximation=approximation, history=history,
                    sweeps=len(history) - 1,
                    rank_observed=int(np.linalg.matrix_rank(approximation)),
                    stationarity=stationarity)

    if rank == 0 or target_energy == 0:
        return result(np.zeros_like(target), [float(np.sqrt(target_energy))])
    if rank == min(target.shape):
        return result(target.copy(), [0.0])

    target_action = problem.operator(target)
    u, singular, vh = np.linalg.svd(target, full_matrices=False)
    rng = np.random.default_rng(seed)
    best = None
    start_uppers = []
    for start in range(starts):
        if start == 0:
            lfactor = u[:, :rank] * singular[:rank]
            rfactor = vh[:rank].T.copy()
        else:
            rfactor = np.linalg.qr(rng.standard_normal((target.shape[1], rank)),
                                   mode="reduced")[0]
            lfactor = _solve_factor(problem.left, problem.right, problem.potential,
                                    target_action, rfactor)
        approximation = lfactor @ rfactor.T
        squared = max(problem.energy(target - approximation), 0.0)
        history = [float(np.sqrt(squared))]
        for _ in range(max_sweeps):
            new_l, new_r = _als_sweep(problem, target_action, rfactor)
            candidate = new_l @ new_r.T
            new_squared = max(problem.energy(target - candidate), 0.0)
            # Reject numerical increases rather than altering reported errors.
            if new_squared > squared:
                break
            improvement = squared - new_squared
            lfactor, rfactor, approximation = new_l, new_r, candidate
            squared = new_squared
            history.append(float(np.sqrt(squared)))
            if improvement <= tol * max(squared, np.finfo(float).eps * target_energy):
                break
        gradient = problem.operator(approximation - target)
        left_frame = np.linalg.qr(lfactor, mode="reduced")[0]
        stationarity = float(np.sqrt(np.sum((gradient @ rfactor) ** 2)
                                    + np.sum((gradient.T @ left_frame) ** 2)))
        candidate_result = result(approximation.copy(), history, stationarity)
        start_uppers.append(candidate_result["upper"])
        if best is None or candidate_result["upper"] < best["upper"]:
            best = candidate_result
    best["start_uppers"] = start_uppers
    return best


def solve_cut(left: np.ndarray, right: np.ndarray, potential: np.ndarray,
              rhs: np.ndarray, rank: int, initial: np.ndarray | None = None,
              starts: int = 2, max_sweeps: int = 50, tol: float = 1e-10,
              seed: int = 0) -> dict:
    """Minimize J(X)=energy(X)/2-<rhs,X> in a rank-constrained matrix space.

    Only the operator and forcing enter this solver. It neither requires nor
    constructs a reference solution. ``initial`` is an optional proxy matrix;
    its truncated SVD initializes the first start. Otherwise SVD(rhs) is used.
    Further starts use random orthonormal right frames. ``history`` records
    actual J values, including initialization, for the best retained start.
    Exact block solves give a feasible local result, not global optimality.
    """
    rhs = _real_array(rhs, "rhs")
    problem = CutProblem(np.zeros_like(rhs), left, right, potential)
    rank = _rank(rank, rhs.shape)
    _validate_controls(starts, max_sweeps, tol)
    initial = rhs if initial is None else _real_array(initial, "initial")
    if initial.shape != rhs.shape:
        raise ValueError("initial must have the same shape as rhs")

    def objective(matrix):
        return float(0.5 * problem.energy(matrix) - np.sum(rhs * matrix))

    def result(approximation, history):
        return dict(approximation=approximation, objective=objective(approximation),
                    history=history, sweeps=len(history) - 1,
                    rank_observed=int(np.linalg.matrix_rank(approximation)))

    if rank == 0 or not np.any(rhs):
        return result(np.zeros_like(rhs), [0.0])
    u, singular, vh = np.linalg.svd(initial, full_matrices=False)
    rng = np.random.default_rng(seed)
    scale = max(float(np.sum(rhs ** 2)), np.finfo(float).tiny)
    best = None
    start_objectives = []
    for start in range(starts):
        if start == 0:
            lfactor = u[:, :rank] * singular[:rank]
            rfactor = vh[:rank].T.copy()
        else:
            rfactor = np.linalg.qr(rng.standard_normal((rhs.shape[1], rank)),
                                   mode="reduced")[0]
            lfactor = _solve_factor(problem.left, problem.right, problem.potential,
                                    rhs, rfactor)
        approximation = lfactor @ rfactor.T
        value = objective(approximation)
        history = [value]
        for _ in range(max_sweeps):
            new_l, new_r = _als_sweep(problem, rhs, rfactor)
            candidate = new_l @ new_r.T
            new_value = objective(candidate)
            if new_value > value:
                break
            improvement = value - new_value
            lfactor, rfactor, approximation = new_l, new_r, candidate
            value = new_value
            history.append(value)
            if improvement <= tol * max(abs(value), np.finfo(float).eps * scale):
                break
        candidate_result = result(approximation.copy(), history)
        gradient = problem.operator(approximation) - rhs
        left_frame = np.linalg.qr(lfactor, mode="reduced")[0]
        candidate_result["stationarity"] = float(np.sqrt(
            np.sum((gradient @ rfactor) ** 2) + np.sum((gradient.T @ left_frame) ** 2)))
        start_objectives.append(candidate_result["objective"])
        if best is None or candidate_result["objective"] < best["objective"]:
            best = candidate_result
    best["start_objectives"] = start_objectives
    return best
