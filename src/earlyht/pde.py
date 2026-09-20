"""Small SPD elliptic problems and reference-free conjugate-gradient probes.

All norms and energies in this module use the Euclidean interior-grid inner
product. Multiplication by the common cell volume is deliberately omitted.
The benchmark inputs are right-hand sides, not manufactured solution labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, cg


BALANCED_CUTS = ((0, 1), (0, 2), (0, 3))
FloatArray = NDArray[np.float64]


def _axis_order(ndim: int, cut: Iterable[int]) -> tuple[int, ...]:
    cut = tuple(cut)
    if not cut or len(cut) >= ndim or len(set(cut)) != len(cut):
        raise ValueError("A cut must be a nonempty proper subset of axes.")
    if any(axis < 0 or axis >= ndim for axis in cut):
        raise ValueError("Cut axis is out of bounds.")
    return cut + tuple(axis for axis in range(ndim) if axis not in cut)


def matricize(tensor: FloatArray, cut: Iterable[int]) -> FloatArray:
    """Unfold with cut axes first, preserving their specified order (C order)."""
    tensor = np.asarray(tensor)
    cut = tuple(cut)
    order = _axis_order(tensor.ndim, cut)
    nrow = int(np.prod([tensor.shape[axis] for axis in cut]))
    return tensor.transpose(order).reshape(nrow, -1)


def unmatricize(
    matrix: FloatArray, shape: tuple[int, ...], cut: Iterable[int]
) -> FloatArray:
    """Invert :func:`matricize`, including cuts with non-sorted axes."""
    shape, cut = tuple(shape), tuple(cut)
    order = _axis_order(len(shape), cut)
    expected = (
        int(np.prod([shape[axis] for axis in cut])),
        int(np.prod([shape[axis] for axis in order[len(cut) :]])),
    )
    matrix = np.asarray(matrix)
    if matrix.shape != expected:
        raise ValueError(f"Expected unfolding shape {expected}, got {matrix.shape}.")
    return matrix.reshape(tuple(shape[axis] for axis in order)).transpose(
        np.argsort(order)
    )


def _mode_product(operator: FloatArray, tensor: FloatArray, axis: int) -> FloatArray:
    moved = np.moveaxis(tensor, axis, 0)
    product = operator @ moved.reshape(moved.shape[0], -1)
    return np.moveaxis(product.reshape(moved.shape), 0, axis)


def _kronecker_sum(operators: Iterable[FloatArray]) -> FloatArray:
    operators = tuple(operators)
    total = int(np.prod([operator.shape[0] for operator in operators]))
    result = np.zeros((total, total), dtype=np.float64)
    for active, operator in enumerate(operators):
        term = np.ones((1, 1), dtype=np.float64)
        for index, other in enumerate(operators):
            term = np.kron(term, operator if index == active else np.eye(len(other)))
        result += term
    return result


def _axis_coercivity_lower(operator: FloatArray) -> float:
    """Use the Dirichlet spectrum when the stored stencil matches exactly.

    General SPD matrices use a smallest-eigenvalue estimate reduced by a
    conservative backward-error allowance. This allowance is a numerical
    diagnostic, not an interval enclosure or a formal LAPACK error bound.
    """
    n = len(operator)
    stencil = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    scale = float(operator[0, 0] / 2)
    if scale > 0 and np.array_equal(operator, scale * stencil):
        # scale = a / h**2 for the standard centered Dirichlet difference.
        # Exact comparison prevents small general perturbations from silently
        # inheriting the unperturbed operator's analytic eigenvalue.
        lower = float(4 * scale * np.sin(np.pi / (2 * (n + 1))) ** 2)
    else:
        eigenvalues = np.linalg.eigvalsh(operator)
        if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0:
            raise ValueError("Axis operators must be positive definite.")
        spectral_norm = float(np.max(np.abs(eigenvalues)))
        allowance = 16 * np.finfo(np.float64).eps * n * spectral_norm
        lower = float(eigenvalues[0] - allowance)
        if lower <= 0:
            raise ValueError(
                "Axis operator is too ill-conditioned for a positive "
                "conservative coercivity estimate."
            )
    if not np.isfinite(lower) or lower <= 0:
        raise ValueError("Axis coercivity estimate must be finite and positive.")
    return lower


@dataclass
class PDEProblem:
    """Centered finite-difference diffusion plus a nonnegative nodal potential."""

    name: str
    shape: tuple[int, ...]
    axis_operators: tuple[FloatArray, ...]
    potential: FloatArray
    rhs: FloatArray
    metadata: dict[str, Any] = field(default_factory=dict)
    _coercivity_lower: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.shape = tuple(self.shape)
        if not self.shape or any(n <= 0 for n in self.shape):
            raise ValueError("Grid dimensions must be positive.")
        if len(self.axis_operators) != len(self.shape):
            raise ValueError("One axis operator is required for each dimension.")
        self.axis_operators = tuple(
            np.array(operator, dtype=np.float64, copy=True)
            for operator in self.axis_operators
        )
        lower = 0.0
        for n, operator in zip(self.shape, self.axis_operators):
            if operator.shape != (n, n) or not np.all(np.isfinite(operator)):
                raise ValueError("Each axis operator must be a finite square matrix.")
            if not np.allclose(operator, operator.T, rtol=1e-12, atol=1e-12):
                raise ValueError("Axis operators must be symmetric.")
            lower += _axis_coercivity_lower(operator)
        self.potential = np.array(self.potential, dtype=np.float64, copy=True)
        self.rhs = np.array(self.rhs, dtype=np.float64, copy=True)
        if self.potential.shape != self.shape or self.rhs.shape != self.shape:
            raise ValueError("Potential and forcing must match the tensor grid shape.")
        if not np.all(np.isfinite(self.rhs)) or not np.all(np.isfinite(self.potential)):
            raise ValueError("Potential and forcing must contain finite values.")
        if np.any(self.potential < 0):
            raise ValueError("This benchmark requires a nonnegative potential.")
        self._coercivity_lower = lower + float(np.min(self.potential))

    @property
    def coercivity_lower(self) -> float:
        """Axis lower estimates plus min(V), for residual error diagnostics.

        Exact Dirichlet stencils use their analytic smallest eigenvalues;
        general SPD axes use backward-error-adjusted numerical estimates.
        Floating-point evaluation is not a validated interval certificate.
        """
        return self._coercivity_lower

    def apply(self, tensor: FloatArray) -> FloatArray:
        tensor = np.asarray(tensor, dtype=np.float64)
        if tensor.shape != self.shape:
            raise ValueError(f"Expected tensor shape {self.shape}, got {tensor.shape}.")
        result = self.potential * tensor
        for axis, operator in enumerate(self.axis_operators):
            result = result + _mode_product(operator, tensor, axis)
        return result

    def as_linear_operator(self) -> LinearOperator:
        """Flattening and unflattening both use NumPy C order."""
        size = int(np.prod(self.shape))

        def matvec(vector: FloatArray) -> FloatArray:
            return self.apply(np.asarray(vector).reshape(self.shape)).ravel()

        return LinearOperator((size, size), matvec=matvec, rmatvec=matvec, dtype=float)

    def diagonal(self) -> FloatArray:
        diagonal = self.potential.copy()
        for axis, operator in enumerate(self.axis_operators):
            broadcast_shape = [1] * len(self.shape)
            broadcast_shape[axis] = self.shape[axis]
            diagonal += np.diag(operator).reshape(broadcast_shape)
        return diagonal

    def energy(self, tensor: FloatArray) -> float:
        """Return squared energy norm <tensor, A tensor>, without a factor 1/2."""
        tensor = np.asarray(tensor, dtype=np.float64)
        return float(np.sum(tensor * self.apply(tensor)))

    def cut_operators(
        self, cut: Iterable[int]
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return Ar, Ac, P with unfold(Aw)=Ar W + W Ac + P*W.

        The final multiplication is elementwise. All axis operators are
        symmetric, so no transpose of Ac is needed in this identity.
        """
        cut = tuple(cut)
        order = _axis_order(len(self.shape), cut)
        complement = order[len(cut) :]
        row = _kronecker_sum(self.axis_operators[axis] for axis in cut)
        col = _kronecker_sum(self.axis_operators[axis] for axis in complement)
        return row, col, matricize(self.potential, cut)


def _dirichlet_laplacian(n: int) -> FloatArray:
    h = 1.0 / (n + 1)
    return (
        2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    ) / h**2


def build_suite(n: int = 8, seeds: Iterable[int] = (0, 1, 2)) -> list[PDEProblem]:
    """Build up to nine small four-dimensional problems, using only source data.

    At most three seeds are used for paired forcing and at most two for each
    of competing-frequency forcing, anisotropic diffusion, and coupled
    potential. Different seeds change variable pairings and forcing widths.
    No exact or numerical solution is computed here.
    """
    if n < 3:
        raise ValueError("The suite needs at least three interior points per axis.")
    seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))[:3]
    if not seeds:
        raise ValueError("At least one seed is required.")
    grid = np.arange(1, n + 1, dtype=float) / (n + 1)
    x = np.meshgrid(grid, grid, grid, grid, indexing="ij", sparse=True)
    shape = (n,) * 4
    base = _dirichlet_laplacian(n)
    envelope = np.ones(shape)
    high_envelope = np.ones(shape)
    high_frequency = min(3, n)
    for coordinate in x:
        envelope *= np.sin(np.pi * coordinate)
        high_envelope *= np.sin(high_frequency * np.pi * coordinate)
    suite = []
    for family in ("paired", "competing", "anisotropic", "potential"):
        family_seeds = seeds if family == "paired" else seeds[:2]
        for seed in family_seeds:
            rng = np.random.default_rng(seed)
            p = tuple(int(axis) for axis in rng.permutation(4))
            beta = float(rng.uniform(7.0, 13.0))
            low_kernel = np.exp(
                -beta * ((x[p[0]] - x[p[1]]) ** 2 + (x[p[2]] - x[p[3]]) ** 2)
            )
            high_kernel = np.exp(
                -beta * ((x[p[0]] - x[p[2]]) ** 2 + (x[p[1]] - x[p[3]]) ** 2)
            )
            low = envelope * low_kernel
            low /= np.linalg.norm(low)
            high = high_envelope * high_kernel
            high /= np.linalg.norm(high)
            high_amplitude = 0.0 if family == "paired" else 4.0
            rhs = low + high_amplitude * high
            diffusion = np.ones(4)
            if family == "anisotropic":
                diffusion[list(p)] = (1.0, 2.0, 5.0, 9.0)
            potential = np.zeros(shape)
            if family == "potential":
                coupled = (
                    1.0
                    + 0.5 * np.sin(2 * np.pi * (x[p[0]] + x[p[2]]))
                    + 0.5 * np.cos(2 * np.pi * (x[p[1]] - x[p[3]]))
                )
                potential = np.broadcast_to(10.0 + 120.0 * coupled**2, shape).copy()
            suite.append(
                PDEProblem(
                    name=f"{family}_s{seed}",
                    shape=shape,
                    axis_operators=tuple(a * base for a in diffusion),
                    potential=potential,
                    rhs=rhs,
                    metadata={
                        "family": family,
                        "seed": seed,
                        "n": n,
                        "boundary": "homogeneous Dirichlet",
                        "forcing_construction": "prescribed source; no solution supplied",
                        "axis_permutation": list(p),
                        "low_pairs": [[p[0], p[1]], [p[2], p[3]]],
                        "high_pairs": [[p[0], p[2]], [p[1], p[3]]],
                        "kernel_beta": beta,
                        "high_frequency": high_frequency,
                        "high_amplitude": high_amplitude,
                        "diffusion": diffusion.tolist(),
                        "inner_product": "unweighted Euclidean interior grid",
                    },
                )
            )
    return suite


def reference_solve(
    problem: PDEProblem, rtol: float = 1e-12, maxiter: int = 2000
) -> dict[str, Any]:
    """Evaluation-only tight SciPy CG solve; never called by source/probe builders."""
    if rtol <= 0 or maxiter <= 0:
        raise ValueError("rtol and maxiter must be positive.")
    start = perf_counter()
    operator = problem.as_linear_operator()
    diagonal = problem.diagonal().ravel()
    preconditioner = LinearOperator(
        operator.shape, matvec=lambda vector: vector / diagonal, dtype=float
    )
    iterations = 0

    def count_iteration(_: FloatArray) -> None:
        nonlocal iterations
        iterations += 1

    vector, info = cg(
        operator,
        problem.rhs.ravel(),
        rtol=rtol,
        atol=0.0,
        maxiter=maxiter,
        M=preconditioner,
        callback=count_iteration,
    )
    w = vector.reshape(problem.shape)
    residual_norm = float(np.linalg.norm(problem.rhs - problem.apply(w)))
    rhs_norm = float(np.linalg.norm(problem.rhs))
    relative = residual_norm / rhs_norm if rhs_norm else residual_norm
    return {
        "w": w,
        "residual_norm": residual_norm,
        "relative_residual": relative,
        "elapsed": perf_counter() - start,
        "iterations": iterations,
        "converged": bool(info == 0 and relative <= max(10 * rtol, 1e-14)),
        "solver_info": int(info),
    }


def cg_probes(
    problem: PDEProblem,
    maxiter: int = 120,
    rtol: float = 1e-10,
    checkpoints: Iterable[int] = (0, 1, 2, 4, 8, 12, 20, 32, 48, 64, 96, 120),
    preconditioner: str = "jacobi",
) -> list[dict[str, Any]]:
    """Record PCG iterates with certified residual-based energy-error radii.

    At every saved iterate, delta=||b-Aw||_2/sqrt(lambda_lower) bounds
    ||u-w||_A for the exact solution u of this discrete system. These
    certificates use neither a reference solution nor future iterates.
    The initial and final iterates are always included.
    """
    if maxiter < 0 or rtol <= 0:
        raise ValueError("maxiter must be nonnegative and rtol positive.")
    if preconditioner not in ("jacobi", "none"):
        raise ValueError("preconditioner must be 'jacobi' or 'none'.")
    checkpoints = {int(iteration) for iteration in checkpoints}
    start = perf_counter()
    w = np.zeros(problem.shape, dtype=float)
    residual = problem.rhs.copy()
    rhs_norm = float(np.linalg.norm(problem.rhs))
    target = rtol * rhs_norm
    diagonal = problem.diagonal() if preconditioner == "jacobi" else 1.0
    root_lower = np.sqrt(problem.coercivity_lower)
    snapshots: list[dict[str, Any]] = []

    def save(iteration: int) -> bool:
        true_residual_norm = float(np.linalg.norm(problem.rhs - problem.apply(w)))
        converged = true_residual_norm <= target
        snapshots.append(
            {
                "iteration": iteration,
                "w": w.copy(),
                "residual_norm": true_residual_norm,
                "relative_residual": (
                    true_residual_norm / rhs_norm if rhs_norm else true_residual_norm
                ),
                "delta": true_residual_norm / root_lower,
                "elapsed": perf_counter() - start,
                "converged": bool(converged),
            }
        )
        return bool(converged)

    if save(0) or maxiter == 0:
        return snapshots
    z = residual / diagonal
    direction = z.copy()
    rz = float(np.sum(residual * z))
    for iteration in range(1, maxiter + 1):
        applied = problem.apply(direction)
        curvature = float(np.sum(direction * applied))
        if curvature <= 0 or not np.isfinite(curvature):
            raise FloatingPointError("PCG lost positive curvature on an SPD problem.")
        alpha = rz / curvature
        w += alpha * direction
        residual -= alpha * applied
        candidate_converged = np.linalg.norm(residual) <= target
        if iteration in checkpoints or iteration == maxiter or candidate_converged:
            if save(iteration):
                break
        z = residual / diagonal
        next_rz = float(np.sum(residual * z))
        direction = z + (next_rz / rz) * direction
        rz = next_rz
    return snapshots
