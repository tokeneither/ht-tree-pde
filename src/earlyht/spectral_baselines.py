"""Separable direct solve and spectral-preconditioned CG benchmarks.

The setup detects *exactly stored* scaled Dirichlet difference stencils and
uses orthonormal DST-I on those axes. Other SPD axes use orthogonal eigenbases.
All setup work is included in each solver's reported elapsed time. The cached
axis transforms and eigenvalue sum are reused throughout a PCG call.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from scipy.fft import dstn
from scipy.sparse.linalg import LinearOperator, cg

from .pde import PDEProblem


def _mode_multiply(matrix: np.ndarray, tensor: np.ndarray, axis: int) -> np.ndarray:
    moved = np.moveaxis(tensor, axis, 0)
    result = matrix @ moved.reshape(moved.shape[0], -1)
    return np.moveaxis(result.reshape(moved.shape), 0, axis)


class SeparableSpectralInverse:
    """Cached inverse of ``sum_i A_i + shift * I`` on the tensor grid.

    Eigenvectors act separately in each mode. This operator is an exact
    separable solve in floating-point arithmetic, not a low-rank truncation.
    The nonseparable potential is intentionally absent from its action.
    """

    def __init__(self, problem: PDEProblem, shift: float = 0.0):
        start = perf_counter()
        if not np.isfinite(shift) or shift < 0:
            raise ValueError("The spectral shift must be finite and nonnegative.")
        self.shape = problem.shape
        self.shift = float(shift)
        self.eigenvalues = []
        self.eigenvectors = []
        self.dst_axes = []
        for axis, matrix in enumerate(problem.axis_operators):
            n = matrix.shape[0]
            stencil = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
            scale = float(matrix[0, 0] / 2)
            if scale > 0 and np.array_equal(matrix, scale * stencil):
                frequencies = np.arange(1, n + 1, dtype=float)
                values = 4 * scale * np.sin(np.pi * frequencies / (2 * (n + 1))) ** 2
                vectors = None
                self.dst_axes.append(axis)
            else:
                values, vectors = np.linalg.eigh(matrix)
            if values[0] <= 0 or not np.all(np.isfinite(values)):
                raise ValueError("Axis operators must have finite positive spectra.")
            self.eigenvalues.append(values)
            self.eigenvectors.append(vectors)
        self.dst_axes = tuple(self.dst_axes)
        self.denominator = np.full(self.shape, self.shift)
        for axis, values in enumerate(self.eigenvalues):
            broadcast_shape = [1] * len(self.shape)
            broadcast_shape[axis] = self.shape[axis]
            self.denominator += values.reshape(broadcast_shape)
        self.setup_seconds = perf_counter() - start

    def _check(self, tensor: np.ndarray) -> np.ndarray:
        tensor = np.asarray(tensor, dtype=float)
        if tensor.shape != self.shape:
            raise ValueError(f"Expected shape {self.shape}, got {tensor.shape}.")
        return tensor

    def forward(self, tensor: np.ndarray) -> np.ndarray:
        """Transform a tensor to the orthonormal product eigenbasis."""
        result = self._check(tensor)
        if self.dst_axes:
            result = dstn(result, type=1, axes=self.dst_axes, norm="ortho")
        for axis, vectors in enumerate(self.eigenvectors):
            if vectors is not None:
                result = _mode_multiply(vectors.T, result, axis)
        return result

    def inverse_transform(self, coefficients: np.ndarray) -> np.ndarray:
        """Undo :meth:`forward`; orthonormal DST-I is its own inverse."""
        result = self._check(coefficients)
        if self.dst_axes:
            result = dstn(result, type=1, axes=self.dst_axes, norm="ortho")
        for axis, vectors in enumerate(self.eigenvectors):
            if vectors is not None:
                result = _mode_multiply(vectors, result, axis)
        return result

    def apply(self, tensor: np.ndarray) -> np.ndarray:
        return self.inverse_transform(self.forward(tensor) / self.denominator)

    @property
    def transform(self) -> str:
        if len(self.dst_axes) == len(self.shape):
            return "orthonormal DST-I"
        if not self.dst_axes:
            return "orthogonal axis eigendecompositions"
        return "mixed DST-I and axis eigendecompositions"


def _residual_summary(
    problem: PDEProblem, solution: np.ndarray, *, rtol: float, atol: float
) -> dict[str, Any]:
    residual_norm = float(np.linalg.norm(problem.rhs - problem.apply(solution)))
    rhs_norm = float(np.linalg.norm(problem.rhs))
    threshold = max(atol, rtol * rhs_norm)
    return {
        "residual_norm": residual_norm,
        "relative_residual": residual_norm / rhs_norm if rhs_norm else residual_norm,
        "residual_threshold": threshold,
        "true_residual_converged": bool(residual_norm <= threshold),
    }


def spectral_solve(problem: PDEProblem, rtol: float = 1e-12) -> dict[str, Any]:
    """Direct spectral solve, allowed only for an exactly constant potential.

    ``iterations`` is zero because no iterative solve is performed. The
    reported operator application is the final independent residual check.
    """
    start = perf_counter()
    if not np.isfinite(rtol) or rtol <= 0:
        raise ValueError("rtol must be finite and positive.")
    constant = float(problem.potential.flat[0])
    if not np.all(problem.potential == constant):
        raise ValueError("The direct spectral solver requires a constant potential.")
    inverse = SeparableSpectralInverse(problem, constant)
    solve_start = perf_counter()
    solution = inverse.apply(problem.rhs)
    solve_seconds = perf_counter() - solve_start
    residual = _residual_summary(problem, solution, rtol=rtol, atol=0.0)
    return {
        "w": solution,
        "method": "spectral direct solve",
        "transform": inverse.transform,
        "shift": constant,
        "iterations": 0,
        "operator_applications": 1,
        "preconditioner_applications": 0,
        "setup_seconds": inverse.setup_seconds,
        "solve_seconds": solve_seconds,
        "elapsed": perf_counter() - start,
        "converged": residual["true_residual_converged"],
        **residual,
    }


def spectral_pcg(
    problem: PDEProblem,
    rtol: float = 1e-10,
    atol: float = 0.0,
    maxiter: int = 2000,
    shift: float | None = None,
    x0: np.ndarray | None = None,
) -> dict[str, Any]:
    """CG with the cached SPD inverse of ``A0 + mean(V) * I`` by default.

    ``shift`` can override the announced source-only default. All eigenbasis
    setup, denominator allocation, transforms, and true residual checks are
    included in ``elapsed``. ``converged`` requires the independently checked
    true residual as well as SciPy's successful exit status.
    """
    start = perf_counter()
    if not np.isfinite(rtol) or rtol <= 0 or not np.isfinite(atol) or atol < 0:
        raise ValueError("rtol must be positive and atol nonnegative, both finite.")
    if not isinstance(maxiter, (int, np.integer)) or maxiter <= 0:
        raise ValueError("maxiter must be a positive integer.")
    if x0 is not None:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != problem.shape or not np.all(np.isfinite(x0)):
            raise ValueError("x0 must be a finite tensor on the problem grid.")
    shift = float(np.mean(problem.potential)) if shift is None else float(shift)
    inverse = SeparableSpectralInverse(problem, shift)
    size = int(np.prod(problem.shape))
    counts = {"operator_applications": 0, "preconditioner_applications": 0,
              "iterations": 0}

    def apply_operator(vector: np.ndarray) -> np.ndarray:
        counts["operator_applications"] += 1
        return problem.apply(np.asarray(vector).reshape(problem.shape)).ravel()

    def apply_inverse(vector: np.ndarray) -> np.ndarray:
        counts["preconditioner_applications"] += 1
        return inverse.apply(np.asarray(vector).reshape(problem.shape)).ravel()

    def callback(_: np.ndarray) -> None:
        counts["iterations"] += 1

    operator = LinearOperator((size, size), matvec=apply_operator, dtype=float)
    preconditioner = LinearOperator((size, size), matvec=apply_inverse, dtype=float)
    solve_start = perf_counter()
    vector, info = cg(
        operator, problem.rhs.ravel(),
        x0=None if x0 is None else x0.ravel(),
        M=preconditioner, rtol=rtol, atol=atol, maxiter=int(maxiter), callback=callback,
    )
    solve_seconds = perf_counter() - solve_start
    solution = vector.reshape(problem.shape)
    residual = _residual_summary(problem, solution, rtol=rtol, atol=atol)
    counts["operator_applications"] += 1
    return {
        "w": solution,
        "method": "spectral-preconditioned CG",
        "transform": inverse.transform,
        "shift": shift,
        "scipy_info": int(info),
        "setup_seconds": inverse.setup_seconds,
        "solve_seconds": solve_seconds,
        "elapsed": perf_counter() - start,
        "converged": bool(info == 0 and residual["true_residual_converged"]),
        **counts,
        **residual,
    }
