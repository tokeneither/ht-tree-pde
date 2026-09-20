"""Dense per-core energy minimization for a fixed multilevel HT tree.

The operator is diagonal in the supplied tensor-product coordinate basis. Each
update solves only a small normal system for one leaf or transfer core; the
full solution ``forcing / diagonal`` is never formed. This is a local optimizer,
not a global-optimality or scalable-memory claim.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np

from .ht_primal import edge_cuts, leaves, unfold, validate_tree


def _reconstruct(cores: dict, tree, shape: tuple[int, ...]) -> np.ndarray:
    """Contract leaf bases and transfer cores; restore the original axis order."""
    def frame(node):
        axes = leaves(node)
        if isinstance(node, int):
            return cores[axes]
        left, right = frame(node[0]), frame(node[1])
        core = cores[axes]
        if node == tree:
            return left @ core @ right.T
        return np.kron(left, right) @ core.reshape(left.shape[1] * right.shape[1], -1)

    order = leaves(tree)
    ordered = frame(tree).reshape(tuple(shape[i] for i in order))
    return ordered.transpose(np.argsort(order))


def _orthogonalize(cores: dict, tree) -> None:
    """Bottom-up gauge changes, absorbing each QR factor into its parent."""
    def visit(node, parent=None, child_index=None):
        if isinstance(node, tuple):
            visit(node[0], node, 0)
            visit(node[1], node, 1)
        if parent is None:
            return
        axes = leaves(node)
        core = cores[axes]
        matrix = core.reshape(-1, core.shape[-1])
        orthogonal, triangular = np.linalg.qr(matrix, mode="reduced")
        cores[axes] = orthogonal.reshape(core.shape)
        parent_axes = leaves(parent)
        upper = cores[parent_axes]
        if child_index == 0:
            cores[parent_axes] = np.tensordot(triangular, upper, axes=(1, 0))
        else:
            transformed = np.tensordot(triangular, upper, axes=(1, 1))
            cores[parent_axes] = np.swapaxes(transformed, 0, 1)

    visit(tree)


def fit_ht_energy(forcing: np.ndarray, diagonal: np.ndarray, rounded: dict,
                  max_sweeps: int = 4, tol: float = 1e-10,
                  diagnostics: bool = True) -> dict:
    """Improve a feasible ``ht_round`` result by all-core alternating solves.

    Minimize ``J(x) = 0.5 * <x, diagonal*x> - <forcing,x>`` on the fixed
    HT representation. Exact-arithmetic conditional minimizations are monotone;
    floating-point updates are accepted only when the independently evaluated
    objective does not increase. Singular normal systems use an eigenvalue
    pseudoinverse. This yields a feasible captured-energy *lower* bound only.

    ``cores`` in the return value includes the singleton leaf bases, unlike the
    input rounding dictionary, where they are stored in ``frames``.
    """
    if not np.isrealobj(forcing) or not np.isrealobj(diagonal):
        raise ValueError("forcing and diagonal must be real")
    forcing = np.asarray(forcing, dtype=float)
    diagonal = np.asarray(diagonal, dtype=float)
    if (forcing.ndim < 2 or diagonal.shape != forcing.shape
            or not np.all(np.isfinite(forcing)) or not np.all(np.isfinite(diagonal))
            or np.any(diagonal <= 0)):
        raise ValueError("forcing must be finite and diagonal must be finite, positive and matching")
    if isinstance(max_sweeps, bool) or not isinstance(max_sweeps, Integral) or max_sweeps < 0:
        raise ValueError("max_sweeps must be a nonnegative integer")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    tree = validate_tree(rounded["tree"], forcing.ndim)
    cores = {axes: np.array(core, dtype=float, copy=True)
             for axes, core in rounded["cores"].items()}
    for axes, frame in rounded["frames"].items():
        if len(axes) == 1:
            cores[axes] = np.array(frame, dtype=float, copy=True)
    root = leaves(tree)
    order = (root,) + edge_cuts(tree)
    fvector, weight = forcing.ravel(), diagonal.ravel()

    def objective(candidate):
        vector = candidate.ravel()
        return float(0.5 * np.dot(vector * weight, vector) - np.dot(fvector, vector))

    approximation = _reconstruct(cores, tree, forcing.shape)
    if approximation.shape != forcing.shape or not np.all(np.isfinite(approximation)):
        raise ValueError("the initial HT representation is invalid")
    # Scalar optimization is an admissible warm start, even with zero sweeps.
    denominator = float(np.sum(diagonal * approximation**2))
    alpha = float(np.sum(forcing * approximation)) / denominator if denominator > 0 else 0.0
    cores[root] *= alpha
    approximation = _reconstruct(cores, tree, forcing.shape)
    current = objective(approximation)
    history = [current]
    updates = []
    accepted_updates = 0
    attempted_updates = 0

    for _ in range(int(max_sweeps)):
        previous_sweep = current
        for axes in order:
            previous_core = cores[axes].copy()
            basis = np.empty((forcing.size, previous_core.size))
            for column in range(previous_core.size):
                unit = np.zeros_like(previous_core)
                unit.flat[column] = 1.0
                cores[axes] = unit
                basis[:, column] = _reconstruct(cores, tree, forcing.shape).ravel()
            cores[axes] = previous_core
            normal = basis.T @ (weight[:, None] * basis)
            normal = (normal + normal.T) * 0.5
            rhs = basis.T @ fvector
            eigenvalues, eigenvectors = np.linalg.eigh(normal)
            cutoff = (64 * np.finfo(float).eps * max(normal.shape)
                      * max(float(eigenvalues[-1]), 0.0))
            active = eigenvalues > cutoff
            if np.any(active):
                vectors = eigenvectors[:, active]
                solution = vectors @ ((vectors.T @ rhs) / eigenvalues[active])
            else:
                solution = np.zeros_like(rhs)
            cores[axes] = solution.reshape(previous_core.shape)
            candidate = _reconstruct(cores, tree, forcing.shape)
            value = objective(candidate)
            attempted_updates += 1
            if np.isfinite(value) and value <= current:
                approximation, current = candidate, value
                accepted_updates += 1
            else:
                cores[axes] = previous_core
            updates.append(current)

        # Prevent growing gauge condition numbers without altering the tensor.
        saved = {axes: core.copy() for axes, core in cores.items()}
        _orthogonalize(cores, tree)
        gauged = _reconstruct(cores, tree, forcing.shape)
        gauged_objective = objective(gauged)
        if np.isfinite(gauged_objective) and gauged_objective <= current:
            approximation, current = gauged, gauged_objective
        else:
            cores = saved
        history.append(current)
        if previous_sweep - current <= tol * max(abs(current), np.finfo(float).tiny):
            break

    result = dict(
        tree=tree, approximation=approximation, cores=cores, objective=current,
        objective_history=history, update_history=updates,
        captured_gain=float(-2 * current), sweeps=len(history) - 1,
        accepted_updates=accepted_updates, attempted_updates=attempted_updates,
        theoretical_coefficients=int(sum(core.size for core in cores.values())),
    )
    if diagnostics:
        result["edge_ranks"] = {}
        for axes in edge_cuts(tree):
            matrix = unfold(approximation, axes)
            singular = np.linalg.svd(matrix, compute_uv=False)
            cutoff = 10 * max(matrix.shape) * np.finfo(float).eps * singular[0]
            result["edge_ranks"][axes] = int(np.count_nonzero(singular > cutoff))
    return result
