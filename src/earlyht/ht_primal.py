"""Feasible dense hierarchical-Tucker projections for arbitrary binary trees.

This module constructs *nested* frames. Truncating unrelated matricizations
independently would not preserve all tree-edge ranks. Its dense implementation
is intended for small verification problems, not as a scalable HT library.
"""
from __future__ import annotations

from numbers import Integral
from typing import Union

import numpy as np


Tree = Union[int, tuple["Tree", "Tree"]]


def leaves(tree: Tree) -> tuple[int, ...]:
    """Axis order induced by the left-to-right traversal of a binary tree."""
    if isinstance(tree, Integral) and not isinstance(tree, bool):
        return (int(tree),)
    if not isinstance(tree, tuple) or len(tree) != 2:
        raise ValueError("a tree must be an integer leaf or a binary tuple")
    return leaves(tree[0]) + leaves(tree[1])


def validate_tree(tree: Tree, ndim: int) -> Tree:
    """Normalize integer leaves and require exactly the axes 0, ..., ndim-1."""
    if isinstance(ndim, bool) or not isinstance(ndim, Integral) or ndim < 2:
        raise ValueError("at least two tensor dimensions are required")
    axes = leaves(tree)
    if len(axes) != ndim or sorted(axes) != list(range(ndim)):
        raise ValueError("tree leaves must contain each tensor axis exactly once")

    def normalize(node):
        if isinstance(node, Integral):
            return int(node)
        return normalize(node[0]), normalize(node[1])

    return normalize(tree)


def edge_cuts(tree: Tree) -> tuple[tuple[int, ...], ...]:
    """Every nonroot node's axes, in bottom-up order.

    Both complementary children of the root are included because they are
    distinct HT frames, even though their matricizations share singular values.
    """
    result = []

    def visit(node, root=False):
        if isinstance(node, tuple):
            visit(node[0])
            visit(node[1])
        if not root:
            result.append(leaves(node))

    visit(tree, root=True)
    return tuple(result)


def unfold(tensor: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """C-order matricization with exactly the supplied row-axis order."""
    tensor = np.asarray(tensor)
    axes = tuple(axes)
    if (not axes or len(set(axes)) != len(axes)
            or any(not isinstance(i, Integral) or isinstance(i, bool)
                   or i < 0 or i >= tensor.ndim for i in axes)):
        raise ValueError("axes must be distinct valid tensor axes")
    complement = tuple(i for i in range(tensor.ndim) if i not in axes)
    rows = int(np.prod([tensor.shape[i] for i in axes], dtype=int))
    return np.transpose(tensor, axes + complement).reshape(rows, -1)


def balanced_tree(order) -> Tree:
    """Split a supplied permutation into two nearly equal parts recursively."""
    order = tuple(order)
    if not order or len(set(order)) != len(order):
        raise ValueError("order must contain distinct integer axes")
    if any(not isinstance(i, Integral) or isinstance(i, bool) or i < 0
           for i in order):
        raise ValueError("order must contain distinct nonnegative integer axes")
    if len(order) == 1:
        return int(order[0])
    middle = len(order) // 2
    return balanced_tree(order[:middle]), balanced_tree(order[middle:])


def _canonical_tree(tree: Tree) -> Tree:
    if isinstance(tree, Integral):
        return int(tree)
    left, right = (_canonical_tree(tree[0]), _canonical_tree(tree[1]))
    return ((left, right) if tuple(sorted(leaves(left))) < tuple(sorted(leaves(right)))
            else (right, left))


def tree_label(tree: Tree) -> str:
    """Compact, one-based, human-readable parenthesized tree."""
    if isinstance(tree, Integral):
        return str(int(tree) + 1)
    return f"({tree_label(tree[0])},{tree_label(tree[1])})"


def candidate_trees(ndim: int, count: int = 8, seed: int = 0,
                    planted: Tree | None = None) -> tuple[Tree, ...]:
    """Reproducible distinct balanced trees, optionally including one given tree.

    The planted tree is an explicitly supplied experimental control, not a
    structure inferred from PDE data. If fewer balanced trees exist than
    requested, the finite random search returns the trees it found.
    """
    if isinstance(ndim, bool) or not isinstance(ndim, Integral) or ndim < 2:
        raise ValueError("ndim must be an integer >= 2")
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
        raise ValueError("count must be a positive integer")
    initial = balanced_tree(range(ndim)) if planted is None else validate_tree(planted, ndim)
    result = [_canonical_tree(initial)]
    seen = {result[0]}
    rng = np.random.default_rng(seed)
    for _ in range(max(100, 100 * count)):
        if len(result) >= count:
            break
        candidate = _canonical_tree(balanced_tree(rng.permutation(ndim)))
        if candidate not in seen:
            result.append(candidate)
            seen.add(candidate)
    return tuple(result)


def ht_round(tensor: np.ndarray, tree: Tree, rank: int,
             diagnostics: bool = True) -> dict:
    """Construct a tensor with every nonroot HT rank at most ``rank``.

    At a leaf, retain the leading left singular vectors of its matricization.
    At an internal node, first project onto the Kronecker product of its child
    frames, then retain the leading left singular vectors of the projected
    matricization. The root is projected onto its two child frames without an
    additional rank constraint. The HT *root rank* is always one; ``rank``
    constrains each nonroot edge, including both children of the root.

    Returned frames are dense inspection aids; ``theoretical_coefficients``
    counts the leaf bases and transfer cores of the compressed representation,
    not peak memory used by this implementation. ``edge_ranks`` uses a relative
    numerical singular-value tolerance and is omitted when diagnostics=False.
    """
    if not np.isrealobj(tensor):
        raise ValueError("tensor must be real")
    tensor = np.asarray(tensor, dtype=float)
    if tensor.ndim < 2 or min(tensor.shape) == 0 or not np.all(np.isfinite(tensor)):
        raise ValueError("tensor must be finite with at least two nonempty axes")
    if isinstance(rank, bool) or not isinstance(rank, Integral) or rank < 1:
        raise ValueError("rank must be a positive integer")
    tree = validate_tree(tree, tensor.ndim)
    frames = {}
    cores = {}
    discarded_squared = {}
    coefficients = 0

    def build(node):
        nonlocal coefficients
        axes = leaves(node)
        matrix = unfold(tensor, axes)
        if isinstance(node, int):
            vectors, singular, _ = np.linalg.svd(matrix, full_matrices=False)
            kept = min(int(rank), len(singular))
            frame = vectors[:, :kept]
            coefficients += frame.size
        else:
            left = build(node[0])
            right = build(node[1])
            product = np.kron(left, right)
            projected = product.T @ matrix
            vectors, singular, _ = np.linalg.svd(projected, full_matrices=False)
            kept = min(int(rank), len(singular))
            core = vectors[:, :kept]
            frameshape = (left.shape[1], right.shape[1], kept)
            cores[axes] = core.reshape(frameshape)
            coefficients += core.size
            frame = product @ core
        frames[axes] = frame
        discarded_squared[axes] = float(np.sum(singular[kept:] ** 2))
        return frame

    left = build(tree[0])
    right = build(tree[1])
    order = leaves(tree)
    # Unlike unfold(), this root matrix follows the right subtree's own order.
    root_shape = (left.shape[0], right.shape[0])
    matrix = tensor.transpose(order).reshape(root_shape)
    root_core = left.T @ matrix @ right
    cores[order] = root_core
    coefficients += root_core.size
    ordered_shape = tuple(tensor.shape[i] for i in order)
    reconstructed = (left @ root_core @ right.T).reshape(ordered_shape)
    approximation = reconstructed.transpose(np.argsort(order)).copy()
    result = dict(
        tree=tree, tree_label=tree_label(tree), rank=int(rank),
        approximation=approximation, frames=frames, cores=cores,
        representation_ranks={axes: frame.shape[1] for axes, frame in frames.items()},
        theoretical_coefficients=int(coefficients),
        dense_tensor_coefficients=int(tensor.size),
        frobenius_error=float(np.linalg.norm(tensor - approximation)),
        discarded_squared=discarded_squared,
        discarded_error_bound=float(np.sqrt(sum(discarded_squared.values()))),
    )
    if diagnostics:
        spectra = {axes: np.linalg.svd(unfold(approximation, axes), compute_uv=False)
                   for axes in edge_cuts(tree)}
        # A tolerance slightly above machine precision avoids counting the
        # reconstruction roundoff as extra HT rank in near-zero singular modes.
        observed = {}
        for axes, singular in spectra.items():
            scale = singular[0] if len(singular) else 0.0
            tolerance = max(unfold(approximation, axes).shape) * np.finfo(float).eps * scale * 10
            observed[axes] = int(np.count_nonzero(singular > tolerance))
        result.update(edge_singular_values=spectra, edge_ranks=observed)
    return result
