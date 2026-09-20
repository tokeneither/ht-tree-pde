"""Reuse family-gain bounds for a fixed operator and affine forcing space.

The caller supplies valid bounds for the same candidate cones, rank budgets,
coordinates and SPD operator A at every reference.  It must also establish
0 < B <= A for the inverse minorant used to assemble the parameter Gram
matrix.  Neither condition can be inferred from a small Gram matrix alone.

For forcing f(theta) = sum_j theta[j] basis[j], sqrt(G_T) is 1-Lipschitz in
the A-inverse norm.  A constant affine offset is represented by a basis vector
whose coefficient is always one.  Optional homogeneous transport additionally
uses G_T(a f) = a**2 G_T(f), including negative a.  This is valid because the
families are closed under all real scalar multiplications.

All bounds are exact-arithmetic statements.  Diagonal Gram inflation and the
relative decision guard are floating-point safeguards, not interval validation.
In particular, these routines do not verify user-supplied minorants or gains.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


_EPS = np.finfo(float).eps
_INPUT_TOLERANCE = 1e-12


def _real_array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(value, name):
    result = _real_array(value, name)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    result = result.copy()
    result.setflags(write=False)
    return result


def _validated_gram(gram):
    gram = _real_array(gram, "gram")
    if gram.ndim != 2 or gram.shape[0] == 0 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be a nonempty square matrix")
    scale = float(np.linalg.norm(gram, ord=np.inf))
    if not np.isfinite(scale):
        raise ValueError("gram is too large for floating-point evaluation")
    if np.linalg.norm(gram - gram.T, ord=np.inf) > _INPUT_TOLERANCE * scale:
        raise ValueError("gram must be symmetric up to rounding error")
    result = 0.5 * gram + 0.5 * gram.T
    minimum = float(np.linalg.eigvalsh(result)[0])
    if minimum < -_INPUT_TOLERANCE * scale:
        raise ValueError("gram must be positive semidefinite")
    # Repair only rounding-sized negative eigenvalues and round upward.  This
    # does not certify that an arbitrary supplied matrix bounds the true Gram.
    padding = max(0.0, -minimum) + 64 * _EPS * gram.shape[0] * scale
    result = result + padding * np.eye(gram.shape[0])
    result.setflags(write=False)
    return result


def _forcing_basis(basis):
    basis = _real_array(basis, "basis")
    if basis.ndim < 2 or basis.shape[0] == 0 or basis.size == 0:
        raise ValueError("basis must have shape (parameters, *forcing_shape)")
    return basis


def _apply_action(vector, *, diagonal, action, name):
    if (diagonal is None) == (action is None):
        raise ValueError(f"supply exactly one diagonal or callable for {name}")
    if action is not None:
        result = _real_array(action(vector.copy()), name)
        if result.shape != vector.shape:
            raise ValueError(f"{name} must preserve the forcing shape")
        return result
    diagonal = _real_array(diagonal, name)
    if np.any(diagonal <= 0):
        raise ValueError(f"{name} diagonal must be strictly positive")
    try:
        diagonal = np.broadcast_to(diagonal, vector.shape)
    except ValueError as error:
        raise ValueError(f"{name} diagonal has incompatible shape") from error
    result = vector * diagonal
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} result is not finite")
    return result


def parameter_gram(basis, *, inverse_diagonal=None,
                   apply_inverse_minorant: Callable | None = None):
    """Assemble an inflated PSD Gram Q_ij = <basis_i, B^-1 basis_j>.

    ``inverse_diagonal`` contains diagonal entries of B^-1, not of B, and
    broadcasts to the forcing shape.  Alternatively the supplied callable
    applies B^-1 once per basis vector.  The caller must establish B <= A.
    Construction touches the full basis only once; queries use Q alone.
    """
    basis = _forcing_basis(basis)
    inverse_basis = np.stack([
        _apply_action(vector, diagonal=inverse_diagonal,
                      action=apply_inverse_minorant, name="inverse minorant")
        for vector in basis
    ])
    flat = basis.reshape(basis.shape[0], -1)
    inverse_flat = inverse_basis.reshape(basis.shape[0], -1)
    return _validated_gram(flat @ inverse_flat.T)


def feasible_direction_correlations(basis, direction, *, operator_diagonal=None,
                                    apply_operator: Callable | None = None):
    """Return c_j = <basis_j, v> / sqrt(<v, A v>) for a feasible v != 0.

    Feasibility in the claimed family is the caller's responsibility.  The
    lower gain at a query is |theta @ c|**2, realized after optimizing the
    scalar amplitude of v.  It is generally NOT the gain of the unchanged v.
    """
    basis = _forcing_basis(basis)
    direction = _real_array(direction, "direction")
    if direction.shape != basis.shape[1:]:
        raise ValueError("direction and forcing basis have incompatible shapes")
    image = _apply_action(direction, diagonal=operator_diagonal,
                          action=apply_operator, name="operator")
    energy = float(np.sum(direction * image))
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError("direction must have finite, strictly positive A-energy")
    correlations = basis.reshape(basis.shape[0], -1) @ direction.ravel()
    return _vector(correlations / np.sqrt(energy), "direction correlations")


@dataclass(frozen=True)
class ReferenceCertificate:
    """Gain bounds and an optional feasible ray at one parameter vector.

    ``winner`` indexes the family of ``direction_correlations``; it need not
    already have a strict certificate.  Arrays are copied and made read-only.
    A positive infinite upper bound is allowed for an unresolved family.
    """

    theta: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    winner: int | None = None
    direction_correlations: np.ndarray | None = None

    def __post_init__(self):
        theta = _vector(self.theta, "theta")
        lower = _vector(self.lower, "lower")
        if np.iscomplexobj(self.upper):
            raise ValueError("upper must be real")
        upper = np.asarray(self.upper, dtype=float).copy()
        if upper.shape != lower.shape or np.any(np.isnan(upper)):
            raise ValueError("upper must have the same shape as lower and no NaNs")
        if np.any(lower < 0) or np.any(upper < 0):
            raise ValueError("gain bounds must be nonnegative")
        finite_upper = upper[np.isfinite(upper)]
        scale = max(float(np.max(lower)), float(np.max(finite_upper, initial=0.0)))
        if np.any(lower - upper > _INPUT_TOLERANCE * scale):
            raise ValueError("lower gain bound exceeds upper gain bound")
        # Weaken a lower bound in a rounding-sized overlap, never tighten U.
        lower = np.minimum(lower, upper)
        lower.setflags(write=False)
        upper.setflags(write=False)
        winner = self.winner
        if winner is None:
            winner = int(np.argmax(lower))
        if (isinstance(winner, (bool, np.bool_)) or not isinstance(winner, (int, np.integer))
                or not 0 <= winner < lower.size):
            raise ValueError("winner must index an existing candidate")
        correlations = self.direction_correlations
        if correlations is not None:
            correlations = _vector(correlations, "direction correlations")
            if correlations.shape != theta.shape:
                raise ValueError("direction correlations and theta must have the same shape")
            ray_gain = float(theta @ correlations)**2
            if ray_gain > upper[winner] + _INPUT_TOLERANCE * max(scale, ray_gain):
                raise ValueError("stored direction gain exceeds its reference upper bound")
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "winner", int(winner))
        object.__setattr__(self, "direction_correlations", correlations)


def _quadratic_upper(vector, gram):
    value = float(vector @ gram @ vector)
    cancellation_scale = float(np.abs(vector) @ np.abs(gram) @ np.abs(vector))
    result = max(0.0, value) + 64 * _EPS * cancellation_scale
    if not np.isfinite(result):
        raise ValueError("parameter quadratic form is too large for floating point")
    return result


class CertificateLibrary:
    """Intersect transported intervals from a library of reference points.

    Set ``homogeneous_scaling=True`` to also transport from a*theta0, where a
    minimizes ||theta-a theta0||_Q.  Both the original a=1 and this scaled
    transport are retained, so the option cannot weaken an interval.  A global
    upper bound theta^T Q theta is always used (equivalently transport from
    zero).  Thus an empty library safely abstains and returns finite bounds.

    ``query`` certifies a family, not the accuracy of an unchanged stored
    tensor.  Transported family lower bounds do not construct a tensor that
    attains them at the query.  Directional lower bounds require ray rescaling.
    """

    def __init__(self, gram, n_candidates, *, relative_guard=1e-9,
                 homogeneous_scaling=False):
        self.gram = _validated_gram(gram)
        if (isinstance(n_candidates, (bool, np.bool_))
                or not isinstance(n_candidates, (int, np.integer)) or n_candidates < 2):
            raise ValueError("n_candidates must be an integer of at least two")
        if not np.isfinite(relative_guard) or relative_guard < 0:
            raise ValueError("relative_guard must be finite and nonnegative")
        self.n_candidates = int(n_candidates)
        self.relative_guard = float(relative_guard)
        self.homogeneous_scaling = bool(homogeneous_scaling)
        self._references = []

    @property
    def references(self):
        """Immutable view of stored reference certificates."""
        return tuple(self._references)

    def add_reference(self, theta, lower, upper, *, winner=None,
                      direction_correlations=None):
        """Store bounds for all candidates and return the reference index."""
        reference = ReferenceCertificate(theta, lower, upper, winner,
                                         direction_correlations)
        if reference.theta.size != self.gram.shape[0]:
            raise ValueError("theta has incompatible parameter dimension")
        if reference.lower.size != self.n_candidates:
            raise ValueError("reference must bound every candidate")
        self._references.append(reference)
        return len(self._references) - 1

    def query(self, theta):
        """Return strict-winner decision and transported bounds as JSON data.

        ``selected_index`` is None on abstention; ``candidate_index`` always
        identifies the largest available lower bound.  ``directional_lower``
        separately records ray-optimized bounds, not unchanged-tensor gains.
        Distances and scale factors are ordered by stored reference index.
        """
        theta = _vector(theta, "theta")
        if theta.size != self.gram.shape[0]:
            raise ValueError("theta has incompatible parameter dimension")
        energy_upper = _quadratic_upper(theta, self.gram)
        lower_root = np.zeros(self.n_candidates)
        upper_root = np.full(self.n_candidates, np.sqrt(energy_upper))
        directional_root = np.zeros(self.n_candidates)
        reference_distances = []
        reference_scale_factors = []
        scaled_reference_distances = []

        for reference in self._references:
            delta = np.sqrt(_quadratic_upper(theta-reference.theta, self.gram))
            reference_distances.append(float(delta))
            scales = [(1.0, delta)]
            factor = 1.0
            scaled_delta = delta
            if self.homogeneous_scaling:
                denominator = float(reference.theta @ self.gram @ reference.theta)
                if denominator > 0:
                    factor = float(theta @ self.gram @ reference.theta) / denominator
                else:
                    factor = 0.0
                scaled_delta = np.sqrt(_quadratic_upper(
                    theta-factor*reference.theta, self.gram))
                if factor != 1.0:
                    scales.append((factor, scaled_delta))
            reference_scale_factors.append(float(factor))
            scaled_reference_distances.append(float(scaled_delta))
            for factor, distance in scales:
                if factor == 0:
                    # Avoid the undefined floating-point product 0 * inf.
                    transported_lower = np.zeros(self.n_candidates)
                    transported_upper = np.full(self.n_candidates, distance)
                else:
                    transported_lower = np.maximum(
                        abs(factor)*np.sqrt(reference.lower)-distance, 0.0)
                    transported_upper = abs(factor)*np.sqrt(reference.upper)+distance
                lower_root = np.maximum(lower_root, transported_lower)
                upper_root = np.minimum(upper_root, transported_upper)
            if reference.direction_correlations is not None:
                ray_root = abs(float(theta @ reference.direction_correlations))
                family = reference.winner
                directional_root[family] = max(directional_root[family], ray_root)
                lower_root[family] = max(lower_root[family], ray_root)

        lower = lower_root**2
        upper = upper_root**2
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("transported bounds overflowed floating point")
        scale = max(float(np.max(upper)), float(np.max(lower)))
        guard = self.relative_guard * scale
        consistency_guard = max(self.relative_guard, _INPUT_TOLERANCE) * scale
        if np.any(lower > upper + consistency_guard):
            raise ArithmeticError("transported intervals are inconsistent; check supplied bounds and metric")
        lower = np.minimum(lower, upper)
        candidate = int(np.argmax(lower))
        competing_upper = float(np.max(np.delete(upper, candidate)))
        margin = float(lower[candidate] - competing_upper)
        certified = margin > guard
        return {
            "certified": bool(certified),
            "selected_index": candidate if certified else None,
            "candidate_index": candidate,
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "directional_lower": (directional_root**2).tolist(),
            "margin": margin,
            "guard": float(guard),
            "global_energy_upper": float(energy_upper),
            "n_references": len(self._references),
            "reference_distances": reference_distances,
            "reference_scale_factors": reference_scale_factors,
            "scaled_reference_distances": scaled_reference_distances,
            "homogeneous_scaling": self.homogeneous_scaling,
        }
