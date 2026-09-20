"""Tree selection from spectral CP forcing without a full-grid tensor.

The input factors are coefficients in given orthonormal axis eigenbases.
Only the retained product space is materialized. Bounds are floating-point
evaluations of exact-arithmetic inequalities, not interval certificates.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from time import perf_counter

import numpy as np

from .direct_selection import choose_from_gains, select_direct
from .pde import PDEProblem


@dataclass
class SpectralCPInput:
    """Four-mode forcing ``sum_s weights[s] prod_i factors[i][:,s]``.

    Axis eigenvalues must be positive and nondecreasing. Each factor has one
    row per eigenvalue and a common number of columns. Constant nonnegative
    potentials are supported; no physical eigenvector matrices are needed.
    """

    eigenvalues: tuple[np.ndarray, ...]
    factors: tuple[np.ndarray, ...]
    weights: np.ndarray
    potential: float = 0.0
    name: str = "spectral_cp"

    def __post_init__(self):
        if len(self.eigenvalues) != 4 or len(self.factors) != 4:
            raise ValueError("The compressed selector requires four modes.")
        self.eigenvalues = tuple(np.asarray(a, dtype=float) for a in self.eigenvalues)
        self.factors = tuple(np.asarray(a, dtype=float) for a in self.factors)
        self.weights = np.asarray(self.weights, dtype=float)
        if self.weights.ndim != 1 or not self.weights.size or not np.all(np.isfinite(self.weights)):
            raise ValueError("weights must be a nonempty finite vector.")
        for values, factor in zip(self.eigenvalues, self.factors):
            if (values.ndim != 1 or not values.size or not np.all(np.isfinite(values))
                    or np.any(values <= 0) or np.any(np.diff(values) < 0)):
                raise ValueError("Axis eigenvalues must be finite, positive and nondecreasing.")
            if factor.shape != (len(values), len(self.weights)) or not np.all(np.isfinite(factor)):
                raise ValueError("Each finite factor must have shape (axis size, CP term count).")
        self.potential = float(self.potential)
        if not np.isfinite(self.potential) or self.potential < 0:
            raise ValueError("The constant potential must be finite and nonnegative.")

    @property
    def shape(self):
        return tuple(len(values) for values in self.eigenvalues)

    @property
    def terms(self):
        return len(self.weights)


def _norm_from_grams(grams, weights):
    product = np.ones((len(weights), len(weights)))
    for gram in grams:
        product *= gram
    summands = weights[:, None] * product * weights[None, :]
    raw = float(np.sum(summands))
    # This flag exposes possible cancellation. It is not an interval proof.
    return max(0.0, raw), bool(raw < 0), float(np.sum(np.abs(summands)))


def cp_norm_squared(source):
    """Evaluate the CP norm from Gram matrices, without materializing it."""
    grams = [factor.T @ factor for factor in source.factors]
    return _norm_from_grams(grams, source.weights)[0]


def _retained_indices(source, modes, retained_indices):
    if retained_indices is None:
        if (isinstance(modes, bool) or not isinstance(modes, Integral)
                or not 1 <= modes <= min(source.shape)):
            raise ValueError("modes must be a feasible positive integer.")
        result = tuple(np.arange(modes) for _ in range(4))
    else:
        requested = list(retained_indices)
        if requested and all(isinstance(i, Integral) and not isinstance(i, bool) for i in requested):
            requested = [requested] * 4
        if len(requested) != 4:
            raise ValueError("Provide shared indices or one index vector per axis.")
        result = []
        for entries, n in zip(requested, source.shape):
            entries = list(entries)
            if (not entries or any(isinstance(i, bool) or not isinstance(i, Integral)
                                   for i in entries)):
                raise ValueError("Retained indices must be nonempty integer vectors.")
            indices = np.asarray(entries, dtype=int)
            if np.any(indices < 0) or np.any(indices >= n) or len(set(entries)) != len(entries):
                raise ValueError("Retained indices must be distinct valid axis indices.")
            result.append(np.sort(indices))
        result = tuple(result)
        if len({len(indices) for indices in result}) != 1:
            raise ValueError("The current reduced selector needs equal retained mode counts.")
    return result


def compress_cp(source, modes=4, retained_indices=None):
    """Build only the small retained PDE and bound omitted forcing energy.

    ``retained_indices`` overrides ``modes`` and may be one shared list or
    four equal-length lists. Arbitrary spectral indices preserve HT ranks.
    Disjoint first-omitted-axis blocks avoid subtracting nearly equal norms.
    """
    start = perf_counter()
    indices = _retained_indices(source, modes, retained_indices)
    retained_factors = tuple(factor[index] for factor, index in zip(source.factors, indices))
    all_grams = tuple(factor.T @ factor for factor in source.factors)
    keep_grams = tuple(factor.T @ factor for factor in retained_factors)
    norm_squared, norm_clipped, _ = _norm_from_grams(all_grams, source.weights)
    retained_norm_squared, keep_clipped, _ = _norm_from_grams(keep_grams, source.weights)
    tail_norm_squared = 0.0
    tail_upper_partition = 0.0
    lambda_out = float("inf")
    blocks = []
    floating_clipped = norm_clipped or keep_clipped
    for axis in range(4):
        mask = np.ones(source.shape[axis], dtype=bool)
        mask[indices[axis]] = False
        if not np.any(mask):
            continue
        omitted = source.factors[axis][mask]
        grams = keep_grams[:axis] + (omitted.T @ omitted,) + all_grams[axis + 1:]
        block_norm, clipped, absolute_sum = _norm_from_grams(grams, source.weights)
        floating_clipped |= clipped
        # Allowance for Gram products/summation; numerical diagnostic only.
        allowance = (32 * np.finfo(float).eps * (sum(source.shape) + source.terms ** 2)
                     * absolute_sum)
        first_omitted = float(source.eigenvalues[axis][mask].min())
        common_lower = source.potential + first_omitted + sum(
            float(source.eigenvalues[j][0]) for j in range(4) if j != axis)
        block_lower = source.potential + first_omitted + sum(
            float(source.eigenvalues[j][indices[j][0] if j < axis else 0])
            for j in range(4) if j != axis)
        lambda_out = min(lambda_out, common_lower)
        tail_norm_squared += block_norm
        tail_upper_partition += (block_norm + allowance) / block_lower
        blocks.append(dict(first_omitted_axis=axis, norm_squared=block_norm,
                           roundoff_allowance=allowance, eigenvalue_lower=block_lower,
                           floating_clipped=clipped))
    total_allowance = sum(block["roundoff_allowance"] for block in blocks)
    tail_upper_global = ((tail_norm_squared + total_allowance) / lambda_out
                         if blocks else 0.0)
    tail_upper = min(tail_upper_global, tail_upper_partition)
    # The only tensor materialization is m^4. Explicit output indices prevent
    # einsum from constructing a physical full-grid forcing tensor.
    reduced = np.einsum("s,is,js,ks,ls->ijkl", source.weights, *retained_factors,
                        optimize=True)
    m = len(indices[0])
    problem = PDEProblem(
        source.name + "_retained", (m,) * 4,
        tuple(np.diag(values[index]) for values, index in zip(source.eigenvalues, indices)),
        np.full((m,) * 4, source.potential), reduced,
        metadata=dict(input_format="spectral_cp", full_shape=source.shape,
                      retained_indices=[index.tolist() for index in indices]),
    )
    full_entries = int(np.prod(source.shape, dtype=object))
    input_scalars = int(sum(f.size for f in source.factors) + source.weights.size
                        + sum(values.size for values in source.eigenvalues))
    return dict(problem=problem, modes=m, retained_indices=[index.tolist() for index in indices],
                tail_upper=float(tail_upper), tail_upper_global=float(tail_upper_global),
                tail_upper_partition=float(tail_upper_partition),
                tail_norm_squared=float(tail_norm_squared), norm_squared=norm_squared,
                retained_norm_squared=retained_norm_squared,
                lambda_out=None if not blocks else float(lambda_out), tail_blocks=blocks,
                floating_clipped=bool(floating_clipped),
                tail_roundoff_allowance=float(total_allowance),
                full_shape=source.shape, full_tensor_entries=full_entries,
                input_scalar_count=input_scalars, retained_tensor_entries=m ** 4,
                full_tensor_materialized=False, elapsed=perf_counter() - start)


def select_cp(source, rank=1, modes=4, retained_indices=None, sweeps=(0, 2, 6, 16)):
    """Compare full-space trees using a compact spectral CP input.

    Lower bounds come from feasible retained tensors. The omitted-energy
    upper bound is added to every reduced upper bound before deciding.
    A reduced certificate alone is never reported as a full-space one.
    """
    start = perf_counter()
    if isinstance(rank, bool) or not isinstance(rank, Integral) or rank <= 0:
        raise ValueError("rank must be a positive integer.")
    sweeps = tuple(sweeps)
    if not sweeps or any(isinstance(stage, bool) or not isinstance(stage, Integral)
                         or stage < 0 for stage in sweeps):
        raise ValueError("sweeps must contain nonnegative integer budgets.")
    data = compress_cp(source, modes=modes, retained_indices=retained_indices)
    reduced = select_direct(data["problem"], rank=rank, modes=data["modes"], sweeps=sweeps)
    bounds = {name: dict(entry, upper=entry["upper"] + data["tail_upper"],
                         reduced_upper=entry["upper"])
              for name, entry in reduced["bounds"].items()}
    decision = choose_from_gains(bounds)
    metadata = {key: value for key, value in data.items() if key != "problem"}
    return dict(**decision, bounds=bounds, rank=rank, modes=data["modes"],
                compression=metadata, reduced_certified=reduced["certified"],
                reduced_trace=reduced["trace"], spectral_tail_bound=data["tail_upper"],
                setup_seconds=data["elapsed"], reduced_selection_seconds=reduced["elapsed"],
                elapsed=perf_counter() - start, full_pde_iterations=0,
                full_tensor_materialized=False,
                scope="three balanced full-leaf families; spectral CP input; floating-point bounds")


def paired_spectral_cp(n, interaction_modes=2, pairing=(0, 1), band_start=0,
                       diffusivities=(1.0, 1.0, 1.0, 1.0), decay=0.0,
                       potential=0.0, name=None):
    """Analytic paired forcing with ``interaction_modes**2`` CP terms.

    Each pair uses a diagonal sum of equal-frequency sine products. The
    source is normalized in the discrete orthonormal spectral basis. Its
    source pairing is a construction label, not an assumed solution winner.
    ``band_start`` is zero-based; 10 selects sine modes 11, 12, ... .
    """
    for value, label in ((n, "n"), (interaction_modes, "interaction_modes")):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{label} must be a positive integer.")
    if (isinstance(band_start, bool) or not isinstance(band_start, Integral)
            or band_start < 0 or band_start + interaction_modes > n):
        raise ValueError("The requested frequency band must fit the axis size.")
    pairing = tuple(pairing)
    if len(pairing) != 2 or len(set(pairing)) != 2 or any(i not in range(4) for i in pairing):
        raise ValueError("pairing must contain two distinct axes from 0 to 3.")
    diffusivities = np.asarray(diffusivities, dtype=float)
    if diffusivities.shape != (4,) or np.any(diffusivities <= 0) or not np.all(np.isfinite(diffusivities)):
        raise ValueError("Four positive finite diffusivities are required.")
    if not np.isfinite(decay) or decay < 0:
        raise ValueError("decay must be finite and nonnegative.")
    frequency = np.arange(1, n + 1)
    base = 4 * (n + 1) ** 2 * np.sin(np.pi * frequency / (2 * (n + 1))) ** 2
    eigenvalues = tuple(diffusion * base for diffusion in diffusivities)
    count = interaction_modes ** 2
    factors = [np.zeros((n, count)) for _ in range(4)]
    weights = np.empty(count)
    for k in range(interaction_modes):
        for ell in range(interaction_modes):
            term = k * interaction_modes + ell
            weights[term] = np.exp(-decay * (k + ell))
            for axis in range(4):
                factors[axis][band_start + (k if axis in pairing else ell), term] = 1.0
    weights /= np.linalg.norm(weights)
    label = name or f"paired_{pairing[0] + 1}{pairing[1] + 1}_band{band_start + 1}_k{interaction_modes}"
    return SpectralCPInput(eigenvalues, tuple(factors), weights, potential, label)


def concatenate_cp(sources, scales=None, name="cp_mixture"):
    """Sum compatible spectral CP sources by concatenating their factors."""
    sources = tuple(sources)
    if not sources:
        raise ValueError("At least one source is required.")
    scales = np.ones(len(sources)) if scales is None else np.asarray(scales, dtype=float)
    if scales.shape != (len(sources),) or not np.all(np.isfinite(scales)):
        raise ValueError("One finite scale is required per source.")
    first = sources[0]
    for source in sources[1:]:
        if (source.potential != first.potential or
                any(not np.array_equal(a, b) for a, b in zip(source.eigenvalues, first.eigenvalues))):
            raise ValueError("Sources must use identical eigenvalues and constant potentials.")
    return SpectralCPInput(
        first.eigenvalues,
        tuple(np.concatenate([source.factors[i] for source in sources], axis=1) for i in range(4)),
        np.concatenate([scale * source.weights for scale, source in zip(scales, sources)]),
        first.potential, name,
    )
