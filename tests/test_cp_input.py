"""Independent dense and analytic checks of compressed CP tree selection."""

import unittest
from unittest.mock import patch

import numpy as np

from earlyht.cp_input import (
    SpectralCPInput, compress_cp, concatenate_cp, cp_norm_squared,
    paired_spectral_cp, select_cp,
)
from earlyht.pde import BALANCED_CUTS, PDEProblem, matricize
from earlyht.selection import cut_name


def dense_forcing(source):
    # Independent termwise construction, used only for tiny test tensors.
    tensor = np.zeros(source.shape)
    for term, weight in enumerate(source.weights):
        part = np.asarray(weight)
        for factor in source.factors:
            part = np.multiply.outer(part, factor[:, term])
        tensor += part
    return tensor


def random_source(n, seed=128):
    rng = np.random.default_rng(seed)
    return SpectralCPInput(
        tuple(np.arange(1, n + 1, dtype=float) ** 2 + axis for axis in range(4)),
        tuple(rng.normal(size=(n, 3)) for _ in range(4)),
        np.array([1.0, -0.7, 0.4]), potential=2.0,
    )


class CPInputTests(unittest.TestCase):
    def test_cp_norm_and_reduced_tensor_match_independent_dense_construction(self):
        for n in (3, 4):
            source = random_source(n)
            dense = dense_forcing(source)
            compressed = compress_cp(source, modes=2)
            np.testing.assert_allclose(compressed["problem"].rhs, dense[:2, :2, :2, :2],
                                       rtol=1e-13, atol=1e-13)
            self.assertAlmostEqual(cp_norm_squared(source), float(np.sum(dense ** 2)), places=9)
            self.assertAlmostEqual(compressed["tail_norm_squared"],
                                   float(np.sum(dense ** 2) - np.sum(dense[:2, :2, :2, :2] ** 2)),
                                   places=9)

    def test_arbitrary_product_indices_and_tail_bound(self):
        source = random_source(4)
        dense = dense_forcing(source)
        indices = [[0, 2], [1, 3], [0, 3], [1, 2]]
        compressed = compress_cp(source, retained_indices=indices)
        retained = np.ix_(*indices)
        np.testing.assert_allclose(compressed["problem"].rhs, dense[retained], rtol=1e-13)
        denominator = np.full(source.shape, source.potential)
        for axis, eigenvalues in enumerate(source.eigenvalues):
            denominator += eigenvalues.reshape((1,) * axis + (4,) + (1,) * (3 - axis))
        density = dense ** 2 / denominator
        density[retained] = 0.0
        exact_tail = float(np.sum(density))
        self.assertGreaterEqual(compressed["tail_upper"], exact_tail * (1 - 1e-12))
        self.assertLessEqual(compressed["tail_upper"], compressed["tail_upper_global"])
        omitted_mask = np.ones(source.shape, dtype=bool)
        omitted_mask[retained] = False
        self.assertAlmostEqual(compressed["lambda_out"], float(denominator[omitted_mask].min()))

    def test_disjoint_tail_retains_small_values_without_norm_subtraction(self):
        factors = tuple(np.array([[1.0], [1e-10], [0.0]]) for _ in range(4))
        source = SpectralCPInput(tuple(np.arange(1.0, 4.0) for _ in range(4)),
                                 factors, np.array([1.0]))
        compressed = compress_cp(source, modes=1)
        # The naive norm difference rounds to zero; block norms remain positive.
        self.assertEqual(compressed["norm_squared"] - compressed["retained_norm_squared"], 0.0)
        self.assertGreater(compressed["tail_norm_squared"], 3.9e-20)
        self.assertGreater(compressed["tail_upper"], 0.0)

    def test_scalar_operator_has_exact_analytic_tree_gain_bounds(self):
        source = paired_spectral_cp(3, pairing=(0, 2), potential=1.0)
        source.eigenvalues = tuple(np.ones(3) * (i + 1) for i in range(4))
        dense = dense_forcing(source)
        result = select_cp(source, modes=2, sweeps=(0,))
        self.assertTrue(result["certified"])
        self.assertEqual(result["selected"], "13|24")
        self.assertEqual(result["spectral_tail_bound"], 0.0)
        for cut in BALANCED_CUTS:
            singular = np.linalg.svd(matricize(dense, cut), compute_uv=False)
            exact = singular[0] ** 2 / 11.0
            self.assertAlmostEqual(result["bounds"][cut_name(cut)]["lower"], exact, places=12)
            self.assertAlmostEqual(result["bounds"][cut_name(cut)]["upper"], exact, places=12)

    def test_full_certificate_uses_tail_and_not_only_reduced_winner(self):
        low = paired_spectral_cp(4, pairing=(0, 1))
        high = paired_spectral_cp(4, pairing=(0, 2), band_start=2)
        source = concatenate_cp((low, high), scales=(1.0, 20.0))
        result = select_cp(source, modes=2, sweeps=(0,))
        self.assertTrue(result["reduced_certified"])
        self.assertFalse(result["certified"])
        self.assertGreater(result["spectral_tail_bound"], 0.0)

    def test_large_input_constructs_only_small_pde_and_calls_no_reference_solver(self):
        n = 4096
        source = concatenate_cp((paired_spectral_cp(n, pairing=(0, 1)),
                                 paired_spectral_cp(n, pairing=(0, 2), band_start=10)),
                                scales=(1.0, 4.0))
        constructed_shapes = []

        def checked_problem(*args, **kwargs):
            constructed_shapes.append(args[1])
            self.assertEqual(args[1], (4, 4, 4, 4))
            return PDEProblem(*args, **kwargs)

        with patch("earlyht.cp_input.PDEProblem", side_effect=checked_problem), \
             patch("earlyht.pde.reference_solve", side_effect=AssertionError("reference leak")), \
             patch("earlyht.pde.cg_probes", side_effect=AssertionError("CG iteration")):
            result = select_cp(source, retained_indices=[0, 1, 10, 11], sweeps=(0,))
        self.assertEqual(constructed_shapes, [(4, 4, 4, 4)])
        self.assertTrue(result["certified"])
        self.assertEqual(result["selected"], "12|34")
        self.assertEqual(result["compression"]["full_tensor_entries"], n ** 4)
        self.assertEqual(result["compression"]["retained_tensor_entries"], 4 ** 4)
        self.assertFalse(result["full_tensor_materialized"])

    def test_complete_retention_has_zero_tail_and_zero_forcing_abstains(self):
        source = random_source(3)
        self.assertEqual(compress_cp(source, modes=3)["tail_upper"], 0.0)
        source.weights[:] = 0.0
        result = select_cp(source, modes=2, sweeps=(0,))
        self.assertFalse(result["certified"])
        self.assertEqual(result["spectral_tail_bound"], 0.0)

    def test_invalid_inputs_and_controls_are_rejected(self):
        source = random_source(4)
        for kwargs in (dict(modes=0), dict(modes=True), dict(retained_indices=[0, 0]),
                       dict(retained_indices=[0, 4]), dict(retained_indices=[0.0, 1.0]),
                       dict(retained_indices=[[0], [0, 1], [0], [0]])):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                compress_cp(source, **kwargs)
        for kwargs in (dict(rank=True), dict(sweeps=()), dict(sweeps=(0.5,))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_cp(source, modes=2, **kwargs)
        with self.assertRaises(ValueError):
            SpectralCPInput(source.eigenvalues, source.factors, source.weights, potential=-1)
        with self.assertRaises(ValueError):
            paired_spectral_cp(4, interaction_modes=3, band_start=2)


if __name__ == "__main__":
    unittest.main()
