"""Independent numerical checks for the compression-resistant example."""

from itertools import combinations
import unittest

import numpy as np

from earlyht.compression_gap import (
    compressed_scalar_bound,
    compression_gap_lower_bound,
    optimal_suffix_compression,
    optimized_scalar_dual,
    optimized_scalar_primal,
    polynomial_spectrum,
)
from earlyht.direct_selection import metric_bound


class CompressionGapTests(unittest.TestCase):
    def test_primal_matches_independent_simplex_dual(self):
        rng = np.random.default_rng(73510)
        cases = [np.ones(1), np.ones(6), np.array([1., 4.]),
                 np.array([1., 1., 4., 9.]), polynomial_spectrum(32)]
        cases.extend(np.exp(rng.uniform(-5., 6., size)) for size in (3, 7, 19))
        for d in cases:
            primal = optimized_scalar_primal(d)
            dual = optimized_scalar_dual(d)
            with self.subTest(d=d):
                self.assertAlmostEqual(primal.value, dual.value, places=10)
                self.assertTrue(np.all(dual.weights >= 0))
                self.assertAlmostEqual(float(dual.weights.sum()), 1., places=14)
                self.assertGreaterEqual(primal.residual_threshold,
                                        np.sqrt(float(d.min())) - 1e-14)

    def test_boundary_and_interior_kkt_cases(self):
        boundary = optimized_scalar_primal([1., 4.])
        self.assertAlmostEqual(boundary.value, 4. / 3., places=14)
        self.assertAlmostEqual(boundary.residual_threshold, 1., places=14)
        interior = optimized_scalar_primal(polynomial_spectrum(16))
        self.assertGreater(interior.residual_threshold, 1.)
        dual = optimized_scalar_dual(polynomial_spectrum(16))
        self.assertGreater(dual.multiplier, 1.)
        self.assertAlmostEqual(dual.weights[0], 0., places=14)

    def test_diagonal_center_has_correct_full_matrix_objective(self):
        for d in (np.array([.2, 1., 4., 9.]), polynomial_spectrum(9)):
            primal = optimized_scalar_primal(d)
            n = len(d)
            value, _ = metric_bound(np.diag(np.sqrt(d)), d / 2, d / 2,
                                    np.full(n, d.min()), np.ones(n), 1,
                                    np.diag(primal.center_diagonal))
            self.assertAlmostEqual(value, primal.value, places=11)

    def test_product_bound_and_actual_rank_one_energy_are_one(self):
        rng = np.random.default_rng(571)
        d = np.exp(rng.uniform(-2., 5., 9))
        forcing = np.diag(np.sqrt(d))
        product, _ = metric_bound(forcing, d / 2, d / 2,
                                  np.sqrt(d), np.sqrt(d), 1)
        self.assertAlmostEqual(product, 1., places=13)
        actual_diagonal = (d[:, None] + d[None, :]) / 2
        for index in (0, 4, 8):
            x = np.zeros_like(forcing)
            x[index, index] = 1 / np.sqrt(d[index])
            self.assertEqual(np.linalg.matrix_rank(x), 1)
            gain = 2 * np.sum(forcing * x) - np.sum(actual_diagonal * x ** 2)
            self.assertAlmostEqual(gain, 1., places=13)
        # The product upper bound must also cover arbitrary rank-one points.
        for _ in range(15):
            x = np.outer(rng.normal(size=len(d)), rng.normal(size=len(d)))
            rescaled_gain = np.sum(forcing * x) ** 2 / np.sum(actual_diagonal * x ** 2)
            self.assertLessEqual(rescaled_gain, product + 1e-12)

    def test_suffix_optimum_matches_all_subsets(self):
        cases = [polynomial_spectrum(size) for size in range(1, 9)]
        cases.extend([np.array([9., 1., 4., 4., 16.]),
                      np.array([.1, 1.7, .3, 30., 2., 80.])])
        for d in cases:
            full_subset_minimum = min(
                compressed_scalar_bound(d, subset)
                for size in range(len(d) + 1)
                for subset in combinations(range(len(d)), size))
            suffix = optimal_suffix_compression(d)
            with self.subTest(d=d):
                self.assertAlmostEqual(suffix.value, full_subset_minimum, places=11)
                self.assertAlmostEqual(
                    suffix.value, compressed_scalar_bound(d, suffix.retained_indices),
                    places=11)

    def test_compression_gap_grows_and_respects_theorem(self):
        gaps = []
        for size in (64, 128, 256):
            result = optimal_suffix_compression(polynomial_spectrum(size))
            self.assertGreaterEqual(result.value, compression_gap_lower_bound(size))
            self.assertGreater(result.tail_energy, 0.)
            self.assertLess(result.value, result.suffix_totals[0])
            gaps.append(result.value)
        self.assertTrue(np.all(np.diff(gaps) > 0.))

    def test_common_spectral_scaling_preserves_values(self):
        d = polynomial_spectrum(17)
        expected = optimal_suffix_compression(d)
        scaled = optimal_suffix_compression(d * 37.2)
        self.assertAlmostEqual(expected.value, scaled.value, places=11)
        np.testing.assert_array_equal(expected.retained_indices, scaled.retained_indices)

    def test_rejects_invalid_model_data(self):
        for d in ([], [[1., 2.]], [0., 1.], [-1., 1.], [1., np.inf]):
            with self.assertRaises(ValueError):
                optimized_scalar_primal(d)
            with self.assertRaises(ValueError):
                optimized_scalar_dual(d)
        for retained in ([0, 0], [-1], [3], [0.5]):
            with self.assertRaises(ValueError):
                compressed_scalar_bound([1., 4., 9.], retained)


if __name__ == "__main__":
    unittest.main()
