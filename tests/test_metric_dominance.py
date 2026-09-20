"""Quantitative checks of the balanced-metric dominance propositions."""

import unittest

import numpy as np

from earlyht.direct_selection import metric_bound, product_metrics


def balanced_metric(a, b):
    shift = (float(np.min(b)) - float(np.min(a))) / 2
    return 2 * np.sqrt(a + shift), np.sqrt(b - shift)


class MetricDominanceTests(unittest.TestCase):
    def test_balanced_metric_sandwich_with_unequal_spectral_minima(self):
        for a, b in ((np.array([.2, 2., 80.]), np.array([7., 11.])),
                     (np.array([9., 20.]), np.array([.01, .5, 900.]))):
            left, right = balanced_metric(a, b)
            metric = left[:, None] * right[None, :]
            coercivity = a.min() + b.min()
            self.assertGreaterEqual(float(metric.min()), coercivity - 1e-12)
            self.assertTrue(np.all(metric <= a[:, None] + b[None, :] + 1e-12))
            produced = [l[:, None] * r[None, :] for l, r, _ in product_metrics(a, b)]
            self.assertTrue(any(np.allclose(value, metric, rtol=1e-13, atol=1e-13)
                                for value in produced))

    def test_variational_identity_allows_a_full_rank_center(self):
        rng = np.random.default_rng(180)
        a, b = np.array([.3, 1.1, 4., 19.]), np.array([2., 7., 12.])
        forcing, center = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
        self.assertEqual(np.linalg.matrix_rank(center), 3)
        left, right = balanced_metric(a, b)
        upper, optimizer = metric_bound(forcing, a, b, left, right, 1, center)
        diagonal = a[:, None] + b[None, :]
        residual = forcing - diagonal * center
        center_gain = 2 * np.sum(forcing * center) - np.sum(diagonal * center ** 2)
        displacement = optimizer - center
        equivalent = (center_gain + 2 * np.sum(residual * displacement)
                      - np.sum(left[:, None] * right[None, :] * displacement ** 2))
        self.assertAlmostEqual(upper, equivalent, places=11)

    def test_product_bound_dominates_scalar_for_every_sampled_center(self):
        rng = np.random.default_rng(905)
        a, b = np.array([.3, 1., 7., 40.]), np.array([2., 6., 12.])
        left, right = balanced_metric(a, b)
        coercivity = a.min() + b.min()
        for rank in (1, 2, 3):
            for scale in (0., .01, 1., 100.):
                forcing = rng.normal(size=(4, 3))
                center = scale * rng.normal(size=(4, 3))
                product, _ = metric_bound(forcing, a, b, left, right, rank, center)
                scalar, _ = metric_bound(forcing, a, b,
                                          np.full(4, coercivity), np.ones(3),
                                          rank, center)
                with self.subTest(rank=rank, scale=scale):
                    self.assertLessEqual(product, scalar + 1e-10 * max(1., scalar))

    def test_raw_gap_equals_spectral_contrast(self):
        k, rank, m, high = 6, 2, 1., 1000.
        a = np.r_[m / 2, np.full(k, high / 2)]
        forcing = np.diag(np.r_[0., np.ones(k)])
        left, right = balanced_metric(a, a)
        product, _ = metric_bound(forcing, a, a, left, right, rank)
        scalar, _ = metric_bound(forcing, a, a, np.full(k + 1, m),
                                  np.ones(k + 1), rank)
        self.assertAlmostEqual(product, rank / high, places=14)
        self.assertAlmostEqual(scalar / product, high / m, places=10)

    def test_optimized_scalar_gap_matches_closed_form(self):
        k, rank, m, high = 5, 2, .7, 80.
        a = np.r_[m / 2, np.full(k, high / 2)]
        forcing = np.diag(np.r_[0., np.ones(k)])
        t = 1 / (high + m * (k / rank - 1))
        optimal_center = np.diag(np.r_[0., np.full(k, t)])
        expected_scalar = k / (high + m * (k / rank - 1))
        scalar_left, scalar_right = np.full(k + 1, m), np.ones(k + 1)
        scalar, _ = metric_bound(forcing, a, a, scalar_left, scalar_right,
                                  rank, optimal_center)
        self.assertAlmostEqual(scalar, expected_scalar, places=12)
        rng = np.random.default_rng(88)
        for scale in (.001, .01, 1.):
            for _ in range(8):
                center = optimal_center + scale * rng.normal(size=(k + 1, k + 1))
                other, _ = metric_bound(forcing, a, a, scalar_left, scalar_right,
                                         rank, center)
                self.assertGreaterEqual(other + 1e-12, expected_scalar)
        left, right = balanced_metric(a, a)
        product, _ = metric_bound(forcing, a, a, left, right, rank)
        self.assertAlmostEqual(product, rank / high, places=14)
        self.assertGreater(scalar / product, 2.4)

    def test_optimized_relaxation_gap_grows_with_dimension(self):
        # D/m = k^2 makes the optimized scalar/product ratio approach k.
        ratios = []
        for k in (2, 4, 8, 16):
            high = float(k ** 2)
            a = np.r_[.5, np.full(k, high / 2)]
            forcing = np.diag(np.r_[0., np.ones(k)])
            center = np.diag(np.r_[0., np.full(k, 1 / (high + k - 1))])
            scalar, _ = metric_bound(forcing, a, a, np.ones(k + 1),
                                      np.ones(k + 1), 1, center)
            left, right = balanced_metric(a, a)
            product, _ = metric_bound(forcing, a, a, left, right, 1)
            ratio = scalar / product
            self.assertAlmostEqual(ratio, k * high / (high + k - 1), places=11)
            self.assertGreaterEqual(ratio, k / 2)
            ratios.append(ratio)
        self.assertTrue(np.all(np.diff(ratios) > 0))
        self.assertGreater(ratios[-1], 15.)


    def test_nonuniform_band_jensen_bound_by_enumerating_rank_subsets(self):
        from itertools import combinations

        k, rank, m = 5, 2, 2.
        high_diagonal = np.array([20., 24., 30., 36., 50.])
        a = np.r_[m / 2, high_diagonal / 2]
        forcing = np.diag(np.r_[0., np.ones(k)])
        projector = forcing.copy()
        diagonal = a[:, None] + a[None, :]
        remainder = diagonal - m
        t = 1 / (np.mean(high_diagonal) + m * (k / rank - 1))
        expected_lower = k * t
        candidates = []
        for subset in combinations(range(1, k + 1), rank):
            candidate = np.zeros_like(forcing)
            candidate[list(subset), list(subset)] = (k / rank) * t
            self.assertEqual(np.linalg.matrix_rank(candidate), rank)
            candidates.append(candidate)
        np.testing.assert_allclose(np.mean(candidates, axis=0), t * projector,
                                   atol=1e-15)
        self.assertAlmostEqual(np.mean([np.sum(x ** 2) for x in candidates]),
                               (k ** 2 / rank) * t ** 2, places=15)
        rng = np.random.default_rng(124)
        for scale in (0., .01, 1.):
            center = scale * rng.normal(size=forcing.shape)
            base = float(np.sum(remainder * center ** 2))
            shifted = forcing - remainder * center
            average = base + np.mean([2 * np.sum(shifted * x) - m * np.sum(x ** 2)
                                      for x in candidates])
            square = float(np.sum(remainder * (center - t * projector) ** 2))
            self.assertAlmostEqual(average, expected_lower + square, places=11)
            upper, _ = metric_bound(forcing, a, a, np.full(k + 1, m),
                                      np.ones(k + 1), rank, center)
            self.assertGreaterEqual(upper + 1e-11, average)

    def test_dirichlet_high_band_has_quantitative_scalar_relaxation_gap(self):
        n, rank = 8, 1
        frequencies = np.arange(1., n + 1.)
        axis_spectrum = 4 * (n + 1) ** 2 * np.sin(frequencies * np.pi / (2 * (n + 1))) ** 2
        a = (axis_spectrum[:, None] + axis_spectrum[None, :]).ravel()
        indices = np.arange(n ** 2).reshape(n, n)[n // 2:, n // 2:].ravel()
        k, m = len(indices), 2 * a.min()
        forcing = np.zeros((n ** 2, n ** 2))
        forcing[indices, indices] = 1.
        left, right = balanced_metric(a, a)
        product, _ = metric_bound(forcing, a, a, left, right, rank)
        high_diagonal = 2 * a[indices]
        exact = float(np.sum(np.sort(1 / high_diagonal)[-rank:]))
        self.assertAlmostEqual(product, exact, places=14)
        scalar_inf_lower = k / (np.mean(high_diagonal) + m * (k / rank - 1))
        self.assertGreaterEqual(scalar_inf_lower / exact,
                                2 * n ** 2 / (16 * rank + np.pi ** 2))
        # A nonuniform diagonal center attains the rank-one scalar dual optimum;
        # the Jensen bound remains below this computed feasible dual value.
        harmonic = np.sum(1 / (high_diagonal - m))
        coefficient = 1 / (1 + m * harmonic)
        center = np.zeros_like(forcing)
        center[indices, indices] = coefficient / (high_diagonal - m)
        scalar, _ = metric_bound(forcing, a, a, np.full(n ** 2, m),
                                  np.ones(n ** 2), rank, center)
        self.assertAlmostEqual(scalar, harmonic / (1 + m * harmonic), places=12)
        self.assertGreaterEqual(scalar + 1e-13, scalar_inf_lower)

if __name__ == '__main__':
    unittest.main()
