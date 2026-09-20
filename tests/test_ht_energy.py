"""Check feasible, monotone multilevel HT energy optimization."""
import unittest

import numpy as np

from earlyht.ht_energy import fit_ht_energy
from earlyht.ht_primal import ht_round


class HTEnergyTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(114)

    def test_eight_dimensional_updates_improve_and_preserve_every_rank(self):
        tree = (((7, 0), (5, 2)), ((3, 6), (1, 4)))
        forcing = self.rng.normal(size=(3,) * 8)
        axis = np.arange(3, dtype=float) ** 2 + 1
        diagonal = np.zeros_like(forcing)
        for i in range(8):
            shape = [1] * 8
            shape[i] = 3
            diagonal += (i + 1) * axis.reshape(shape)
        initial = ht_round(forcing, tree, 2)
        fitted = fit_ht_energy(forcing, diagonal, initial, max_sweeps=3)
        self.assertTrue(all(rank <= 2 for rank in fitted["edge_ranks"].values()))
        self.assertEqual(fitted["theoretical_coefficients"], 100)
        self.assertTrue(np.all(np.diff(fitted["objective_history"]) <= 0))
        self.assertTrue(np.all(np.diff(fitted["update_history"]) <= 0))
        self.assertLess(fitted["objective"], 1.1 * fitted["objective_history"][0])
        approximation = fitted["approximation"]
        direct = float(0.5 * np.sum(diagonal * approximation**2)
                       - np.sum(forcing * approximation))
        self.assertAlmostEqual(fitted["objective"], direct, places=11)
        self.assertAlmostEqual(fitted["captured_gain"], -2 * direct, places=11)
        self.assertLessEqual(fitted["captured_gain"], float(np.sum(forcing**2 / diagonal)) + 1e-11)
        # The input representation is not mutated by fitting.
        np.testing.assert_array_equal(initial["approximation"], ht_round(forcing, tree, 2)["approximation"])

    def test_full_rank_tree_matches_independent_diagonal_solution(self):
        tree = ((2, 0), (3, 1))
        forcing = self.rng.normal(size=(2,) * 4)
        diagonal = self.rng.uniform(1, 7, size=forcing.shape)
        initial = ht_round(forcing, tree, 4)
        fitted = fit_ht_energy(forcing, diagonal, initial, max_sweeps=2)
        np.testing.assert_allclose(fitted["approximation"], forcing / diagonal,
                                   rtol=2e-11, atol=2e-12)
        self.assertAlmostEqual(fitted["captured_gain"], float(np.sum(forcing**2 / diagonal)), places=11)

    def test_zero_sweeps_perform_only_feasible_scalar_optimization(self):
        forcing = self.rng.normal(size=(3,) * 4)
        diagonal = self.rng.uniform(1, 5, size=forcing.shape)
        initial = ht_round(forcing, ((0, 2), (3, 1)), 2)
        before = initial["approximation"]
        alpha = np.sum(forcing * before) / np.sum(diagonal * before**2)
        fitted = fit_ht_energy(forcing, diagonal, initial, max_sweeps=0)
        np.testing.assert_allclose(fitted["approximation"], alpha * before, atol=3e-14)
        self.assertEqual(fitted["sweeps"], 0)
        self.assertEqual(fitted["attempted_updates"], 0)
        self.assertAlmostEqual(fitted["captured_gain"],
                               float(np.sum(forcing * before)**2 / np.sum(diagonal * before**2)),
                               places=11)

    def test_zero_forcing_and_invalid_operator(self):
        forcing = np.zeros((2,) * 4)
        initial = ht_round(forcing, ((0, 1), (2, 3)), 2)
        fitted = fit_ht_energy(forcing, np.ones_like(forcing), initial, max_sweeps=2)
        self.assertEqual(fitted["captured_gain"], 0)
        self.assertTrue(all(rank == 0 for rank in fitted["edge_ranks"].values()))
        for diagonal in (np.zeros_like(forcing), -np.ones_like(forcing), np.ones((2, 2))):
            with self.assertRaises(ValueError):
                fit_ht_energy(forcing, diagonal, initial)
        for sweeps in (-1, 0.5, True):
            with self.assertRaises(ValueError):
                fit_ht_energy(forcing, np.ones_like(forcing), initial, max_sweeps=sweeps)


if __name__ == "__main__":
    unittest.main()
