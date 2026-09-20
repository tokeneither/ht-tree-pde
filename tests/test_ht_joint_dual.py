"""Validity, gradients, support handling and strict multi-edge improvement."""

import unittest

import numpy as np

from earlyht.ht_bounds import CutBoundCache
from earlyht.ht_joint_dual import JointCutDual, optimize_joint_bound
from earlyht.ht_primal import balanced_tree, ht_round


class JointHTDualTests(unittest.TestCase):
    def test_gradient_matches_directional_difference_away_from_ties(self):
        rng = np.random.default_rng(620)
        forcing = rng.normal(size=(3, 2, 3, 2))
        values = tuple(np.linspace(.5 + axis, 6. + axis, n)
                       for axis, n in enumerate(forcing.shape))
        dual = JointCutDual(forcing, values, ((0, 1), (0, 2)), 1,
                            center=rng.normal(size=forcing.shape) * .03, alpha=.37)
        point = rng.normal(size=dual.variables) * .1
        direction = rng.normal(size=dual.variables)
        direction /= np.linalg.norm(direction)
        value, gradient = dual.value_gradient(point)
        eps = 1e-6
        difference = (dual.value_gradient(point + eps * direction)[0]
                      - dual.value_gradient(point - eps * direction)[0]) / (2 * eps)
        self.assertTrue(np.isfinite(value))
        self.assertAlmostEqual(difference, float(gradient @ direction), places=7)

    def test_arbitrary_allocations_cover_known_feasible_ht_solution(self):
        rng = np.random.default_rng(814)
        shape = (3,) * 4
        tree = balanced_tree(range(4))
        solution = ht_round(rng.normal(size=shape), tree, 2)["approximation"]
        values = tuple(np.array([1. + i, 4. + i, 9. + i]) for i in range(4))
        diagonal = CutBoundCache(np.zeros(shape), values, 2).diagonal
        forcing = diagonal * solution  # Testing oracle only, never a benchmark source.
        optimum = float(np.sum(forcing * solution))
        dual = JointCutDual(forcing, values, ((0, 1), (0,)), 2,
                            center=rng.normal(size=shape), alpha=.23)
        for _ in range(5):
            upper, _ = dual.value_gradient(rng.normal(size=dual.variables))
            self.assertGreaterEqual(upper + 1e-10 * optimum, optimum)
        result = optimize_joint_bound(forcing, values, ((0, 1), (0,)), 2,
                                      center=solution * .8, maxiter=25)
        self.assertGreaterEqual(result["upper"] + 1e-10 * optimum, optimum)
        self.assertLessEqual(result["upper"], result["initial_upper"])

    def test_exact_allocation_improves_both_exact_single_edge_relaxations(self):
        forcing = np.zeros((2, 2, 2))
        forcing[1, 0, 0] = forcing[0, 1, 0] = forcing[0, 0, 1] = 1.
        values = (np.full(2, 1 / 3),) * 3
        cache = CutBoundCache(forcing, values, 1)
        self.assertAlmostEqual(cache.bound((0,))["upper"], 2., places=12)
        self.assertAlmostEqual(cache.bound((1,))["upper"], 2., places=12)
        dual = JointCutDual(forcing, values, ((0,), (1,)), 1)
        # Flattened nonzero positions are 001, 010, 100.
        value, _ = dual.value_gradient(np.array([0., -1 / 8, 1 / 8]))
        self.assertAlmostEqual(value, 25 / 16, places=12)
        result = optimize_joint_bound(forcing, values, ((0,), (1,)), 1, maxiter=80)
        self.assertLessEqual(result["upper"], 25 / 16 + 1e-10)
        # A concrete common rank-one candidate attains gain 4/3.
        self.assertGreaterEqual(result["upper"], 4 / 3)

    def test_support_compression_agrees_with_full_matrix_evaluation(self):
        rng = np.random.default_rng(3)
        forcing = np.zeros((3, 3, 3, 3))
        forcing[0, 1, 0, 2] = 1.
        forcing[2, 0, 2, 1] = -.7
        center = np.zeros_like(forcing)
        center[1, 1, 0, 0] = .4  # Center-only support must also be retained.
        values = tuple(np.array([1., 3., 9.]) for _ in range(4))
        restricted = JointCutDual(forcing, values, ((0, 1), (0, 2)), 1, center)
        dense = JointCutDual(forcing, values, ((0, 1), (0, 2)), 1, center, support="full")
        allocation = rng.normal(size=restricted.variables)
        full_allocation = np.zeros(dense.variables)
        full_allocation[restricted.support_indices] = allocation
        small_value, small_gradient = restricted.value_gradient(allocation)
        full_value, full_gradient = dense.value_gradient(full_allocation)
        self.assertEqual(restricted.variables, 3)
        self.assertAlmostEqual(small_value, full_value, places=12)
        np.testing.assert_allclose(small_gradient, full_gradient[restricted.support_indices],
                                   atol=1e-12)

    def test_exact_scalar_feasible_solution_and_early_exclusion(self):
        rng = np.random.default_rng(19)
        forcing = np.einsum("i,j,k,l->ijkl", *(rng.normal(size=2) for _ in range(4)))
        values = (np.full(2, .25),) * 4
        optimum = float(np.sum(forcing ** 2))
        result = optimize_joint_bound(forcing, values, ((0, 1), (0,)), 1,
                                      target=2 * optimum, maxiter=20)
        self.assertAlmostEqual(result["upper"], optimum, places=10)
        self.assertTrue(result["excluded"])
        self.assertEqual(result["iterations"], 0)
        self.assertEqual(result["full_pde_iterations"], 0)

    def test_zero_forcing_and_invalid_parameters(self):
        forcing = np.zeros((2, 2, 2))
        values = (np.ones(2),) * 3
        result = optimize_joint_bound(forcing, values, ((0,), (1,)), 1)
        self.assertEqual(result["upper"], 0.)
        self.assertEqual(result["allocation_variables"], 0)
        for cuts in (((0,),), ((0,), (1, 2))):
            with self.assertRaises(ValueError):
                JointCutDual(forcing, values, cuts, 1)
        for alpha in (0., 1., np.nan):
            with self.assertRaises(ValueError):
                JointCutDual(forcing, values, ((0,), (1,)), 1, alpha=alpha)
        with self.assertRaises(ValueError):
            optimize_joint_bound(forcing, values, ((0,), (1,)), 1, maxiter=-1)


if __name__ == "__main__":
    unittest.main()
