"""Pointwise feasibility and independent energy checks for potential bounds."""

import unittest
from unittest.mock import patch

import numpy as np

from earlyht.potential_bounds import additive_minorants, minorant_lower_bound
from earlyht.trees import CutProblem, rank_lower_bound


class PotentialBoundTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(816)

    def problem(self, m=6, n=5):
        left = self.rng.normal(size=(m, m))
        right = self.rng.normal(size=(n, n))
        return CutProblem(
            self.rng.normal(size=(m, n)),
            left.T @ left + np.eye(m),
            right.T @ right + np.eye(n),
            self.rng.uniform(0.5, 12.0, size=(m, n)),
        )

    def test_minorants_are_pointwise_feasible(self):
        examples = [
            self.rng.uniform(0, 15, size=(7, 5)),
            np.array([[1.0, 6.0, 3.0], [4.0, 2.0, 9.0]]),
            np.zeros((3, 4)),
            np.full((2, 3), 7.0),
        ]
        for potential in examples:
            before = potential.copy()
            generated = list(additive_minorants(potential))
            self.assertEqual(len(generated), 4)
            for p, q, label in generated:
                with self.subTest(shape=potential.shape, label=label):
                    self.assertEqual(p.shape, (potential.shape[0],))
                    self.assertEqual(q.shape, (potential.shape[1],))
                    self.assertTrue(np.all(p >= 0))
                    self.assertTrue(np.all(q >= 0))
                    self.assertTrue(np.all(p[:, None] + q[None, :] <= potential + 1e-13))
            np.testing.assert_array_equal(potential, before)

    def test_row_and_column_constructions_recover_additive_potential(self):
        p = np.array([1.0, 4.0, 2.0, 3.0])
        q = np.array([0.2, 1.8, 0.5])
        potential = p[:, None] + q[None, :]
        for row, col, label in additive_minorants(potential):
            if label != "constant":
                np.testing.assert_allclose(row[:, None] + col[None, :], potential)

    def test_bound_dominates_base_and_is_below_independent_feasible_errors(self):
        for _ in range(4):
            problem = self.problem()
            for rank in (0, 1, 2, 4):
                with self.subTest(rank=rank):
                    base = rank_lower_bound(problem, rank)
                    strengthened = minorant_lower_bound(problem, rank)
                    self.assertGreaterEqual(strengthened, base)
                    # No ALS implementation is used to obtain these candidates.
                    u, singular, vh = np.linalg.svd(problem.matrix, full_matrices=False)
                    candidates = [
                        (u[:, :rank] * singular[:rank]) @ vh[:rank],
                        np.zeros_like(problem.matrix),
                    ]
                    candidates.extend(
                        self.rng.normal(size=(6, rank))
                        @ self.rng.normal(size=(5, rank)).T
                        for _ in range(3)
                    )
                    for candidate in candidates:
                        error = np.sqrt(problem.energy(problem.matrix - candidate))
                        self.assertLessEqual(strengthened, error + 2e-11)

    def test_scalar_operators_and_constant_potential_give_exact_svd_bound(self):
        matrix = self.rng.normal(size=(7, 5))
        problem = CutProblem(
            matrix, 2 * np.eye(7), 3 * np.eye(5), np.full((7, 5), 7.0)
        )
        singular = np.linalg.svd(matrix, compute_uv=False)
        for rank in (0, 1, 2, 4, 5):
            expected = np.sqrt(12 * np.sum(singular[rank:] ** 2))
            self.assertAlmostEqual(minorant_lower_bound(problem, rank), expected, places=11)
        self.assertGreater(minorant_lower_bound(problem, 2), rank_lower_bound(problem, 2))

    def test_bound_does_not_modify_cut_problem(self):
        problem = self.problem()
        attributes = ("matrix", "left", "right", "potential", "_left_root", "_right_root")
        before = {name: getattr(problem, name).copy() for name in attributes}
        minorant_lower_bound(problem, 2)
        for name in attributes:
            np.testing.assert_array_equal(getattr(problem, name), before[name])

    def test_zero_potential_and_full_rank_skip_extra_factorizations(self):
        problem = self.problem()
        empty_potential = CutProblem(
            problem.matrix, problem.left, problem.right, np.zeros_like(problem.potential)
        )
        with patch("earlyht.potential_bounds.cholesky", side_effect=AssertionError("extra Cholesky")):
            self.assertEqual(
                minorant_lower_bound(empty_potential, 2), rank_lower_bound(empty_potential, 2)
            )
            self.assertEqual(minorant_lower_bound(problem, 100), 0)

    def test_invalid_inputs_follow_existing_rank_convention(self):
        problem = self.problem()
        for rank in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                minorant_lower_bound(problem, rank)
        for potential in (
            -np.ones((2, 3)), np.full((2, 3), np.nan),
            np.ones(3), np.zeros((0, 3)), np.ones((2, 3), dtype=complex),
        ):
            with self.assertRaises(ValueError):
                list(additive_minorants(potential))


if __name__ == "__main__":
    unittest.main()
