"""Independent guarantees for cut bounds and feasible alternating fits."""
import unittest

import numpy as np

from earlyht.trees import CutProblem, fit_cut, rank_lower_bound, solve_cut


class CutBoundsTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2026)

    def problem(self, m=6, n=5):
        left = self.rng.normal(size=(m, m))
        right = self.rng.normal(size=(n, n))
        return CutProblem(self.rng.normal(size=(m, n)),
                          left.T @ left + np.eye(m),
                          right.T @ right + np.eye(n),
                          self.rng.uniform(0, 3, (m, n)))

    def test_operator_matches_dense_vectorization(self):
        problem = self.problem(3, 4)
        dense = (np.kron(problem.left, np.eye(4))
                 + np.kron(np.eye(3), problem.right)
                 + np.diag(problem.potential.ravel()))
        matrix = self.rng.normal(size=(3, 4))
        np.testing.assert_allclose(problem.operator(matrix).ravel(),
                                   dense @ matrix.ravel(), atol=2e-13)
        self.assertAlmostEqual(problem.energy(matrix),
                               float(matrix.ravel() @ dense @ matrix.ravel()), places=11)

    def test_isotropic_case_recovers_exact_svd_tail(self):
        matrix = self.rng.normal(size=(7, 5))
        problem = CutProblem(matrix, 2 * np.eye(7), 3 * np.eye(5), np.zeros((7, 5)))
        singular = np.linalg.svd(matrix, compute_uv=False)
        expected = np.sqrt(5 * np.sum(singular[2:] ** 2))
        fitted = fit_cut(problem, 2, starts=3, max_sweeps=6)
        self.assertAlmostEqual(rank_lower_bound(problem, 2), expected, places=11)
        self.assertAlmostEqual(fitted["upper"], expected, places=11)

    def test_orthogonal_eigenspace_example_rank_one(self):
        # A matched low-frequency block and four high-frequency singular pairs.
        epsilon, frequency = 0.1, 20
        matrix = np.diag([1.0] + [epsilon / 2] * 4)
        half_laplacian = np.diag([2.0] + [2.0 * frequency ** 2] * 4)
        problem = CutProblem(matrix, half_laplacian, half_laplacian, np.zeros((5, 5)))
        exact_squared = 4 + 4 * frequency ** 2 * epsilon ** 2 - max(
            4, frequency ** 2 * epsilon ** 2)
        self.assertAlmostEqual(rank_lower_bound(problem, 1), np.sqrt(exact_squared), places=12)
        fitted = fit_cut(problem, 1, starts=3, max_sweeps=30)
        self.assertLessEqual(fitted["lower"], fitted["upper"] + 1e-11)
        self.assertLessEqual(fitted["rank_observed"], 1)

    def test_als_monotone_feasible_and_true_error(self):
        problem = self.problem()
        fitted = fit_cut(problem, 2, starts=3, max_sweeps=30, tol=1e-12)
        self.assertTrue(np.all(np.diff(fitted["history"]) <= 0))
        self.assertLessEqual(fitted["rank_observed"], 2)
        self.assertLessEqual(fitted["lower"], fitted["upper"] + 1e-11)
        self.assertAlmostEqual(fitted["upper"],
                               np.sqrt(problem.energy(problem.matrix - fitted["approximation"])),
                               places=12)
        self.assertAlmostEqual(fitted["upper"], min(fitted["start_uppers"]), places=12)
        self.assertEqual(fitted["sweeps"], len(fitted["history"]) - 1)

    def test_lower_bound_holds_for_independent_feasible_candidates(self):
        problem = self.problem()
        bound = rank_lower_bound(problem, 2)
        for _ in range(6):
            approximation = (self.rng.normal(size=(6, 2))
                             @ self.rng.normal(size=(5, 2)).T)
            self.assertLessEqual(bound, np.sqrt(problem.energy(problem.matrix - approximation)))

    def test_zero_and_full_rank_edges(self):
        problem = self.problem(4, 3)
        zero = fit_cut(problem, 0)
        self.assertEqual(zero["rank_observed"], 0)
        self.assertAlmostEqual(zero["upper"], np.sqrt(problem.energy(problem.matrix)))
        full = fit_cut(problem, 100)
        np.testing.assert_array_equal(full["approximation"], problem.matrix)
        self.assertEqual(full["lower"], 0)
        self.assertEqual(full["upper"], 0)
        empty = CutProblem(np.zeros((4, 3)), problem.left, problem.right, problem.potential)
        self.assertEqual(fit_cut(empty, 2)["upper"], 0)

    def test_input_assumptions_are_enforced(self):
        problem = self.problem()
        for rank in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                fit_cut(problem, rank)
        with self.assertRaises(ValueError):
            CutProblem(np.eye(2), np.eye(2), np.eye(2), -np.ones((2, 2)))
        with self.assertRaises(ValueError):
            CutProblem(np.eye(2), np.diag([1, -1]), np.eye(2), np.zeros((2, 2)))

    def test_forcing_only_solver_matches_full_rank_dense_solution(self):
        problem = self.problem(4, 3)
        rhs = self.rng.normal(size=(4, 3))
        dense = (np.kron(problem.left, np.eye(3))
                 + np.kron(np.eye(4), problem.right)
                 + np.diag(problem.potential.ravel()))
        expected = np.linalg.solve(dense, rhs.ravel()).reshape(rhs.shape)
        solved = solve_cut(problem.left, problem.right, problem.potential,
                           rhs, rank=3, starts=1)
        np.testing.assert_allclose(solved["approximation"], expected, rtol=2e-12, atol=2e-13)
        self.assertAlmostEqual(solved["objective"],
                               -0.5 * float(np.sum(rhs * expected)), places=12)

    def test_forcing_only_solver_rank_constraint_and_proxy_start(self):
        problem = self.problem()
        rhs = self.rng.normal(size=problem.matrix.shape)
        solved = solve_cut(problem.left, problem.right, problem.potential,
                           rhs, rank=2, initial=rhs / 20, starts=3, max_sweeps=40)
        self.assertLessEqual(solved["rank_observed"], 2)
        self.assertTrue(np.all(np.diff(solved["history"]) <= 0))
        self.assertAlmostEqual(solved["objective"], min(solved["start_objectives"]), places=12)
        actual = solved["approximation"]
        self.assertAlmostEqual(solved["objective"],
                               0.5 * problem.energy(actual) - np.sum(rhs * actual), places=12)
        zero = solve_cut(problem.left, problem.right, problem.potential,
                         np.zeros_like(rhs), rank=2)
        self.assertEqual(zero["objective"], 0)
        np.testing.assert_array_equal(zero["approximation"], np.zeros_like(rhs))


if __name__ == "__main__":
    unittest.main()
