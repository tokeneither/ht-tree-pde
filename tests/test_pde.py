"""Independent operator identities and residual certificates on small systems."""

import unittest
from unittest.mock import patch

import numpy as np

from earlyht.pde import (
    BALANCED_CUTS,
    PDEProblem,
    build_suite,
    cg_probes,
    matricize,
    reference_solve,
    unmatricize,
)


def small_problem(shape=(2, 3, 2, 3), seed=4):
    rng = np.random.default_rng(seed)
    operators = []
    for n in shape:
        raw = rng.standard_normal((n, n))
        operators.append(raw.T @ raw + np.eye(n))
    return PDEProblem(
        "small", shape, tuple(operators), rng.random(shape), rng.normal(size=shape)
    )


def explicit_operator(problem):
    size = int(np.prod(problem.shape))
    operator = np.diag(problem.potential.ravel())
    for active, axis_matrix in enumerate(problem.axis_operators):
        term = np.ones((1, 1))
        for axis, n in enumerate(problem.shape):
            term = np.kron(term, axis_matrix if axis == active else np.eye(n))
        operator += term
    assert operator.shape == (size, size)
    return operator


class PDETests(unittest.TestCase):
    def test_suite_uses_analytic_dirichlet_coercivity_without_eigensolver(self):
        with patch("earlyht.pde.np.linalg.eigvalsh",
                   side_effect=AssertionError("Dirichlet spectrum is analytic")):
            suite = build_suite(n=8)
        for problem in suite:
            n = problem.shape[0]
            expected = (
                4 * (n + 1) ** 2 * np.sin(np.pi / (2 * (n + 1))) ** 2
                * sum(problem.metadata["diffusion"])
                + np.min(problem.potential)
            )
            np.testing.assert_allclose(problem.coercivity_lower, expected, rtol=1e-15)

    def test_perturbed_dirichlet_stencil_uses_general_spectral_estimate(self):
        n = 5
        operator = 36. * (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
        operator[-1, -1] -= 1e-10
        eigvalsh = np.linalg.eigvalsh
        with patch("earlyht.pde.np.linalg.eigvalsh", wraps=eigvalsh) as spectral:
            problem = PDEProblem("perturbed", (n,), (operator,),
                                 np.zeros(n), np.ones(n))
        spectral.assert_called_once()
        self.assertLess(problem.coercivity_lower, eigvalsh(operator)[0])

    def test_general_coercivity_allows_small_upward_eigenvalue_roundoff(self):
        operator = np.diag([1., 4., 9.])
        with patch("earlyht.pde.np.linalg.eigvalsh",
                   return_value=np.array([1. + 1e-14, 4., 9.])):
            problem = PDEProblem("general", (3,), (operator,),
                                 np.zeros(3), np.ones(3))
        self.assertGreater(problem.coercivity_lower, 0.)
        self.assertLess(problem.coercivity_lower, 1.)

    def test_unresolved_positive_coercivity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ill-conditioned"):
            PDEProblem("near-singular", (2,), (np.diag([1e-16, 1.]),),
                       np.zeros(2), np.ones(2))

    def test_apply_and_energy_match_independent_kronecker_assembly(self):
        problem = small_problem()
        operator = explicit_operator(problem)
        vector = np.random.default_rng(10).normal(size=np.prod(problem.shape))
        tensor = vector.reshape(problem.shape)
        np.testing.assert_allclose(problem.apply(tensor).ravel(), operator @ vector)
        np.testing.assert_allclose(problem.as_linear_operator() @ vector, operator @ vector)
        np.testing.assert_allclose(problem.energy(tensor), vector @ operator @ vector)
        self.assertLessEqual(problem.coercivity_lower, np.linalg.eigvalsh(operator)[0] + 1e-12)

    def test_cut_operator_identity_including_nonsorted_cut(self):
        problem = small_problem()
        tensor = np.random.default_rng(8).normal(size=problem.shape)
        for cut in [*BALANCED_CUTS, (2, 0), (1,)]:
            with self.subTest(cut=cut):
                matrix = matricize(tensor, cut)
                np.testing.assert_array_equal(unmatricize(matrix, problem.shape, cut), tensor)
                row, col, potential = problem.cut_operators(cut)
                np.testing.assert_allclose(
                    matricize(problem.apply(tensor), cut),
                    row @ matrix + matrix @ col + potential * matrix,
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_residual_energy_bound_uses_true_residual(self):
        problem = small_problem()
        operator = explicit_operator(problem)
        exact = np.linalg.solve(operator, problem.rhs.ravel()).reshape(problem.shape)
        for preconditioner in ["none", "jacobi"]:
            with self.subTest(preconditioner=preconditioner):
                probes = cg_probes(
                    problem,
                    maxiter=100,
                    rtol=1e-11,
                    checkpoints=(0, 1, 2, 4, 8, 12),
                    preconditioner=preconditioner,
                )
                self.assertEqual(probes[0]["iteration"], 0)
                self.assertTrue(probes[-1]["converged"])
                for probe in probes:
                    error = exact - probe["w"]
                    energy_error = np.sqrt(problem.energy(error))
                    self.assertLessEqual(energy_error, probe["delta"] * (1 + 1e-9) + 1e-13)
                    residual_norm = np.linalg.norm(problem.rhs - problem.apply(probe["w"]))
                    np.testing.assert_allclose(probe["residual_norm"], residual_norm)
                self.assertTrue(all(
                    first["elapsed"] <= second["elapsed"]
                    for first, second in zip(probes, probes[1:])
                ))

    def test_reference_solve_matches_dense_solution(self):
        problem = small_problem()
        reference = reference_solve(problem)
        expected = np.linalg.solve(explicit_operator(problem), problem.rhs.ravel())
        self.assertTrue(reference["converged"])
        self.assertLess(reference["relative_residual"], 1e-11)
        np.testing.assert_allclose(reference["w"].ravel(), expected, rtol=1e-10, atol=1e-12)

    def test_probe_final_iterate_is_retained_and_zero_rhs_is_handled(self):
        problem = small_problem()
        probes = cg_probes(problem, maxiter=3, checkpoints=(0, 1), rtol=1e-14)
        self.assertEqual([probe["iteration"] for probe in probes], [0, 1, 3])
        problem.rhs[:] = 0
        probes = cg_probes(problem)
        self.assertEqual(len(probes), 1)
        self.assertTrue(probes[0]["converged"])
        self.assertEqual(probes[0]["delta"], 0)

    def test_suite_and_probes_do_not_call_reference(self):
        with patch("earlyht.pde.reference_solve", side_effect=AssertionError("Reference leak")):
            suite = build_suite(n=4)
            self.assertEqual(len(suite), 9)
            self.assertEqual(len({problem.name for problem in suite}), len(suite))
            self.assertEqual(
                {problem.metadata["family"] for problem in suite},
                {"paired", "competing", "anisotropic", "potential"},
            )
            self.assertTrue(all(np.min(problem.potential) >= 0 for problem in suite))
            for problem in suite:
                self.assertEqual(problem.rhs.shape, (4,) * 4)
                self.assertGreater(np.linalg.norm(problem.rhs), 0)
                cg_probes(problem, maxiter=2)

    def test_suite_is_reproducible(self):
        first, second = build_suite(n=4, seeds=(7,)), build_suite(n=4, seeds=(7,))
        self.assertEqual(len(first), 4)
        self.assertEqual(len(second), 4)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left.rhs, right.rhs)
            np.testing.assert_array_equal(left.potential, right.potential)
            self.assertEqual(left.metadata, right.metadata)

    def test_invalid_cut_and_negative_potential_are_rejected(self):
        problem = small_problem()
        with self.assertRaises(ValueError):
            matricize(problem.rhs, (1, 1))
        with self.assertRaises(ValueError):
            PDEProblem(
                "negative", problem.shape, problem.axis_operators,
                -np.ones(problem.shape), problem.rhs,
            )


if __name__ == "__main__":
    unittest.main()
