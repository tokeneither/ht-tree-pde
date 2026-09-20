"""Compare the spectral benchmarks with independently assembled dense systems."""

import unittest
from unittest.mock import patch

import numpy as np

from earlyht.pde import PDEProblem, build_suite
from earlyht.spectral_baselines import (
    SeparableSpectralInverse,
    spectral_pcg,
    spectral_solve,
)


def dense_operator(problem):
    result = np.diag(problem.potential.ravel())
    for axis, active in enumerate(problem.axis_operators):
        term = np.ones((1, 1))
        for index, n in enumerate(problem.shape):
            term = np.kron(term, active if index == axis else np.eye(n))
        result += term
    return result


def generic_problem(constant=False):
    rng = np.random.default_rng(423)
    shape = (2, 3, 2)
    axes = []
    for n in shape:
        raw = rng.normal(size=(n, n))
        axes.append(raw.T @ raw + np.eye(n))
    potential = np.full(shape, 1.25) if constant else 2 * rng.random(shape)
    return PDEProblem("spectral-test", shape, tuple(axes), potential,
                      rng.normal(size=shape))


class SpectralBaselineTests(unittest.TestCase):
    def test_generic_direct_solve_matches_dense_and_preserves_input(self):
        problem = generic_problem(constant=True)
        source = problem.rhs.copy()
        result = spectral_solve(problem)
        expected = np.linalg.solve(dense_operator(problem), source.ravel())
        np.testing.assert_allclose(result["w"].ravel(), expected, rtol=2e-13, atol=1e-14)
        np.testing.assert_array_equal(problem.rhs, source)
        self.assertTrue(result["converged"])
        self.assertEqual(result["iterations"], 0)
        self.assertGreaterEqual(result["elapsed"], result["setup_seconds"])

    def test_dirichlet_direct_solver_uses_dst_without_eigendecomposition(self):
        problem = build_suite(n=3, seeds=(1,))[0]
        with patch("earlyht.spectral_baselines.np.linalg.eigh",
                   side_effect=AssertionError("No dense eigensolver for stencil axes")):
            result = spectral_solve(problem)
        expected = np.linalg.solve(dense_operator(problem), problem.rhs.ravel())
        np.testing.assert_allclose(result["w"].ravel(), expected, rtol=2e-13, atol=1e-15)
        self.assertEqual(result["transform"], "orthonormal DST-I")
        self.assertTrue(result["converged"])

    def test_mixed_transform_inverse_and_diagonalization(self):
        generic = generic_problem(constant=True)
        axes = list(generic.axis_operators)
        axes[0] = np.array([[8., -4.], [-4., 8.]])
        problem = PDEProblem("mixed", generic.shape, tuple(axes), generic.potential,
                             generic.rhs)
        spectral = SeparableSpectralInverse(problem, 1.25)
        coefficients = spectral.forward(problem.rhs)
        np.testing.assert_allclose(spectral.inverse_transform(coefficients), problem.rhs,
                                   rtol=2e-13, atol=1e-14)
        np.testing.assert_allclose(spectral.forward(problem.apply(problem.rhs)),
                                   spectral.denominator * coefficients,
                                   rtol=2e-13, atol=1e-13)
        self.assertEqual(spectral.dst_axes, (0,))

    def test_variable_potential_pcg_matches_dense_and_counts_work(self):
        problem = generic_problem()
        result = spectral_pcg(problem, rtol=1e-12, maxiter=100)
        expected = np.linalg.solve(dense_operator(problem), problem.rhs.ravel())
        self.assertTrue(result["converged"])
        np.testing.assert_allclose(result["w"].ravel(), expected, rtol=2e-11, atol=1e-13)
        self.assertEqual(result["shift"], float(np.mean(problem.potential)))
        self.assertEqual(result["operator_applications"], result["iterations"] + 1)
        self.assertEqual(result["preconditioner_applications"], result["iterations"])
        self.assertGreaterEqual(result["elapsed"],
                                result["setup_seconds"] + result["solve_seconds"])
        np.testing.assert_allclose(result["residual_norm"],
                                   np.linalg.norm(problem.rhs - problem.apply(result["w"])))

    def test_constant_potential_pcg_finishes_in_one_iteration(self):
        problem = generic_problem(constant=True)
        result = spectral_pcg(problem, rtol=1e-12)
        self.assertTrue(result["converged"])
        self.assertEqual(result["iterations"], 1)

    def test_zero_rhs_and_unsuccessful_exit_are_explicit(self):
        problem = generic_problem()
        limited = spectral_pcg(problem, rtol=1e-14, maxiter=1)
        self.assertFalse(limited["converged"])
        self.assertEqual(limited["iterations"], 1)
        self.assertEqual(limited["scipy_info"], 1)
        problem.rhs[:] = 0
        zero = spectral_pcg(problem, rtol=1e-12)
        self.assertTrue(zero["converged"])
        self.assertEqual(zero["iterations"], 0)
        self.assertEqual(zero["relative_residual"], 0)

    def test_rejects_variable_direct_solve_and_invalid_parameters(self):
        problem = generic_problem()
        with self.assertRaisesRegex(ValueError, "constant"):
            spectral_solve(problem)
        with self.assertRaises(ValueError):
            spectral_pcg(problem, shift=-1)
        with self.assertRaises(ValueError):
            spectral_pcg(problem, maxiter=0)
        with self.assertRaises(ValueError):
            spectral_pcg(problem, rtol=np.nan)
        with self.assertRaises(ValueError):
            spectral_pcg(problem, x0=np.zeros(3))


if __name__ == "__main__":
    unittest.main()
