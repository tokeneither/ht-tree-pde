"""Independent two-cone oracles for parameter-certificate transport."""
import json
import unittest

import numpy as np

from earlyht.reuse import (
    CertificateLibrary,
    ReferenceCertificate,
    feasible_direction_correlations,
    parameter_gram,
)


class ReuseTests(unittest.TestCase):
    # A = diag(4, 1), K0 = span(e0), K1 = span(e1).  The exact family
    # gains are f0**2 / 4 and f1**2, without any numerical optimization.
    operator = np.array([4., 1.])
    basis = np.array([[4., 1.], [2., 0.], [0., 1.]])
    theta0 = np.array([1., 0., 0.])

    def make_library(self, *, direction=False, homogeneous_scaling=False):
        gram = parameter_gram(self.basis, inverse_diagonal=1/self.operator)
        library = CertificateLibrary(gram, 2, homogeneous_scaling=homogeneous_scaling)
        correlations = None
        if direction:
            correlations = feasible_direction_correlations(
                self.basis, [7., 0.], operator_diagonal=self.operator)
        library.add_reference(self.theta0, [4., 1.], [4., 1.], winner=0,
                              direction_correlations=correlations)
        return library

    def exact_gains(self, theta):
        forcing = np.asarray(theta) @ self.basis
        return forcing**2 / self.operator

    def assert_valid_intervals(self, result, gains):
        lower, upper = np.array(result['lower']), np.array(result['upper'])
        scale = max(1., float(np.max(gains)))
        self.assertTrue(np.all(lower <= gains + 1e-11*scale))
        self.assertTrue(np.all(upper >= gains - 1e-11*scale))
        if result['certified']:
            self.assertEqual(result['selected_index'], int(np.argmax(gains)))
            self.assertNotEqual(gains[0], gains[1])

    def test_gram_matches_independent_inverse_and_callable(self):
        expected = self.basis @ np.diag([.25, 1.]) @ self.basis.T
        diagonal = parameter_gram(self.basis, inverse_diagonal=[.25, 1.])
        calls = []

        def inverse(vector):
            calls.append(vector.copy())
            return np.linalg.solve(np.diag(self.operator), vector)

        action = parameter_gram(self.basis, apply_inverse_minorant=inverse)
        self.assertEqual(len(calls), 3)
        np.testing.assert_allclose(action, diagonal, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(diagonal, expected, rtol=1e-12, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(diagonal-expected)[0], -1e-14)
        # The dependent basis has a null direction; Gram construction retains
        # safety by adding a tiny positive shift, not a negative eigenvalue.
        self.assertGreaterEqual(np.linalg.eigvalsh(diagonal)[0], 0.)

    def test_certified_region_and_real_switch(self):
        library = self.make_library()
        result = library.query([1., -.3, .2])
        self.assertTrue(result['certified'])
        self.assertAlmostEqual(result['reference_distances'][0], np.sqrt(.13), places=11)
        self.assert_valid_intervals(result, np.array([2.89, 1.44]))
        switched = library.query([1., -.6, .6])
        self.assertFalse(switched['certified'])
        self.assertIsNone(switched['selected_index'])
        self.assert_valid_intervals(switched, np.array([1.96, 2.56]))

    def test_weaker_minorant_is_conservative(self):
        weaker = parameter_gram(self.basis, inverse_diagonal=[.5, 1.])
        library = CertificateLibrary(weaker, 2)
        library.add_reference(self.theta0, [4., 1.], [4., 1.])
        result = library.query([1., -.3, .2])
        self.assertAlmostEqual(result['reference_distances'][0]**2, .22, places=10)
        self.assertTrue(result['certified'])
        self.assert_valid_intervals(result, np.array([2.89, 1.44]))

    def test_arbitrary_queries_never_false_certify(self):
        rng = np.random.default_rng(71093)
        for homogeneous_scaling in (False, True):
            library = self.make_library(direction=True,
                                        homogeneous_scaling=homogeneous_scaling)
            # Include changing overall amplitude, signs, and large excursions,
            # instead of checking only points inside the sufficient ball.
            for theta in rng.normal(size=(200, 3)) * [2., 4., 4.]:
                result = library.query(theta)
                self.assert_valid_intervals(result, self.exact_gains(theta))

    def test_non_diagonal_spd_with_multiple_exact_ray_families(self):
        rng = np.random.default_rng(30081)
        matrix = rng.normal(size=(7, 7))
        operator = matrix.T @ matrix + np.eye(7)
        basis = rng.normal(size=(4, 7))
        directions = rng.normal(size=(5, 7))
        energies = np.einsum('ni,ij,nj->n', directions, operator, directions)
        gram = parameter_gram(basis,
            apply_inverse_minorant=lambda v: np.linalg.solve(operator, v))
        library = CertificateLibrary(gram, 5, homogeneous_scaling=True)

        def exact(theta):
            return (directions @ (theta @ basis))**2 / energies

        for theta in rng.normal(size=(3, 4)):
            gains = exact(theta)
            winner = int(np.argmax(gains))
            correlations = feasible_direction_correlations(basis, directions[winner],
                apply_operator=lambda v: operator @ v)
            library.add_reference(theta, gains, gains, winner=winner,
                                  direction_correlations=correlations)
        for theta in rng.normal(size=(50, 4)):
            result, gains = library.query(theta), exact(theta)
            self.assertTrue(np.all(np.array(result['lower']) <= gains+1e-10))
            self.assertTrue(np.all(np.array(result['upper']) >= gains-1e-10))
            if result['certified']:
                self.assertEqual(result['selected_index'], int(np.argmax(gains)))

    def test_exact_tie_and_sufficient_region_boundary_abstain(self):
        library = self.make_library()
        # Here delta=1/2 and both transported bounds equal 9/4.  The real
        # family winner is still strict, but the sufficient test must abstain.
        boundary = library.query([1., -.5, 0.])
        self.assertFalse(boundary['certified'])
        self.assertLessEqual(boundary['margin'], boundary['guard'])
        # Actual gains are equal to 9/4; adding exact anchors cannot certify.
        library.add_reference([1., -.5, .5], [2.25, 2.25], [2.25, 2.25])
        self.assertFalse(library.query([1., -.5, .5])['certified'])
        self.assertFalse(library.query([0., 0., 0.])['certified'])

    def test_empty_library_safe_fallback_and_strict_json(self):
        gram = parameter_gram(self.basis, inverse_diagonal=1/self.operator)
        library = CertificateLibrary(gram, 2)
        result = library.query([1., 0., 0.])
        self.assertFalse(result['certified'])
        self.assertEqual(result['n_references'], 0)
        self.assertEqual(result['lower'], [0., 0.])
        self.assertTrue(all(value >= 5. for value in result['upper']))
        json.dumps(result, allow_nan=False)

    def test_directional_region_accepts_beyond_symmetric_ball(self):
        plain = self.make_library()
        direction = self.make_library(direction=True)
        query = [1., 0., .6]
        self.assertFalse(plain.query(query)['certified'])
        result = direction.query(query)
        self.assertTrue(result['certified'])
        self.assertEqual(result['directional_lower'], [4., 0.])
        self.assert_valid_intervals(result, self.exact_gains(query))

    def test_direction_gain_requires_ray_optimization(self):
        vector = np.array([7., 0.])
        c = feasible_direction_correlations(self.basis, vector,
            apply_operator=lambda v: self.operator*v)
        theta = np.array([1., 0., .6])
        forcing = theta @ self.basis
        ray_gain = float(theta @ c)**2
        unchanged_gain = float(2*forcing @ vector - vector @ (self.operator*vector))
        rescaled = (forcing @ vector)/(vector @ (self.operator*vector))*vector
        actual_gain = float(2*forcing @ rescaled - rescaled @ (self.operator*rescaled))
        self.assertEqual(ray_gain, 4.)
        self.assertEqual(actual_gain, ray_gain)
        self.assertLess(unchanged_gain, 0.)

    def test_homogeneous_transport_handles_signed_amplitudes(self):
        library = self.make_library(homogeneous_scaling=True)
        reference = library.query(self.theta0)
        for amplitude in (1e-6, .01, .5, 2., 1000., -1., -3.):
            result = library.query(amplitude*self.theta0)
            self.assertTrue(result['certified'])
            self.assertEqual(result['selected_index'], 0)
            np.testing.assert_allclose(result['lower'], amplitude**2*np.array(reference['lower']),
                                       rtol=1e-12, atol=1e-24)
            np.testing.assert_allclose(result['upper'], amplitude**2*np.array(reference['upper']),
                                       rtol=1e-12, atol=1e-24)
            self.assertAlmostEqual(result['reference_scale_factors'][0], amplitude)
            self.assert_valid_intervals(result, self.exact_gains(amplitude*self.theta0))
        self.assertFalse(library.query(np.zeros(3))['certified'])

    def test_homogeneous_transport_never_weakens_intervals(self):
        plain, scaled = self.make_library(), self.make_library(homogeneous_scaling=True)
        rng = np.random.default_rng(309)
        for theta in rng.normal(size=(40, 3)):
            before, after = plain.query(theta), scaled.query(theta)
            self.assertTrue(np.all(np.array(after['lower']) >= np.array(before['lower'])-1e-12))
            self.assertTrue(np.all(np.array(after['upper']) <= np.array(before['upper'])+1e-12))

    def test_multiple_references_intersect_and_allow_new_winner(self):
        gram = np.diag([.25, 1.])
        library = CertificateLibrary(gram, 2, homogeneous_scaling=True)
        library.add_reference([4., 1.], [4., 1.], [4., 1.])
        old = library.query([1., 4.])
        library.add_reference([1., 4.], [.25, 16.], [.25, 16.], winner=1,
                              direction_correlations=[0., 1.])
        result = library.query([1., 4.])
        self.assertEqual(result['n_references'], 2)
        self.assertTrue(result['certified'])
        self.assertEqual(result['selected_index'], 1)
        self.assertTrue(np.all(np.array(result['upper']) <= np.array(old['upper'])))
        self.assertTrue(np.all(np.array(result['lower']) >= np.array(old['lower'])))
        self.assert_valid_intervals(result, np.array([.25, 16.]))

    def test_affine_offset_and_null_parameter_directions(self):
        library = self.make_library(homogeneous_scaling=True)
        # [1, -2, -1] is a forcing-null parameter direction.
        result = library.query(self.theta0 + np.array([1., -2., -1.]))
        self.assertTrue(result['certified'])
        self.assert_valid_intervals(result, np.array([4., 1.]))
        self.assertLess(result['reference_distances'][0], 1e-5)
        zeros = CertificateLibrary(np.zeros((2, 2)), 2, homogeneous_scaling=True)
        zeros.add_reference([1., 2.], [0., 0.], [0., np.inf])
        outcome = zeros.query([8., -3.])
        self.assertFalse(outcome['certified'])
        self.assertEqual(outcome['upper'], [0., 0.])
        json.dumps(outcome, allow_nan=False)

    def test_reference_copies_inputs_and_validates_dimensions(self):
        theta, lower, upper = np.array([1., 0.]), np.array([4., 1.]), np.array([4., 1.])
        library = CertificateLibrary(np.diag([4., 1.]), 2)
        library.add_reference(theta, lower, upper)
        theta[0], lower[0], upper[1] = 100., 100., 0.
        result = library.query([1., 0.])
        self.assertTrue(result['certified'])
        np.testing.assert_array_equal(library.references[0].theta, [1., 0.])
        with self.assertRaises(ValueError):
            library.references[0].theta[0] = 4.
        with self.assertRaises(ValueError):
            library.add_reference([1.], [1., 0.], [1., 0.])
        with self.assertRaises(ValueError):
            library.add_reference([1., 0.], [1.], [1.])
        with self.assertRaises(ValueError):
            library.query([1.])

    def test_invalid_gram_bounds_directions_and_callables(self):
        for gram in ([[1., 2.], [0., 1.]], [[1., 0.], [0., -1.]],
                     [[np.nan]], [[np.inf]], [1., 2.], [], np.eye(2)*1j):
            with self.subTest(gram=gram), self.assertRaises(ValueError):
                CertificateLibrary(gram, 2)
        for count in (0, 1, 2.5, True):
            with self.assertRaises(ValueError):
                CertificateLibrary(np.eye(2), count)
        for guard in (-1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                CertificateLibrary(np.eye(2), 2, relative_guard=guard)
        for lower, upper in (([-1., 0.], [1., 1.]), ([2., 0.], [1., 1.]),
                             ([np.nan, 0.], [1., 1.]), ([0., 0.], [1., np.nan])):
            with self.assertRaises(ValueError):
                ReferenceCertificate([1., 0.], lower, upper)
        with self.assertRaises(ValueError):
            ReferenceCertificate([1., 0.], [1., 0.], [1., 1.], winner=2)
        with self.assertRaises(ValueError):
            ReferenceCertificate([1., 0.], [1., 0.], [1., 1.], winner=0,
                                 direction_correlations=[2., 0.])
        with self.assertRaises(ValueError):
            parameter_gram(self.basis)
        with self.assertRaises(ValueError):
            parameter_gram(self.basis, inverse_diagonal=[1., 1.],
                           apply_inverse_minorant=lambda v: v)
        with self.assertRaises(ValueError):
            parameter_gram(self.basis, inverse_diagonal=[1., -1.])
        with self.assertRaises(ValueError):
            parameter_gram(self.basis, apply_inverse_minorant=lambda v: np.zeros(3))
        with self.assertRaises(ValueError):
            parameter_gram(self.basis, apply_inverse_minorant=lambda v: v[::-1])
        with self.assertRaises(ValueError):
            feasible_direction_correlations(self.basis, [0., 0.],
                                            operator_diagonal=self.operator)
        with self.assertRaises(ValueError):
            feasible_direction_correlations(self.basis, [1., 0., 0.],
                                            operator_diagonal=self.operator)

    def test_inconsistent_cross_reference_bounds_raise(self):
        library = CertificateLibrary(np.eye(2), 2)
        library.add_reference([1., 0.], [.9, .1], [1., .2])
        library.add_reference([1., 0.], [.1, .9], [.2, 1.])
        with self.assertRaises(ArithmeticError):
            library.query([1., 0.])


if __name__ == '__main__':
    unittest.main()
