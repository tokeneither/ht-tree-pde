"""Independent checks of direct variational bounds and projection certificates."""

import unittest
from unittest.mock import patch

import numpy as np

from earlyht.direct_selection import (
    choose_from_gains,
    full_feasible_gain,
    metric_bound,
    product_metrics,
    scaled_gain,
    select_direct,
    spectral_data,
)
from earlyht.pde import PDEProblem, _mode_product, matricize, unmatricize


def diagonal_problem(n=3, seed=10, variable_potential=False):
    rng = np.random.default_rng(seed)
    operators = tuple(np.diag(np.arange(1., n + 1.) ** 2 + axis)
                      for axis in range(4))
    shape = (n,) * 4
    potential = (rng.uniform(.3, 2., size=shape) if variable_potential
                 else np.full(shape, .3))
    return PDEProblem('test_direct', shape, operators, potential,
                      rng.normal(size=shape))


def explicit_separable_operator(problem, shift):
    total = int(np.prod(problem.shape))
    result = shift * np.eye(total)
    for active, operator in enumerate(problem.axis_operators):
        term = np.ones((1, 1))
        for axis, n in enumerate(problem.shape):
            term = np.kron(term, operator if axis == active else np.eye(n))
        result += term
    return result


class DirectMetricTests(unittest.TestCase):
    def test_product_metrics_are_positive_loewner_minorants(self):
        # Widely unequal spectra exercise endpoint and interior products.
        a = np.array([.03, .2, 8., 90.])
        b = np.array([.001, .6, 3.])
        diagonal = a[:, None] + b[None, :]
        count = 0
        for left, right, label in product_metrics(a, b):
            with self.subTest(metric=label):
                self.assertTrue(np.all(left > 0))
                self.assertTrue(np.all(right > 0))
                self.assertGreaterEqual(
                    float(np.min(diagonal - left[:, None] * right[None, :])),
                    -1e-12)
                count += 1
        self.assertGreaterEqual(count, 2)

    def test_nonminorant_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'minorant'):
            metric_bound(np.ones((2, 2)), np.ones(2), np.ones(2),
                         np.full(2, 3.), np.ones(2), 1)

    def test_geometric_bound_equals_known_diagonal_global_optimum(self):
        a = np.array([1., 4., 9.])
        diagonal_f = np.array([3., 7., 2.])
        forcing = np.diag(diagonal_f)
        exact_terms = diagonal_f ** 2 / (2 * a)
        for rank in (1, 2, 3):
            selected = np.argsort(exact_terms)[-rank:]
            optimum = np.zeros_like(forcing)
            optimum[selected, selected] = diagonal_f[selected] / (2 * a[selected])
            upper, _ = metric_bound(forcing, a, a, 2 * np.sqrt(a),
                                    np.sqrt(a), rank, optimum)
            self.assertAlmostEqual(upper, float(np.sum(exact_terms[selected])),
                                   places=12)

    def test_centered_bound_can_close_a_loose_scalar_metric(self):
        # A = 10 I and B = 5 I. The raw B bound is twice the optimum.
        forcing = np.diag([6., 3., 1.])
        a, b = np.full(3, 4.), np.full(3, 6.)
        center = np.diag([.6, .3, 0.])
        raw, _ = metric_bound(forcing, a, b, np.full(3, 5.),
                              np.ones(3), 2)
        centered, _ = metric_bound(forcing, a, b, np.full(3, 5.),
                                   np.ones(3), 2, center)
        self.assertAlmostEqual(raw, 9.)
        self.assertAlmostEqual(centered, 4.5)

    def test_random_feasible_gains_never_exceed_centered_upper(self):
        rng = np.random.default_rng(44)
        a, b = np.array([.4, 1.1, 7., 12.]), np.array([.2, 2., 9.])
        forcing = rng.normal(size=(4, 3))
        diagonal = a[:, None] + b[None, :]
        center = rng.normal(size=(4, 3))
        for rank in (1, 2):
            feasible = [rng.normal(size=(4, rank)) @ rng.normal(size=(rank, 3))
                        for _ in range(20)]
            gains = [float(np.sum(forcing * x)) ** 2 /
                     float(np.sum(diagonal * x ** 2)) for x in feasible]
            for left, right, label in product_metrics(a, b):
                upper, candidate = metric_bound(forcing, a, b, left, right,
                                                rank, center)
                with self.subTest(rank=rank, metric=label):
                    self.assertGreaterEqual(upper + 1e-12, max(gains))
                    self.assertLessEqual(np.linalg.matrix_rank(candidate, tol=1e-10), rank)

    def test_optimal_rescaling_uses_actual_energy(self):
        forcing = np.array([[2., -1.], [3., 4.]])
        diagonal = np.array([[2., 3.], [5., 8.]])
        candidate = np.outer([1., -2.], [.3, .7])
        old_gain = 2 * np.sum(forcing * candidate) - np.sum(diagonal * candidate ** 2)
        gain, fitted = scaled_gain(forcing, diagonal, candidate)
        actual_gain = 2 * np.sum(forcing * fitted) - np.sum(diagonal * fitted ** 2)
        self.assertAlmostEqual(gain, actual_gain, places=12)
        self.assertGreaterEqual(gain, old_gain)
        self.assertAlmostEqual(float(np.sum((forcing - diagonal * fitted) * fitted)),
                               0., places=12)


class DirectProjectionTests(unittest.TestCase):
    def test_tail_matches_independent_full_inverse_energy(self):
        rng = np.random.default_rng(19)
        n = 3
        operators = []
        for axis in range(4):
            q, _ = np.linalg.qr(rng.normal(size=(n, n)))
            operators.append(q @ np.diag([1. + axis, 3. + axis, 8. + axis]) @ q.T)
        problem = PDEProblem('rotated', (n,) * 4, tuple(operators),
                             rng.uniform(.4, 2., size=(n,) * 4),
                             rng.normal(size=(n,) * 4))
        data = spectral_data(problem, modes=2)
        retained = np.zeros(problem.shape)
        retained[(slice(0, 2),) * 4] = data['forcing']
        for axis, q in enumerate(data['eigenvectors']):
            retained = _mode_product(q, retained, axis)
        omitted = (problem.rhs - retained).ravel()
        operator = explicit_separable_operator(problem, data['vmin'])
        exact_tail = float(omitted @ np.linalg.solve(operator, omitted))
        self.assertAlmostEqual(data['tail'], exact_tail, places=11)
        self.assertEqual(spectral_data(problem, modes=n)['tail'], 0.)

    def test_full_feasible_gain_enforces_rank_before_physical_energy(self):
        template = diagonal_problem(variable_potential=True)
        rng = np.random.default_rng(32)
        operators = []
        for axis, diagonal in enumerate(template.axis_operators):
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            operators.append(q @ diagonal @ q.T)
        problem = PDEProblem('rotated_feasible', template.shape, tuple(operators),
                             template.potential, template.rhs)
        data = spectral_data(problem, modes=2)
        cut = (0, 2)
        # Deliberately full rank: only its leading component may support a
        # rank-one family lower bound. This catches numerical-rank heuristics.
        candidate = np.diag([4., 3., 2., 1.])
        truncated = np.diag([4., 0., 0., 0.])
        reduced = unmatricize(truncated, (2,) * 4, cut)
        physical = reduced
        for axis, q in enumerate(data['eigenvectors']):
            physical = np.moveaxis(np.tensordot(q[:, :2], physical, axes=(1, axis)), 0, axis)
        pairing = float(np.sum(problem.rhs * physical))
        expected = pairing ** 2 / problem.energy(physical)
        got = full_feasible_gain(problem, data, cut, candidate, rank=1)
        self.assertAlmostEqual(got, expected, places=12)

    def test_tail_upper_covers_known_full_grid_rank_one_solution(self):
        # A known solution is legitimate here as an independent testing oracle;
        # the selector receives only the resulting forcing, never that solution.
        for variable_potential in (False, True):
            problem = diagonal_problem(variable_potential=variable_potential)
            rng = np.random.default_rng(8)
            solution = np.einsum('i,j,k,l->ijkl',
                                 *(rng.normal(size=3) for _ in range(4)))
            problem.rhs = problem.apply(solution)
            optimum = problem.energy(solution)
            result = select_direct(problem, rank=1, modes=1, sweeps=(0,))
            self.assertGreater(result['spectral_tail_bound'], 0.)
            self.assertFalse(result['certified'])  # All three families contain u.
            for family, bound in result['bounds'].items():
                with self.subTest(potential=variable_potential, family=family):
                    self.assertLessEqual(bound['lower'], optimum + 1e-10)
                    self.assertGreaterEqual(bound['upper'] + 1e-10, optimum)


class DirectSelectionTests(unittest.TestCase):
    def test_analytic_pairing_theorem_certifies_before_als_or_pde_solve(self):
        k = 3
        eigenvalues = np.arange(1., k + 1.) ** 2
        forcing = np.einsum('ij,kl->ijkl', np.eye(k), np.eye(k)) / k
        problem = PDEProblem('pairing_theorem', (k,) * 4,
                             tuple(np.diag(eigenvalues) for _ in range(4)),
                             np.zeros((k,) * 4), forcing)
        forbidden = AssertionError('selection must finish before solving')
        with patch('earlyht.direct_selection.solve_cut', side_effect=forbidden), \
                patch('earlyht.pde.cg_probes', side_effect=forbidden), \
                patch('earlyht.pde.reference_solve', side_effect=forbidden):
            result = select_direct(problem, rank=1, modes=k)
        self.assertTrue(result['certified'])
        self.assertEqual(result['selected'], '12|34')
        self.assertEqual(result['full_pde_iterations'], 0)
        self.assertEqual([step['stage'] for step in result['trace']], [0])
        self.assertGreaterEqual(result['bounds']['12|34']['lower'] + 1e-14,
                                1 / (4 * np.mean(eigenvalues)))
        for tree in ('13|24', '14|23'):
            self.assertAlmostEqual(result['bounds'][tree]['upper'],
                                   1 / (4 * k ** 2 * eigenvalues[0]), places=12)

    def test_gain_intervals_require_global_separation(self):
        tied = choose_from_gains({'a': {'lower': 1., 'upper': 2.},
                                  'b': {'lower': .9, 'upper': 1.5}})
        self.assertEqual(tied['selected'], 'a')
        self.assertFalse(tied['certified'])
        separated = choose_from_gains({'a': {'lower': 1.6, 'upper': 2.},
                                       'b': {'lower': .9, 'upper': 1.5}})
        self.assertTrue(separated['certified'])
        self.assertAlmostEqual(separated['margin'], .1)

    def test_shared_exact_family_abstains(self):
        forcing = np.ones((2,) * 4)
        problem = PDEProblem('tie', (2,) * 4, tuple(np.eye(2) for _ in range(4)),
                             np.zeros((2,) * 4), forcing)
        result = select_direct(problem, rank=1, modes=2, sweeps=(0,))
        self.assertFalse(result['certified'])
        for bound in result['bounds'].values():
            self.assertAlmostEqual(bound['lower'], 4.)
            self.assertAlmostEqual(bound['upper'], 4.)


if __name__ == '__main__':
    unittest.main()
