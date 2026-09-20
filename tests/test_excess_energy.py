"""Independent energy-error checks for approximate selection and tie handling."""
import unittest

import numpy as np

from earlyht.certificates import excess_energy_certificate
from earlyht.ht_bounds import canonical_tree_cuts
from earlyht.ht_search import all_balanced_trees, search_ht
from earlyht.ht_primal import unfold


class ExcessEnergyTests(unittest.TestCase):
    def test_regret_matches_independent_best_matrix_approximation(self):
        # A=I and a matrix rank-one family with exactly known best error 1.
        f = np.diag([3., 1.])
        candidate = np.array([[0., 0.], [0., 1.]])
        gain = 2 * np.sum(f * candidate) - np.sum(candidate**2)
        result = excess_energy_certificate(gain, [9.], tolerance=8., relative_guard=0.)
        actual_excess = np.sum((f-candidate)**2) - 1.
        self.assertEqual(result['excess_energy_bound'], actual_excess)
        self.assertEqual(actual_excess, 8.)
        self.assertTrue(result['epsilon_certified'])

    def test_exact_tie_can_stop_with_solution_quality_but_no_unique_tree(self):
        f = np.ones((2,)*4) / 4
        spectra = tuple(np.full(2, .25) for _ in range(4))
        result = search_ht(f, spectra, rank=1, stages=(0,2), fits_per_stage=3,
                           excess_energy_tolerance=1e-8, return_approximation=True)
        self.assertFalse(result['certified'])
        self.assertTrue(result['epsilon_certified'])
        self.assertEqual(result['termination_reason'], 'excess_energy')
        self.assertEqual(result['tree_fits'], 1)
        self.assertEqual(result['proxy_operator_steps'], 0)
        # Here u=f, and every tree contains it. Verify the ACTUAL tensor.
        error = float(np.sum((f-result['approximation'])**2))
        self.assertLessEqual(error, result['excess_energy_bound'])
        gain = float(2*np.sum(f*result['approximation']) - np.sum(result['approximation']**2))
        self.assertAlmostEqual(gain, result['candidate_gain'], places=13)

    def test_nonzero_regret_and_returned_witness_for_sum_of_separable_terms(self):
        # This orthogonally decomposable tensor has best all-rank-one gain 9
        # for each tree, and optimal squared error 1, independently of search.
        f = np.zeros((2,)*4)
        f[(0,)*4], f[(1,)*4] = 3., 1.
        spectra = tuple(np.full(2, .25) for _ in range(4))
        result = search_ht(f, spectra, rank=1, stages=(0,), fits_per_stage=3,
                           excess_energy_tolerance=1e-7, return_approximation=True)
        self.assertTrue(result['epsilon_certified'])
        candidate = result['approximation']
        self.assertLessEqual(float(np.sum((f-candidate)**2))-1.,
                             result['excess_energy_bound'])
        tree = all_balanced_trees(tuple(range(4)))[result['selected_index']]
        for cut in canonical_tree_cuts(tree,4):
            singular = np.linalg.svd(unfold(candidate,cut),compute_uv=False)
            self.assertLess(float(np.linalg.norm(singular[1:])),1e-10)

    def test_strict_family_does_not_imply_small_candidate_regret(self):
        # A family can beat every rival while its own bound is still loose.
        result = excess_energy_certificate(2., [10., 1.], tolerance=.1)
        self.assertGreater(2., 1.)
        self.assertFalse(result['epsilon_certified'])
        self.assertGreaterEqual(result['excess_energy_bound'], 8.)

    def test_default_still_abstains_on_tie_and_returns_serializable_metadata(self):
        import json
        f = np.ones((2,)*4) / 4
        spectra = tuple(np.full(2,.25) for _ in range(4))
        result = search_ht(f,spectra,rank=1,stages=(0,),fits_per_stage=1)
        self.assertFalse(result['certified'])
        self.assertFalse(result['epsilon_certified'])
        self.assertNotIn('approximation',result)
        json.dumps(result)

    def test_guard_and_invalid_bounds(self):
        result = excess_energy_certificate(1., [1.], tolerance=0.)
        self.assertFalse(result['epsilon_certified'])
        self.assertGreater(result['excess_energy_bound'],0.)
        self.assertFalse(excess_energy_certificate(1., [float('inf')],
                                                   tolerance=1.)['epsilon_certified'])
        with self.assertRaises(ArithmeticError):
            excess_energy_certificate(2., [1.])
        for tolerance in (-1., float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                excess_energy_certificate(0.,[1.],tolerance=tolerance)


if __name__ == '__main__':
    unittest.main()
