import unittest

from earlyht.selection import compare_bounds


class CertificateTests(unittest.TestCase):
    def test_separated_intervals_certify(self):
        result = compare_bounds({'a': {'lower': 1., 'upper': 1.1},
                                 'b': {'lower': 2., 'upper': 2.4}}, .2)
        self.assertTrue(result['certified'])
        self.assertEqual(result['selected'], 'a')
        self.assertAlmostEqual(result['margin'], .5)

    def test_close_contenders_abstain(self):
        result = compare_bounds({'a': {'lower': 1., 'upper': 1.1},
                                 'b': {'lower': 1.2, 'upper': 1.3}}, .2)
        self.assertFalse(result['certified'])

    def test_small_upper_alone_is_not_certificate(self):
        result = compare_bounds({'a': {'lower': 0., 'upper': .1},
                                 'b': {'lower': 0., 'upper': 20.}}, 0.)
        self.assertFalse(result['certified'])

    def test_distance_lipschitz_intervals(self):
        # Distances to singleton approximation families in Euclidean 2-space.
        import numpy as np
        actual, proxy = np.array([1., 1.]), np.array([1.05, 1.])
        families = {'a': np.array([1., 0.]), 'b': np.array([4., 1.])}
        errors = {key: float(np.linalg.norm(proxy-value)) for key, value in families.items()}
        decision = compare_bounds({key: {'lower': e, 'upper': e} for key, e in errors.items()},
                                  float(np.linalg.norm(actual-proxy)))
        self.assertTrue(decision['certified'])
        actual_errors = {key: float(np.linalg.norm(actual-value)) for key,value in families.items()}
        self.assertEqual(decision['selected'], min(actual_errors, key=actual_errors.get))

    def test_screened_selection_on_known_rank_separation(self):
        import numpy as np
        from earlyht.pde import PDEProblem
        from earlyht.selection import select_tree
        pair=np.eye(2)/np.sqrt(2.)
        tensor=np.einsum('ij,kl->ijkl',pair,pair)
        problem=PDEProblem('known',(2,)*4,tuple(np.eye(2) for _ in range(4)),
                           np.zeros((2,)*4),4*tensor)
        decision=select_tree(problem,tensor,0.,1,strategy='screened')
        self.assertTrue(decision['certified'])
        self.assertEqual(decision['selected'],'12|34')
        self.assertEqual(decision['als_refinements'],0)
        for name in ('13|24','14|23'):
            self.assertAlmostEqual(decision['bounds'][name]['lower'],np.sqrt(3.))

    def test_screened_selection_abstains_on_shared_exact_representation(self):
        import numpy as np
        from earlyht.pde import PDEProblem
        from earlyht.selection import select_tree
        tensor=np.ones((2,)*4)
        problem=PDEProblem('tie',(2,)*4,tuple(np.eye(2) for _ in range(4)),
                           np.zeros((2,)*4),4*tensor)
        decision=select_tree(problem,tensor,0.,1,strategy='screened')
        self.assertFalse(decision['certified'])
        self.assertEqual(decision['als_refinements'],0)


if __name__ == '__main__':
    unittest.main()
