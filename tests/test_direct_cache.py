"""Independent oracle and repeated-forcing checks for cached direct setup."""
import unittest
import numpy as np
from earlyht.direct_full import prepare_direct_operator, select_direct_full
from earlyht.pde import PDEProblem, matricize, BALANCED_CUTS
from earlyht.selection import cut_name


class DirectCacheTests(unittest.TestCase):
    def test_repeated_forcing_preserves_intervals_and_actual_candidate_gain(self):
        rng = np.random.default_rng(874)
        p = PDEProblem('cache', (2,)*4, tuple(np.eye(2) for _ in range(4)),
                       np.full((2,)*4, 2.), np.zeros((2,)*4))
        cache = prepare_direct_operator(p, metric_policy='compact')
        for scale in (1., 3., 0.):
            p.rhs[:] = scale*rng.normal(size=p.shape)
            cached = select_direct_full(p, sweeps=(0,2), metric_policy='compact',
                                         cache=cache, return_approximation=True)
            fresh = select_direct_full(p, sweeps=(0,2), metric_policy='compact')
            self.assertEqual(cached['bounds'], fresh['bounds'])
            self.assertEqual(cached['selected'], fresh['selected'])
            self.assertFalse(cached['setup_included'])
            x = cached['approximation']
            gain = float(2*np.sum(p.rhs*x)-p.energy(x))
            self.assertAlmostEqual(gain,cached['bounds'][cached['selected']]['lower'],places=11)
            for cut in BALANCED_CUTS:
                exact = np.linalg.svd(matricize(p.rhs,cut),compute_uv=False)[0]**2/6
                self.assertAlmostEqual(cached['bounds'][cut_name(cut)]['upper'],exact,places=11)

    def test_cache_rejects_different_problem_or_policy(self):
        p=PDEProblem('one',(2,)*4,tuple(np.eye(2) for _ in range(4)),
                     np.zeros((2,)*4),np.ones((2,)*4))
        other=PDEProblem('two',p.shape,p.axis_operators,p.potential,p.rhs)
        cache=prepare_direct_operator(p,metric_policy='compact')
        with self.assertRaises(ValueError):
            select_direct_full(other,cache=cache,metric_policy='compact')
        with self.assertRaises(ValueError):
            select_direct_full(p,cache=cache,metric_policy='full')


if __name__ == '__main__':
    unittest.main()
