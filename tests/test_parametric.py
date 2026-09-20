"""Oracle checks for offline affine solutions and distance-to-gain conversion."""
import unittest
import numpy as np
from earlyht.parametric import AffineSolutionBaseline, NAMES
from earlyht.pde import PDEProblem, BALANCED_CUTS, matricize


class ParametricBaselineTests(unittest.TestCase):
    def test_affine_reconstruction_matches_independent_dense_solve(self):
        rng=np.random.default_rng(601)
        shape=(2,)*4
        axes=tuple(np.array([[3.+i,-.4],[-.4,4.+i]]) for i in range(4))
        potential=rng.uniform(0,4,size=shape)
        basis=rng.normal(size=(3,*shape))
        p=PDEProblem('dense',shape,axes,potential,basis[0])
        full=np.diag(potential.ravel())
        for axis,op in enumerate(axes):
            term=np.ones((1,1))
            for j in range(4):
                term=np.kron(term,op if j==axis else np.eye(2))
            full+=term
        solver=AffineSolutionBaseline(p,basis,rtol=1e-12)
        for theta in (np.array([1.,.3,-.4]),np.array([-1.,2.,.2]),np.zeros(3)):
            rhs=np.tensordot(theta,basis,axes=1)
            exact=np.linalg.solve(full,rhs.ravel()).reshape(shape)
            np.testing.assert_allclose(solver.solution(theta),exact,rtol=1e-10,atol=1e-12)
            lo,hi,_=solver.energy_interval(theta)
            true=float(np.sum(rhs*exact))
            self.assertLessEqual(lo,true+1e-12)
            self.assertGreaterEqual(hi,true-1e-12)

    def test_distance_gain_intervals_contain_exact_svd_family_gains(self):
        rng=np.random.default_rng(799)
        shape=(2,)*4
        basis=rng.normal(size=(3,*shape))
        p=PDEProblem('isotropic',shape,tuple(np.eye(2) for _ in range(4)),
                     np.full(shape,2.),basis[0])
        solver=AffineSolutionBaseline(p,basis)
        solver.prepare_certificate()
        for theta in (np.array([1.,.1,-.2]),np.array([.3,-1.,.7])):
            p.rhs[:]=np.tensordot(theta,basis,axes=1)
            result=solver.certificate(theta)
            gains=[]
            for i,cut in enumerate(BALANCED_CUTS):
                exact=np.linalg.svd(matricize(p.rhs,cut),compute_uv=False)[0]**2/6.
                gains.append(exact)
                self.assertLessEqual(result['lower'][i],exact+1e-11)
                self.assertGreaterEqual(result['upper'][i],exact-1e-11)
            if result['certified']:
                self.assertEqual(result['selected_index'],int(np.argmax(gains)))


if __name__=='__main__':
    unittest.main()
