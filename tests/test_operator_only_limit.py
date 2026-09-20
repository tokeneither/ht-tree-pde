"""The same Dirichlet operator admits sources with distinct best rank-one cuts."""

import unittest

import numpy as np

from earlyht.pde import BALANCED_CUTS, PDEProblem, matricize, reference_solve
from earlyht.trees import CutProblem, rank_lower_bound


class OperatorOnlyLimitTests(unittest.TestCase):
    def test_delta_pairings_have_distinct_unique_best_cuts(self):
        for n in (2, 3):
            identity = np.eye(n)
            targets = (
                (np.einsum("ij,kl->ijkl", identity, identity), (0, 1)),
                (np.einsum("ik,jl->ijkl", identity, identity), (0, 2)),
            )
            axis = (n + 1) ** 2 * (
                2 * identity - np.eye(n, k=1) - np.eye(n, k=-1)
            )
            shape = (n,) * 4
            problem = PDEProblem("fixed-dirichlet", shape, (axis,) * 4,
                                 np.zeros(shape), np.zeros(shape))
            expected_gap = np.sqrt(problem.coercivity_lower * (n * n - 1))
            for target, matching_cut in targets:
                for cut in BALANCED_CUTS:
                    with self.subTest(n=n, matching=matching_cut, cut=cut):
                        matrix = matricize(target, cut)
                        left, right, potential = problem.cut_operators(cut)
                        lower = rank_lower_bound(
                            CutProblem(matrix, left, right, potential), 1
                        )
                        if cut == matching_cut:
                            self.assertEqual(np.linalg.matrix_rank(matrix), 1)
                            self.assertLess(lower, 1e-12)
                        else:
                            self.assertEqual(np.linalg.matrix_rank(matrix), n * n)
                            np.testing.assert_allclose(
                                matrix @ matrix.T, np.eye(n * n), atol=0, rtol=0
                            )
                            self.assertGreaterEqual(lower, expected_gap * (1 - 1e-12))

    def test_both_constructed_sources_recover_targets_with_the_same_operator(self):
        n = 3
        identity = np.eye(n)
        axis = 16. * (2 * identity - np.eye(n, k=1) - np.eye(n, k=-1))
        shape = (n,) * 4
        problem = PDEProblem("fixed-dirichlet", shape, (axis,) * 4,
                             np.zeros(shape), np.zeros(shape))
        targets = (
            np.einsum("ij,kl->ijkl", identity, identity),
            np.einsum("ik,jl->ijkl", identity, identity),
        )
        sources = [problem.apply(target) for target in targets]
        self.assertFalse(np.array_equal(sources[0], sources[1]))
        for target, source in zip(targets, sources):
            problem.rhs[:] = source
            recovered = reference_solve(problem)
            self.assertTrue(recovered["converged"])
            np.testing.assert_allclose(recovered["w"], target, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
