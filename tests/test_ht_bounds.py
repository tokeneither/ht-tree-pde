"""General HT gain bounds: independent energy oracles and multi-edge failures."""

import unittest

import numpy as np

from earlyht.ht_bounds import (CutBoundCache, canonical_tree_cuts, ht_gain_bounds,
                               _nonzero_singular_values)
from earlyht.ht_primal import balanced_tree, candidate_trees, ht_round


def diagonal_from_axes(values, potential=0.):
    shape = tuple(len(value) for value in values)
    diagonal = np.full(shape, float(potential))
    for i, value in enumerate(values):
        reshape = [1] * len(shape)
        reshape[i] = len(value)
        diagonal += value.reshape(reshape)
    return diagonal


class GeneralHTBoundTests(unittest.TestCase):
    def test_exact_zero_compression_preserves_all_nonzero_singular_values(self):
        rng = np.random.default_rng(392)
        small = rng.normal(size=(7, 5))
        padded = np.zeros((23, 19))
        padded[np.ix_([0, 2, 4, 9, 14, 18, 21], [1, 3, 8, 11, 17])] = small
        compressed = _nonzero_singular_values(padded)
        dense = np.linalg.svd(padded, compute_uv=False)
        np.testing.assert_allclose(compressed, dense[:5], rtol=1e-13, atol=1e-14)
        self.assertLess(np.linalg.norm(dense[5:]), 1e-13)
        self.assertEqual(len(_nonzero_singular_values(np.zeros((7, 9)))), 0)

    def test_edge_cuts_remove_complements_but_keep_leaf_constraints(self):
        tree = balanced_tree(range(8))
        cuts = canonical_tree_cuts(tree, 8)
        self.assertEqual(len(cuts), 13)
        self.assertEqual(sum(len(cut) == 1 for cut in cuts), 8)
        self.assertEqual(sum(len(cut) == 2 for cut in cuts), 4)
        self.assertEqual(sum(len(cut) == 4 for cut in cuts), 1)
        reversed_tree = (tree[1], tree[0])
        self.assertEqual(cuts, canonical_tree_cuts(reversed_tree, 8))

    def test_eight_mode_bounds_cover_known_feasible_solution(self):
        rng = np.random.default_rng(240)
        values = tuple(np.array([.3 + i, 2. + i, 7. + i]) for i in range(8))
        diagonal = diagonal_from_axes(values, .7)
        for tree in candidate_trees(8, count=2, seed=52):
            solution = ht_round(rng.normal(size=(3,) * 8), tree, 2)["approximation"]
            forcing = diagonal * solution  # Independent manufactured test oracle only.
            exact_energy = float(np.sum(solution * forcing))
            result = ht_gain_bounds(forcing, values, tree, 2, solution, potential=.7)
            self.assertAlmostEqual(result["lower"] / exact_energy, 1., places=12)
            self.assertGreaterEqual(result["upper"] + 1e-10 * exact_energy, exact_energy)
            self.assertLess(result["max_relative_rank_tail"], 1e-12)
            self.assertEqual(result["full_pde_iterations"], 0)

    def test_scalar_operator_closes_bounds_for_every_feasible_ht_solution(self):
        rng = np.random.default_rng(43)
        values = tuple(np.ones(2) * (i + 1) / 36 for i in range(8))
        tree = balanced_tree([3, 6, 1, 7, 0, 4, 2, 5])
        solution = ht_round(rng.normal(size=(2,) * 8), tree, 2)["approximation"]
        result = ht_gain_bounds(solution, values, tree, 2, solution)
        expected = float(np.sum(solution ** 2))
        self.assertAlmostEqual(result["lower"], expected, places=11)
        self.assertAlmostEqual(result["upper"], expected, places=11)
        self.assertEqual(result["bounded_edges"], 5)

    def test_root_feasible_but_inner_edge_infeasible_candidate_is_rejected(self):
        rng = np.random.default_rng(96)
        # A product across the root has root rank 1, but each arbitrary
        # four-mode factor generally has pair-cut rank 4 > requested rank 2.
        candidate = np.multiply.outer(rng.normal(size=(2,) * 4),
                                      rng.normal(size=(2,) * 4))
        values = tuple(np.ones(2) for _ in range(8))
        with self.assertRaisesRegex(ValueError, "violates the HT rank"):
            ht_gain_bounds(np.ones_like(candidate), values, balanced_tree(range(8)),
                           2, candidate)

    def test_shared_unrestricted_centers_cache_all_315_balanced_trees(self):
        rng = np.random.default_rng(15)
        forcing = rng.normal(size=(2,) * 8)
        values = tuple(np.array([1., 4. + i]) for i in range(8))
        center = rng.normal(size=forcing.shape) * .01  # Full rank is allowed.
        cache = CutBoundCache(forcing, values, 2, center=center)
        trees = candidate_trees(8, count=315, seed=62)
        self.assertEqual(len(trees), 315)
        for tree in trees:
            interval = ht_gain_bounds(forcing, values, tree, 2, cache=cache,
                                      centered_cuts=0)
            self.assertLessEqual(interval["upper"], interval["zero_center_upper"] + 1e-12)
        self.assertEqual(len(cache._states), 63)
        self.assertEqual(cache.svd_evaluations, 2 * 63)
        self.assertGreater(cache.shared_cache_hits, 1000)

    def test_new_shared_center_intersects_existing_bounds_without_recomputation(self):
        rng = np.random.default_rng(113)
        forcing = rng.normal(size=(2,) * 8)
        values = tuple(np.array([1., 5.]) for _ in range(8))
        cache = CutBoundCache(forcing, values, 2)
        cut = (0, 1, 2, 3)
        first = cache.upper_for_cut(cut)
        self.assertEqual(cache.svd_evaluations, 1)
        self.assertEqual(cache.upper_for_cut(tuple(range(4, 8))), first)
        self.assertEqual(cache.svd_evaluations, 1)
        cache.add_center(.01 * forcing)
        second = cache.upper_for_cut(cut)
        self.assertEqual(cache.svd_evaluations, 2)
        self.assertLessEqual(second["upper"], first["upper"])
        self.assertEqual(cache.upper_for_cut(cut), second)
        self.assertEqual(cache.svd_evaluations, 2)

    def test_minimum_cut_relaxation_can_have_unbounded_ht_gap(self):
        # Matrix multiplication tensor: F_(ij),(jk),(ki)=1. Its injective norm
        # is 1 by the Frobenius trace inequality, while each unfolding has
        # largest squared singular value n. Five fixed modes embed it in 8D.
        for n in (2, 3):
            forcing = np.zeros((n * n,) * 3 + (1,) * 5)
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        forcing[i * n + j, j * n + k, k * n + i, 0, 0, 0, 0, 0] = 1.
            values = tuple(np.full(size, 1 / 8) for size in forcing.shape)
            candidate = np.zeros_like(forcing)
            candidate[(0,) * 8] = 1.
            result = ht_gain_bounds(forcing, values, balanced_tree(range(8)), 1, candidate)
            self.assertAlmostEqual(result["lower"], 1., places=12)
            self.assertAlmostEqual(result["upper"], float(n), places=11)
            # B=A means every center gives exactly the same edge relaxation.
            cache = CutBoundCache(forcing, values, 1, center=np.ones_like(forcing))
            shared = ht_gain_bounds(forcing, values, balanced_tree(range(8)), 1,
                                    candidate, cache=cache)
            self.assertAlmostEqual(shared["upper"], float(n), places=11)

    def test_more_metrics_never_weaken_upper_bounds(self):
        rng = np.random.default_rng(80)
        forcing = rng.normal(size=(2,) * 6)
        values = tuple(np.array([.3 + i, 12. + i]) for i in range(6))
        tree = balanced_tree(range(6))
        bounds = [ht_gain_bounds(forcing, values, tree, 1, metric_policy=policy)["upper"]
                  for policy in ("scalar", "geometric", "compact", "full")]
        self.assertTrue(np.all(np.diff(bounds) <= 1e-12))

    def test_invalid_inputs_and_cache_mismatch_are_rejected(self):
        forcing = np.ones((2, 2, 2))
        values = tuple(np.ones(2) for _ in range(3))
        tree = ((0, 1), 2)
        with self.assertRaises(ValueError):
            canonical_tree_cuts(((0, 0), 2), 3)
        with self.assertRaises(ValueError):
            CutBoundCache(forcing, values, 0)
        with self.assertRaises(ValueError):
            CutBoundCache(forcing, values, 1, potential=-1)
        cache = CutBoundCache(forcing, values, 1)
        with self.assertRaisesRegex(ValueError, "different forcing"):
            ht_gain_bounds(2 * forcing, values, tree, 1, cache=cache)
        with self.assertRaises(ValueError):
            cache.upper_for_cut((0, 1, 2))
        with self.assertRaises(ValueError):
            cache.add_center(np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
