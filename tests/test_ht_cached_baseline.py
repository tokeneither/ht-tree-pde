"""The optimized exhaustive comparator must reproduce actual HT tensors."""
import unittest

import numpy as np

from earlyht.ht_energy import fit_ht_energy
from earlyht.ht_primal import ht_round, candidate_trees, edge_cuts, unfold
from earlyht.ht_search import all_balanced_trees
from scripts.benchmark_ht_v3 import shared_rounder


class CachedHTBaselineTests(unittest.TestCase):
    def test_cached_actual_tensors_and_cores_match_independent_rounding(self):
        rng = np.random.default_rng(881)
        for shape in ((2,) * 8, (3,) * 8, (2, 3, 4, 2, 3)):
            target = rng.standard_normal(shape)
            rounded_cached, _ = shared_rounder(target, 2)
            trees = candidate_trees(len(shape), count=13, seed=82)
            for tree in trees:
                cached = rounded_cached(tree)
                ordinary = ht_round(target, tree, 2, diagnostics=False)
                np.testing.assert_allclose(cached["approximation"], ordinary["approximation"],
                                           rtol=2e-12, atol=2e-13)
                self.assertEqual(cached["frames"].keys(), ordinary["frames"].keys())
                self.assertEqual(cached["cores"].keys(), ordinary["cores"].keys())
                for axes in cached["cores"]:
                    np.testing.assert_allclose(cached["cores"][axes], ordinary["cores"][axes],
                                               rtol=2e-12, atol=2e-13)

    def test_shared_cache_does_not_change_after_energy_fit(self):
        rng = np.random.default_rng(17)
        forcing = rng.standard_normal((3,) * 8)
        diagonal = rng.uniform(1, 3, forcing.shape)
        round_tree, stats = shared_rounder(forcing / diagonal, 2)
        tree = candidate_trees(8, count=1)[0]
        before = round_tree(tree)
        saved = before["approximation"].copy()
        frame_snapshot = {axes: frame.copy() for axes, frame in before["frames"].items()}
        fit = fit_ht_energy(forcing, diagonal, before, max_sweeps=1, diagnostics=False)
        after = round_tree(tree)
        np.testing.assert_array_equal(saved, after["approximation"])
        for axes in frame_snapshot:
            np.testing.assert_array_equal(frame_snapshot[axes], after["frames"][axes])
        for axes in edge_cuts(tree):
            singular = np.linalg.svd(unfold(fit["approximation"], axes), compute_uv=False)
            self.assertLessEqual(np.count_nonzero(singular > 1e-9 * singular[0]), 2)
        self.assertGreater(stats["subtree_cache_hits"], 0)

    def test_all_315_trees_reuse_only_identical_subtree_work(self):
        rng = np.random.default_rng(44)
        target = rng.standard_normal((2,) * 8)
        round_tree, stats = shared_rounder(target, 2)
        for tree in all_balanced_trees(tuple(range(8))):
            cached = round_tree(tree)["approximation"]
            independent = ht_round(target, tree, 2, diagnostics=False)["approximation"]
            np.testing.assert_allclose(cached, independent, rtol=2e-12, atol=2e-13)
        # 8 leaves + C(8,2) pairs + 3*C(8,4) quartet subtrees.
        self.assertEqual(stats["subtree_svd_evaluations"], 8 + 28 + 210)


if __name__ == "__main__":
    unittest.main()
