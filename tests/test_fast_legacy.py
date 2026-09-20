"""Fast original-criterion calculations compared with full singular spectra."""

import unittest

import numpy as np

from earlyht.fast_legacy import (gram_tail_squared, prepare_fast_legacy,
                                 select_tree_fast, truncated_candidate)
from earlyht.pde import PDEProblem, build_suite, cg_probes
from earlyht.selection import select_tree


class FastLegacyTests(unittest.TestCase):
    def test_tail_lower_bound_and_small_tail_cancellation(self):
        rng = np.random.default_rng(524)
        for shape in ((40, 32), (32, 40), (3, 5)):
            matrix = rng.normal(size=shape)
            singular = np.linalg.svd(matrix, compute_uv=False)
            for rank in (0, 1, 2, min(shape)):
                lower = gram_tail_squared(matrix, rank)
                exact = float(np.sum(singular[rank:] ** 2))
                self.assertLessEqual(lower, exact + 1e-12)
                self.assertLessEqual(exact - lower, 2e-9 * np.sum(matrix ** 2))
        matrix = np.outer(rng.normal(size=40), rng.normal(size=32))
        self.assertEqual(gram_tail_squared(matrix, 1), 0.0)

    def test_partial_candidates_match_full_svd_optimum_and_rank(self):
        rng = np.random.default_rng(127)
        for shape in ((40, 32), (32, 40), (4, 5)):
            matrix = rng.normal(size=shape)
            singular = np.linalg.svd(matrix, compute_uv=False)
            for backend in ("gram", "svds"):
                candidate, observed, info = truncated_candidate(matrix, 2, backend)
                self.assertLessEqual(np.linalg.matrix_rank(candidate), 2)
                self.assertLessEqual(observed, 2)
                np.testing.assert_allclose(np.sum((matrix - candidate) ** 2),
                                           np.sum(singular[2:] ** 2), rtol=2e-12)
                self.assertTrue(np.isfinite(info["residual"]))

    def test_cached_original_and_fast_decisions_agree_on_suite_probes(self):
        for problem in build_suite(n=4, seeds=(0,)):
            for potential_bound in ("discard", "minorants"):
                cache = prepare_fast_legacy(problem, potential_bound)
                probes = cg_probes(problem, maxiter=8, checkpoints=(2, 4, 8))
                for probe in probes:
                    if probe["iteration"] == 0:
                        continue
                    for rank in (1, 2):
                        original = select_tree(problem, probe["w"], probe["delta"], rank,
                                               strategy="screened", potential_bound=potential_bound,
                                               starts=1, max_sweeps=8)
                        fast = select_tree_fast(problem, probe["w"], probe["delta"], rank,
                                                potential_bound=potential_bound, cache=cache,
                                                starts=1, max_sweeps=8)
                        self.assertEqual(fast["selected"], original["selected"])
                        self.assertEqual(fast["certified"], original["certified"])
                        for name in original["bounds"]:
                            self.assertLessEqual(fast["bounds"][name]["lower"],
                                                 original["bounds"][name]["lower"] + 1e-12)
                        self.assertFalse(fast["setup_included"])

    def test_known_separation_and_shared_exact_fit(self):
        pair = np.eye(2) / np.sqrt(2)
        tensor = np.einsum("ij,kl->ijkl", pair, pair)
        problem = PDEProblem("known", tensor.shape, tuple(np.eye(2) for _ in range(4)),
                             np.zeros_like(tensor), 4 * tensor)
        separated = select_tree_fast(problem, tensor, 0, 1)
        self.assertTrue(separated["certified"])
        self.assertEqual(separated["selected"], "12|34")
        tie = select_tree_fast(problem, np.ones_like(tensor), 0, 1)
        self.assertFalse(tie["certified"])

    def test_invalid_cache_and_controls_rejected(self):
        problem = build_suite(n=3, seeds=(0,))[0]
        cache = prepare_fast_legacy(problem)
        for kwargs in (dict(delta=-1), dict(rank=-1), dict(backend="unknown"),
                       dict(cache=cache, potential_bound="discard")):
            args = dict(delta=0.0, rank=1)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                select_tree_fast(problem, problem.rhs, **args)


if __name__ == "__main__":
    unittest.main()
