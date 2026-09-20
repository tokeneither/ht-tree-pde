"""Validity checks for full-operator centered gain bounds."""

import unittest
from unittest.mock import patch

import numpy as np

from earlyht.direct_full import select_direct_full
from earlyht.pde import BALANCED_CUTS, PDEProblem, matricize, unmatricize
from earlyht.selection import cut_name


def scalar_problem(seed=92):
    rng = np.random.default_rng(seed)
    shape = (2, 2, 2, 2)
    axes = tuple((i + 1.0) * np.eye(2) for i in range(4))
    return PDEProblem("scalar", shape, axes, np.full(shape, 3.0), rng.normal(size=shape))


class DirectFullTests(unittest.TestCase):
    def test_scalar_operator_matches_exact_svd_gain_for_each_tree(self):
        problem = scalar_problem()
        for rank in (1, 2):
            result = select_direct_full(problem, rank=rank, sweeps=(0,))
            exact = {}
            for cut in BALANCED_CUTS:
                name = cut_name(cut)
                singular = np.linalg.svd(matricize(problem.rhs, cut), compute_uv=False)
                exact[name] = np.sum(singular[:rank] ** 2) / 13.0
                self.assertAlmostEqual(result["bounds"][name]["lower"], exact[name], places=12)
                self.assertAlmostEqual(result["bounds"][name]["upper"], exact[name], places=12)
            self.assertEqual(result["selected"], max(exact, key=exact.get))
            self.assertTrue(result["certified"])

    def test_nonseparable_potential_wraps_known_rank_one_optimum(self):
        rng = np.random.default_rng(183)
        shape = (2, 2, 2, 2)
        axes = tuple(np.array([[3.0 + i, -1.0], [-1.0, 3.0 + i]]) for i in range(4))
        potential = rng.uniform(0.0, 10.0, size=shape)
        left, right = rng.normal(size=4), rng.normal(size=4)
        exact = unmatricize(np.outer(left, right), shape, (0, 2))
        problem = PDEProblem("known", shape, axes, potential, np.zeros(shape))
        problem.rhs[:] = problem.apply(exact)
        full_gain = float(np.sum(exact * problem.rhs))
        result = select_direct_full(problem, rank=1, sweeps=(0, 2, 6, 16))
        exact_family = result["bounds"][cut_name((0, 2))]
        self.assertLessEqual(exact_family["lower"], full_gain * (1 + 2e-12))
        self.assertGreaterEqual(exact_family["upper"], full_gain * (1 - 2e-12))
        for entry in result["bounds"].values():
            self.assertLessEqual(entry["lower"], full_gain * (1 + 2e-12))
            self.assertLessEqual(entry["lower"], entry["upper"] + 1e-10)
        if result["certified"]:
            self.assertEqual(result["selected"], cut_name((0, 2)))

    def test_every_upper_bound_dominates_independent_feasible_samples(self):
        problem = scalar_problem()
        rng = np.random.default_rng(28)
        problem.potential[:] = rng.uniform(0, 20, size=problem.shape)
        result = select_direct_full(problem, sweeps=(0, 2, 6))
        for cut in BALANCED_CUTS:
            upper = result["bounds"][cut_name(cut)]["upper"]
            for _ in range(30):
                matrix = np.outer(rng.normal(size=4), rng.normal(size=4))
                tensor = unmatricize(matrix, problem.shape, cut)
                pairing = np.sum(problem.rhs * tensor)
                gain = pairing ** 2 / problem.energy(tensor)
                self.assertLessEqual(gain, upper + 1e-11)
        for previous, current in zip(result["trace"], result["trace"][1:]):
            for name in previous["bounds"]:
                self.assertGreaterEqual(current["bounds"][name]["lower"],
                                        previous["bounds"][name]["lower"])
                self.assertLessEqual(current["bounds"][name]["upper"],
                                     previous["bounds"][name]["upper"])

    def test_zero_source_abstains_and_no_reference_solver_is_used(self):
        problem = scalar_problem()
        problem.rhs[:] = 0
        with patch("earlyht.pde.reference_solve", side_effect=AssertionError("reference leak")), \
             patch("earlyht.pde.cg_probes", side_effect=AssertionError("full-grid iteration")):
            result = select_direct_full(problem, sweeps=(0, 2))
        self.assertFalse(result["certified"])
        self.assertEqual(result["full_pde_iterations"], 0)
        self.assertTrue(all(entry["lower"] == entry["upper"] == 0
                            for entry in result["bounds"].values()))

    def test_optional_dual_refinement_preserves_valid_bounds(self):
        problem = scalar_problem()
        rng = np.random.default_rng(171)
        problem.potential[:] = rng.uniform(0, 30, size=problem.shape)
        initial = select_direct_full(problem, sweeps=(0,), metric_policy="geometric")
        refined = select_direct_full(problem, sweeps=(0,), metric_policy="geometric",
                                     dual_refinements=20)
        for cut in BALANCED_CUTS:
            name = cut_name(cut)
            before, after = initial["bounds"][name], refined["bounds"][name]
            self.assertLessEqual(after["upper"], before["upper"])
            self.assertGreaterEqual(after["lower"], before["lower"])
            self.assertLessEqual(after["lower"], after["upper"] + 1e-11)
            for _ in range(20):
                tensor = unmatricize(np.outer(rng.normal(size=4), rng.normal(size=4)),
                                     problem.shape, cut)
                gain = np.sum(problem.rhs * tensor) ** 2 / problem.energy(tensor)
                self.assertLessEqual(gain, after["upper"] + 1e-11)
        self.assertGreater(refined["dual_svd_evaluations"], 0)

    def test_cheaper_metric_policies_retain_exact_scalar_case(self):
        problem = scalar_problem()
        for policy, count in (("geometric", 3), ("compact", 9)):
            result = select_direct_full(problem, sweeps=(0,), metric_policy=policy)
            self.assertTrue(result["certified"])
            self.assertEqual(result["metric_evaluations"], count)
            for cut in BALANCED_CUTS:
                exact = np.linalg.svd(matricize(problem.rhs, cut), compute_uv=False)[0] ** 2 / 13
                self.assertAlmostEqual(result["bounds"][cut_name(cut)]["upper"], exact, places=12)

    def test_invalid_controls_are_rejected(self):
        problem = scalar_problem()
        for kwargs in (dict(rank=0), dict(rank=True), dict(rank=5), dict(sweeps=()),
                       dict(sweeps=(-1,)), dict(sweeps=(0.5,)), dict(dual_refinements=-1),
                       dict(metric_policy="unknown")):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_direct_full(problem, **kwargs)


if __name__ == "__main__":
    unittest.main()
