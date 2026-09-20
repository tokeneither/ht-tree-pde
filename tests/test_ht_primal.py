"""Structural and approximation guarantees of genuine multilevel HT rounding."""
import unittest

import numpy as np

from earlyht.ht_primal import (balanced_tree, candidate_trees, edge_cuts,
                               ht_round, leaves, unfold, validate_tree)


def random_ht_tensor(shape, tree, rank, seed=17):
    """Independent random transfer-core construction, with no rounding call."""
    rng = np.random.default_rng(seed)

    def frame(node):
        if isinstance(node, int):
            return np.linalg.qr(rng.normal(size=(shape[node], rank)), mode="reduced")[0]
        first, second = frame(node[0]), frame(node[1])
        product = np.kron(first, second)
        transfer = np.linalg.qr(rng.normal(size=(product.shape[1], rank)),
                               mode="reduced")[0]
        return product @ transfer

    left, right = frame(tree[0]), frame(tree[1])
    core = rng.normal(size=(left.shape[1], right.shape[1]))
    order = leaves(tree)
    ordered = (left @ core @ right.T).reshape(tuple(shape[i] for i in order))
    return ordered.transpose(np.argsort(order))


class HTPrimalTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(41)

    def test_every_edge_rank_in_eight_dimensions(self):
        # Scrambled labels expose accidental reliance on contiguous axis groups.
        tree = (((7, 0), (5, 2)), ((3, 6), (1, 4)))
        target = self.rng.normal(size=(3,) * 8)
        rounded = ht_round(target, tree, 2)
        self.assertEqual(len(rounded["edge_ranks"]), 14)
        self.assertTrue(all(value <= 2 for value in rounded["edge_ranks"].values()))
        self.assertEqual(rounded["theoretical_coefficients"], 8 * 3 * 2 + 6 * 2**3 + 2**2)
        self.assertLess(rounded["theoretical_coefficients"], target.size)
        for axes, frame in rounded["frames"].items():
            np.testing.assert_allclose(frame.T @ frame, np.eye(frame.shape[1]), atol=3e-14)
            self.assertEqual(frame.shape[0], int(np.prod([target.shape[i] for i in axes])))

    def test_exactness_for_independently_generated_ht_tensor(self):
        tree = (((2, 7), (4, 0)), ((5, 1), (6, 3)))
        target = random_ht_tensor((3,) * 8, tree, 2)
        rounded = ht_round(target, tree, 2)
        np.testing.assert_allclose(rounded["approximation"], target, rtol=2e-12, atol=2e-14)
        self.assertTrue(all(rank == 2 for rank in rounded["edge_ranks"].values()))
        # The tensor is not merely CP rank two: an incompatible cut has rank > 2.
        self.assertGreater(np.linalg.matrix_rank(unfold(target, (0, 1, 2, 3))), 2)

    def test_projection_pythagoras_and_fixed_frame_optimality(self):
        tree = ((3, 0), (2, 1))
        target = self.rng.normal(size=(4, 3, 2, 3))
        rounded = ht_round(target, tree, 2)
        approximation = rounded["approximation"]
        residual = target - approximation
        self.assertAlmostEqual(float(np.sum(residual * approximation)), 0.0, places=11)
        self.assertAlmostEqual(float(np.sum(target**2)),
                               float(np.sum(residual**2) + np.sum(approximation**2)), places=10)
        left = rounded["frames"][leaves(tree[0])]
        right = rounded["frames"][leaves(tree[1])]
        order = leaves(tree)
        matrix = target.transpose(order).reshape(left.shape[0], right.shape[0])
        minimum = np.linalg.norm(residual)
        for _ in range(5):
            core = self.rng.normal(size=(left.shape[1], right.shape[1]))
            self.assertLessEqual(minimum, np.linalg.norm(matrix - left @ core @ right.T) + 1e-12)

    def test_discarded_tail_bound_for_several_shapes(self):
        for shape, order, rank in (((3,) * 8, (5, 0, 6, 1, 7, 2, 4, 3), 2),
                                   ((2, 3, 4, 2, 3), (4, 0, 3, 1, 2), 2),
                                   ((3,) * 4, (0, 1, 2, 3), 1)):
            target = self.rng.normal(size=shape)
            rounded = ht_round(target, balanced_tree(order), rank)
            self.assertLessEqual(rounded["frobenius_error"],
                                 rounded["discarded_error_bound"] + 1e-12)
            # Projection contractions imply the computed discarded tail at
            # every node is bounded by the original matricization's tail.
            for axes, tail_squared in rounded["discarded_squared"].items():
                singular = np.linalg.svd(unfold(target, axes), compute_uv=False)
                self.assertLessEqual(tail_squared, float(np.sum(singular[rank:] ** 2)) + 1e-10)

    def test_full_rank_zero_and_rank_one_limits(self):
        shape = (2, 3, 2, 3)
        tree = ((2, 0), (3, 1))
        target = self.rng.normal(size=shape)
        np.testing.assert_allclose(ht_round(target, tree, 100)["approximation"], target, atol=2e-13)
        zero = ht_round(np.zeros(shape), tree, 2)
        self.assertEqual(zero["frobenius_error"], 0)
        self.assertTrue(all(value == 0 for value in zero["edge_ranks"].values()))
        factors = [self.rng.normal(size=n) for n in shape]
        separable = np.einsum("i,j,k,l->ijkl", *factors)
        np.testing.assert_allclose(ht_round(separable, tree, 1)["approximation"], separable,
                                   atol=3e-14, rtol=3e-13)

    def test_nested_transfer_reconstruction(self):
        tree = (((3, 7), (1, 5)), ((0, 6), (2, 4)))
        rounded = ht_round(self.rng.normal(size=(2,) * 8), tree, 2)

        def verify(node):
            if isinstance(node, int):
                return rounded["frames"][(node,)]
            left, right = verify(node[0]), verify(node[1])
            axes = leaves(node)
            core = rounded["cores"][axes]
            if node == tree:
                return left @ core @ right.T
            actual = np.kron(left, right) @ core.reshape(left.shape[1] * right.shape[1], -1)
            np.testing.assert_allclose(actual, rounded["frames"][axes], atol=2e-14)
            return actual

        order = leaves(tree)
        reconstructed = verify(tree).reshape((2,) * 8).transpose(np.argsort(order))
        np.testing.assert_allclose(reconstructed, rounded["approximation"], atol=2e-14)

    def test_tree_candidates_are_reproducible_and_unique(self):
        planted = (((7, 1), (5, 0)), ((2, 6), (4, 3)))
        candidates = candidate_trees(8, count=12, seed=9, planted=planted)
        self.assertEqual(candidates, candidate_trees(8, count=12, seed=9, planted=planted))
        self.assertEqual(len(set(candidates)), 12)
        for tree in candidates:
            self.assertEqual(sorted(leaves(validate_tree(tree, 8))), list(range(8)))
            self.assertEqual(len(edge_cuts(tree)), 14)
        canonical_cuts = lambda tree: {frozenset(cut) for cut in edge_cuts(tree)}
        self.assertEqual(canonical_cuts(candidates[0]), canonical_cuts(planted))

    def test_unfold_and_input_validation(self):
        target = np.arange(24).reshape(2, 3, 4)
        np.testing.assert_array_equal(unfold(target, (2, 0)), target.transpose(2, 0, 1).reshape(8, 3))
        for invalid_tree in (((0, 1), (1, 2)), (0, (1, 4)), [0, [1, 2]]):
            with self.assertRaises(ValueError):
                ht_round(target, invalid_tree, 2)
        for rank in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                ht_round(target, (0, (1, 2)), rank)
        with self.assertRaises(ValueError):
            ht_round(target.astype(complex), (0, (1, 2)), 2)
        with self.assertRaises(ValueError):
            ht_round(np.full((2, 2), np.nan), (0, 1), 2)


if __name__ == "__main__":
    unittest.main()
