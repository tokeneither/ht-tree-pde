"""Search-level correctness: complete families, exact optimum, and abstention."""
import unittest
import numpy as np

from earlyht.ht_primal import balanced_tree, tree_label
from earlyht.ht_search import all_balanced_trees, prescribed_ht_forcing, search_ht


class HTSearchTests(unittest.TestCase):
    def test_balanced_tree_enumeration_is_complete_and_has_no_duplicates(self):
        trees=all_balanced_trees(tuple(range(8)))
        self.assertEqual(len(trees),315)
        self.assertEqual(len(set(tree_label(tree) for tree in trees)),315)
        self.assertEqual(len(all_balanced_trees(tuple(range(4)))),3)

    def test_scalar_operator_known_global_optimum_certifies_scrambled_tree(self):
        tree=balanced_tree((0,3,1,6,2,7,4,5))
        forcing=prescribed_ht_forcing((2,)*8,tree,seed=419)
        # A=I: this independently known u=f is exactly HT-feasible in the
        # prescribed family. No u argument is passed to the search.
        values=tuple(np.full(2,1/8) for _ in range(8))
        result=search_ht(forcing,values,stages=(0,),fits_per_stage=3)
        self.assertTrue(result['certified'])
        winner=result['selected_index']
        self.assertAlmostEqual(result['lower'][winner],1.,places=12)
        self.assertLess(max(u for i,u in enumerate(result['upper']) if i!=winner),1.)
        self.assertEqual(result['proxy_operator_steps'],0)

    def test_equal_family_optima_do_not_produce_a_strict_certificate(self):
        forcing=np.ones((2,)*8)/16
        values=tuple(np.full(2,1/8) for _ in range(8))
        result=search_ht(forcing,values,stages=(0,),fits_per_stage=1)
        self.assertFalse(result['certified'])
        self.assertLessEqual(result['margin'],1e-12)


if __name__=='__main__':
    unittest.main()
