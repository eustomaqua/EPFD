# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from utils_const import FIXED_SEED
from utils_const import check_equal
import numpy as np


class TestEntropy(unittest.TestCase):
    def test_entropy(self):
        from data_entropy import (H, H1, H2, I, MI, VI,
                                TDAC, TDAS1, TDAS2)
        prng = np.random.RandomState(FIXED_SEED)
        X, Y = prng.randint(5, size=(2, 17)).tolist()

        self.assertTrue(H(0) >= 0 and H(1) >= 0 and H(0.5) >= 0)
        self.assertTrue(H1(X) >= 0 and H1(Y) >= 0)
        self.assertTrue(H2(X, Y) >= 0)

        self.assertIsInstance(I(X, Y), float)
        self.assertIsInstance(MI(X, Y), float)
        self.assertIsInstance(VI(X, Y), float)

        self.assertAlmostEqual(I(X, Y), I(Y, X))
        self.assertAlmostEqual(MI(X, Y), MI(Y, X))
        self.assertAlmostEqual(VI(X, Y), VI(Y, X))

        L = prng.randint(5, size=17).tolist()
        lam = 0.5
        self.assertIsInstance(TDAC(X, Y, L, lam), float)

        S = prng.randint(5, size=(17, 4)).tolist()
        ans1 = TDAS1(S, L, lam)
        ans2  = TDAS2(S, L, lam)
        self.assertAlmostEqual(ans1, ans2)


    def test_COMEP(self):
        from data_entropy import tdac_sum, arg_max_p, COMEP
        prng = np.random.RandomState(FIXED_SEED + 1)
        L = prng.randint(5, size=17).tolist()
        lam = 0.5

        T = prng.randint(5, size=(17, 4)).tolist()
        k = 2
        S = COMEP(T, k, L, lam)
        self.assertEqual(np.sum(S), k)

        idx = arg_max_p(T, S, L, lam)
        self.assertIsInstance(idx, np.integer)
        T = np.array(T)
        p = T[:, 0].tolist()
        S = T[:, 1:].tolist()
        ans = tdac_sum(p, S, L, lam)
        self.assertIsInstance(ans, float)
        del T, k, S, idx, p, ans


    def test_DOMEP(self):
        from data_entropy import (find_idx_in_sub, DOMEP,
                                randomly_partition)
        prng = np.random.RandomState(FIXED_SEED + 2)
        L = prng.randint(5, size=17).tolist()
        lam = 0.5

        N = prng.randint(5, size=(17, 6)).tolist()
        k, m = 2, 2
        S = DOMEP(N, k, m, L, lam)
        self.assertEqual(np.sum(S), k)

        Tl = randomly_partition(6, m); Tl = np.array(Tl)
        i = prng.choice(np.unique(Tl))
        N = np.array(N)
        ans = find_idx_in_sub(i, Tl, N, k, L, lam)
        # assert isinstance(ans, np.ndarray) and isinstance(ans[0], np.integer)
        self.assertIsInstance(ans, np.ndarray)
        self.assertIsInstance(ans[0], np.integer)
        del N, k, m, S, Tl, i, ans


