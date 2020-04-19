# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from utils_const import FIXED_SEED
from utils_const import check_equal
import numpy as np


def test_entropy():
    from data_entropy import (H, H1, H2, I, MI, VI,
                            TDAC, TDAS1, TDAS2)
    prng = np.random.RandomState(FIXED_SEED)
    X, Y = prng.randint(5, size=(2, 17)).tolist()

    assert H(0) >= 0 and H(1) >= 0 and H(0.5) >= 0
    assert H1(X) >= 0 and H1(Y) >= 0
    assert H2(X, Y) >= 0

    assert isinstance(I(X, Y), float)
    assert isinstance(MI(X, Y), float)
    assert isinstance(VI(X, Y), float)

    assert check_equal(I(X, Y), I(Y, X))
    assert check_equal(MI(X, Y), MI(Y, X))
    assert check_equal(VI(X, Y), VI(Y, X))

    L = prng.randint(5, size=17).tolist()
    lam = 0.5
    assert isinstance(TDAC(X, Y, L, lam), float)

    S = prng.randint(5, size=(17, 4)).tolist()
    ans1 = TDAS1(S, L, lam)
    ans2 = TDAS2(S, L, lam)
    assert check_equal(ans1, ans2)


def test_COMEP():
    from data_entropy import tdac_sum, arg_max_p, COMEP
    prng = np.random.RandomState(FIXED_SEED + 1)
    L = prng.randint(5, size=17).tolist()
    lam = 0.5

    T = prng.randint(5, size=(17, 4)).tolist()
    k = 2
    S = COMEP(T, k, L, lam)
    assert np.sum(S) == k

    idx = arg_max_p(T, S, L, lam)
    assert isinstance(idx, np.integer)
    T = np.array(T)
    p = T[:, 0].tolist()
    S = T[:, 1:].tolist()
    ans = tdac_sum(p, S, L, lam)
    assert isinstance(ans, float)
    del T, k, S, idx, p, ans


def test_DOMEP():
    from data_entropy import (find_idx_in_sub, DOMEP,
                            randomly_partition)
    prng = np.random.RandomState(FIXED_SEED + 2)
    L = prng.randint(5, size=17).tolist()
    lam = 0.5

    N = prng.randint(5, size=(17, 6)).tolist()
    k, m = 2, 2
    S = DOMEP(N, k, m, L, lam)
    assert np.sum(S) == k

    Tl = randomly_partition(6, m); Tl = np.array(Tl)
    i = prng.choice(np.unique(Tl))
    N = np.array(N)
    ans = find_idx_in_sub(i, Tl, N, k, L, lam)
    assert isinstance(ans, np.ndarray) and isinstance(ans[0], np.integer)
    del N, k, m, S, Tl, i, ans


