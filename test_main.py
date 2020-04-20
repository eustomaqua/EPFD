# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
from main import load_iris


class TestMain(unittest.TestCase):
    def test_load_iris(self):
        X_trn, y_trn, X_tst, y_tst = load_iris()
        self.assertTrue(len(np.unique(y_trn)) == 2)
        self.assertTrue(len(np.unique(y_tst)) == 2)
        self.assertEqual(len(X_trn), len(y_trn))
        self.assertEqual(len(X_tst), len(y_tst))

