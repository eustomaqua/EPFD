# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
from main import define_params, load_iris, main

class TestMain(unittest.TestCase):
    def test_prelim(self):
        args = define_params()
        self.assertIsInstance(args.nb_cls, int)
        self.assertIsInstance(args.nb_pru, int)
        self.assertIsInstance(args.name_pru, str)
        self.assertIsInstance(args.distributed, bool)
        self.assertIsInstance(args.lam, float)
        self.assertIsInstance(args.m, int)
        self.assertTrue(args.m >= 2)

    def test_data(self):
        _, y_trn, _, y_tst = load_iris()
        self.assertTrue(len(np.unique(y_trn)) == 2)
        self.assertTrue(len(np.unique(y_tst)) == 2)

    def test_main(self):
        args = define_params()
        args.name_pru = 'COMEP'
        main(args)
        args.name_pru = 'DOMEP'
        main(args)

        args.name_pru = 'PEP'
        args.distributed = True
        main(args)
