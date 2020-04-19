# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import unittest

from utils_const import FIXED_SEED
import numpy as np

from data_distributed import (distributed_last_criterion,
                              distributed_single_pruning,
                              distributed_pruning_methods)


class TestDistributed(unittest.TestCase):
    """docstring for TestDistributed"""

    def test_prelim(self):
        y = np.random.randint(4, size=50)
        yt = np.random.randint(4, size=(17, 50))

        accsg = distributed_last_criterion(y, yt)
        # assert 0. <= accsg <= 1.
        self.assertTrue(0. <= accsg <= 1.)


    def check_centralized_distributed(self, name_pru, y, yt, nb_pru):
        nb_cls = len(yt)
        rho = nb_pru / nb_cls
        Pc = distributed_single_pruning(
                        name_pru, y, yt, nb_cls, nb_pru, rho=rho)
        Pd = distributed_pruning_methods(
                        y, yt, nb_pru, 2, name_pru, rho=rho)
        assert Pc and Pd


    def test_centralized_distributed(self):
        y = np.random.randint(4, size=50)
        yt = np.random.randint(4, size=(11, 50))
        nb_pru = 5

        self.check_centralized_distributed('ES', y, yt, nb_pru)
        self.check_centralized_distributed('KP', y, yt, nb_pru)
        self.check_centralized_distributed('KL', y, yt, nb_pru)
        self.check_centralized_distributed('RE', y, yt, nb_pru)
        self.check_centralized_distributed('OO', y, yt, nb_pru)
        ## check_centralized_distributed('GMA', y, yt, nb_pru)
        ## check_centralized_distributed('LCS', y, yt, nb_pru)
        # check_centralized_distributed('OEP', y, yt, nb_pru)
        # check_centralized_distributed('SEP', y, yt, nb_pru)
        # check_centralized_distributed('PEP', y, yt, nb_pru)

    def test_binary_situation(self):
        y = np.random.randint(2, size=50)
        yt = np.random.randint(2, size=(11, 50))
        nb_pru = 5

        self.check_centralized_distributed('ES', y, yt, nb_pru)
        self.check_centralized_distributed('KP', y, yt, nb_pru)
        self.check_centralized_distributed('KL', y, yt, nb_pru)
        self.check_centralized_distributed('RE', y, yt, nb_pru)
        self.check_centralized_distributed('OO', y, yt, nb_pru)
        ## check_centralized_distributed('GMA', y, yt, nb_pru)
        ## check_centralized_distributed('LCS', y, yt, nb_pru)
        ## check_centralized_distributed('DREP', y, yt, nb_pru)
        # check_centralized_distributed('OEP', y, yt, nb_pru)
        # check_centralized_distributed('SEP', y, yt, nb_pru)
        # check_centralized_distributed('PEP', y, yt, nb_pru)



    def generate_data(self, nb_clas, nb_inst, nb_cls):
        prng = np.random.RandomState(FIXED_SEED + 3)
        y = prng.randint(nb_clas, size=nb_inst)
        yt = prng.randint(nb_clas, size=(nb_cls, nb_inst))
        return y, yt

    def test_OEP(self):
        nb_pru = 9
        y, yt = self.generate_data(4, 50, 17)
        self.check_centralized_distributed('OEP', y, yt, nb_pru)
        y, yt = self.generate_data(2, 50, 17)
        self.check_centralized_distributed('OEP', y, yt, nb_pru)

    def test_SEP(self):
        nb_pru = 9
        y, yt = self.generate_data(4, 50, 17)
        self.check_centralized_distributed('SEP', y, yt, nb_pru)
        y, yt = self.generate_data(2, 50, 17)
        self.check_centralized_distributed('SEP', y, yt, nb_pru)

    def test_PEP(self):
        nb_pru = 9
        y, yt = self.generate_data(4, 50, 17)
        self.check_centralized_distributed('PEP', y, yt, nb_pru)
        y, yt = self.generate_data(2, 50, 17)
        self.check_centralized_distributed('PEP', y, yt, nb_pru)


    def test_composable(self):
        nb_pru = 9
        y, yt = self.generate_data(2, 50, 17)
        self.check_centralized_distributed('DREP', y, yt, nb_pru)
        #
        # check_centralized_distributed('GMA', y, yt, nb_pru)
        # check_centralized_distributed('LCS', y, yt, nb_pru)
        #
        # y, yt = generate_data(4, 50, 17)
        # check_centralized_distributed('GMA', y, yt, nb_pru)
        # check_centralized_distributed('LCS', y, yt, nb_pru)

    def test_GMA(self):
        nb_pru = 5
        y, yt = self.generate_data(2, 50, 11)
        self.check_centralized_distributed('GMA', y, yt, nb_pru)
        y, yt = self.generate_data(4, 50, 11)
        self.check_centralized_distributed('GMA', y, yt, nb_pru)

    def test_LCS(self):
        nb_pru = 5
        y, yt = self.generate_data(2, 50, 11)
        self.check_centralized_distributed('LCS', y, yt, nb_pru)
        y, yt = self.generate_data(4, 50, 11)
        self.check_centralized_distributed('LCS', y, yt, nb_pru)

