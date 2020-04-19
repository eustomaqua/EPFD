# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from utils_const import FIXED_SEED
from utils_const import check_equal
import numpy as np

from data_distributed import (distributed_last_criterion,
                              distributed_single_pruning,
                              distributed_find_idx_in_sub,
                              distributed_pruning_methods)


def test_prelim():
    y = np.random.randint(4, size=50)
    yt = np.random.randint(4, size=(17, 50))

    accsg = distributed_last_criterion(y, yt)
    assert 0. <= accsg <= 1.


def check_centralized_distributed(name_pru, y, yt, nb_pru):
    nb_cls = len(yt)
    rho = nb_pru / nb_cls
    Pc = distributed_single_pruning(
                    name_pru, y, yt, nb_cls, nb_pru, rho=rho)
    Pd = distributed_pruning_methods(
                    y, yt, nb_pru, 2, name_pru, rho=rho)
    assert Pc and Pd


def test_centralized_distributed():
    y = np.random.randint(4, size=50)
    yt = np.random.randint(4, size=(17, 50))
    nb_pru = 9

    check_centralized_distributed('ES', y, yt, nb_pru)
    check_centralized_distributed('KP', y, yt, nb_pru)
    check_centralized_distributed('KL', y, yt, nb_pru)
    check_centralized_distributed('RE', y, yt, nb_pru)
    check_centralized_distributed('OO', y, yt, nb_pru)
    check_centralized_distributed('GMA', y, yt, nb_pru)
    check_centralized_distributed('LCS', y, yt, nb_pru)
    check_centralized_distributed('OEP', y, yt, nb_pru)
    check_centralized_distributed('SEP', y, yt, nb_pru)
    check_centralized_distributed('PEP', y, yt, nb_pru)

def test_binary_situation():
    y = np.random.randint(2, size=50)
    yt = np.random.randint(2, size=(17, 50))
    nb_pru = 9

    check_centralized_distributed('ES', y, yt, nb_pru)
    check_centralized_distributed('KP', y, yt, nb_pru)
    check_centralized_distributed('KL', y, yt, nb_pru)
    check_centralized_distributed('RE', y, yt, nb_pru)
    check_centralized_distributed('OO', y, yt, nb_pru)
    check_centralized_distributed('GMA', y, yt, nb_pru)
    check_centralized_distributed('LCS', y, yt, nb_pru)
    check_centralized_distributed('DREP', y, yt, nb_pru)
    check_centralized_distributed('OEP', y, yt, nb_pru)
    check_centralized_distributed('SEP', y, yt, nb_pru)
    check_centralized_distributed('PEP', y, yt, nb_pru)


