# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import time

from sklearn import datasets
from sklearn import neighbors

import numpy as np
np.random.seed(4567)

from pyensemble.classify import BaggingEnsembleAlgorithm
from data_distributed import (distributed_single_pruning,
                              distributed_pruning_methods)
from data_distributed import COMEP_Pruning, DOMEP_Pruning



def define_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb-cls", type=int, default=17,
        help='Number of individual classifiers in the ensemble')
    parser.add_argument("--nb-pru", type=int, default=5,
        help='Number of members in the pruned sub-ensemble')
    parser.add_argument("--name-pru", type=str, default='COMEP',
        choices=['ES', 'KP', 'KL', 'RE', 'OO',
                 'DREP', 'SEP', 'OEP', 'PEP',
                 'GMA', 'LCS', 'COMEP', 'DOMEP'],
        help='Name of the expected ensemble pruning method')
    parser.add_argument("--distributed", action="store_true",
        help='Whether to use EPFD (framework)')
    parser.add_argument("--lam", type=float, default=0.5,
        help="lambda")
    parser.add_argument("--m", type=int, default=2,
        help='Number of Machines')

    args = parser.parse_args()
    return args


def load_iris():
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    choice = list(range(len(y)))
    np.random.shuffle(choice)
    X = X[choice]
    y = y[choice]

    choice = np.random.choice([0, 1, 2])
    cindex = y != choice
    y = y[cindex]
    X = X[cindex]

    cindex = list(range(len(y)))
    np.random.shuffle(cindex)
    choice = len(cindex) // 2
    X_trn = X[: choice].tolist()
    y_trn = y[: choice].tolist()
    X_tst = X[choice :].tolist()
    y_tst = y[choice :].tolist()

    return X_trn, y_trn, X_tst, y_tst


def main(args):
    nb_cls = args.nb_cls
    nb_pru = args.nb_pru
    name_pru = args.name_pru
    distributed = args.distributed
    lam = args.lam
    m = args.m

    X_trn, y_trn, _, _ = load_iris()


    name_cls = neighbors.KNeighborsClassifier()
    _, clfs = BaggingEnsembleAlgorithm(
        X_trn, y_trn, name_cls, nb_cls)

    y_insp = [i.predict(X_trn).tolist() for i in clfs]
    rho = nb_pru / nb_cls


    if name_pru not in ['COMEP', 'DOMEP']:
        since = time.time()
        Pc = distributed_single_pruning(name_pru, y_trn, y_insp,
                nb_cls, nb_pru, rho=rho)
        Tc = time.time() - since
        print("{:5s}: {:.4f}s, get {}".format(name_pru, Tc, Pc))

        if distributed:
            since = time.time()
            Pd = distributed_pruning_methods(y_trn, y_insp,
                nb_pru, m, name_pru, rho=rho)
            Td = time.time() - since
            print("{:5s}: {:.4f}s, get {}".format(name_pru, Td, Pd))

    elif name_pru == 'COMEP':
        since = time.time()
        Pc = COMEP_Pruning(np.array(y_insp).T.tolist(), nb_pru, y_trn, lam)
        Tc = time.time() - since
        print("{:5s}: {:.4f}s, get {}".format(name_pru, Tc, Pc))

    elif name_pru == 'DOMEP':
        since = time.time()
        Pd = DOMEP_Pruning(np.array(y_insp).T.tolist(), nb_pru, m, y_trn, lam)
        Td = time.time() - since
        print("{:5s}: {:.4f}s, get {}".format(name_pru, Td, Pd))

    else:
        raise ValueError("Please check the `name_pru`.")


if __name__ == "__main__":
    args = define_params()
    main(args)

