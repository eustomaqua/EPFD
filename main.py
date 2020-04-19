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
from data_entropy import COMEP, DOMEP



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
parser.add_argument("--lambda", type=float, default=0.5,
    help="lambda")
parser.add_argument("--m", type=int, default=2,
    help='Number of Machines')

args = parser.parse_args()
nb_cls = args.nb_cls
nb_pru = args.nb_pru
name_pru = args.name_pru
distributed = args.distributed
lam = args.lambda
m = args.m



# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

choice = np.choice([0, 1, 2])
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


name_cls = neighbors.KNeighborsClassifier()
coef, clfs = BaggingEnsembleAlgorithm(
    X_trn, y_trn, name_cls, nb_cls)

y_insp = [i.predict(X_trn).tolist() for i in clfs]
y_pred = [i.predict(X_tst).tolist() for i in clfs]
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
        Td = time.time() - since()
        print("{:5s}: {:.4f}s, get {}".format(name_pru, Td, Pd))

elif name_pru == 'COMEP':
    since = time.time()
    Pc = COMEP(np.array(y_insp).T.tolist(), nb_pru, y_trn, lam, rho=rho)
    Tc = time.time() - since
    print("{:5s}: {:.4f}s, get {}".format(name_pru, Tc, Pc))

elif name_pru == 'DOMEP':
    since = time.time()
    Pd = DOMEP(np.array(y_insp).T.tolist(), nb_pru, m, y_trn, lam, rho=rho)
    print("{:5s}: {:.4f}s, get {}".format(name_pru, Td, Pd))

else:
    raise ValueError("Please check the `name_pru`.")




