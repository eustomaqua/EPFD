# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from copy import deepcopy
import gc
import time
gc.enable()

import numpy as np
from pathos import multiprocessing as pp
# from pympler.asizeof import asizeof


from data_entropy import COMEP, DOMEP
from utils_const import DTY_INT, DTY_BOL

import pyensemble.classify.voting as dv
import pyensemble.pruning as dp
import data_entropy as dt



# ----------  Baselines  -----------
# Existing Ensemble Pruning Methods
#


def element_contrastive_pruning_via_validation(name_pru,
                nb_cls, nb_pru, y_val, y_cast, epsilon=1e-3, rho=0.5):
    # since = time.time()
    if nb_pru >= nb_cls:
        return list(range(nb_cls)), None

    if name_pru == 'ES':
        ys_cast, P       = dp.Early_Stopping(              y_cast,        nb_cls, nb_pru)
    elif name_pru == 'KL':
        ys_cast, P       = dp.KL_divergence_Pruning(       y_cast,        nb_cls, nb_pru)
    elif name_pru == 'KP':
        ys_cast, P       = dp.Kappa_Pruning(               y_cast, y_val, nb_cls, nb_pru)
    elif name_pru == 'OO':
        ys_cast, P, flag = dp.Orientation_Ordering_Pruning(y_cast, y_val)
    elif name_pru == 'RE':
        ys_cast, P       = dp.Reduce_Error_Pruning(        y_cast, y_val, nb_cls, nb_pru)
    elif name_pru == 'GMA' or name_pru == 'GMM':  #'GMM_Algorithm'
        ys_cast, P       = dp.GMM_Algorithm(               y_cast, y_val, nb_cls, nb_pru)
    elif name_pru == 'LCS':  #'Local_Search':
        ys_cast, P       = dp.Local_Search(                y_cast, y_val, nb_cls, nb_pru, epsilon)
    elif name_pru == 'DREP':
        ys_cast, P       = dp.DREP_Pruning(                y_cast, y_val, nb_cls,         rho)
    elif name_pru == 'SEP':
        ys_cast, P       = dp.SEP_Pruning(                 y_cast, y_val, nb_cls,         rho)
    elif name_pru == 'OEP':
        ys_cast, P       = dp.OEP_Pruning(                 y_cast, y_val, nb_cls)
    elif name_pru == 'PEP':
        ys_cast, P       = dp.PEP_Pruning(                 y_cast, y_val, nb_cls,         rho)
    else:
        raise UserWarning("Error occurred in `contrastive_pruning_methods`.")

    if name_pru != "OO":
        flag = None
    P = np.where(P)[0].tolist()
    del ys_cast
    return deepcopy(P), flag



# ----------  Framework EPFD  -----------
# Ensemble Pruning Framework in a Distributed Setting (EPFD)
#


def distributed_last_criterion(y, yt):
    # could be replaced by another criterion
    fens = dv.plurality_voting(y, yt)
    accsg = np.mean(np.array(fens) == np.array(y))
    del fens
    gc.collect()
    return accsg  # scalar


def distributed_single_pruning(name_pru, y, yt, nb_cls, nb_pru, epsilon=1e-3, rho=0.5):
    P, _ = element_contrastive_pruning_via_validation(
        name_pru, nb_cls, nb_pru, y, yt, epsilon, rho)
    return deepcopy(P)



def distributed_find_idx_in_sub(i,Tl, y,yt,k, name_pru, epsilon,rho):
    # k = nb_pru;   rho = pr_thin
    sub_idx_in_N = np.where(np.array(Tl) == i)[0]
    sub_yt = np.array(yt)[sub_idx_in_N].tolist()
    nb_cls = len(sub_idx_in_N)
    nb_pru = k
    sub_idx_pruning = distributed_single_pruning(name_pru, y,sub_yt, nb_cls,nb_pru, epsilon,rho)
    ans = sub_idx_in_N[ sub_idx_pruning ]
    del sub_yt, nb_cls, nb_pru
    del sub_idx_in_N, sub_idx_pruning
    gc.collect()
    return deepcopy(ans)  # np.ndarray of np.integer


def distributed_pruning_methods(y,yt, k,m, name_pru,epsilon=1e-3,rho=0.5):
    yt = np.array(yt);  n = yt.shape[0]  # n = len(yt)
    Tl = dt.randomly_partition(n=n, m=m);   Tl = np.array(Tl)
    Sl = np.zeros(n, dtype=DTY_INT) - 1  # init

    # concurrent selection
    pool = pp.ProcessingPool(nodes = m)
    sub_idx = pool.map( distributed_find_idx_in_sub,
                        list(range(m)), [Tl]*m, [y]*m, [yt]*m, [k]*m,
                        [name_pru]*m, [epsilon]*m, [rho]*m)
    del pool, Tl

    for i in range(m):
        Sl[ sub_idx[i] ] = i
    del sub_idx
    sub_all_in_N = np.where(Sl != -1)[0]
    sub_all_single = distributed_single_pruning(
                            name_pru, y, yt[sub_all_in_N].tolist(),
                            len(sub_all_in_N),k, epsilon,rho)

    final_S = np.zeros(n, dtype=DTY_BOL)
    final_S[ sub_all_in_N[ sub_all_single ] ] = True  # 1
    del sub_all_in_N, sub_all_single

    # can be replaced to another criterion
    acc_temS = distributed_last_criterion(y, yt[final_S].tolist())
    acc_Sl = [ distributed_last_criterion(y, yt[Sl == i].tolist())  for i in range(m)]
    if np.sum(np.array(acc_Sl) > acc_temS) >= 1:
        tem_argmax_l = acc_Sl.index( np.max(acc_Sl) )
        final_S = (Sl == tem_argmax_l)
        del tem_argmax_l

    del acc_temS, acc_Sl, yt, n, m, Sl
    final_S = np.where(final_S)[0]
    final_S = final_S.tolist()
    gc.collect()
    return deepcopy(final_S)

