# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


DTY_FLT = 'float32'
DTY_INT = 'int32'
DTY_BOL = 'bool'

CONST_ZERO = 1e-16
CONST_DIFF = 1e-6

GAP_INF = 2 ** 31 - 1
GAP_MID = 1e8
GAP_NAN = 1e-16

RANDOM_SEED = None
FIXED_SEED = 4567


def check_zero(temp):
    return temp if temp != 0. else CONST_ZERO

def check_equal(tem_A, tem_B):
    return True if abs(tem_A - tem_B) <= CONST_DIFF else False
