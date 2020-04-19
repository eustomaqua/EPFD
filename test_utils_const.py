# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from utils_const import check_zero
from utils_const import check_equal


def test_check():
    assert check_zero(0.0)
    assert check_zero(1e-18)
    assert check_equal(1e-7, 1e-6)
