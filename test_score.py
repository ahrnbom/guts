"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.


    Tests related to how scoring is computed
"""

import numpy as np 
from score import iou3D

def are_close(x, y):
    diff = x - y 
    return abs(diff) < 0.0001

def are_somewhat_close(x, y):
    diff = x - y
    return abs(diff) < 0.1

def test_iou3D():
    a = np.array([10, 20, 3, 4, 1.0])
    b = np.array([10, 20, 3, 4, 1.0])

    res = iou3D(a, b)
    assert are_close(res, 1.0)

    b[-1] = 1.1
    res = iou3D(a, b)
    assert are_somewhat_close(res, 0.9)

    a[-1] = 0.9
    res2 = iou3D(a, b)
    assert are_somewhat_close(res2, 0.85)
    assert res2 < res 

    a = np.array([11, 18, 2.9, 5.0, 0.87])
    res3 = iou3D(a, b)
    assert res3 < res2 
    assert res3 < res 
    
    a = np.array([10, 20, 3, 4, np.pi-0.01])
    b = np.array([10, 20, 3, 4, -np.pi+0.01])
    res = iou3D(a, b)
    assert are_somewhat_close(res, 1.0)

    b = np.array([10, 10, 3, 4, -np.pi+0.01])
    res = iou3D(a, b)
    assert are_close(res, 0)

    b = np.array([10, 10, 3, 40, -np.pi+0.01])
    res = iou3D(a, b)
    assert res > 0
    assert are_somewhat_close(res, 0)

    a = np.array([1, 0, 4, 1, 0])
    b = np.array([-1, 0, 4, 1, 0])
    res = iou3D(a, b)
    assert are_close(res, 0.3333333)

    a = np.array([0, 0, 2, 1, 0])
    b = np.array([0, 0, 2, 1, np.pi/2])
    res = iou3D(a, b)
    assert are_close(res, 0.3333333)
