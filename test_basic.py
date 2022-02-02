"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.

    Tests for the most basic of assumptions,
    mostly about how numpy handles angles
"""
import numpy as np

from util import vector_dist, angle_distance

def test_angle_diff():
    tests = np.array([[0.1, 0.2, 0.1],
                      [0.1, 0.2+np.pi*2, 0.1],
                      [0.1, 0.2-np.pi*2, 0.1],
                      [0.1+np.pi*2, 0.2, 0.1],
                      [0.1-np.pi*2, 0.2, 0.1],
                      [0.2, 0.1, -0.1],
                      [0.2, 0.1-np.pi*2, -0.1],
                      [0.2, 0.1+np.pi*2, -0.1],
                      [0.2+np.pi*2, 0.1, -0.1],
                      [0.2-np.pi*2, 0.1, -0.1],
                      [np.pi-0.01, -np.pi+0.01, 0.02]], dtype=np.float32)
    
    n = tests.shape[0]
    for i in range(n):
        test = tests[i, :]
        a, b, gt = test 
        dist = angle_distance(b, a)

        diff = abs(dist - gt)
        assert diff < 0.001

def test_angles():
    rng = np.random.default_rng()
    for _ in range(100):
        angle = rng.uniform(low=-1.0, high=1.0)*np.pi

        x = np.cos(angle)
        y = np.sin(angle)

        new_angle = np.arctan2(y,x)
        # Note: angles are in [-pi,pi)

        assert abs(angle-new_angle) < 0.0001

def test_arctan2():
    rng = np.random.default_rng()
    for _ in range(100):
        x1 = rng.uniform(low=-1.0, high=1.0)
        y1 = rng.uniform(low=-1.0, high=1.0)

        x2 = rng.uniform(low=-1.0, high=1.0)
        y2 = rng.uniform(low=-1.0, high=1.0)

        A = np.array([x1, y1], dtype=np.float32)
        B = np.array([x2, y2], dtype=np.float32)

        diff = B - A
        dist = np.linalg.norm(diff)
        angle = np.arctan2(diff[1], diff[0])

        dx = np.cos(angle)
        dy = np.sin(angle)

        BB = A + np.array([dx, dy], dtype=np.float32)*dist 

        assert vector_dist(B, BB) < 0.001

if __name__=="__main__":
    test_angle_diff()