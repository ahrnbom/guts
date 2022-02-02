"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.

    Tests of Kalman filters
"""


import numpy as np 

from util import vector_dist
from filter import filter2D, filter3D
from options import default_3D_params_guts, default_2D_params

def test_filter3D(do_test=True, **kwargs):
    s = np.array([10, 10], dtype=np.float32)
    trackpath = np.genfromtxt('data/trackpath.csv', delimiter=',')
    n = trackpath.shape[1]

    c1 = trackpath[:, 0]
    c2 = trackpath[:, 1]

    phi = np.arctan2(c2[1]-c1[1], c2[0]-c1[0])
    v = vector_dist(c2, c1)
    p = default_3D_params_guts()
    f = filter3D(c2, s, phi, v, kappa=p.kappa, P_factor=p.P_factor,
                 Q_c=p.Q_c, Q_s=p.Q_s, Q_phi=p.Q_phi, Q_v=p.Q_v, 
                 Q_omega=p.Q_omega, Q_cov=p.Q_cov,
                 R_c=p.R_c, R_s=p.R_s)

    oldc = c1

    filter_dists = np.zeros((n,), dtype=np.float32)
    naive_dists = np.zeros((n,), dtype=np.float32)

    for i in range(2, n):
        f.predict()

        pc = f.x[0:2]

        c = trackpath[:, i]

        filter_dists[i] = vector_dist(pc, c)
        naive_dists[i] = vector_dist(oldc, c)

        phi = np.arctan2(c[1]-oldc[1], c[0]-oldc[0])

        oldc = c 
        z = np.array([*c, *s, phi], dtype=np.float32)

        f.update(z)
    
    better = np.sum(filter_dists < naive_dists) / n 

    fmean = np.mean(filter_dists)
    nmean = np.mean(naive_dists)
    
    if do_test:
        print(better, fmean, nmean, nmean/fmean)
        
        # We should, more often than not, be closer than "last position" prediction
        assert(better > 0.5)

        # The average distance should be lower than "last position" prediction
        assert(fmean < nmean)
    else:
        score = nmean/fmean if better > 0.5 else 0
        return score 

def test_filter2D(do_test=True):
    s = [10, 10]

    trackpath = np.genfromtxt('data/trackpath.csv', delimiter=',')
    n = trackpath.shape[1]

    c = trackpath[:, 0]
    p = default_2D_params()
    f = filter2D(c, s, P_factor=p.P_factor, Q_c=p.Q_c, Q_s=p.Q_s, Q_v=p.Q_v, 
                 Q_ds=p.Q_ds, Q_a=p.Q_a, Q_cov=p.Q_cov, Q_scov=p.Q_scov,
                 R_s=p.R_s, R_c=p.R_c)

    oldc = np.zeros((2,), dtype=np.float32)

    filter_dists = np.zeros((n,), dtype=np.float32)
    naive_dists = np.zeros((n,), dtype=np.float32)

    for i in range(1, n):
        f.predict()
        pc = f.x[0:2]
        
        c = trackpath[:, i]

        filter_dists[i] = vector_dist(pc, c)
        naive_dists[i] = vector_dist(oldc, c)
        
        oldc = c 

        z = np.array([*c, *s], dtype=np.float32)
        f.update(z, 0.01)

    better = np.sum(filter_dists < naive_dists) / n 

    fmean = np.mean(filter_dists)
    nmean = np.mean(naive_dists)

    if do_test:
        print(better, fmean, nmean, nmean/fmean)

        # We should, more often than not, be closer than "last position" prediction
        assert(better > 0.5)
        # The average distance should be lower than "last position" prediction
        assert(fmean < nmean)
    else:
        score = nmean/fmean if better > 0.5 else 0
        return score 


if __name__ == "__main__":
    test_filter3D()