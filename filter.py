"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.


    This module describes the Kalman filters used by GUTS/UTS
"""


import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, JulierSigmaPoints
from filterpy.kalman import ExtendedKalmanFilter

from util import angle_distance
from sqrt import sqrt_func

default_tau = 1.0/24

class LinearizingEKF:
    def __init__(self, dim_x:int, dim_z:int, fx, 
                 Hx:np.ndarray, Hjac:np.ndarray,
                 Fjac):

        self.ekf = ExtendedKalmanFilter(dim_x, dim_z)
        self.fx = fx 
        self.Hx = Hx 
        self.Hjac = Hjac
        self.Fjac = Fjac 

        self.dim_x = dim_x
        self.dim_z = dim_z

    def predict(self):
        self.ekf.predict()

    def update(self, z, tau):
        self.ekf.F = self.linearizeF(tau)
        self.ekf.update(z, self.Hjac, self.Hx)

    def linearizeF(self, tau):
        # Create state transition matrix that is a linearization of the 
        # function fx
        x1 = self.ekf.x
        F = self.Fjac(x1, tau)
        return F
    
    def __getattr__(self, name:str):
        # These attributes from the EKF can be directly accessed
        if name in ('x', 'P'): 
            return self.ekf.__dict__[name]
        
        raise ValueError(f"Unknown attribute {name} in Linearizing EKF")

def filter2D(init_c, init_s, init_v=[0,0], init_ds=[0,0], init_a=[0,0],
             P_factor=1.0,
             Q_c=3.0, Q_s=0.5, Q_v=1.0, Q_ds=0.1, Q_a=0.5, 
             Q_cov=0.01, Q_scov=0.001,
             R_c=0.01, R_s=0.001) -> LinearizingEKF:
    
    def hx(x):
        return x[0:4]
    
    def Hjac(x):
        Hj = np.zeros((4,10), dtype=np.float32)
        for i in range(4):
            Hj[i,i] = 1.0

        return Hj 

    def fx(x, tau):
        c = x[0:2]
        s = x[2:4]
        v = x[4:6]
        ds = x[6:8]
        a = x[8:10]

        new_c = c + v*tau + 0.5*a*(tau**2)
        new_s = s * np.exp(ds*tau)
        new_v = v + a*tau
        
        new_x = x.copy()
        new_x[0:2] = new_c
        new_x[2:4] = new_s
        new_x[4:6] = new_v
        
        return new_x 

    def Fjac(x, tau):
        c = x[0:2]
        s = x[2:4]
        v = x[4:6]
        ds = x[6:8]
        a = x[8:10]
        Fj = np.diag([1, 1, np.exp(ds[0]*tau), np.exp(ds[1]*tau), 
                      1, 1, 1, 1, 1, 1])
        Fj[0,4] = tau 
        Fj[1,5] = tau 
        Fj[2,6] = s[0]*tau*np.exp(ds[0]*tau)
        Fj[3,7] = s[1]*tau*np.exp(ds[1]*tau)
        Fj[4,8] = tau
        Fj[5,9] = tau 
        Fj[0,8] = tau**2
        Fj[1,9] = tau**2

        return Fj 
    
    filter = LinearizingEKF(10, 4, fx, hx, Hjac, Fjac)

    x = np.zeros((10,), dtype=np.float32)
    x[0] = init_c[0]
    x[1] = init_c[1]
    x[2] = init_s[0]
    x[3] = init_s[1]
    x[4] = init_v[0]
    x[5] = init_v[1]
    x[6] = init_ds[0]
    x[7] = init_ds[1]
    x[8] = init_a[0]
    x[9] = init_a[1]
    filter.ekf.x = x 
    
    filter.ekf.P *= P_factor
    filter.ekf.R = np.diag([R_c, R_c, R_s, R_s])
    Q = np.diag([Q_c, Q_c, Q_s, Q_s, Q_v, Q_v, Q_ds, Q_ds,
                            Q_a, Q_a])
    Q[0,4] = Q[4,0] = Q_cov 
    Q[1,5] = Q[5,1] = Q_cov 
    Q[0,8] = Q[8,0] = Q_cov 
    Q[1,9] = Q[9,1] = Q_cov 
    Q[2,6] = Q[6,2] = Q_scov
    Q[3,7] = Q[7,3] = Q_scov
    filter.ekf.Q = Q 

    return filter

def filter3D(init_c, init_s, init_phi, init_v, init_omega=0.0, tau=default_tau,
             kappa=0.00000005,
             P_factor=10.0,
             Q_c=0.0003, Q_s=0.0001, Q_phi=0.0001, Q_v=0.001, Q_omega=0.5,
             Q_cov=0.00001,
             R_c=0.00001, R_s=0.000001, R_phi=0.0001,
             min_v_for_rotate=0.1):
    
    def hx(x):
        return x[0:5]
    
    def fx(x, tau):
        c = x[0:2]
        phi = x[4]
        v = x[5]
        omega = x[6]

        if abs(v) < min_v_for_rotate:
            omega = 0.0 # avoid spinning in place 
        
        if abs(omega) > 0.0001:
            new_c1 = c[0] + (v/omega) * (np.sin(phi + omega*tau) - np.sin(phi))
            new_c2 = c[1] + (v/omega) * (np.cos(phi) - np.cos(phi + omega*tau))
        else:
            new_c1 = c[0] + v * tau * np.cos(phi)
            new_c2 = c[1] + v * tau * np.sin(phi)
        
        new_phi = phi + omega*tau
        new_omega = omega 

        new_x = x.copy()
        new_x[0] = new_c1 
        new_x[1] = new_c2
        new_x[4] = new_phi
        new_x[6] = new_omega
        return new_x 

    def res_x(x1, x2):
        diff = x1 - x2   
        diff[4] = angle_distance(x1[4], x2[4])
        return diff 
    
    def res_z(z1, z2):
        diff = z1 - z2
        diff[4] = angle_distance(z1[4], z2[4])
        return diff

    def mean_x(sigmas, weights):
        # Computes the mean of several x vectors, taking into account
        # that x[4] is an angle

        x = np.zeros((7,), dtype=np.float32)
        sin_sum, cos_sum = 0.0, 0.0
        for sigma, weight in zip(sigmas, weights):
            x += sigma * weight 
            sin_sum += np.sin(sigma[4])*weight
            cos_sum += np.cos(sigma[4])*weight
        # Overwrite incorrect x[4] with actual angle 
        x[4] = np.arctan2(sin_sum, cos_sum)
        return x 
    
    def mean_z(sigmas, weights):
        x = np.zeros((5,), dtype=np.float32)
        sin_sum, cos_sum = 0.0, 0.0
        for sigma, weight in zip(sigmas, weights):
            x += sigma * weight 
            sin_sum += np.sin(sigma[4])*weight
            cos_sum += np.cos(sigma[4])*weight
        # Overwrite incorrect x[4] with actual angle 
        x[4] = np.arctan2(sin_sum, cos_sum)
        return x 

    #-points = MerweScaledSigmaPoints(7, alpha, beta, kappa,
    #                                sqrt_method=sqrt_func, subtract=res_x)
    points = JulierSigmaPoints(7, kappa=kappa, sqrt_method=sqrt_func, 
                               subtract=res_x)
    filter = UnscentedKalmanFilter(7, 5, tau, hx, fx, points,
                                   sqrt_fn=sqrt_func, residual_x=res_x, 
                                   residual_z=res_z,
                                   x_mean_fn=mean_x, z_mean_fn=mean_z)
    
    # Improves numerical stability 
    filter.inv = np.linalg.pinv
    
    x = np.zeros((7,), dtype=np.float32)
    x[0:2] = init_c.flatten()
    x[2:4] = init_s[0:2].flatten()
    x[4] = init_phi
    x[5] = init_v
    x[6] = init_omega
    filter.x = x

    filter.P *= P_factor # Initial uncertainty

    filter.R = np.diag([R_c, R_c, R_s, R_s, R_phi])
    Q = np.diag([Q_c, Q_c, Q_s, Q_s, Q_phi, Q_v, Q_omega])
    Q[4,6] = Q[6,4] = Q[0,5] = Q[5,0] = Q[1,5] = Q[5,1] = Q_cov     
    Q[0,6] = Q[6,0] = Q[1,6] = Q[6,1] = -Q_cov
    Q[0,4] = Q[4,0] = Q[1,4] = Q[4,1] = Q_cov
    filter.Q = Q

    return filter 


