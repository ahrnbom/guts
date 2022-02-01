"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.


    Module for describing a ground surface based on a set of points on it.
    Also supports flat grounds at a constant z-level 
"""


import numpy as np
from scipy.linalg import null_space
from scipy.optimize import least_squares
from options import Options

from util import vector_dist, ldiv, pflat, dlt, multi_flatten
import panic 

"""
    Generalized ground surface representation, either as a number of points
    we can interpolate between, or as just a plane
"""
class Ground():
    def __init__(self, points, options:Options, flat_z=None):
        self.is_flat = False
        self.options = options 

        if flat_z is not None:
            self.is_flat = True
            self.flat_z = flat_z 
        else:
            assert isinstance(points, np.ndarray)
            assert len(points.shape) == 2 
            assert points.shape[0] == 4

            self.points = points
            self.alpha = options.ground_alpha
            self.local_ground_cache = list()

    def flatten(self):
        self.flat_z = np.mean(self.points[2,:])
        self.is_flat = True 

    # v: list of 2D points, one per camera, in homogeneous coordinates
    # PP: list of camera projection matrices (3x4)
    # K: intrinsic matrix of all cameras (3x3)
    # h: Height above the ground (in metres). Note z_dir is used automagically!
    # opt_iters: If above zero, the 3D position will be optimized using PANIC
    def triangulate(self, v, PP, K, h=0.0, opt_iters=0):
        n = len(v)
    
        assert(n == len(PP))        
        assert(n >= 1)

        z_dir = self.options.z_dir
        height = h*z_dir 

        # Moves the entire ground up or down, used to find points 
        # above/below ground plane
        if abs(height) > 0.000001: 
            G = self.points.copy()
            G[2,:] += height
        else:
            G = self.points

        M = np.zeros((3*n, 4+n), dtype=np.float64)
        for i in range(n):
            M[(3*i):(3*i + 3), 0:4] = ldiv(K, PP[i])
            M[(3*i):(3*i + 3), 4+i] = -pflat(ldiv(K, np.squeeze(v[i])))
        w = dlt(M)
        
        X0 = pflat(w[0:4])
        
        if not np.all(np.isfinite(X0)):
            return None # no use continuing if data is unstable
            
        # Get central position of all cameras
        centers = [pflat(null_space(P)) for P in PP]
        if len(centers) == 1:
            CC = np.array(centers[0])
        else:
            CC = np.mean(np.array(centers), axis=0)
        CC = np.squeeze(CC)
        
        mz = np.mean(G[2,:])
        
        # Assume ground is just z=mz
        b = CC[2] - X0[2]
        a = np.sqrt( (CC[0] - X0[0])**2 + (CC[1] - X0[1])**2 )
        B = CC[2] - mz
        A = a*B/b
        D = X0[0:2] - CC[0:2]
        D /= np.linalg.norm(D)
        
        X1 = np.array([CC[0], CC[1], 0, 1]) + np.hstack([A*D, np.array([0, 0])])
        init = X1[0:2]

        if opt_iters==0 or self.is_flat:
            x, y = init
            z = self.get_z(x, y, h)
            return x, y, z
            
        g = self.get_local_ground(*init)
        if abs(height) > 0.000001: 
            g = g.copy()
            g[2,:] += height
        
        alpha = self.alpha

        # Now we have what we need to optimize over the position
        def _res(x):
            inputs = multi_flatten([x, *PP, *v, g[0:3, :], alpha])
            inputs = inputs.astype(np.float64)
            out = np.zeros((6,), dtype=np.float64)
            panic_path = f"generated/gtriangul{n}_res.mc"
            r = panic.run(panic_path, 'main', [out, inputs])[0]
            return r
        
        def _jac(x):
            inputs = multi_flatten([x, *PP, *v, g[0:3, :], alpha])
            inputs = inputs.astype(np.float64)
            out = np.zeros((6,2), dtype=np.float64)
            panic_path = f"generated/gtriangul{n}_jac.mc"
            j = panic.run(panic_path, 'main', [out, inputs])[0]
            return j
        
        if not np.all(np.isfinite(init)):
            return None # no use continuing if data is unstable
        
        out = least_squares(_res, init, jac=_jac, method='lm', 
                            max_nfev=opt_iters)
        opt = out.x

        x, y = opt
        z = self.get_z(x, y, h)
        return x, y, z

    def get_z(self, x, y, height=0.0, fast=True):
        z_dir = self.options.z_dir

        if self.is_flat:
            return self.flat_z + z_dir*height
        else:
            points = self.points
            if fast:
                points = self.get_local_ground(x, y)
            z = ground_z(points, self.alpha, x, y, height=z_dir*height)
            return z 
    
    def get_local_ground(self, x, y):
        assert not self.is_flat

        for item in self.local_ground_cache:
            x_cache, y_cache, g_cache = item 
            if vector_dist([x,y], [x_cache,y_cache]) < 0.05:
                return g_cache 
        
        # Not found in cache 
        g = local_ground_model(x, y, self.points)
        new_item = (x, y, g)
        self.local_ground_cache.append(new_item)

        if len(self.local_ground_cache) > 32:
            # Remove the oldest 
            self.local_ground_cache.pop(0)
        
        return g 
            
def local_ground_model(x, y, G):
    # Sort the ground based on squared distance from (x, y)
    
    n = G.shape[1]
    if n < 8:
        raise ValueError("Cannot create local ground with fewer than 8 points")
    
    dx = G[0, :] - x
    dy = G[1, :] - y
    sq_dists = dx**2 + dy**2
    
    indices = np.argsort(sq_dists)
    
    # Pick the 8 closest
    indices = indices[:8]
    g = G[:, indices]
    return g
    

def ground_z(ground, alpha, x, y, height):
    n = ground.shape[1]
    sum_w = 0.0
    z = 0.0
    for i in range(n):
        gx = ground[0, i]
        gy = ground[1, i]
        gz = ground[2, i] + height
        
        dist = np.sqrt( (gx-x)**2 + (gy-y)**2 )
        w = np.exp(-alpha*dist)
        sum_w += w
        z += w*gz
    z /= sum_w
    
    return z