"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.


    This is a reimplementation of the "active corners" strategy for converting
    pixel coordinate detections into world coordinates, used by UTS.
    It's designed to only work for cars/trucks/buses, it only supports flat
    ground surfaces and in many situations provides output of questionable
    quality. SAMHNet and GUTS were designed to avoid these issues.

    The implementation is probably not a perfect recreation of the original
    UTS authors' code, but it's based on both the paper and e-mail discussions
    with the authors. This is my honest best attempt to recreate the method.
"""

import numpy as np

from position import Position
from roughmodels import get_rough_model
from util import aabb_centroid, pflat, rotation_matrix 
from world import World

# Can return None in case it fails
def activecorners(pos1:Position, pos2:Position, class_name:str, world:World,
                  dt:float):
    assert(world.ground.is_flat)
    assert(dt>0.000000001)

    c1_ic = aabb_centroid(pos1.aabb)
    c2_ic = aabb_centroid(pos2.aabb)

    c1_ic = np.array([*c1_ic, 1.0], dtype=np.float32).reshape((3,1))
    c2_ic = np.array([*c2_ic, 1.0], dtype=np.float32).reshape((3,1))

    shape_prior = world.shape_priors[class_name]
    h2 = shape_prior[2]/2

    if world.options.activecorners_onecam:
        cams = [world.camera]
    else:
        cams = world.cameras

    K = world.intrinsics
    c1_wc = world.ground.triangulate([c1_ic], cams, K, h=h2, opt_iters=0)
    c2_wc = world.ground.triangulate([c2_ic], cams, K, h=h2, opt_iters=0)

    if c1_wc is None or c2_wc is None:
        return None 

    # To homogeneous coordinates
    c1_wc = np.array([*c1_wc, 1.0], dtype=np.float32).reshape((4,1))
    c2_wc = np.array([*c2_wc, 1.0], dtype=np.float32).reshape((4,1))

    # Place back on ground surface
    c1_wc[2] -= h2*world.options.z_dir
    c2_wc[2] -= h2*world.options.z_dir 

    diff = c2_wc - c1_wc
    v = np.linalg.norm(diff[0:2]) / dt 
    phi = np.arctan2(diff[1], diff[0]) # [-pi,pi), 0 means the x-direction

    if v < world.options.activecorners_minv:
        # Road user is standing still
        X, Y = c1_wc[0:2]
        return X, Y, None, None, None, v, None 

    # Greatly simplified, rough approximate version for debugging
    #l, w, h = shape_prior
    #return c2_wc[0], c2_wc[1], l, w, h, v, phi

    # Let's find the position based on the latest detection 
    c_wc = c2_wc 
    
    # Get 3D model
    assert world.options.models3D == 'uts'
    model = get_rough_model(class_name, options=world.options, pos=c_wc,
                            angle=phi, shape=shape_prior)

    # Find active edges 
    P = world.camera
    model2D = pflat(P @ model)
    top_edge = np.argmin(model2D[1,:])
    bot_edge = np.argmax(model2D[1,:])
    left_edge = np.argmin(model2D[0,:])
    right_edge = np.argmax(model2D[0,:])

    aabbs = [pos1.aabb, pos2.aabb]

    Ms = list()
    bs = list()
    for t in range(2):
        aabb = aabbs[t]
        x1, y1, x2, y2 = aabb

        # First element is model2D index
        # Second element is 1 if top/down and 0 if left/right
        # Third element is the corresponding 2D coordinate from the AABB
        edges = [(top_edge, 1, y1), (bot_edge, 1, y2), 
                 (left_edge, 0, x1), (right_edge, 0, x2)]

        for edge in edges:
            index, i, xi = edge 
            dl, dw, dh = deltas(index)        
            M_part, b_part = make_parts(P, i, phi, xi, dl, dw, dh, t*dt, 
                                        z_dir=world.options.z_dir,
                                        Z=world.ground.flat_z)
            Ms.append(M_part)
            bs.append(b_part)
    
    # Add equations for shape normalization
    shape_prior_weight = world.options.activecorners_spw
    Ms.append([0,0,shape_prior_weight,0,0,0])
    bs.append(shape_prior[0] * shape_prior_weight)
    Ms.append([0,0,0,shape_prior_weight,0,0])
    bs.append(shape_prior[1] * shape_prior_weight)
    Ms.append([0,0,0,0,shape_prior_weight,0])
    bs.append(shape_prior[2] * shape_prior_weight)

    M = np.vstack(Ms)
    b = np.vstack(bs)
    
    # Solve it!
    solution = np.linalg.lstsq(M, b, rcond=None)[0]
    X, Y, l, w, h, v2 = solution 
    l = abs(l)
    w = abs(w)
    h = abs(h)
    v2 = abs(v2)
    
    # Move forward to where we think the RU is now
    #TODO does this really help?
    dx = np.cos(phi)
    dy = np.sin(phi)
    X += v*dt*dx 
    Y += v*dt*dy     

    return X, Y, l, w, h, v, phi

def deltas(index):
    # This just depends on the order of the points in the rough UTS 3D model
    if (index//2)%2 == 0:
        dl = 1.0
    else:
        dl = -1.0
    
    if index%2 == 0:
        dw = 1.0
    else:
        dw = -1.0
    
    if index<4:
        dh = 0
    else:
        dh = 2.0

    dl /= 2
    dw /= 2
    dh /= 2

    return dl, dw, dh 

def make_parts(P, i, phi, xi, dl, dw, dh, dt, z_dir, Z=0.0):
    Pi = P[i, :]
    Pz = P[2, :]

    R = rotation_matrix(phi)
    r11 = R[0,0]
    r12 = R[0,1]
    r21 = R[1,0]
    r22 = R[1,1]

    xp1 = (Pi - xi*Pz).T
    xp2 = (xi*Pz - Pi).T

    MM = np.array([[1, 0, dl*r11, dw*r12, 0, dt*r11],
                  [0, 1, dl*r21, dw*r22, 0, dt*r21],
                  [0, 0, 0, 0, z_dir*dh, 0],
                  [0, 0, 0, 0, 0, 0]], dtype=np.float32)
    
    M = xp1 @ MM
    b = xp2 @ np.array([0, 0, Z, 1], dtype=np.float32).reshape((4,1))

    return M, b 
