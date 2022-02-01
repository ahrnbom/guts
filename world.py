"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.


    Module describing a 'world', including cameras and ground surface
"""

from pathlib import Path 
import numpy as np
from scipy.linalg.decomp_svd import null_space 

from ground import Ground
from util import natural_choice, pflat, vector_normalize
from options import Options, profile_options
from utocs import UTOCS
from shape_priors import utocs_shape_priors

class World:
    def __init__(self, ground:Ground, cameras:list, 
                 instrinsics:np.ndarray, frame_rate:float,
                 options:Options):
        self.cameras = cameras
        self.ground = ground
        self.options = options 
        self.intrinsics = instrinsics
        self.frame_rate = frame_rate

        self.camera = natural_choice(cameras)

        self.shape_priors = dict()
        # Note: Shape priors must be set separately!!
    
    def set_shape_priors(self, priors):
        self.shape_priors = priors
    
    def get_gt(self, seq_num, frame_no):
        raise NotImplementedError("world.get_gt should be overwritten!")

    def compute_score(self, seq_num, tracks):
        raise NotImplementedError("world.compute_score should be overwritten!")

    def get_camera_center(self):
        return pflat(null_space(self.camera))

    def get_camera_direction(self):
        view_dir = vector_normalize(self.camera[2, 0:3].flatten())
        return view_dir 

    def triangulate(self, x, y, height=0.0, opt_iters=0, reset_height=False):
        v = np.array([x, y, 1], dtype=np.float32)
        out = self.ground.triangulate([v], [self.camera], self.intrinsics, 
                                       h=height, opt_iters=opt_iters)
        
        if out is None:
            return None
        
        x, y, z = out 
        if reset_height:
            z -= height 
        
        return x, y, z 

    def multi_triangulate(self, xs, ys, cams, height=0.0, opt_iters=0, 
                          reset_height=False):
        vs = list()
        for x, y in zip(xs, ys):
            v = np.array([x, y, 1], dtype=np.float32)
            vs.append(v)
        
        cameras = [self.cameras[c] for c in cams]
        out = self.ground.triangulate(vs, cameras, self.intrinsics, h=height,
                                      opt_iters=opt_iters)
        
        if out is None:
            return None
        
        x, y, z = out 
        if reset_height:
            z -= height 
        
        return x, y, z 
    
    # Get a disparity map along the ground surface, from cam1 to cam2
    def get_disparity_map(self, name, cam1, cam2, N=64):
        if self.ground.is_flat:
            raise ValueError("Cannot build disparity map of flat ground")

        cache_file = Path('disparity_cache') / f"{name}_{cam1}{cam2}.csv"
        if not cache_file.parent.is_dir():
            cache_file.parent.mkdir()
        
        if cache_file.is_file():
            return np.genfromtxt(cache_file, delimiter=',')
        
        # Find x,y range of the ground
        G = self.ground.points 
        minx = np.min(G[0, :])
        maxx = np.max(G[0, :])
        miny = np.min(G[1, :])
        maxy = np.max(G[1, :])

        # Sample N^2 points
        disparities = np.zeros((4, N**2), dtype=np.float64)
        P1 = self.cameras[cam1]
        P2 = self.cameras[cam2] 
        i = 0 
        for iy in np.linspace(0.0, 1.0, N):
            y = miny + (maxy-miny)*iy 

            for ix in np.linspace(0.0, 1.0, N):
                x = minx + (maxx-minx)*ix
                z = self.ground.get_z(x, y)

                X = np.array([x, y, z, 1.0], dtype=np.float64).reshape((4,1))
                x1 = pflat(P1 @ X).flatten()
                x2 = pflat(P2 @ X).flatten()

                delta = x2 - x1 

                pxx, pxy, _ = x1 
                dx, dy, _ = delta 
                disparities[:, i] = [pxx, pxy, dx, dy]
                i += 1 

        # Save in cache 
        np.savetxt(cache_file, disparities, delimiter=',')

        return disparities


def utocs_world(utocs_root:Path, seq_num:int, options_profile:str='uts',
                shape_priors=None, utocs=None, options=None):
    if options is None:
        options = profile_options(options_profile)

    im_shape = (720,1280,3)
    options.im_shape = im_shape
    options.max_dist_from_camera = 50.0
    options.pixel_height_limit = 15

    if utocs_root is None:
        utocs_root = options.utocs_root

    if utocs is None:
        utocs = UTOCS(utocs_path=utocs_root, options=options)

    PP, K = utocs.get_cameras(seq_num)
    G = utocs.get_ground(seq_num)

    ground = Ground(G, options)
    world = World(ground, PP, K, 25.0, options)
    if world.options.flat_ground:
        world.ground.flatten()

    # In utocs experiments, we should use camera 0 as the main camera 
    world.camera = PP[0]

    if shape_priors is None:
        shape_priors = utocs_shape_priors()[0]
    world.set_shape_priors(shape_priors)

    world.get_gt = lambda seqnum, frame_no: utocs.get_gt(seqnum, frame_no)
    world.compute_score = lambda seqnum,tracks: utocs.score(seqnum, tracks)

    return world 