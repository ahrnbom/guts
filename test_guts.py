"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.


    Test for GUTS on a demo video inside the folder data/demo. It comes from
    the UTOCS dataset. There's ground truth, so it is tested all the way 
    including MOTA score in the end. Furthermore, the results will be visualized
    in the folder 'output'.

"""

from pathlib import Path 
import numpy as np 

from guts import guts 
from images import VideoSequence
from world import World
from options import profile_options
from utocs import build_camera_matrices
from ground import Ground

def test_guts():
    output_folder = Path('output')
    output_folder.mkdir(exist_ok=True)

    folder = Path('data') / 'demo'
    imseq = VideoSequence(folder / 'vid.mp4')
    
    # Build custom world
    options = profile_options('guts')
    im_shape = (720,1280,3)
    options.im_shape = im_shape
    options.max_dist_from_camera = 50.0
    options.pixel_height_limit = 15
    options.detector_cache = False 
    PP, K = build_camera_matrices(folder, output_K=True)

    ground_points = np.genfromtxt(folder / 'ground_points.txt', 
                                      delimiter=',', dtype=np.float32).T
    n = ground_points.shape[1]
    new_ground = np.ones((4, n), dtype=np.float32)
    new_ground[0:3, :] = ground_points
    G = new_ground

    ground = Ground(G, options)
    world = World(ground, PP, K, 25.0, options)

    tracks = guts(world, imseq, 'demo')

    print(tracks)


if __name__=="__main__":
    test_guts()