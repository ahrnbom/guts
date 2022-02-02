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
from scipy.linalg import null_space
import imageio as iio

from guts import guts, track_to_obj
from images import VideoSequence
from world import World
from options import profile_options
from utocs import build_camera_matrices, gtis_from_json
from ground import Ground
from util import long_str, pflat 
from score import evaluate_tracks
from visualize import render_pixel_frame, category_colors

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

    # Run GUTS on the data
    tracks = guts(world, imseq, 'demo')

    # Evaluate those tracks against ground truth
    center_pos = pflat(null_space(PP[0]))
    def get_gt(frame_no):
        json_file = folder / 'positions' / f"{long_str(frame_no+2260,6)}.json"
        gtis = gtis_from_json(json_file)
        return gtis 
    
    gt = gt = {fn: get_gt(fn) for fn in range(0, 201)}
    mota = evaluate_tracks(tracks, gt, center_pos, options, 'demo')
    print(f"MOTA: {mota}")
    assert mota > 0.5 
    
    # Visualize results 
    colors = category_colors(options.classes)
    with iio.get_writer(output_folder / 'demo.mp4', fps=20) as out_vid:
        for frame_no in range(201):
            relevant = [t for t in tracks if frame_no in t.history]
            objs = [track_to_obj(t, frame_no, world) for t in relevant]
            for obj in objs:
                obj['X'] = np.array(obj['X'], dtype=np.float32)
            
            im = imseq.load(frame_no)
            im = render_pixel_frame(im, options.classes, frame_no, objs, [], 
                                    PP[0], colors, G)

            out_vid.append_data(im)

            if frame_no%20 == 0:
                print(f"Rendering video... {frame_no/2.0:.2f}%")

if __name__=="__main__":
    test_guts()