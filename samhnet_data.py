"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.
    
   
    Prepare data for SAMHNet training/validation
"""

import numpy as np 
from pathlib import Path
import json
from typing import List 
from multiprocessing import Pool 

from options import profile_options
from score import GTInstance, gti_to_text
from util import clamp, intr, long_str, nice, pflat
from utocs import UTOCS
from mask_encoding import decode_one
from position import Position
from bicyclists import create_bicyclists
from world import World, utocs_world

# Extracts a rectangular region from numpy array close to a point 
# The resulting region can be smaller than width/height if it's close to borders
def get_neighbourhood(mask, x, y, width=8, height=8):
    mask_h, mask_w = mask.shape 

    if x < 0 or x >= mask_w or y < 0 or y >= mask_h:
        return None 

    minx = clamp(intr(x - width/2), 0, mask_w-1)
    maxx = clamp(intr(x + width/2), 0, mask_w-1)
    miny = clamp(intr(y - height/2), 0, mask_h-1)
    maxy = clamp(intr(y + height/2), 0, mask_h-1)

    hood = mask[miny:maxy, minx:maxx]
    return hood 

def match_positions(poss2D:List[Position], poss3D: List[GTInstance], P:np.ndarray):
    matches = list()
    for pos3D in poss3D:
        found2D = None 
        most_pixels = 0

        x2D = pflat(P @ pos3D.X).flatten()
        for pos2D in poss2D:
            if not (pos2D.class_name == pos3D.type):
                continue 
            
            # Check if the 2D point is inside the mask or close to it 
            x = intr(x2D[0])
            y = intr(x2D[1])
            neighbourhood = get_neighbourhood(pos2D.mask, x, y)
            if neighbourhood is not None:
                score = np.sum(neighbourhood)
                if score > most_pixels:
                    found2D = pos2D
                    most_pixels = score 
                    # Just in case there are multiple matches, pick the
                    # most promising one 
        
        if found2D is not None:
            matches.append( (found2D, pos3D) )
    
    return matches 

def prepare_utocs(seq_name:str, which_set:str, utocs:UTOCS, conf_thresh=0.35):
    # Verify that instance segmentation detections are in the cache 
    seq_num = int(seq_name)
    cache_folder = Path('detections_cache') / 'detectron2' / f"utocs{seq_num}"
    if not cache_folder.is_dir():
        raise ValueError(f"Sequence {seq_name} is not in detections cache!")
    
    # For now, just use the default camera (_0)
    cache_files = [f for f in cache_folder.glob('*_0.txt')]
    cache_files.sort()

    # Prepare a folder to store training data
    folder = Path('samhnet_cache') / which_set / seq_name
    folder.mkdir(parents=True, exist_ok=True)

    # Build the world
    world = utocs_world(utocs.root_path, seq_num, utocs=utocs, 
                        options_profile='guts')

    for cache_file in cache_files:
        frame_no = int(cache_file.stem.split('_')[0])
        
        lines = [l for l in cache_file.read_text().split('\n') if l]
        positions = list()
        for line in lines:
            obj = json.loads(line)
            mask_str = obj['mask']
            im_size = tuple(obj['imres'])
            mask = decode_one(mask_str, im_size)
            class_name = obj['class_name']
            conf = obj['confidence']
            if conf >= conf_thresh:
                pos = Position(mask=mask, class_name=class_name, 
                               confidence=conf)
                positions.append(pos)

        # Important to make sure we have "bicyclist" and not "bicycle"!
        positions = create_bicyclists(positions)
        
        positions3D = utocs.get_gt(seq_num, frame_no)

        PP, _ = utocs.get_cameras(seq_num)
        P = PP[0]

        matched = match_positions(positions, positions3D, P)
        out_lines = list()
        for match in matched:
            pos2D:Position = match[0]
            pos3D:GTInstance = match[1]

            mh = compute_middle_height(pos2D, pos3D, world)

            cam_cen = world.get_camera_center().flatten().tolist()
            cam_dir = world.get_camera_direction().flatten().tolist()
            cam_cen = ','.join([str(float(c)) for c in cam_cen])
            cam_dir = ','.join([str(float(c)) for c in cam_dir])

            line = '_&&&&_'.join([pos2D.to_text(), gti_to_text(pos3D), str(mh),
                                  cam_cen, cam_dir])
            out_lines.append(line)
        
        out_file = folder / f"{long_str(frame_no)}.txt"
        out_file.write_text('\n'.join(out_lines))

        if frame_no%20 == 0:
            print(out_file)

def same_sign(a, b):
    sa = np.sign(a)
    sb = np.sign(b)
    return abs(sa - sb) < 0.001

# Use bisection method to find the optimal middle height
def compute_middle_height(pos2D:Position, pos3D:GTInstance, world:World):
    p = np.array(pos2D.get_center_point(), dtype=np.float32)
    w = pos3D.X.flatten()[0:2]
    c = world.get_camera_center().flatten()[0:2]

    # c: camera position
    # w: ground truth position of object 
    # t: triangulated position given some middle height
    # The idea is to project t to the line from c to w
    # If the projection is close to 1.0, t is close to w
    # and taking 1.0 - proj is a signed loss, perfect for bisection
    # Furthermore, it is reasonable to assume that mh=0 gives negative and 
    # mh=1 gives positive loss 

    dir_cw = w - c 
    divisor = np.linalg.norm(dir_cw)**2

    iters = world.options.samhnet_opt_iters

    def f(h): 
        out = world.triangulate(p[0], p[1], h*pos3D.shape[2], opt_iters=iters)
        if out is None:
            return np.nan
        else:
            tx, ty = out[0:2]
            t = np.array([tx, ty], dtype=np.float32)
            dir_ct = t - c 
            proj = np.dot(dir_cw, dir_ct) / divisor 
            return float(1.0 - proj)

    # Bisection method
    low = 0.0
    high = 1.0

    low_score = f(low)
    high_score = f(high)

    okay = False 
    while not okay:
        mid_point = (low + high)/2.0
        mid_score = f(mid_point)

        if (abs(mid_score) < 0.001) or ((high-low) < 0.0001):
            okay = True 
        else:
            if same_sign(low_score, mid_score):
                mid_score = low_score 
                low = mid_point 
            elif same_sign(high_score, mid_score):
                mid_score = high_score 
                high = mid_point 
            else:
                break 
    
    return mid_point 

def utocs_data():
    options = profile_options('guts')
    utocs = UTOCS(options=options)

    seqs = utocs.sets['training']
    with Pool(4) as pool:
        pool.starmap(prepare_utocs, [(seq, 'training', utocs) for seq in 
                                     utocs.sets['training']])
        # This helps other script know if this needs to be called again or not
        (Path('samhnet_cache') / 'training' / 'done').write_text("Yes!")

        pool.starmap(prepare_utocs, [(seq, 'validation', utocs) for seq in 
                                     utocs.sets['validation']])
        (Path('samhnet_cache') / 'validation' / 'done').write_text("Yes!")

if __name__=="__main__":
    nice()
    utocs_data()