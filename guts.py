"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.


    Module for GUTS and the UTS reimplementation. Contains the overarching 
    structure of these methods, and can be run as a script.
"""

from pathlib import Path 
import numpy as np 
from typing import List 
import json 
import argparse

from images import ImageSequence, FolderSequence
from options import profile_options, default_3D_params_guts, set_guts_options
from position import Position
from utocs import UTOCS
from world import World, utocs_world
from detectors import YOLOv3Detector, Detectron2Detector
from detector3D import GUTSDetector
from bicyclists import create_bicyclists
from hungarian import Hungarian
from track import reset_id
from util import long_str, nice, vector_dist
from shape_priors import utocs_shape_priors

def guts(world:World, seq:ImageSequence, name:str, verbose=True, seq_num=None):
    options = world.options

    detector_str = options.detector
    if detector_str == 'yolo':
        detector = YOLOv3Detector(cache_name=name, options=options)
    elif detector_str == 'detectron2':
        detector = Detectron2Detector(cache_name=name, options=options)
    elif detector_str == 'detectron2_and_samhnet':
        detector = GUTSDetector(cache_name=name, world=world)
    else:
        raise ValueError(f"Unknown detector type {detector_str}")

    n_frames = seq.number_of_frames()
    start_frame = seq.start_frame()

    hungarian = Hungarian(options, world, initial_frame_no=start_frame)
    reset_id()

    for frame_no in range(start_frame, start_frame+n_frames):
        if options.detector_cache and detector.is_in_cache(frame_no, 0):
            # No need to waste time loading images that won't be used
            im = None
        else:
            im = seq.load(frame_no)
        
        positions = detector.detect(im, frame_no)

        # If these are 2D positions, we should fix bicyclists here
        # if these are 3D positions, this has already been done in the detector
        if positions and isinstance(positions[0], Position):
            positions = create_bicyclists(positions, options)

            # Remove small detections (same limit as dataset)
            positions = [pos for pos in positions if 
                         pos.get_size()[1] >= options.pixel_height_limit]
        
        dt = 1.0/world.frame_rate
        hungarian.associate_detections(positions, dt)
        
        if verbose and frame_no % 20 == 0:
            print(f"{100*(frame_no-start_frame)/(n_frames):.1f}%, " \
                  f"number of tracks: {len(hungarian.tracks)}, seq {seq_num}")

    tracks = hungarian.get_final_tracks()
    return tracks

def export_tracks(tracks:List, seq:str, name:str, start_frame:int, end_frame:int,
                  world:World, verbose=True, export_options=False):
    output = Path('.') / 'output' / 'tracks' / name / seq 
    output.mkdir(parents=True, exist_ok=True)

    if export_options:
        (output / '..' / 'options.txt').write_text(str(world.options))

    center_point = world.get_camera_center()
    max_dist = world.options.max_dist_from_camera

    for frame_no in range(start_frame, end_frame+1):
        relevant = [t for t in tracks if frame_no in t.history]
        if max_dist < np.inf:
            c = center_point[0:2]
            def track_ok(t):
                pos = t.vector_for_scoring(frame_no)[0:2]
                return vector_dist(c, pos) < max_dist
            relevant = [t for t in relevant if track_ok(t)]
        
        objs = list()
        for track in relevant:
            vec = track.vector_for_scoring(frame_no)
            x, y, l, w, phi = [float(v) for v in vec]
            z = world.ground.get_z(x, y)
            fx = float(np.cos(phi))
            fy = float(np.sin(phi))
            fz = 0.0
            h = float(track.height)
            obj = {'x': x, 'y': y, 'z': z, 'l':l, 'w':w, 'h':h,
                   'forward_x': fx, 'forward_y': fy, 'forward_z': fz, 
                   'type': track.class_name, 'id': track.id}
            
            objs.append(obj)

        json_file = output / f"{long_str(frame_no, 6)}.json"
        json_file.write_text(json.dumps(objs, indent=2))

    if verbose:
        print(f"Written JSON files in {output}")

def main(dataset_name:str, extra_name="", which_set='training',
         force_detectron2=False, force_samhnet=False, 
         force_vrus=False, profile='uts', lowconf=False):
    
    options = profile_options(profile)

    if force_detectron2:
        options.detector = 'detectron2'

    if force_samhnet:
        options.detector = 'detectron2_and_samhnet'
        options.flat_ground = False 
        options.system3D = 'samhnet'
        options.tracks2D = False 
        options.hungarian2D = False 
        options.models3D = 'basic'
        options.params2D = None 
        options.params3D = default_3D_params_guts()
        options = set_guts_options(options)

    if force_vrus:
        options.classes = ['car', 'truck', 'bus', 'pedestrian', 'bicyclist']

    if lowconf:
        options.detector_conf_thresh = 0.2

    if dataset_name == 'utocs':
        dataset = UTOCS(options=options)
        shape_priors = utocs_shape_priors()[0]
    else:
        # Add your own dataset here?
        raise ValueError("Unknown dataset")

    scores = list()
    run_name = f"{dataset_name}{extra_name}"

    should_export_options = True # only once 
    for seq_str in dataset.sets[which_set]:
        seq_num = int(seq_str)

        if dataset_name == 'utocs':
            world = utocs_world(dataset.root_path, seq_num, utocs=dataset, 
                                options=options, shape_priors=shape_priors)
            
            # We use cam0 as the "main" camera
            seq = FolderSequence(dataset.root_path / 'scenarios' / \
                                long_str(seq_num, 4) / 'images' / 'cam0')
        else:
            raise ValueError("Unknown dataset")

        run_name_seq = f"{dataset_name}{seq_num}"

        tracks = guts(world, seq, name=run_name_seq, 
                      seq_num=seq_num)
        
        start, stop = dataset.get_frames(seq_num)
        export_tracks(tracks, seq_str, run_name, 
                      start_frame=start, end_frame=stop, world=world, 
                      export_options=should_export_options)
        should_export_options = False

        score = world.compute_score(seq_num, tracks)
        print(dataset_name, seq_str, score)
        scores.append(score)
    print(f"Mean score: {np.mean(scores)}")

if __name__ == "__main__":
    nice()

    args = argparse.ArgumentParser()
    args.add_argument("--name", help="Specific name for this run", 
                      type=str, default="")
    args.add_argument("--profile", help="Which options profile to use",
                      type=str, default='uts')
    args.add_argument("--dataset", help="Must be 'utocs'",
                      type=str, default="utocs")
    args.add_argument("--set", help="'training', 'validation', 'test' or 'all'", 
                      type=str)
    args.add_argument("--det2", action='store_true', 
                      help="Include to force Detectron2 as the detector")
    args.add_argument("--samhnet", action='store_true', 
                      help="Include to force usage of SAMHNet and Detectron2")
    args.add_argument("--vru", action='store_true',
                      help="Include to force inclusion of VRU classes " \
                           "(pedestrians and bicyclists)")
    args.add_argument("--lowconf", action='store_true',
                      help="Include to run with low confidence thresh (0.2). "\
                           "This is useful for building detector caches.")
    args = args.parse_args()

    sets = [args.set]
    if args.set == 'all':
        sets = ['training', 'validation', 'test']

    for the_set in sets:
        main(args.dataset, extra_name=args.name, which_set=the_set, 
             force_detectron2=args.det2, force_samhnet=args.samhnet,
             force_vrus=args.vru, profile=args.profile, lowconf=args.lowconf)
