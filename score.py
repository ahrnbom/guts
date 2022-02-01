"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.


    Use MOTMetrics to compute score between ground truth and a list of tracks
"""

from typing import List, Dict
import motmetrics as mm 
from dataclasses import dataclass
import numpy as np 
import shapely.geometry
import shapely.affinity
import json 

from util import vector_dist
from options import Options

@dataclass
class GTInstance:
    X:np.ndarray
    type:str
    id:int
    shape:np.ndarray
    phi:float

def gti_to_text(gti:GTInstance):
    obj = {'X': [float(x) for x in gti.X.flatten()],
           'type': gti.type, 
           'id': int(gti.id),
           'shape': [float(x) for x in gti.shape.flatten()],
           'phi': float(gti.phi)}
    return json.dumps(obj)

def text_to_gti(text:str):
    obj = json.loads(text)
    X = np.array(obj['X'], dtype=np.float32).reshape((4,1))
    shape = np.array(obj['shape'], dtype=np.float32)
    return GTInstance(X, obj['type'], obj['id'], shape, obj['phi'])

def compute_dist(gt:GTInstance, track, frame_no:int) -> float:
    if gt.type == track.class_name:
        return vector_dist(gt.X[0:2], track.vector_for_scoring(frame_no)[0:2])
    else:
        return np.nan

def iou_dist3D(gt:GTInstance, track, frame_no:int, min_iou:float):
    if gt.type == track.class_name:
        gt_vec = np.array([*gt.X[0:2], *gt.shape[0:2], gt.phi], 
                          dtype=np.float32)
        
        if np.any(np.isnan(gt_vec)):
            return np.nan

        tr_vec = track.vector_for_scoring(frame_no)
        iou = iou3D(gt_vec, tr_vec)
        if iou < min_iou:
            return np.nan
        
        return 1.0 - iou # IOU "distance" is defined like this
    else:
        return np.nan

def iou3D(A:np.ndarray, B:np.ndarray):
    a_rect = rotated_rectangle(A)
    b_rect = rotated_rectangle(B)
    intersection = a_rect.intersection(b_rect).area
    union = a_rect.union(b_rect).area
    iou = intersection / union 
    return iou


def rotated_rectangle(vec):
    # "Native" shapely solution, more than twice as slow for some reason
    #x, y, l, w, phi = vec
    #box = shapely.geometry.box(-l/2.0, -w/2.0, l/2.0, w/2.0)
    #box = shapely.affinity.rotate(box, 360.0*phi/(2.0*np.pi))
    #box = shapely.affinity.translate(box, x, y)

    if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
        raise ValueError(f"Incorrect vector {vec}")

    assert abs(vec[2]) > 0
    assert abs(vec[3]) > 0 


    positions = list()
    base_pos = vec[:2]
    phi = vec[-1]
    forward = np.array([np.cos(phi), np.sin(phi)], dtype=np.float32)
    pi2 = np.pi/2.0
    right = np.array([np.cos(phi+pi2), np.sin(phi+pi2)], dtype=np.float32)
    for il, ii in zip((-0.5, 0.5), (1.0, -1.0)):
        ll = il * vec[2] * forward 
        for iw in (-0.5, 0.5):
            ww = ii*iw * vec[3] * right 
            new_pos = base_pos + ll + ww 
            positions.append((new_pos[0], new_pos[1]))
    box = shapely.geometry.Polygon(positions)
    return box 


# tracks here are of type Track3D
# cannot type hint to due circular import that I don't want to fix lol
def evaluate_tracks(tracks:List, gt:Dict[int,List[GTInstance]],
                    center_point:np.ndarray, options:Options,
                    name:str='some_tracker'):
    max_dist = options.max_dist_from_camera
    min_iou = options.min_iou_for_match

    acc = mm.MOTAccumulator(auto_id=True)

    frames = list(gt.keys())
    frames.sort()

    for frame_no in frames:
        gt_instances = gt[frame_no]
        gt_ids = [g.id for g in gt_instances]

        relevant_tracks = [t for t in tracks if frame_no in t.history]

        if max_dist < np.inf:
            c = center_point[0:2]
            def track_ok(t):
                pos = t.vector_for_scoring(frame_no)[0:2]
                return vector_dist(c, pos) < max_dist
            relevant_tracks = [t for t in relevant_tracks if track_ok(t)]
        
        track_ids = [t.id for t in relevant_tracks]
            
        # Present distances to the format expected by MOTMetrics
        dists = list()
        for g in gt_instances:
            these_dists = list()
            for t in relevant_tracks:
                #these_dists.append(compute_dist(g, t, frame_no))
                dist = iou_dist3D(g, t, frame_no, min_iou)
                these_dists.append(dist)
            dists.append(these_dists)

        acc.update(gt_ids, track_ids, dists)
    
    # Compute the metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp'], name=name)

    return float(summary['mota'])


