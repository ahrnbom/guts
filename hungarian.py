""" 
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.
    
    
    A module for Hungarian association logic, in either 2D (pixel) or 3D (world)
    coordinates. This is basically a shell for a reasonable multi-target 
    tracking solution.
"""

from scipy.optimize import linear_sum_assignment
import numpy as np
DISALLOWED = np.inf
from typing import List, Dict, Tuple

from options import Options
from util import aabb_centroid, vector_dist, pflat
from track import Track2D, Track3D
from world import World 

class Hungarian:
    def __init__(self, options:Options, world:World, initial_frame_no:int=0):
        self.options = options
        self.world = world
        self.tracks = list()
        self.old_tracks = list()
        self.frame_no = initial_frame_no
        self.time = 0.0

    def get_final_tracks(self):
        # This list copy allows this function to be called every frame
        # if you need that for some reason
        final_tracks = self.old_tracks + self.tracks

        # 2D tracks should not be included, they're just temporary
        final_tracks = [t for t in final_tracks if t.type=='3D']

        # Only include tracks that have been around for a while
        min_frames = self.world.frame_rate * self.world.options.min_time
        final_tracks = [t for t in final_tracks if len(t.history) > min_frames]

        for track in final_tracks:
            track.finalize()

        return final_tracks

    # Sufficient Amount Of Motion, see if 2D tracks should be converted to 3D
    def saom(self):
        if self.options.tracks2D and self.options.system3D == 'activecorners':
            # We're running a mix of 2D and 3D tracks!
            for track_index, track in enumerate(self.tracks):
                if track.type == '2D' and track.saom(self.time):
                    new_track = track.to3D(self.time)
                    self.tracks[track_index] = new_track

    def kill_old(self):
        to_remove = list()

        for track_index, track in enumerate(self.tracks):
            latest = track.last_updated
            time_difference = self.time - latest
            too_old = time_difference >= self.options.time_without_detections

            if track.should_die or too_old:
                to_remove.append(track_index)
            
        # Remove the tracks 
        for i in reversed(to_remove):
            track = self.tracks.pop(i)
            self.old_tracks.append(track)

    # Removes any pedestrians too close to bikes
    # This is because sometimes the bicyclist logic doesn't work
    # like if the bicycle isn't detected in that particular frame
    def suppress_pedestrians(self, dets):
        kill_range = self.options.pedestrian_suppression_range

        bicyclists = [t for t in self.tracks if t.class_name=='bicyclist']
        if not bicyclists:
            return dets

        # loop through indices backwards to allow removing from list
        for i in reversed(range(len(dets))):
            det = dets[i]
            should_die = False
            if det.class_name == 'pedestrian':
                for track in bicyclists:
                    dist = vector_dist(det.pos3D[0:2], track.get_x()[0:2])
                    if dist < kill_range:
                        should_die = True
            
            if should_die:
                dets.pop(i)
        
        return dets 

    # Process a list of detections and associated with tracks
    def associate_detections(self, dets:list, dt:float):
        self.frame_no += 1
        self.time += dt 

        # If we use a mix of 2D and 3D tracks, convert 2D to 3D
        self.saom()

        # Any tracks that are too old should be removed
        self.kill_old()

        # Predict new positions to compare with 
        for track in self.tracks:
            track.predict()
            track.store_history(self.frame_no, self.time)
        
        # Remove any detections of classes not supported by the method
        dets = [d for d in dets if d.class_name in self.options.classes]

        if (self.options.pedestrian_suppression_range > 0.00000001) and \
           (not self.options.hungarian2D):
            dets = self.suppress_pedestrians(dets)

        # Compute costs
        N = len(self.tracks)
        M = len(dets)
        if N>0 and M>0:
            mat = np.zeros((N,M), dtype=np.float32)
            for i_track in range(N):
                for i_det in range(M):
                    cost = self.compute_cost(i_track, dets[i_det])
                    mat[i_track, i_det] = cost 
        
            # Actual association
            try:
                track_indices, det_indices = linear_sum_assignment(mat)
            except ValueError:
                # Some invalid or incompatible values, try fallback method
                # Any unassigned detections will just get new tracks
                track_indices, det_indices = fallback_assignment(mat)
                
                # Old behaviour
                #track_indices = list()
                #det_indices = list()
            finally:
                dets_associated = set()

                for i_track, i_det in zip(track_indices, det_indices):
                    det = dets[i_det]
                    self.update_track(i_track, det, dt)
                    dets_associated.add(i_det)
                
                # Create new tracks for any leftover detections
                for i_det in range(len(dets)):
                    if not i_det in dets_associated:
                        self.new_track(dets[i_det])
        else:
            # Could there be no tracks, but some detections?
            if len(self.tracks) == 0 and len(dets) > 0:
                # Create tracks for each detection!
                for det in dets:
                    self.new_track(det)
    
    # Cost of associating a track with a detection
    def compute_cost(self, track_index:int, det):
        track = self.tracks[track_index]

        # Tracks and detections must be of compatible types
        # Possible optimization, treat each class separately?
        if track.class_name != det.class_name:
            return DISALLOWED

        # Find the position of the track, in correct coordinate system
        if track.type == '2D':
            track_pos = track.get_x()[0:2]
        elif track.type == '3D':
            if self.options.hungarian2D:
                # In a mixed 2D/3D setting, 3D tracks should be converted to 
                # pixel coordinates
                camera = self.world.camera
                world_pos = track.get_x()[0:2]
                
                z = self.world.ground.get_z(*world_pos)
                z += self.world.options.z_dir * track.height/2

                homogeneous = np.array([*world_pos, z, 1], dtype=np.float32)
                homogeneous = homogeneous.reshape((4,1))
                
                pixel_pos = pflat(camera @ homogeneous).flatten()
                track_pos = pixel_pos[0:2]
            else:
                track_pos = track.get_x()[0:2]
        else:
            raise ValueError(f"Track at {track_index} ({track}) has unknown "\
                             f"type {track.type}")

        # Compute the actual distance
        if self.options.hungarian2D:
            dist = vector_dist(track_pos, aabb_centroid(det.aabb))

            # use detection here, because the track could be either 2D or 3D
            mean_size = np.mean(det.get_size()) 
            max_dist = self.options.max_association_cost_factor * mean_size
        else:
            dist = vector_dist(track_pos, det.pos3D[0:2])
            mean_size = np.mean(track.get_x()[2:4])
            max_dist = self.options.max_association_cost_factor * mean_size 

        if dist > max_dist:
            dist = DISALLOWED

        return dist 

    # Update the state of a track based on associated detection
    def update_track(self, track_index:int, det, dt:float):
        track = self.tracks[track_index]
        track.update(det, dt, self.frame_no, self.time)

    # Create a new track for a detection
    def new_track(self, det):
        if self.options.tracks2D:
            track = Track2D(det, class_name=det.class_name, 
                            options=self.options, current_time=self.time,
                            world=self.world)
        else:
            track = Track3D(det.pos3D, det.shape, det.phi, det.v,
                            aabb_history=[], old_times={},
                            class_name=det.class_name,
                            options=self.options,
                            world=self.world,
                            current_time=self.time,
                            det=det)

        self.tracks.append(track)

# A fallback is sometimes needed when the assignment step in scipy fails
# Note: This simple algorithm is not guaranteed to output optimal solution
# but it is very likely to not mess up obvious assignments
def fallback_assignment(mat:np.ndarray):
    # Basic idea: For each track, find the best detection and assign it
    pairs = list()

    # Keep track of already assigned detections
    dists_per_detection = dict()

    # Since np.argmin fails on NaN, we change those to inf
    mat[np.isnan(mat)] = np.inf

    n_tracks, n_dets = mat.shape
    for i_track in range(n_tracks):
        row = mat[i_track, :]
        i_det = assign_row(row, dists_per_detection, pairs)
        if i_det is not None:
            pairs.append((i_track, i_det))
    
    assigned_tracks = [p[0] for p in pairs]
    assigned_dets   = [p[1] for p in pairs]
    return assigned_tracks, assigned_dets

def assign_row(row:np.ndarray, dists_per_detection:Dict, 
               pairs:List[Tuple[int,int]]):
    i_det = np.argmin(row)
    dist = row[i_det]
    if np.isinf(dist):
        return None
        
    # Check if this detection has already been assigned to a track
    if i_det in dists_per_detection:
        previous_dist = dists_per_detection[i_det]
        if previous_dist <= dist:
            row[i_det] = np.inf
            # Recursion without the already occupied detection
            return assign_row(row, dists_per_detection, pairs)
        else:
            # Find the old pair and kill it 
            matching_pair = [(a,b) for a,b in pairs if b==i_det][0]
            pair_index = pairs.index(matching_pair)
            pairs.pop(pair_index)

    dists_per_detection[i_det] = dist
    return i_det