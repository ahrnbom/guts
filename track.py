"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.


    This module describes 2D/3D tracks. GUTS's output is a list of instances
    of these classes.
"""


import numpy as np

from filter import filter2D, filter3D
from options import Options, Filter2DParams, Filter3DParams
from position import Position, Position3D
from util import vector_dist, to_aabb, dict_copy, dict_merge, weighted_angle
from activecorners import activecorners
from world import World

curr_id = 0
def next_id():
    global curr_id
    curr_id += 1
    return curr_id

def reset_id():
    global curr_id
    curr_id = 0  

class Track:
    def __init__(self, options:Options, class_name:str, world:World, 
                 current_time=None, det=None):
        self.options = options
        self.type = None # either "2D" och "3D"
        self.history = dict()
        self.times = dict() # total amount of seconds as a float, for each frame
        self.class_name = class_name
        self.world = world
        self.last_updated = current_time 
        self.last_updated_frame = None
        self.id = next_id()
        self.should_die = False 
    
    def is_numerically_stable(self):
        # Bad numerics can sometimes make Kalman numbers grow very large or NaN
        # We are not interested in such tracks

        c1 = np.any(np.abs(self.filter.x) > 1e8)
        c2 = np.any(np.isnan(self.filter.x))

        return (not c1) and (not c2)
    
    def finalize(self): 
        if self.last_updated_frame is None:
            self.history = {}
        else:
            self.history = {key:val for (key,val) in self.history.items() if 
                            key <= self.last_updated_frame}
            
            self.history = {key:val for (key,val) in self.history.items() if 
                            not np.any(np.isnan(val))}

            # Remove the track if it never moves significantly
            has_significant_motion = False 
            has_significant_motion_counter = 0
            first_pos = None 
            prev_frame = None 
            for frame_no, x_vec in self.history.items():
                pos = x_vec[0:2]

                if first_pos is None:
                    first_pos = pos 
                else:
                    assert frame_no > prev_frame 

                    dist = vector_dist(pos, first_pos)
                    if dist > self.options.significant_motion_distance:
                        has_significant_motion_counter += 1 
                        
                        if has_significant_motion_counter > 8:
                            has_significant_motion = True 
                            break 
                    else:
                        has_significant_motion_counter = 0 
                
                prev_frame = frame_no 
            
            if not has_significant_motion:
                self.history = dict()
                
            
class Track2D(Track):
    def __init__(self, pos:Position, **kwargs):
        super().__init__(**kwargs)        
        self.type = '2D'

        p:Filter2DParams = kwargs['options'].params2D

        x1, y1, x2, y2 = pos.aabb
        x = (x1+x2)/2 
        y = (y1+y2)/2
        self.filter = filter2D([x, y], [x2-x1, y2-y1], 
                               P_factor=p.P_factor, Q_c=p.Q_c, Q_s=p.Q_s, 
                               Q_v=p.Q_v, Q_ds=p.Q_ds, Q_a=p.Q_a, Q_cov=p.Q_cov,
                               Q_scov=p.Q_scov, R_c=p.R_c, R_s=p.R_s)

        if not self.options.tracks2D:
            raise ValueError("Tried to create a 2D track when not allowed")

    def store_history(self, frame_no:int, time:float):
        if frame_no in self.history:
            raise ValueError(f"Frame number {frame_no} already exists!!")
        
        self.history[frame_no] = self.filter.x.copy()
        self.times[frame_no] = time
    
    def predict(self):
        self.filter.predict()

        if not self.is_numerically_stable():
            self.should_die = True
    
    def get_x(self):
        return self.filter.x

    def update(self, det:Position, dt:float, frame_no:int, current_time:float):
        x1, y1, x2, y2 = det.aabb
        w = x2-x1
        h = y2-y1
        x = (x1+x2)/2
        y = (y1+y2)/2
        z = np.array([x, y, w, h], dtype=np.float32)
        self.filter.update(z, dt)

        assert current_time > self.last_updated
        self.last_updated = current_time
        self.last_updated_frame = frame_no

    # Determine if track has sufficient amount of movement to be converted to a 
    # 3D track instead
    def saom(self, current_time:float):
        # 2D tracks need to have been recently updated for SAOM to trigger
        # otherwise drifting nonsense tracks become 3D tracks
        max_time = 2.01*(1.0/self.options.frame_rate)
        if self.history and (current_time-self.last_updated)<=max_time:
            first = min(self.history.keys())
            xf, yf, wf, hf = self.history[first][0:4]
            xn, yn, wn, hn = self.filter.x[0:4]

            typical_size = np.mean([wf, hf, wn, hn])
            dist = vector_dist([xf, yf], [xn, yn])
            ratio = dist/typical_size
            if ratio > self.options.saom_thresh:
                return True 
            
        return False 

    # Convert to 3D track
    def to3D(self, current_time:float):
        first = min(self.history.keys())
        dt = current_time - self.times[first]
        assert dt > 0

        xf, yf, wf, hf = self.history[first][0:4]
        xn, yn, wn, hn = self.filter.x[0:4]

        aabb_first = to_aabb(xf, yf, wf, hf)
        aabb_now = to_aabb(xn, yn, wn, hn)
        pos_first = Position(aabb=aabb_first, class_name=self.class_name)
        pos_now = Position(aabb=aabb_now, class_name=self.class_name)

        out = activecorners(pos1=pos_first, pos2=pos_now, 
                            class_name=self.class_name,
                            world=self.world, dt=dt)
        
        if out is None:
            # Conversion to 3D failed, try again later 
            return self 
        else:
            X, Y, l, w, h, v, phi = out 

            pos3D=np.array([X, Y], dtype=np.float32)
            shape=np.array([l, w, h], dtype=np.float32)

            new_track = Track3D(pos3D, shape, phi, v, 
                                world=self.world, class_name=self.class_name, 
                                options=self.options, current_time=current_time,
                                aabb_history=dict_copy(self.history),
                                old_times=self.times)

            # Same ID to clearly mark that this 3D track inherits from 2D track
            # Unintended side effect is that the track counter is increased
            new_track.id = self.id

            return new_track
                    

class Track3D(Track):
    def __init__(self, pos3D:np.ndarray, shape:np.ndarray, phi:float, 
                 v:float, aabb_history:dict, old_times:dict, **kwargs):
        super().__init__(**kwargs)
        self.type = '3D'
        self.tau = 1.0 / kwargs['world'].frame_rate
        self.options = kwargs['options']
        self.height = shape[-1]
        self.aabb_history = aabb_history
        self.times = dict_merge(self.times, old_times)

        self.previous_detection = None 
        self.old_phi = None 
        
        if phi is None:
            # If the road user is standing still, we still want to let
            # activecorners work (or do we?)
            self.init_filter(pos3D, shape, phi, v, self.tau)
        elif np.isnan(phi):
            # If we don't have phi yet, wait to create filter until we do
            # which should happen at next update
            # For now, just store the position which we'll need to compute phi
            # This is only done in GUTS, active corners should never output NaN
            self.filter = None
            self.previous_detection = kwargs['det']
        else:
            self.init_filter(pos3D, shape, phi, v, self.tau)

    def __repr__(self):
        frames = list(self.history.keys())
        if frames:
            frames.sort()
            start = frames[0]
            stop = frames[-1]
        else:
            start = '?'
            stop = '?'
        return f"Track3D {self.class_name} {self.id}, {start}-{stop}"
        
    def init_filter(self, pos3D, shape, phi, v, tau):
        p:Filter3DParams = self.options.params3D

        self.filter = filter3D(pos3D[0:2], shape[0:2], phi, v, tau=tau,
                               kappa=p.kappa, P_factor=p.P_factor,
                               Q_c=p.Q_c, Q_s=p.Q_s, Q_phi=p.Q_phi, Q_v=p.Q_v, 
                               Q_omega=p.Q_omega, Q_cov=p.Q_cov,
                               R_c=p.R_c, R_s=p.R_s, R_phi=p.R_phi,
                               min_v_for_rotate=self.options.min_v_for_rotate)
        

    def store_history(self, frame_no:int, time:float):
        if self.filter is None:
            return

        if frame_no in self.history:
            raise ValueError(f"Frame number {frame_no} already exists!!")
        
        self.history[frame_no] = self.filter.x.copy()
        self.times[frame_no] = time
    
    def predict(self):
        if self.filter is None:
            return 

        self.filter.predict()

        if not self.is_numerically_stable():
            self.should_die = True
    
    def get_x(self):
        if self.filter is None:
            x = np.array([*self.previous_detection.pos3D.flatten()[0:2], 
                          *self.previous_detection.shape[0:2], float("nan")],
                         dtype=np.float32)
            return x 
        else:
            return self.filter.x

    def vector_for_scoring(self, frame_no:int):
        X = self.history[frame_no]
        # Scoring vector should be x, y, l, w, phi 
        return X[0:5]
    
    def suitable_previous_aabb_time(self, current_time:float):
        good_number_of_frames = 5
        l = len(self.aabb_history)

        if l <= good_number_of_frames:
            frame_no = min(self.aabb_history.keys())
            return frame_no, current_time-self.times[frame_no]
        else:
            frame_nos = list(self.aabb_history.keys())
            frame_nos.sort()

            # Hopefully not too distant and also not too recent..?
            frame_no = frame_nos[-good_number_of_frames]

            return frame_no, current_time-self.times[frame_no]

    def update(self, det, dt:float, frame_no:int, current_time:float):
        assert current_time >= self.last_updated
        self.last_updated = current_time
        self.last_updated_frame = frame_no

        if isinstance(det, Position3D):
            X, Y = det.pos3D[0:2]
            x, y = self.previous_detection.pos3D[0:2]
            dist = vector_dist([X,Y], [x,y])
            if dist > self.options.min_dist_for_phi:
                phi = np.arctan2(Y-y, X-x)
                
                factor = self.options.phi_smoothing_factor
                if factor > 0.0 and (self.old_phi is not None):
                    phi = weighted_angle(self.old_phi, phi, factor)

                if self.filter is None:
                    v = dist/self.tau
                    self.init_filter(det.pos3D, det.shape, phi, v, self.tau)
                else:
                    z = np.array([*det.pos3D[0:2], *det.shape[0:2], phi], 
                            dtype=np.float32)
                    self.filter.update(z)
                    self.old_phi = phi 

        elif isinstance(det, Position):
            before, before_dt = self.suitable_previous_aabb_time(current_time)

            xb, yb, wb, hb = self.aabb_history[before][0:4]
            aabb_before = to_aabb(xb, yb, wb, hb)
            pos_before = Position(aabb=aabb_before, class_name=self.class_name)
            out = activecorners(pos_before, det, 
                                self.class_name, self.world, 
                                before_dt)
            if out is None:
                # Don't update the filter if active corners fail!
                pass 
            else:
                X, Y, l, w, h, v, phi = out 
                if l is None or w is None:
                    l, w = self.filter.x[2:4]
                if h is None:
                    h = self.height
                if phi is None:
                    phi = self.filter.x[4]
                z = np.array([X, Y, l, w, phi], dtype=np.float32).flatten()
                
                self.filter.update(z)
                
                # Gradually update the height
                self.height = 0.9 * self.height + 0.1 * h 

            # Store new AABB in AABB history, because this isn't done elsewhere
            x1, y1, x2, y2 = det.aabb
            xn = (x1+x2)/2.0
            yn = (y1+y2)/2.0
            wn = x2-x1
            hn = y2-y1
            to_be_stored = np.array([xn, yn, wn, hn], dtype=np.float32)
            self.aabb_history[frame_no] = to_be_stored

        else:
            raise ValueError(f"Detection was of unknown type {type(det)}")
        
        self.previous_detection = det 
        