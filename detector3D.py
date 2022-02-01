"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.

    
    Similar to the AABB detectors in detectors.py, this module describes a class
    with a similar interface, but which produces Position3D instead of Position
    by converting positions into world coordinates using SAMHNet. This also 
    means that bicyclist logic must happen inside of here. Supports caching.
"""

from pathlib import Path 
import numpy as np 
from typing import List, Dict 

from position import Position3D
from detectors import Detectron2Detector
from bicyclists import create_bicyclists
from options import Options
from samhnet_run import NetOutput, SAMHNetRunner
from world import World 
from util import long_str, vector_dist

class GUTSDetector:
    def __init__(self, cache_name:str, world:World):
        self.name = 'detectron2_and_samhnet'

        self.world = world
        options = world.options
        self.options = options 
        self.cache_name = cache_name
        self.conf_thresh = options.detector_conf_thresh

        self.networks_are_initialized = False 
        self.use_cache = options.detector_cache

        if self.use_cache:
            self.folder = Path('detections_cache') / self.name / self.cache_name
            self.folder.mkdir(exist_ok=True, parents=True)

    def is_in_cache(self, im_num, cam_num):
        file = self.folder / f"{long_str(im_num)}_{cam_num}.txt"
        return file.is_file()

    def init_networks(self):
        self.det2 = Detectron2Detector(cache_name=self.cache_name, 
                                       options=self.options)
        self.samhnet = SAMHNetRunner(self.options.samhnet_name, self.world)
        self.networks_are_initialized = True

    def detect(self, im:np.ndarray, im_num:int, cam_num:int=0) \
                -> List[Position3D]:

        if self.use_cache:
            file = self.folder / f"{long_str(im_num)}_{cam_num}.txt"
            if file.is_file():
                lines = [l for l in file.read_text().split('\n') if l]
                positions = [Position3D.from_text(l) for l in lines]
            else:
                positions = self._detect(im, im_num, cam_num)
                pos_lines = [p.to_text() for p in positions]
                file.write_text('\n'.join(pos_lines))
        else:
            positions = self._detect(im, im_num, cam_num)
        
        # By filtering here, confidence threshold can be varied even when 
        # loading from cache 
        positions = [p for p in positions 
                     if p.confidence > self.conf_thresh]
        return positions

    def _detect(self, im:np.ndarray, im_num:int, cam_num:int=0):
        if not self.networks_are_initialized:
            self.init_networks()

        # by calling detect, instead of _detect, we can re-use cache 
        pos2Ds = self.det2.detect(im, im_num, cam_num)

        # These need to happen here since we never expose 2D detections 
        pos2Ds = create_bicyclists(pos2Ds, self.options)
        pos2Ds = [pos for pos in pos2Ds if 
                        pos.get_size()[1] >= self.options.pixel_height_limit]

        # Always run in test mode to make sure no input-dropout is applied 
        pos3Ds = self.samhnet.run(pos2Ds, im, 'test')

        return pos3Ds

        
class MultiViewGUTSDetector:
    def __init__(self, cache_name:str, world:World, cams:List[int]):
        self.cache_name = cache_name
        self.cams = cams
        self.world = world
        self.cams_name = ''.join([str(c) for c in cams])
    
        self.name = 'multiview_guts'
        options = world.options
        self.options = options 
        self.conf_thresh = options.detector_conf_thresh

        self.networks_are_initialized = False 
        self.use_cache = options.detector_cache

        if self.use_cache:
            self.folder = Path('detections_cache') / self.name / self.cache_name
            self.folder.mkdir(exist_ok=True, parents=True)

    def init_networks(self):
        self.det2 = Detectron2Detector(cache_name=self.cache_name, 
                                       options=self.options)
        self.samhnet = SAMHNetRunner(self.options.samhnet_name, self.world)
        self.networks_are_initialized = True 
    
    def contains_cache(self, im_num):
        file = self.folder / f"{long_str(im_num)}_{self.cams_name}.txt"
        return file.is_file()
        
    def detect(self, ims:List[np.ndarray], im_num:int, cams:List[int]):
        cams_name = self.cams_name 
        if self.use_cache:
            file = self.folder / f"{long_str(im_num)}_{cams_name}.txt"
            if file.is_file():
                lines = [l for l in file.read_text().split('\n') if l]
                positions = [Position3D.from_text(l) for l in lines]
            else:
                positions = self._detect(ims, im_num)
                pos_lines = [p.to_text() for p in positions]
                file.write_text('\n'.join(pos_lines))
        else:
            positions = self._detect(ims, im_num)
            
        positions = [p for p in positions 
                        if p.confidence > self.conf_thresh]
        return positions

    def _detect(self, ims:List[np.ndarray], im_num:int):
        if not self.networks_are_initialized:
            self.init_networks()

        all_net_outputs = dict()
        for im, cam_num in zip(ims, self.cams):
            # by calling detect, instead of _detect, we can re-use cache 
            pos2Ds = self.det2.detect(im, im_num, cam_num)

            # These need to happen here since we never expose 2D detections 
            pos2Ds = create_bicyclists(pos2Ds)
            pos2Ds = [pos for pos in pos2Ds if 
                            pos.get_size()[1] >= self.options.pixel_height_limit]

            # Always run in test mode to make sure no input-dropout is applied 
            net_outputs = self.samhnet.run_net(pos2Ds, im, 'test')
            all_net_outputs[cam_num] = net_outputs

        matched_outputs = self.stereo_matching(all_net_outputs)
        pos3Ds = self.triangulate(matched_outputs)

        return pos3Ds
    
    def triangulate(self, matched:List[Dict[int, NetOutput]]):
        opt_iters = self.options.samhnet_opt_iters

        pos3Ds = list()
        if not matched:
            return pos3Ds

        for match in matched:
            # Average of several indep endent estimates of the height 
            
            cams = list(match.keys())
            xs = [match[cam].x2D for cam in cams]
            ys = [match[cam].y2D for cam in cams]
            heights = [match[cam].mh * match[cam].shape[-1] for cam in cams]
            height = np.mean(heights)

            out = self.world.multi_triangulate(xs, ys, cams, height=height, 
                                               opt_iters=opt_iters, 
                                               reset_height=True)
            
            if out is not None:
                x, y, z = out 
                shape = match[0].shape
                phi = match[0].phi 
                pos = match[0].pos 

                X = np.array([x, y, z, 1.0], dtype=np.float32)
                if not np.any(np.isnan(X)):    
                    pos3D = Position3D(X, shape, phi, 0.0, 
                                    class_name=pos.class_name, 
                                    confidence=pos.confidence)
                    pos3Ds.append(pos3D)
        
        return pos3Ds 
    
    def stereo_matching(self, all_outputs:Dict[int, List[NetOutput]]):
        if len(self.cams) == 1:
            return [[o] for o in all_outputs]

        main_cam = self.cams[0]
        disparities = dict()
        for cam in self.cams[1:]:
            disparities[cam] = self.world.get_disparity_map(self.cache_name, 
                                                            main_cam, cam)

        corresponances = list()
        # Go through each instance in the main camera, and try to find 
        # corresponding detections in the other cameras 
        for main_output in all_outputs[main_cam]:
            bottom_x, bottom_y = main_output.pos.get_bottom()
            
            candidate = {main_cam: main_output}
            class_name = main_output.pos.class_name

            for cam in self.cams[1:]:
                disparity = disparities[cam]
                dist_x = disparity[0, :] - bottom_x
                dist_y = disparity[1, :] - bottom_y 
                dists = dist_x**2 + dist_y**2
                weights = np.exp(-dists)
                weights /= np.sum(weights)

                dx = np.sum(weights * disparity[2, :])
                dy = np.sum(weights * disparity[3, :])

                # Where we think the road user would be in cam2
                expected = np.array([bottom_x + dx, bottom_y + dy], 
                                    dtype=np.float32)
                
                # Sane maximum distance 
                best_dist = 0.25 * sum(main_output.pos.get_size())/2.0
                best_match = None 

                # Find road users close to here 
                for output in all_outputs[cam]:
                    if not (output.pos.class_name == class_name):
                        continue

                    x, y = output.pos.get_bottom()
                    pos = np.array([x, y], dtype=np.float32)
                    dist = vector_dist(expected, pos)

                    if dist < best_dist:
                        best_dist = dist 
                        best_match = output 
                
                if best_match is not None:
                    candidate[cam] = best_match
            
            corresponances.append(candidate)
        
        return corresponances