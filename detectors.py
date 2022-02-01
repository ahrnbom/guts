"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.

    
    A common interface for YOLOv3 and Detectron2
"""

import numpy as np
from typing import List
import cv2
from pathlib import Path 

import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from pytorchyolo import models as yolomodels
from pytorchyolo import detect as yolodetect

from position import Position
from util import long_str
from options import Options

class Detector:
    def __init__(self, name, options:Options, cache_name:str):
        self.is_initialized = False 
        self.name = name
        self.options = options
        self.min_conf_for_cache = 0.2
        self.conf_thresh = None # overwritten by each detector
        
        self.use_cache = self.init_cache(cache_name)

    def is_in_cache(self, im_num, cam_num):
        file = self.cache_folder / f"{long_str(im_num)}_{cam_num}.txt"
        return file.is_file()

    # im should be np.uint8 RGB
    def detect(self, im:np.ndarray, im_num:int, cam_num:int=0) \
                -> List[Position]:
        
        if self.use_cache:
            file = self.cache_folder / f"{long_str(im_num)}_{cam_num}.txt"
            if file.is_file():
                lines = [l for l in file.read_text().split('\n') if l]
                positions = [Position.from_text(l) for l in lines]
            else:
                positions = self._detect(im)
                pos_lines = [p.to_text() for p in positions]
                file.write_text('\n'.join(pos_lines))
        else:
            positions = self._detect(im)        
        
        # By filtering here, confidence threshold can be varied even when 
        # loading from cache 
        positions = [p for p in positions 
                     if p.confidence > self.conf_thresh]
        return positions

    # Since both YOLOv3 and Detectron2 uses COCO classes, might as well share  
    def get_class_name(self, class_id):
        class_names = {0: 'pedestrian', 1: 'bicycle', 2: 'car', 3: 'bicycle', 
                       5: 'bus', 7: 'truck'}
        if class_id in class_names:
            return class_names[class_id]
        return None
    
    # Initialize cache, if it already exists returns True 
    def init_cache(self, cache_name):
        if cache_name is None or (not self.options.detector_cache):
            return False
        
        folder = Path('detections_cache') / self.name / cache_name
        self.cache_folder = folder 
        return True 
    
class YOLOv3Detector(Detector):
    def __init__(self, cache_name=None, options=None):
        super().__init__('yolo', options=options, cache_name=cache_name)

        if options is None:
            self.conf_thresh = 0.5
        else:
            self.conf_thresh = options.detector_conf_thresh

    def init_net(self):
        self.model = yolomodels.load_model('/yolo/yolov3.cfg', 
                                           '/yolo/yolov3.weights')
        self.is_initialized = True

    def _detect(self, im):
        if not self.is_initialized:
            self.init_net()

        positions = list()

        # According to https://github.com/eriklindernoren/PyTorch-YOLOv3
        # input should be RGB and on range 0-255, dtype uint8
        # same as what we require
        boxes = yolodetect.detect_image(self.model, im)
        
        n = boxes.shape[0]
        for i in range(n):
            _, _, _, _, confidence, class_id = boxes[i, :]
            confidence = float(confidence)
            class_name = self.get_class_name(class_id)
            if class_name is not None and confidence >= self.min_conf_for_cache:
                pos = Position(aabb=boxes[i,0:4], class_name=class_name,
                               confidence=confidence)
                positions.append(pos)
        
        return positions
    

class Detectron2Detector(Detector):
    def __init__(self, cache_name=None, options=None):
        super().__init__('detectron2', options=options, cache_name=cache_name)

        if options is None:
            conf_thresh = 0.45
        else:
            conf_thresh = options.detector_conf_thresh
        self.conf_thresh = conf_thresh

    def init_net(self):
        d2_id = "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(d2_id))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(d2_id)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.min_conf_for_cache

        predictor = DefaultPredictor(cfg)
        
        self.predictor = predictor
        self.cfg = cfg
        
        self.is_initialized = True 

    def _detect(self, im):
        if not self.is_initialized:
            self.init_net()

        positions = list()

        # Detectron2 expects BGR for some reason, so we have to convert
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        outputs = self.predictor(im)
        
        fields = outputs['instances'].to('cpu').get_fields()
        
        all_masks = fields['pred_masks'].numpy()
        all_classes = fields['pred_classes']
        all_confidences = fields['scores']
        n = all_masks.shape[0]
        for i in range(n):
            class_id = int(all_classes[i])
            class_name = self.get_class_name(class_id)
            
            if class_name is not None:
                mask = all_masks[i, :, :]
                if np.sum(mask) > 4:
                    confidence = float(all_confidences[i])
                    pos = Position(mask=mask, class_name=class_name, 
                                confidence=confidence)
                    positions.append(pos)
        
        return positions