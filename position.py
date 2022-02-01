"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.

    Module containing classes that describe generalized positions, in both
    2D and 3D. One could also call these "detections", as the detectors
    produce instances of these classes.
"""


import numpy as np
import json

from util import smallest_box
from mask_encoding import encode_one, decode_one

"""
    Generalized 2D position, either as a segmentation mask, AABB, or both
"""
class Position:
    def __init__(self, aabb=None, mask=None, class_name="", confidence=None):
        if (aabb is None) and (mask is None):
            raise ValueError("Need either a box or a segmentation mask")
        
        if aabb is None:
            aabb = smallest_box(mask)
        
        self.aabb = aabb
        self.mask = mask
        self.class_name = class_name
        self.confidence = confidence

        # aabb is on format x1, y1, x2, y2
    
    def to_text(self):
        aabb = self.aabb.tolist()
        if self.mask is None:
            mask = ''
            imres = []
        else:
            mask = encode_one(self.mask)
            imres = list(self.mask.shape)
        return json.dumps({'class_name': self.class_name, 
                           'aabb': aabb, 'mask': mask,
                           'imres': imres, 'confidence': self.confidence})

    @staticmethod
    def from_text(text):
        obj = json.loads(text)
        aabb = np.array(obj['aabb'])
        if obj['mask']:
            mask = decode_one(obj['mask'], obj['imres'])
        else:
            mask = None 
        pos = Position(aabb=aabb, mask=mask, class_name=obj['class_name'],
                       confidence=obj['confidence'])
        return pos 

    def __repr__(self):
        return f"Position at {self.aabb}, with mask: {self.mask is not None}"
    
    def get_center_point(self):
        x = (self.aabb[0] + self.aabb[2])/2
        y = (self.aabb[1] + self.aabb[3])/2
        return x, y 
    
    def get_size(self):
        w = self.aabb[2] - self.aabb[0]
        h = self.aabb[3] - self.aabb[1]
        return w, h 
    
    def get_bottom(self):
        x = (self.aabb[0] + self.aabb[2])/2
        y = self.aabb[3]
        return x, y 
    
    def merge(self, other):
        if self.mask is None:
            x1 = min(self.aabb[0], other.aabb[0])
            y1 = min(self.aabb[1], other.aabb[1])
            x2 = max(self.aabb[2], other.aabb[2])
            y2 = max(self.aabb[3], other.aabb[3])

            self.aabb[0] = x1 
            self.aabb[1] = y1 
            self.aabb[2] = x2 
            self.aabb[3] = y2 
        else:
            assert other.mask is not None 
            self.mask = self.mask | other.mask
            self.aabb = smallest_box(self.mask)
        
        self.confidence = max(self.confidence, other.confidence)

"""
    3D position/detection, including orientation and size 
"""
class Position3D:
    def __init__(self, pos3D, shape, phi, v, class_name="", confidence=np.nan):
        self.pos3D = pos3D # shape (4,) with [x, y, z, 1.0] in world coordinates
        
        if not isinstance(shape, np.ndarray):
            shape = np.array(shape, dtype=np.float32)
        self.shape = shape # shape (3,), [l, w, h]
        
        self.phi = phi 
        self.v = v 
        self.class_name = class_name 
        self.confidence = confidence

    def has_rotation(self):
        if (self.phi is None) or np.isnan(self.phi):
            return False
        
        return True

    @staticmethod
    def from_text(text):
        obj = json.loads(text)
        assert len(obj['pos3D']) == 4 
        pos3D = np.array(obj['pos3D'], dtype=np.float32)
        assert len(obj['shape']) == 3
        shape = np.array(obj['shape'], dtype=np.float32)
        phi = float(obj['phi'])
        v = float(obj['v'])
        class_name = obj['class_name']
        confidence = float(obj['confidence'])

        return Position3D(pos3D, shape, phi, v, class_name=class_name, 
                          confidence=confidence)
    
    def to_text(self):
        obj = dict()
        obj['pos3D'] = [float(v) for v in self.pos3D]
        obj['shape'] = [float(v) for v in self.shape]
        obj['phi'] = float(self.phi)
        obj['v'] = float(self.v)
        obj['class_name'] = self.class_name
        obj['confidence'] = self.confidence
        return json.dumps(obj)
    
    def __repr__(self):
        return f"3D position of {self.class_name}, at {self.pos3D}"