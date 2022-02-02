"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.

    Tests of 2D detectors. Note that the results need to be verified manually
    by looking at the images in the folder 'test_images'
"""

import imageio as iio
import numpy as np
from position import Position 

from visualize import draw_position
from detectors import YOLOv3Detector, Detectron2Detector
    
def test_yolo():
    y = YOLOv3Detector()
       
    for i in range(4):
        im = iio.imread(f"test_images/t{i+1}.jpg")
        
        positions = y.detect(im, None)
        for pos in positions:
            im = draw_position(im, pos)
        iio.imwrite(f"test_images/o{i+1}_yolo.jpg", im)

def test_detectron2():
    d = Detectron2Detector()
    
    for i in range(4):
        im = iio.imread(f"test_images/t{i+1}.jpg")
        
        positions = d.detect(im, None)
        for pos in positions:
            im = draw_position(im, pos)
        iio.imwrite(f"test_images/o{i+1}_det2.jpg", im)

def test_save_load_positions():
    a = np.zeros((20, 20), dtype=bool)
    a[4, 5:10] = True 
    a[5, 6:11] = True 
    a[6, 7:12] = True 

    pos_a = Position(mask=a, class_name="car", confidence=0.7)
    a_text = pos_a.to_text()

    pos_b = Position.from_text(a_text)
    assert pos_a.class_name == pos_b.class_name
    assert abs(pos_a.confidence-pos_b.confidence) < 0.00001
    assert np.all(pos_a.mask == pos_b.mask)
    assert np.all(pos_a.aabb == pos_b.aabb)

if __name__=="__main__":
    test_save_load_positions()