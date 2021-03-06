"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.


    Module for some classes that describe sequences of images. 
    If your custom dataset stores images in some other way,
    create a subclass of ImageSequence and use it.
"""

from typing import List

import numpy as np
from pathlib import Path 
import imageio as iio

class ImageSequence:
    def load(self, im_num:int) -> np.ndarray:
        pass 

    def number_of_frames(self) -> int:
        pass 

    def start_frame(self) -> int:
        pass 

class FolderSequence(ImageSequence):
    def __init__(self, folder:Path):
        self.images = folder.glob('*.jpg')
        self.images = list(self.images)
        self.images.sort()

    def load(self, im_num:int) -> np.ndarray:
        return iio.imread(self.images[im_num])
    
    def number_of_frames(self) -> int:
        return len(self.images)

    def start_frame(self) -> int:
        return 0 

class VideoSequence(ImageSequence):
    def __init__(self, vid_file:Path):
        assert vid_file.is_file()
        self.vid = iio.get_reader(vid_file)
        self.frame_count = None 

    def __del__(self):
        # Attempt to clean up
        self.vid.close()
    
    def load(self, im_num:int) -> np.ndarray:
        return self.vid.get_data(im_num)
    
    def number_of_frames(self) -> int:
        if self.frame_count is None:
            self.frame_count = self.vid.count_frames()
        
        return self.frame_count

    def start_frame(self) -> int:
        return 0