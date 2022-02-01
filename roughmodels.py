"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.


    Rough 3D models used by UTS (not GUTS)
"""

import numpy as np
from pathlib import Path 

from options import Options
from util import rotation_matrix

def get_rough_model(class_name:str, options:Options, pos:np.ndarray, 
                    angle:float, shape:np.ndarray, append_ones=True):

    if options.models3D == 'uts':
        if not class_name in ['car', 'bus', 'truck']:
            print("WARNING! Unsupported classes!")

        model = np.genfromtxt(Path('data') / 'uts_model.csv',
                                   delimiter=',', dtype=np.float32)
    else:
        raise ValueError("Not implemented yet")

    # Set correct z-direction for this dataset
    model[2,:] *= options.z_dir

    # Rescale based on shape 
    for i in range(3):
        model[i,:] *= shape[i]
    
    # Rotate based on angle 
    model = rotation_matrix(angle) @ model

    # Position out into the world
    for i in range(3):
        model[i,:] += pos[i]
    
    # To allow multiplication with projection matrices 
    if append_ones:
        n,m = model.shape 
        new_model = np.ones((n+1, m), dtype=np.float32)
        new_model[0:n, 0:m] = model 
        model = new_model 

    return model 