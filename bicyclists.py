"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.

    
    Detectors trained on MS COCO does not have a notion of bicyclists, instead 
    they see "bicycle" or "motorbike", and sometimes they see a "pedestrian" as 
    well (if the rider was detected individually).

    The role of this module is to take a list of detections and convert any 
    "bicycle" as well as any "pedestrian" placed above and close enough, 
    into "bicyclist"
"""

from typing import List 

from position import Position
from util import vector_dist
from options import default_options

# Note! This makes changes to the input list!
# We assume you do not want to store the raw detections 
# If you do, just store them before by copying the list
def create_bicyclists(positions: List[Position],
                      options=None) -> List[Position]:
                
    if options is None:
        options = default_options()
    comparable_dist_thresh = options.bicyclist_thresh
    
    to_remove = list()

    for pos in positions:
        if pos.class_name == "bicycle":
            pos.class_name = "bicyclist"

            # Any nearby, above, pedestrians?
            x1, y1 = pos.get_center_point()
            dx = pos.aabb[2] - pos.aabb[0]
            dy = pos.aabb[3] - pos.aabb[1]
            mean_dist = (dx + dy)/2
            
            for other_index, other in enumerate(positions):
                if other_index in to_remove:
                    continue 

                if other.class_name == 'pedestrian':
                    # Is it placed above?
                    x2, y2 = other.get_center_point()
                    if y2 < y1: 
                        # Are they close enough? Compare with dimensions
                        dist = vector_dist((x1, y1), (x2, y2))
                        comparable_dist = dist/mean_dist
                        if comparable_dist < comparable_dist_thresh:
                            pos.merge(other)
                            to_remove.append(other_index)

    # Go through backwards to make index points at the right object 
    to_remove.sort(reverse=True)
    for index in to_remove:
        positions.pop(index)
    
    return positions
                        