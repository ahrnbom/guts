"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.

    
    Extract prior information (mean and standard deviations) for the shapes 
    (length, width, height) of different road user classes from the datasets
"""

import numpy as np 
from pathlib import Path
import json

from options import default_options 
from utocs import UTOCS 

def utocs_shape_priors():
    cache_folder = Path('shape_priors_cache')
    cache_folder.mkdir(exist_ok=True)
    cache_file = cache_folder / 'utocs.txt'
    if cache_file.is_file():
        obj = json.loads(cache_file.read_text())
        means = obj['means']
        stds = obj['stds']
    else:
        options = default_options()
        utocs = UTOCS(options=options)

        gts = utocs.get_gts('training')
        instances = list()
        for some in gts.values():
            instances.extend(some)
        
        means, stds = extract_means_stds(instances)
        cache_file.write_text(json.dumps({'means': means, 'stds': stds}))

    return means, stds

def extract_means_stds(instances):
    means = dict()
    stds = dict()

    ru_classes = [i.type for i in instances]
    ru_classes = list(set(ru_classes))

    for ru_class in ru_classes:
        these = [i for i in instances if i.type==ru_class]
        shapes = [i.shape for i in these]
        shapes = np.stack(shapes)

        lengths = shapes[:, 0]
        widths = shapes[:, 1]
        heights = shapes[:, 2]

        means[ru_class] = [float(np.mean(v)) for v in (lengths, widths, heights)]
        stds[ru_class] = [float(np.std(v)) for v in (lengths, widths, heights)]

    return means, stds

if __name__=="__main__":
    m, s = utocs_shape_priors()
    print(m)