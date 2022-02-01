"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.


    SAMHNet: Shape And Middle Height Network
    The CNN takes a cropped/scaled instance segmentation mask as well as a 
    vector with cropping/scaling information and object class and produces
    as its output the shape (length, width, height) and the Middle Height, a 
    factor between 0 and 1 that, when multiplied with the actual height, gives 
    the height above the ground surface of the middle point in the segmentation 
    mask. 

    Previously, SAMHNet also estimated the angle of the road user, relative to
    the camera's viewing direction, but this never worked very well and thus
    this functionality was disabled, but the code for it still exists here.
"""

import time 
from pathlib import Path
from random import shuffle, random
from shutil import rmtree
from typing import List, Tuple 
import numpy as np
import cv2
from multiprocessing import Pool
from torchinfo import summary as ti_summary
import torch 
import argparse
import json
import imageio as iio 

from position import Position
from score import text_to_gti
from util import intr, list_batches, nice, vector_normalize, all_elements
from options import Options, profile_options
from train import train
from utocs import UTOCS 

# Normalizes a direction so that (1,0) is default_dir
def normalize_direction(direction, default_dir):
    default_dir = vector_normalize(default_dir[0:2])
    right = np.array([[0, -1], [1, 0]], dtype=np.float32) @ default_dir
    new_x = np.dot(direction, default_dir)
    new_y = np.dot(direction, right)
    return np.array([new_x, new_y], dtype=np.float32)

def denormalize_direction(direction, default_dir):
    default_dir = vector_normalize(default_dir[0:2])
    right = np.array([[0, -1], [1, 0]], dtype=np.float32) @ default_dir
    x = direction[0] * default_dir
    y = direction[1] * right
    return x + y 

def summary(net, inp, verbose=True, print_fun=print):
    # For some reason UTF-8 doesn't seem to work 
    output = ti_summary(net, inp, verbose=0)
    text = str(output).encode('ascii', 'ignore').decode('ascii')
    if verbose:
        print_fun(text)
    return text 

class CustomLoss(torch.nn.Module):
    def __init__(self, include_rot, beta=1.0, weights=[1,1,1]):
        super(CustomLoss, self).__init__()
        self.sL1 = torch.nn.SmoothL1Loss(reduction='sum', beta=beta)

        if include_rot:
            assert len(weights) == 3
            self.forward = self.forward_rot
        else:
            assert len(weights) == 2
            self.forward = self.forward_norot
    
        weights = np.array(weights, dtype=np.float32)/np.sum(weights)
        self.weights = torch.from_numpy(weights)
    
    def forward_norot(self, predicted, ground_truth):
        pred_shape, pred_mh = torch.tensor_split(predicted, (3,), dim=1)
        gt_shape, gt_mh = torch.tensor_split(ground_truth, (3,), dim=1)
        
        shape_loss = self.sL1(pred_shape, gt_shape)
        mh_loss = self.sL1(pred_mh, gt_mh)

        w = self.weights
        loss = w[0]*shape_loss + w[1]*mh_loss
        return loss 

    def forward_rot(self, predicted, ground_truth):
        pred_shape, pred_rot, pred_mh = torch.tensor_split(predicted, (3,5), 
                                                           dim=1)
        gt_shape, gt_rot, gt_mh = torch.tensor_split(ground_truth, (3,5), 
                                                     dim=1)
        
        shape_loss   = self.sL1(pred_shape, gt_shape)
        rot_dot = (pred_rot*gt_rot).sum(1) # dot product 
        rot_loss = (1.0 - rot_dot).sum()
        mh_loss = self.sL1(pred_mh, gt_mh)

        w = self.weights 
        loss = w[0]*shape_loss + w[1]*rot_loss + w[2]*mh_loss

        return loss 

class SAMHNet(torch.nn.Module):
    def __init__(self, input_shape, output_rot):
        super(SAMHNet, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.output_rot = output_rot

        n = 7
        channels = 4 
        channels_per_step = 10
        ks = 3
        pd = 1
        height, width = input_shape

        # Some convolutional layers that process segmentation mask 
        for i in range(n):
            if i > 0:
                height = height//2
                width = width//2

            new_channels = channels + channels_per_step 
            self.convs.append(torch.nn.Conv2d(channels, new_channels, 
                                              kernel_size=ks, padding=pd))
            self.convs.append(torch.nn.Conv2d(new_channels, new_channels, 
                                              kernel_size=ks, padding=pd))
            self.convs.append(torch.nn.Conv2d(new_channels, new_channels, 
                                              kernel_size=ks, padding=pd))
            self.convs.append(torch.nn.Conv2d(new_channels, new_channels, 
                                              kernel_size=ks, padding=pd))
            
            channels = new_channels
            
        # The vector input is appended here 
        N = channels*height*width + 9
        
        # Fully connected layers for the last part 
        self.linears = torch.nn.ModuleList()
        for _ in range(3):  
            new_N = intr(N*1.1)
            self.linears.append(torch.nn.Linear(N, new_N))
            N = new_N
            
        # Each output is separate here but will be concatenated later 
        self.fc_shape = torch.nn.Linear(N, 3)
        if output_rot:
            self.fc_rot = torch.nn.Linear(N, 2)
        self.fc_mh = torch.nn.Linear(N, 1)

        # Utilities 
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2)
        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, x1, x2):
        out = x1
        for i in range(0, len(self.convs), 4):
            if i > 0:
                out = self.pool(out)
            
            conv1 = self.convs[i+0]
            conv2 = self.convs[i+1]
            conv3 = self.convs[i+2]
            conv4 = self.convs[i+3]

            out = self.relu(conv1(out))
            early = out 
            out = self.relu(conv2(out))
            out = self.relu(conv3(out))
            out = self.relu(conv4(out))
            
            out = out + early
        
        out = self.flatten(out)
        out = torch.cat((out, x2), dim=1)

        for lin in self.linears:
            out = self.relu(lin(out))
        
        out = self.dropout(out)
        
        out_shape = self.fc_shape(out)
        out_mh = self.fc_mh(out)
        if self.output_rot:
            out_rot = self.fc_rot(out)
            out_rot = torch.nn.functional.normalize(out_rot)
        
            out = torch.cat((out_shape, out_rot, out_mh), dim=1)
        else:
            out = torch.cat((out_shape, out_mh), dim=1)

        return out 

# Produces a numpy array that's 1 at the position where value appears in values
def one_hot(value, values:List):
    assert value in values 

    n = len(values)
    vector = np.zeros((n,), dtype=np.float32)
    index = values.index(value)
    vector[index] = 1.0

    return vector

def crop_mask(mask, aabb):
    x1, y1, x2, y2 = [intr(v) for v in aabb]
    return mask[y1:y2, x1:x2]

def crop_image(im, aabb_ints):
    x1, y1, x2, y2 = aabb_ints
    return im[y1:y2, x1:x2, :]

def imread(im_path, aabb, input_shape):
    cache_folder = Path('samhnet_image_cache')

    aabb_ints = [intr(v) for v in aabb]

    frame_no = int(im_path.stem)
    seq_no = int(im_path.parent.parent.parent.name)
    cache_key = '_'.join([str(v) for v in (seq_no, frame_no, *aabb_ints)])
    cache_file = cache_folder / f"{cache_key}.jpg"

    if cache_file.is_file():
        try:
            im = iio.imread(cache_file)
        except ValueError:
            # In this case, we are probably writing at the same time as reading
            # so just try again a little later
            time.sleep(0.1)
            im = imread(im_path, aabb, input_shape)
    else:
        im = iio.imread(im_path)
        im = crop_image(im, aabb_ints)
        im = cv2.resize(im, input_shape)
        # higher than typical quality to reduce compression issues 
        # we still prefer jpg over png due to drastically faster read/write
        iio.imwrite(cache_file, im, quality=85) 
        
    return im 

# im can be either a path to an image file or an actual image as a (H,W,3) array
def pos2D_to_net_input(pos2D:Position, input_shape:Tuple, options:Options,
                       im, which_set:str, dropout=0.25):
    mask = pos2D.mask 
    imh, imw = mask.shape 
    aabb = pos2D.aabb
    mask = crop_mask(mask, aabb)
    mask = mask.astype(np.float32)
    mask = cv2.resize(mask, input_shape)
    mask = 2.0 * (mask - 0.5) # range -1,1
    mask = np.expand_dims(mask, axis=0)
    
    if options.samhnet_input == 's':
        spatial_input = mask 
    elif options.samhnet_input == 'rgbs':
        if isinstance(im, Path):
            im = imread(im, aabb, input_shape)
        elif isinstance(im, np.ndarray):
            im = crop_image(im, aabb)
            im = cv2.resize(im, input_shape)
        else:
            raise ValueError(f"im is of unknown type {type(im)}")

        im = 2.0 * ((im.astype(np.float32) / 255.0) - 0.5) # from -1 to 1
        im = np.moveaxis(im, -1, 0) # We want (3, H, W)
        spatial_input = np.vstack([im, mask]) # (4, H, W)

        if which_set == 'training':
            if random() < dropout:
                # Choose which one to drop, this way we never drop both 
                if random() > 0.5:
                    spatial_input[0:3, :, :] = 0 # remove RGB
                else:
                    spatial_input[3, :, :] = 0 # remove S 
    else:
        raise ValueError(f"Unknown SAMHNet input type: {options.samhnet_input}")

    cx, cy = pos2D.get_center_point()
    cx /= imw
    cy /= imh 
    pw, ph = pos2D.get_size()
    pw /= imw 
    ph /= imh 
    class_vec = one_hot(pos2D.class_name, options.classes)
    input_vec = np.array([cx, cy, pw, ph, *class_vec], 
                            dtype=np.float32)

    x = (spatial_input, input_vec)
    return x 

def get_data(file:Path, options:Options, input_shape:Tuple, which_set:str, 
             metadata:bool):

    utocs = UTOCS(options=options)
    frame_no = int(file.stem)
    seq_num = int(file.parent.name)
    im_file = utocs.get_impath(seq_num, frame_no)

    lines = [l for l in file.read_text().split('\n') if l]
    examples = list()
    for line in lines:
        pos2D_text, pos3D_text, mh_text, cc_text, cd_text = line.split('_&&&&_')

        cam_dir = np.array([float(v) for v in cd_text.split(',')], 
                           dtype=np.float32)

        mh = np.float32(mh_text)
        pos2D = Position.from_text(pos2D_text)
        pos3D = text_to_gti(pos3D_text)

        x = pos2D_to_net_input(pos2D, input_shape, options, im_file, which_set)

        l, w, h = pos3D.shape
        if options.samhnet_output_phi:
            phi = pos3D.phi
            rot_vec = [np.cos(phi), np.sin(phi)]
            # Normalization here is necessary because the network cannot 
            # guess which direction the x and y axes are, but it should
            # be able to guess the direction compared to where the camera is facing
            rot_vec = normalize_direction(rot_vec, cam_dir)

            y = (np.array([l, w, h, *rot_vec, mh], dtype=np.float32), )
        else:
            y = (np.array([l, w, h, mh], dtype=np.float32), )

        if metadata:
            md = (seq_num, pos2D.class_name)
            examples.append( (x, y, md) )
        else:
            examples.append( (x, y) )

    return examples 

# Across N processes, gather data in a randomized order, while guaranteeing
# that we only go through each data point exactly once.
# When the iterator ends, the dataset has been passed exactly once 
def _data(input_shape, which_set, options, metadata, N=8):
    folder = Path('samhnet_cache') / which_set 

    if not (folder / 'done').is_file():
        print("Preparing data first...")
        from samhnet_data import utocs_data as prepare_data
        prepare_data()

    files = list(folder.glob('**/*.txt'))
    shuffle(files)
    
    with Pool(N) as pool:
        # Should be more than N so that any quickly finished processes
        # can do something also 
        for some_files in list_batches(files, 2*N):    
            inputs = [(f, options, input_shape, which_set, metadata) 
                      for f in some_files]
            examples = pool.starmap(get_data, inputs)
            examples = all_elements(examples)
            shuffle(examples)
            for example in examples:
                yield example 
        
def data(input_shape=(128,128), sets='all', metadata=False, options=None):

    if options is None:
        options = profile_options('guts')
    
    datas = dict()

    # If set to 'training' or 'validation', only do that, otherwise both 
    set_names = ['training', 'validation']
    if sets in set_names:
        set_names = [sets]

    for which_set in set_names:

        def closure(input_shape, which_set, options, metadata):
            return lambda : _data(input_shape, which_set, options,
                                  metadata)
        
        datas[which_set] = closure(input_shape, which_set, options,
                                   metadata)

    return datas 

def main(name, batch_size=128, force=False, input_shape=(150,150), epochs=128):
    # This needs to be emptied before every run 
    image_cache_folder = Path('samhnet_image_cache')
    if image_cache_folder.is_dir():
        rmtree(image_cache_folder)
    image_cache_folder.mkdir()

    options = profile_options('guts')
    output_rot = options.samhnet_output_phi

    # Create a training folder for this run 
    folder = Path('samhnet_training') / name 
    if folder.is_dir():
        if force:
            import shutil 
            shutil.rmtree(folder)
        else:
            raise ValueError(f"Run called {name} already executed! " \
                            "Choose a new name or use --force")

    folder.mkdir(parents=True)
    
    def write(text):
        with (folder / 'log.txt').open('a') as f:
            f.write(text + '\n')
        print(text)

    write(f"Input shape: {input_shape}")

    net = SAMHNet(input_shape, output_rot=output_rot)
    channels = len(options.samhnet_input)
    summary(net, [(batch_size, channels, *input_shape), (batch_size, 9)], 
            print_fun=write)
    
    if output_rot:
        loss_weights = [1,2,1]
    else:
        loss_weights = [1,1]
    write(f"Loss weights: {loss_weights}")
    loss = CustomLoss(include_rot=output_rot, weights=loss_weights)

    examples = data(input_shape)

    (folder / 'input_shape.json').write_text(json.dumps(input_shape))
    (folder / 'output_rot.json').write_text(json.dumps(output_rot))

    train(net, examples, folder, loss, write=write, batch_size=batch_size,
          plot_title=f"SAMHNet Training {name}", epochs=epochs)


if __name__=="__main__":
    nice()

    args = argparse.ArgumentParser()
    args.add_argument("--name", type=str, help="Name of this training run")
    args.add_argument("--force", action="store_true", 
                      help="Include to overwrite previous training run with "\
                           "the same name")
    args = args.parse_args()

    main(name=args.name, force=args.force)
    