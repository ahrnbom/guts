"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.


    A module for running SAMHNet on data. Also contains basic testing logic
    that visualizes the errors as histograms
"""


import argparse
from pathlib import Path
from position import Position, Position3D
from typing import List, Tuple 
import torch 
import numpy as np
import json 
from dataclasses import dataclass

from samhnet import SAMHNet, data, denormalize_direction, pos2D_to_net_input
from util import angle_distance, batches, unique_elements
from plot import histogram_plot
from world import World
from options import profile_options

device = 'cuda' if torch.cuda.is_available() else 'cpu'                

@dataclass
class NetOutput:
    x2D: np.ndarray
    y2D: np.ndarray
    shape: Tuple
    mh: float 
    phi: float 
    pos: Position 

class SAMHNetRunner:
    def __init__(self, name:str, world:World):
        self.name = name 
        self.new_world(world)

        net, input_shape, folder = get_net(name, return_input_shape=True,
                                           return_folder=True)
        
        self.net = net 
        self.input_shape = input_shape
        self.folder = folder 
        self.output_rot = net.output_rot

    def new_world(self, world:World):
        self.world = world 
        self.cam_dir = world.get_camera_direction()
        self.options = world.options
        self.opt_iters = self.options.samhnet_opt_iters
    
    def run_net(self, poss:List[Position], im, which_set:str) -> \
            List[NetOutput]:

        outputs = list()
        if not poss:
            return outputs 

        xs = [pos2D_to_net_input(pos, self.input_shape, self.options, im, 
                                 which_set) for pos in poss]
        xx = [np.stack([x[i] for x in xs]) for i in (0,1)]
        xx = [torch.from_numpy(x).to(device) for x in xx]
        output = self.net(*xx)
        output = output.detach().cpu().numpy()

        n = output.shape[0]
        for i in range(n):
            if self.output_rot:
                l, w, h, fx, fy, mh = output[i, :]
                normalized_dir = np.array([fx, fy], dtype=np.float32)
                direction = denormalize_direction(normalized_dir, self.cam_dir)
                phi = np.arctan2(direction[1], direction[0])
            else:
                l, w, h, mh = output[i, :]
                phi = float("nan")

            shape = (l, w, h)

            pos = poss[i]
            x2D, y2D = pos.get_center_point()
            outputs.append(NetOutput(x2D, y2D, shape, mh, phi, pos))
        
        return outputs 

    def run(self, poss:List[Position], im, which_set:str) -> List[Position3D]:
        # Note that we cannot guarantee that this function returns as many 
        # outputs as inputs, as triangulation can in rare cases fail 

        pos3Ds = list()
        if not poss:
            return pos3Ds 

        net_outputs = self.run_net(poss, im, which_set)

        for net_output in net_outputs:
            x2D = net_output.x2D
            y2D = net_output.y2D
            shape = net_output.shape 
            h = shape[-1] 
            mh = net_output.mh 
            phi = net_output.phi 
            pos = net_output.pos 

            out = self.world.triangulate(x2D, y2D, height=h*mh, 
                                         opt_iters=self.opt_iters,
                                         reset_height=True)
            
            if out is not None:
                x, y, z = out 
                X = np.array([x, y, z, 1.0], dtype=np.float32)
                if not np.any(np.isnan(X)):    
                    pos3D = Position3D(X, shape, phi, 0.0, 
                                    class_name=pos.class_name, 
                                    confidence=pos.confidence)
                    pos3Ds.append(pos3D)
        
        return pos3Ds
        

def get_net(name:str, return_input_shape=False, return_folder=False):
    folder = Path('samhnet_training') / name 
    
    # Read input shape 
    input_shape = tuple(json.loads((folder / 'input_shape.json').read_text()))
    output_rot = bool(json.loads((folder / 'output_rot.json').read_text()))

    net = SAMHNet(input_shape, output_rot=output_rot)

    # Pick best weights 
    weight_files = list(folder.glob('*.pth'))
    weight_files.sort(key=lambda f: float(f.stem.split('vloss')[1]))
    best_pth = weight_files[0]

    print(f"Loading weights file {best_pth} to {device}")

    net.load_state_dict(torch.load(best_pth, map_location=device)) 
    net.eval()
    net.to(device)

    to_return = [net]

    if return_input_shape:
        to_return.append(input_shape)
    
    if return_folder:
        to_return.append(folder)
    
    # Just to prevent ever giving out a list with only one element, annoying
    if len(to_return) > 1:
        return to_return
    else:
        return net 


@dataclass
class Result:
    dl: float 
    dw: float 
    dh: float
    dphi: float 
    dmh: float
    class_name: str
    seq: int

def main(name, batch_size=64):
    net, input_shape, folder = get_net(name, return_input_shape=True, 
                                       return_folder=True)

    output_rot = net.output_rot
    options = profile_options('guts')
    options.samhnet_output_phi = output_rot
    example_data = data(input_shape, sets='validation', metadata=True,
                        options=options)
    examples = example_data['validation']()

    results = list()

    for ib, batch in enumerate(batches(examples, batch_size)):
        xs = [t[0] for t in batch]
        ys = [t[1] for t in batch]
        mds = [t[2] for t in batch]

        nx = len(xs[0])
        xx = [np.stack([x[i] for x in xs]) for i in range(nx)]
        ny = len(ys[0])
        yy = [np.stack([y[i] for y in ys]) for i in range(ny)]
        xx = [torch.from_numpy(x).to(device) for x in xx]

        outputs = net(*xx)
        outputs = outputs.detach().cpu().numpy()

        n = outputs.shape[0]
        for i in range(n):
            if output_rot:
                l, w, h, fx, fy, mh = outputs[i, :]
                phi = np.arctan2(fy, fx)
                
                gt_l, gt_w, gt_h, gt_fx, gt_fy, gt_mh = yy[0][i]
                gt_phi = np.arctan2(gt_fy, gt_fx)

                dphi = angle_distance(gt_phi, phi)
            else:
                l, w, h, mh = outputs[i, :]
                gt_l, gt_w, gt_h, gt_mh = yy[0][i]
                dphi = None

            dl = gt_l - l 
            dw = gt_w - w 
            dh = gt_h - h 
            
            dmh = gt_mh - mh 

            md = mds[i]
            seq, class_name = md 
            result = Result(dl, dw, dh, dphi, dmh, class_name, seq)
            results.append(result)

        if ib%100 == 0:
            print(f"Batch {ib+1}")
    
    for class_name in ['all', 'car', 'bus', 'truck', 'bicyclist', 'pedestrian']:
        out_folder = folder / f"runstats_{class_name}"
        out_folder.mkdir(exist_ok=True)

        if class_name == 'all':
            relevant = results
        else:
            relevant = [r for r in results if r.class_name == class_name]
        
        dls = [r.dl for r in relevant]
        dws = [r.dw for r in relevant]
        dhs = [r.dh for r in relevant]
        dps = [r.dphi for r in relevant]
        dms = [r.dmh for r in relevant]
        histogram_plot(dls, out_folder / 'dl.png', title="Delta lengths")
        histogram_plot(dws, out_folder / 'dw.png', title="Delta widths")
        histogram_plot(dhs, out_folder / 'dh.png', title="Delta heights")
        if output_rot:
            histogram_plot(dps, out_folder / 'dphi.png', title="Delta phi")
        histogram_plot(dms, out_folder / 'dmh.png', title="Delta middle height")
        print(f"Done with {class_name}!")
    
    all_seqs = unique_elements([r.seq for r in results])
    all_seqs.sort()
    for seq in all_seqs:
        out_folder = folder / f"runstats_seq{seq}"
        out_folder.mkdir(exist_ok=True)

        relevant = [r for r in results if r.seq == seq]
        dls = [r.dl for r in relevant]
        dws = [r.dw for r in relevant]
        dhs = [r.dh for r in relevant]
        dps = [r.dphi for r in relevant]
        dms = [r.dmh for r in relevant]
        histogram_plot(dls, out_folder / 'dl.png', title="Delta lengths")
        histogram_plot(dws, out_folder / 'dw.png', title="Delta widths")
        histogram_plot(dhs, out_folder / 'dh.png', title="Delta heights")
        if output_rot:
            histogram_plot(dps, out_folder / 'dphi.png', title="Delta phi")
        histogram_plot(dms, out_folder / 'dmh.png', title="Delta middle height")
        print(f"Done with {seq}!")
    
    print("Done!")

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--name", type=str, help="Name of training run")
    args = args.parse_args()

    main(args.name)