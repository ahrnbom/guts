"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.


    Extract data from the UTOCS dataset
    To run GUTS on a custom dataset, you may want to create a similar
    class for your own dataset
"""

from pathlib import Path 
import numpy as np 
from scipy.linalg import null_space
from typing import List, Dict
import json

from images import FolderSequence
from util import long_str, pflat
from score import GTInstance, evaluate_tracks
from options import Options

class UTOCS:
    def __init__(self, utocs_path:Path=None, options:Options=None):
        if options is None:
            raise ValueError("Must provide at least options!")
        self.options = options
        if utocs_path is None:
            utocs_path = Path(options.utocs_root)

        sets_path = utocs_path / 'sets.txt'
        set_lines = [l for l in sets_path.read_text().split('\n') if l]
        sets = dict()
        for set_line in set_lines:
            key = set_line.split(':')[0]
            sets[key] = set_line.split(': ')[1].split(' ')
        
        self.sets = sets 
        self.root_path = utocs_path

    def get_seqnums_sets(self) -> Dict[str, List[int]]:
        out = dict()
        for the_set, seqs in self.sets.items():
            out[the_set] = [int(v) for v in seqs]
        return out 

    def get_seqnums(self) -> List[int]:
        folder = self.root_path / 'scenarios'
        scenarios = [f for f in folder.glob('*') if f.is_dir()]
        scenarios.sort()

        seq_nums = [int(f.name) for f in scenarios]
        return seq_nums

    def get_cameras(self, seq_num:int):
        sn = long_str(seq_num, 4)
        folder = self.root_path / 'scenarios' / sn
        PP, K = build_camera_matrices(folder, output_K=True)
        return PP, K 

    def get_ground(self, seq_num:int):
        sn = long_str(seq_num, 4)
        folder = self.root_path / 'scenarios' / sn
        ground_points = np.genfromtxt(folder / 'ground_points.txt', 
                                      delimiter=',', dtype=np.float32).T
        n = ground_points.shape[1]
        new_ground = np.ones((4, n), dtype=np.float32)
        new_ground[0:3, :] = ground_points
        G = new_ground
        return G 

    def get_sequence(self, seq_num:int, cam_no:int):
        sn = long_str(seq_num, 4)
        cam = f"cam{cam_no}"
        folder = self.root_path / 'scenarios' / sn / 'images' / cam
        return FolderSequence(folder)

    # Get all train/val/test image sequences 
    def get_sequences(self, which_set:str, cam_no:int=0):
        if which_set.lower() == "all":
            sets = list()
            for more_sets in self.sets.values():
                sets.extend(more_sets)
        else:
            sets = self.sets[which_set]

        seqs = [self.get_sequence(int(seq_num), cam_no) for seq_num in sets]
        return seqs 

    def get_gt(self, seq_num:int, frame_no:int) -> List[GTInstance]:
        sn = long_str(seq_num, 4)
        folder = self.root_path / 'scenarios' / sn / 'positions'
        file = folder / f"{long_str(frame_no)}.json"
        return gtis_from_json(file)

    def get_gts(self, which_set:str):
        sets = self.sets[which_set]
        gts = dict()
        for seq_name in sets:
            seq_num = int(seq_name)
            start, stop = self.get_frames(seq_num)
            for frame_no in range(start, stop):
                gts[(seq_num, frame_no)] = self.get_gt(seq_num, frame_no)
        
        return gts

    def get_impath(self, seq_num:int, frame_no:int, cam_no:int=0):
        im_path = self.root_path / 'scenarios' / long_str(seq_num, 4) / \
                  'images' / f"cam{cam_no}" / f"{long_str(frame_no,6)}.jpg"
        assert im_path.is_file()
        return im_path 

    def get_frames(self, seq_num:int):
        return 0, 2999

        """ More robust but slower implementation:
        sn = long_str(seq_num, 4)
        folder = self.root_path / 'scenarios' / sn / 'positions'
        files = list(folder.glob('*.txt'))
        files.sort(key=lambda f: int(f.stem))

        start = int(files[0].stem)
        stop = int(files[-1].stem)

        return start, stop 
        """
    
    def score(self, seq_num:int, tracks:List):
        start, stop = self.get_frames(seq_num=seq_num)
        gt = {fn: self.get_gt(seq_num, fn) for fn in range(start, stop+1)}
        cams, _ = self.get_cameras(seq_num)
        center_pos = pflat(null_space(cams[0]))
        return evaluate_tracks(tracks, gt, center_pos, self.options)


def gtis_from_json(json_file:Path) -> List[GTInstance]:
    objs = json.loads(json_file.read_text())
    instances = list()
    for obj in objs:
        x, y, z, l, w, h = [obj[key] for key in "xyzlwh"]
        X = np.array([x, y, z, 1], dtype=np.float32).reshape((4,1))
        shape = np.array([l, w, h], dtype=np.float32)
        ru_type = obj['type']
        track_id = obj['id']
        fx, fy = [obj[key] for key in ('forward_x', 'forward_y')]
        phi = np.arctan2(fy, fx)

        instance = GTInstance(X, ru_type, track_id, shape, phi)
        instances.append(instance)
    
    return instances

def build_camera_matrices(folder:Path, output_K=False):
    txt_path = folder / 'cameras.json'
    text = txt_path.read_text()
    cams_obj = json.loads(text)
    f = cams_obj['instrinsics']['f']
    Cx = cams_obj['instrinsics']['Cx']
    Cy = cams_obj['instrinsics']['Cy']

    cameras = dict()
    for cam in cams_obj['cams']:
        values = {'f':f, 'Cx':Cx, 'Cy':Cy}
        for key in ('x', 'y', 'z', 'pitch', 'roll', 'yaw'):
            values[key] = cam[key]
        
        cam_id = int(cam['id'])
        P, K = build_cam(values)
        cameras[cam_id] = P
    
    if output_K:
        return cameras, K

    return cameras

def build_cam(values):
    # Don't ask me why CARLA sets up the cameras this way...
    flip = np.array([[ 0,  1,  0 ], [ 0,  0, -1 ], [ 1,  0,  0 ]], 
                    dtype=np.float32)

    x = values['x']
    y = values['y']
    z = values['z']
    pitch = values['pitch']
    roll = values['roll']
    yaw = values['yaw']
    f = values['f']
    Cx = values['Cx']
    Cy = values['Cy']

    K = np.array([[f, 0, Cx], [0, f, Cy], [0, 0, 1]], dtype=np.float64)

    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))
    matrix = np.identity(4)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    matrix = np.linalg.inv(matrix)
    
    P = K @ flip @ matrix[:3, :]
    
    # Verify that camera's translation is correct 
    cen = np.array([x,y,z,1]).reshape((4,1))
    C = pflat(null_space(P))
    assert(np.allclose(C, cen))

    return P, K


def euler_angles(phi, theta, psi):
    sin = np.sin
    cos = np.cos
    R = [[cos(theta)*cos(psi),  -cos(phi)*sin(psi)+sin(phi)*sin(theta)*cos(psi),  sin(phi)*sin(psi)+cos(phi)*sin(theta)*cos(psi)],
         [cos(theta)*sin(psi), cos(phi)*cos(psi)+sin(phi)*sin(theta)*sin(psi), -sin(phi)*cos(psi)+cos(phi)*sin(theta)*sin(psi)],
         [-sin(theta),            sin(phi)*cos(theta),             cos(phi)*cos(theta)]]
    return np.array(R, dtype=np.float32)