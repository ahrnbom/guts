"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.

    Functionality for visualizing tracks
"""

from typing import List,Dict,Tuple
import cv2 
import imageio as iio
import numpy as np 
from scipy.io import loadmat
from scipy.linalg import null_space 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import argparse
import json 
from pathlib import Path
from math import ceil

from util import cv_point, intr, smallest_box, pflat, long_str, vector_normalize
from position import Position, Position3D
import panic
from utocs import UTOCS
from options import default_options

short_names = {'pedestrian': 'p', 'bicyclist': 'b', 'car': 'c', 'truck': 't',
               'bus': 'B', 'bicycle': 'b'}

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))

def category_colors(categories:List[str]) -> Dict[str, np.ndarray]:
    n = len(categories)
    colors = get_colors(n)
    out = dict()
    for i, cat in enumerate(categories):
        out[cat] = colors[i]
    
    return out

def get_colors(num_colors:int):
    # Generates num_colors many distinct bright colors 

    class_colors = []
    for i in range(0, num_colors):
        # This can probably be written in a more elegant manner
        hue = 255*i/(num_colors+2)
        col = np.empty((1,1,3)).astype("uint8")
        col[0][0][0] = hue
        col[0][0][1] = 128 + (i%9)*9 # Saturation
        col[0][0][2] = 200 + (i%2)*40 # Value (brightness)
        cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
        col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
        class_colors.append(col)
    return class_colors

def int_split2(x):
    if x%2 == 0:
        return x//2, x//2
    else:
        return x//2, (x//2)+1

def autocrop_horizontally(im):
    h, w, _ = im.shape
    white_r = im[:, :, 0]==255
    white_g = im[:, :, 1]==255
    white_b = im[:, :, 2]==255
    white = white_r & white_g & white_b
    all_white_columns = np.all(white, axis=0)
    
    # Find first and last False
    cols = np.where(~all_white_columns)[0]
    first = max(0, int(cols[0]) - 10)
    last = min(w, int(cols[-1]) + 10)

    # Crop
    im = im[:, first:last, :]
    return im 

def good_resize(im, new_h=None, new_w=None):
    orig_im = im

    h, w, _ = im.shape 

    if (new_h is None) and (new_w is None):
        raise ValueError("Needs at least one to scale image")

    if new_h is not None:
        h_scale = new_h/h 
    else:
        h_scale = float("inf")

    if new_w is not None:
        w_scale = new_w/w
    else:
        w_scale = float("inf")

    # Scale by whichever dimension needs the least change, and then pad 
    scale = min(h_scale, w_scale)
    # To ensure compatability with video formats
    w2, h2 = [16*intr(ceil(val*scale/16)) for val in (w, h)]
    im = cv2.resize(im, (w2, h2))
    
    if (new_h is not None) and (new_w is not None):
        dh = int_split2(new_h - h2)
        dw = int_split2(new_w - w2)
        if (dh[0]>=0) and (dh[1]>=0) and (dw[0]>=0) and (dw[1]>=0):
            im = np.pad(im, (dh, dw, (0,0)), constant_values=255)
        else:
            # Fallback: Just resize the original image
            im = cv2.resize(orig_im, (new_w, new_h))
    return im

def draw_position(im, pos:Position, color=(255,255,255), thick=2, 
                  extra_text=""):

    if pos.mask is None:
        im = draw_box(im, pos.aabb, color, thick=thick)
        im = draw_text(im, short_names[pos.class_name] + extra_text, 
                       pos.get_center_point())
    else:
        im = draw_mask(im, pos.mask, color, 
                       short_names[pos.class_name] + extra_text)
    return im 

def draw_position3D(im, pos:Position3D, cam:np.ndarray, color=(255,255,255),
                    extra_text="", z_dir=1.0):
    imh, imw, _ = im.shape 
    x, y, z = [float(v) for v in pos.pos3D][0:3]
    l, w, h = pos.shape 

    text = short_names[pos.class_name] + extra_text

    if (pos.phi is None) or np.isnan(pos.phi):
        # without the angle, 3D boxes don't look very good 
        X = np.array([x,y,z,1.0], dtype=np.float32).reshape((4,1))
        pos2D = pflat(cam @ X)
        point = cv_point(pos2D[0,0], pos2D[1,0])
        cv2.drawMarker(im, point, color, cv2.MARKER_CROSS, 8, 2, cv2.LINE_AA)
        draw_text(im, text, point, color)
    else:
        im = draw_3D_box_phi(im, cam, x, y, z, l, w, h, pos.phi, color, 
                             text=text, z_dir=z_dir)
    return im 
    
def draw_box(im, box, color, thick=2):
    x1, y1, x2, y2 = box
    start = cv_point(x1, y1)
    end = cv_point(x2, y2)
    im = cv2.rectangle(im, start, end, color, thick)
    return im

def draw_text(im, text, point, color=(255,255,255), scale=1.0):
    x, y = point
    point = cv_point(x, y)
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    cv2.putText(im, text, point, font, scale, (0,0,0), 3, line)
    cv2.putText(im, text, point, font, scale, color,   1, line)
    return im 

""" Draw a mask on an image with 50% transparency and a border and centered text """
def draw_mask(im, mask, color, text_all, extra_text_scale=1.0):
    np_color = np.array(color, dtype=np.uint8)
    im = apply_mask_50(im, mask, np_color)
    k = kernel
    mask_border = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, k).astype(bool)
    im = apply_mask_opaque(im, mask_border, np_color)
    
    if text_all:
        box = smallest_box(mask)
        if box is None:
            return im
        
        x = intr((box[0] + box[2])/2)
        y = intr((box[1] + box[3])/2)
        
        font = cv2.FONT_HERSHEY_PLAIN
        text_scale = 0.75*im.shape[0]/480.0
        if text_scale < 1.0:
            text_scale = 1.0
        text_scale *= extra_text_scale
        
        for i_text, text in enumerate(text_all.split("@")):
            yy = intr(y + (i_text*(12*text_scale + 5))) 
            text_w = cv2.getTextSize(text, font, text_scale, 1)[0][0]
            line = cv2.LINE_AA
            cv2.putText(im, text, (x - text_w//2,yy), font, text_scale, (0,0,0), 3, line)
            cv2.putText(im, text, (x - text_w//2,yy), font, text_scale, color,   1, line)
            
    return im

def brighter(color):
    new_color = (intr((color[0]+255)/2),
                 intr((color[1]+255)/2),
                 intr((color[2]+255)/2))
    return new_color

def matplotlib_color(color, alpha=1.0):
    new_color = [c/256.0 for c in color]
    new_color.append(alpha)
    return new_color

def draw_3D_box_phi(im, P, x, y, z, l, w, h, phi, color, text="", z_dir=1.0):
    forward_x = np.cos(phi)
    forward_y = np.sin(phi)
    forward = np.array([forward_x, forward_y, 0.0, 0.0], dtype=np.float32)
    
    up = np.array([0.0, 0.0, z_dir, 0.0], dtype=np.float32)
    right = np.cross(forward[0:3], up[0:3])
    right = np.array([*right, 0.0], dtype=np.float32)

    X = np.array([x, y, z, 1.0], dtype=np.float32)
    im = _draw3Dbox(im, P, X, l, w, h, forward, right, up, color, text=text)
    return im 

def _draw3Dbox(im, P, X, l, w, h, forward, right, up, color, text="",
               line_width=1):
    imh, imw, _ = im.shape

    points3D = build_3D_box(X, l, w, h, forward, right, up) 
    points3D = np.stack(points3D).T
    points2D = pflat(P @ points3D)

    outside_x = (points2D[0,:] < -200) | (points2D[0,:] > imw + 200)
    outside_y = (points2D[1,:] < -200) | (points2D[1,:] > imh + 200)
    if np.any( outside_x | outside_y) or np.any(np.isnan(points2D)):
        # Object outside of image bounds or NaN
        return im 

    for indices in [(0,1,3,2), (4,5,7,6), (0,4,5,1), (2,6,7,3), 
                    (0,4,6,2), (1,5,7,3)]:
        p1 = points2D[:, indices[0]].flatten()
        p2 = points2D[:, indices[1]].flatten()
        p3 = points2D[:, indices[2]].flatten()
        p4 = points2D[:, indices[3]].flatten()

        for pair in [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]:
            a, b = pair 
            
            ax = intr(a[0])
            ay = intr(a[1])

            bx = intr(b[0])
            by = intr(b[1])
            cv2.line(im, (ax, ay), (bx, by), color, line_width, cv2.LINE_AA)

    # Show forward direction
    Xf = X + l/2.0 * forward 
    Xf2D = pflat(P @ Xf)
    xf, yf, _ = Xf2D
    xf = intr(xf)
    yf = intr(yf)
    cv2.drawMarker(im, (xf, yf), brighter(color), cv2.MARKER_TRIANGLE_UP, 8, 2, 
                   cv2.LINE_AA)

    if text:
        text_scale = 1.25*imh/720.0
        draw_text(im, text, (xf, yf), color=color, scale=text_scale)

    return im 

def apply_mask_opaque(im, mask, color):
    for i in range(3):        
        np.copyto(im[:,:,i], color[i] * np.ones_like(im[:,:,i]), where=mask)
    return im
       
def apply_mask_50(im, mask, color):
    # Slower but equivalent 
    #ys, xs = np.where(mask)
    #for y,x in zip(ys, xs):
    #    im[y,x,:] = im[y,x,:]//2 + color//2
    
    h, w, _ = [np.int32(x) for x in im.shape]
    color = np.array(color, dtype=np.uint8)
    im = panic.run('manual_c/apply_mask_50.c', 'main', [im, h, w, mask, color])
    im = im[0]

    return im 

def build_3D_box(X, l, w, h, forward, right, up):
    points3D = list()
    for il in (-0.5, 0.5):
        dl = il * l * forward 
        for iw in (-0.5, 0.5):
            dw = iw * w * right 
            for ih in (0.0, 1.0):
                dh = ih * h * up 
                point3D = X + dl + dw + dh 
                points3D.append(point3D)
    return points3D

def read_positions(pos_path:Path):
    if not pos_path.is_file():
        return None

    text = pos_path.read_text()
    try:
        instances = json.loads(text)
    except:
        if not text:
            return list()
        
        print(f"ERROR: Broken JSON file: {pos_path}")
        print(text)
        raise ValueError("Failed to load JSON!")

    for instance in instances:
        x = instance['x']
        y = instance['y']
        z = instance['z']
        X = np.array([x, y, z, 1], dtype=np.float32)
        instance['X'] = X
        # X is used to make it easy to project into cameras
    return instances

def get_detections(dataset, seq_num, frame_num, cam_num, detector='detectron2',
                   conf_thresh=0.72):

    folder = Path('detections_cache') / detector / f"{dataset}{seq_num}"
    file_path = folder / f"{long_str(frame_num, 6)}_{cam_num}.txt"
    lines = [l for l in file_path.read_text().split('\n') if l]
    detections = [Position.from_text(l) for l in lines]
    detections = [d for d in detections if d.confidence > conf_thresh]
    return detections 

def visualize(folder:Path, gt_folder:Path, dataset:str, classes:List[str], 
              seq_num:int, out_path:Path, include_detections:bool, 
              max_frames=None, hide_gt=False, cam_num=0, start_frame=None,
              scale_factor=None, custom_text=None, video_quality=6):

    if start_frame is None:
        start_frame = 0

    colors = category_colors(classes)
    colors['bicycle'] = colors['bicyclist']

    if dataset == 'utocs':
        im_folder = gt_folder / '..' / 'images' / f"cam{cam_num}"
        images = list(im_folder.glob('*.jpg'))
        images.sort()

        options = default_options()
        utocs = UTOCS(options=options)
        PP, K = utocs.get_cameras(seq_num)
        cam = PP[cam_num]

        ground = np.genfromtxt(gt_folder / '..' / 'ground_points.txt', 
                            delimiter=',', dtype=np.float32).T

        up_dir = 1.0
    else:
        raise ValueError("Unknown dataset")
        # Add your own dataset here?
        
    n_ims = len(images)
    if max_frames is not None:
        n_ims = min(n_ims, max_frames)

    is_first = True
    topdown_dims = None
    frame_index = 0

    with iio.get_writer(out_path, fps=20, quality=video_quality) as vid:
        for im_path in images:
            frame_no = int(im_path.stem.split('_')[0])

            if frame_no < start_frame:
                continue 

            relative_frame_no = frame_no-start_frame
            
            if (max_frames is not None) and (relative_frame_no > max_frames):
                break

            if folder is not None:
                attempt = read_positions(folder /   
                                         f"{long_str(frame_no, 6)}.json")
                gt_mode = False # avoids clutter
            else:
                attempt = None
                gt_mode = True
            gt = read_positions(gt_folder / f"{long_str(frame_no, 6)}.json")

            if gt is None:
                continue

            image = iio.imread(im_path)

            if include_detections:
                detections = get_detections(dataset, seq_num, frame_no, cam_num)
                for detection in detections:
                    color = colors[detection.class_name]
                    image = draw_position(image, detection, color)

            if hide_gt:
                gt = list()
                # We still want to load GT previously, to skip any frames 
                # not covered by ground truth (not relevant for UTOCS)

            frame1 = render_pixel_frame(image, classes, frame_no, attempt, gt, 
                                        cam, colors, ground, gt_mode=gt_mode,
                                        up_dir=up_dir)
            if is_first:
                topdown_dims = frame1.shape
            frame2 = render_topdown_frame(topdown_dims, classes, attempt, gt, 
                                          cam, colors, ground, gt_mode=gt_mode,
                                          is_first=is_first)
            
            frame = np.hstack([frame1, frame2])

            if scale_factor is not None:
                # Just using fx,fy here risks resolution not being divisible 
                # by 16, which FFMPEG doesn't like 
                oldh, oldw, _ = frame.shape 
                newh, neww = [16*(int(v*scale_factor)//16) for v in (oldh, oldw)]
                assert newh%16 == 0
                assert neww%16 == 0
                frame = cv2.resize(frame, (neww, newh), 
                                   interpolation=cv2.INTER_CUBIC)

            # Thicker black first, then thin white, very readable
            text = custom_text or f"Frame {frame_no}"
            h = frame.shape[0]
            lines = text.split('\n')
            for i_line, line in enumerate(lines):
                text_y = h - 60 + i_line*25
                cv2.putText(frame, line, (10,text_y), cv2.FONT_HERSHEY_PLAIN, 
                            1.5, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(frame, line, (10,text_y), cv2.FONT_HERSHEY_PLAIN, 
                            1.5, (255,255,255), 1, cv2.LINE_AA)
                
            vid.append_data(frame)

            is_first = False 
            topdown_dims = frame2.shape

            if frame_index%100 == 0:
                print(f"Frame {frame_no+1}, {100.0*frame_index/n_ims:.2f}%, " \
                      f"Sequence: {seq_num}")

            frame_index += 1 

def rotated_rectangle(x, y, l, w, phi, edge_color, face_color):
    positions = list()
    base_pos = np.array([x, y], dtype=np.float32)
    forward = np.array([np.cos(phi), np.sin(phi)], dtype=np.float32)
    right = np.array([[0, -1], [1, 0]], dtype=np.float32) @ forward 
    for il, ii in zip((-0.5, 0.5), (1.0, -1.0)):
        ll = il * l * forward 
        for iw in (-0.5, 0.5):
            ww = ii*iw * w * right 
            new_pos = base_pos + ll + ww 
            positions.append((new_pos[0], new_pos[1]))
    xy = np.array(positions, dtype=np.float32)
    rect = patches.Polygon(xy, closed=True, ec=edge_color, fc=face_color)
    return rect 

def render_topdown_frame(dims:Tuple, classes:List[str], attempt:List[Dict],
                         ground_truth:List[Dict], cam:np.ndarray, colors:Dict, 
                         ground:np.ndarray, gt_mode=False, is_first=False):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(True)
    ax.set_aspect('equal')

    plt.plot(ground[0,:], ground[1,:], '.', ms=1.0)
    minx, maxx = np.min(ground[0,:]), np.max(ground[0,:])
    miny, maxy = np.min(ground[1,:]), np.max(ground[1,:])

    for gt in ground_truth:
        class_name = gt['type']
        if not class_name in classes:
            continue 

        X, l, w = [gt[key] for key in "Xlw"]
        phi = np.arctan2(gt['forward_y'], gt['forward_x'])
        x, y = X.flatten()[0:2]

        color = colors[class_name]
        alpha = 0.35
        if not gt_mode:
            color = brighter(color)
            alpha = 0.5
        edge_color = matplotlib_color(color)
        face_color = matplotlib_color(color, alpha)
        rect = rotated_rectangle(x, y, l, w, phi, edge_color, face_color)
        ax.add_patch(rect)

        if gt_mode:
            ll = max(1.5, l)
            plt.arrow(x, y, ll*np.cos(phi), ll*np.sin(phi), width=0.005,
                      head_width=1.0, color=edge_color)
            plt.text(x, y, f"{short_names[gt['type']]}{gt['id']}")

        minx = min(minx, x)
        miny = min(miny, y)
        maxx = max(maxx, x)
        maxy = max(maxy, y)

    if attempt is not None:
        for at in attempt:
            class_name = at['type']
            if not class_name in classes:
                continue 

            X, l, w = [at[key] for key in "Xlw"]
            phi = np.arctan2(at['forward_y'], at['forward_x'])
            x, y = X.flatten()[0:2]

            color = colors[class_name]
            edge_color = matplotlib_color(color)
            face_color = matplotlib_color(color, 0.75)
            rect = rotated_rectangle(x, y, l, w, phi, edge_color, face_color)
            ax.add_patch(rect)

            ll = max(1.5, l)
            plt.arrow(x, y, ll*np.cos(phi), ll*np.sin(phi), width=0.005,
                      head_width=1.0, color=edge_color)

            plt.text(x, y, f"{short_names[at['type']]}{at['id']}")

            minx = min(minx, x)
            miny = min(miny, y)
            maxx = max(maxx, x)
            maxy = max(maxy, y)

    # Draw camera 
    cam_cen = pflat(null_space(cam)).flatten()
    cam_dir = vector_normalize(cam[2, 0:3]) * 7.5 
    plt.arrow(cam_cen[0], cam_cen[1], cam_dir[0], cam_dir[1], width=0.01,
              head_width=1.0)
    
    minx = min(minx, cam_cen[0])
    maxx = max(maxx, cam_cen[0])
    miny = min(miny, cam_cen[1])
    maxy = max(maxy, cam_cen[1])

    # Orient plot based on camera angle    
    if cam_dir[1] > 0:
        plt.ylim(miny - 2, maxy + 2)
        plt.xlim(maxx + 2, minx - 2)
    else:
        plt.ylim(maxy + 2, miny - 2)
        plt.xlim(minx - 2, maxx + 2)
        
    # Convert to image as a numpy array
    plt.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, 0:3] # we only need RGB
    buf = autocrop_horizontally(buf)

    # To avoid having different image sizes every frame (not supported by video)
    # we store the width of the first frame and force that width 
    # The height will always be the same, as enforced by the dataset images
    if is_first:
        buf = good_resize(buf, new_h=dims[0])
    else:
        buf = good_resize(buf, new_h=dims[0], new_w=dims[1])

    plt.close()

    return buf 

def render_pixel_frame(image:np.ndarray, classes:List[str], frame_no:int, 
                       attempt:List[Dict], ground_truth:List[Dict], 
                       cam:np.ndarray, colors:Dict, ground:np.ndarray, 
                       gt_mode=False, up_dir=1.0):

    imh, imw, _ = image.shape
    line_width = 2
    if imh > 720:
        line_width = 4

    # Draw ground points 
    n = ground.shape[1]
    new_ground = np.ones((4, n), dtype=np.float32)
    new_ground[0:3, :] = ground
    ground2D = pflat(cam @ new_ground)
    for i in range(n):
        gnd = ground2D[:, i]
        pnt = (intr(gnd[0]), intr(gnd[1]))
        cv2.drawMarker(image, pnt, (255,255,255), cv2.MARKER_CROSS, 2)

    up = np.array([0,0,up_dir,0], dtype=np.float32)

    # Draw ground truth 
    for gt in ground_truth:
        class_name = gt['type']
        if not class_name in classes:
            continue 

        X, l, w, h = [gt[key] for key in "Xlwh"]
        forward = np.array([gt['forward_x'], gt['forward_y'], 
                           gt['forward_z'], 0],
                           dtype=np.float32)
        right = np.array([*np.cross(forward[0:3], up[0:3]), 0.0], 
                         dtype=np.float32)

        color = colors[class_name]
        if not gt_mode:
            color = brighter(color)
        
        text = ""
        if gt_mode:
            text=f"{short_names[gt['type']]}{gt['id']}"
        
        _draw3Dbox(image, cam, X, l, w, h, forward, right, up, color, text=text)
                   
    # Draw the attempted tracks 
    if attempt is not None:
        for at in attempt:
            class_name = at['type']
            if not class_name in classes:
                continue 

            X, l, w, h = [at[key] for key in "Xlwh"]
            forward = np.array([at['forward_x'], at['forward_y'], 
                            at['forward_z'], 0],
                            dtype=np.float32)
            right = np.array([*np.cross(forward[0:3], up[0:3]), 0.0], 
                            dtype=np.float32)

            color = colors[class_name]
            _draw3Dbox(image, cam, X, l, w, h, forward, right, up, color, 
                       text=f"{short_names[at['type']]}{at['id']}",
                       line_width=line_width)

    return image 

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--name", type=str,
                      help="Which run to visualize (name in output/tracks/), " \
                           "can also be left empty to only visualize GT",
                           default="")
    args.add_argument("--gt_folder",  default="./output", type=str,
                      help="Path of ground truth",)
    args.add_argument("--set", type=str, default='test',
                      help="'training', 'validation', 'test' or 'all'")
    args.add_argument("--seqs", type=str, default="", help="Set to only run "\
                      "these sequences. Should be comma-separated, like "\
                      "'0003,0027'.")
    args.add_argument("--classes", type=str, default="",
                      help="Set to a comma-separated list of classes to only " \
                           "visualize those")
    args.add_argument("--processes", help="Number of processes to spawn, " \
                      "0 means single-threaded.", default=0, type=int)
    args.add_argument("--dataset", help="Must be 'utocs'", 
                      type=str, default="utocs")
    args.add_argument("--detections", action="store_true",
                      help="Include to also draw detections")
    args.add_argument("--frames", type=int, help="Limits the number of frames")
    args.add_argument("--hidegt", action="store_true", help="Hide ground truth")
    args = args.parse_args()

    if not args.name:
        folder = None
    else:
        folder = Path('output') / 'tracks' / args.name

    gt_folder = Path(args.gt_folder)
    which_set = args.set

    dataset = args.dataset

    if args.classes:
        classes = args.classes.split(',')
    else:
        classes = ['truck', 'car', 'bus', 'bicyclist', 'pedestrian']

    assert which_set in ['training', 'validation', 'test', 'all']
    if dataset == 'utocs':
        options = default_options()
        utocs = UTOCS(options=options)
        seq_nums = utocs.get_seqnums_sets()
        seq_nums['all'] = seq_nums['training'] + seq_nums['validation'] + \
                          seq_nums['test']
    else:
        raise ValueError("Unknown dataset")
        # Add your own dataset here?

    to_visualize = list()

    seqs_to_run = seq_nums[which_set]
    if args.seqs:
        seqs_to_run = [int(s) for s in args.seqs.split(',')]

    for seq_num in seqs_to_run:
        if dataset == 'utocs':
            seq = gt_folder / 'scenarios' / long_str(seq_num, 4)
            if folder is not None:
                _folder = folder / seq.name
            else:
                _folder = None 
            _gtfolder = seq / 'positions'

        else:
            raise ValueError("Unknown dataset")

        bonus = ''
        if args.detections:
            bonus += '_dets'
        if args.hidegt:
            bonus += '_nogt'
            
        if folder is None:
            out_folder = Path('output') / f"visualized_gt_{dataset}"
            out_folder.mkdir(exist_ok=True, parents=True)
            out_path = out_folder / f"{dataset}_{seq_num}{bonus}.mp4"
        else:
            out_path = folder / f"{dataset}_visualization_{seq_num}{bonus}.mp4"
                
        data = (_folder, _gtfolder, dataset, classes, seq_num, out_path, 
                args.detections, args.frames, args.hidegt)
        to_visualize.append(data)
    
    if args.processes == 0:
        for vis in to_visualize:
            visualize(*vis)
    else:
        from multiprocessing import Pool 
        import os 
        os.nice(10)
        with Pool(args.processes) as pool:
            pool.starmap(visualize, to_visualize)
