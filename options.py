"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.


    Module for the Options class, describing the parameters that control how
    GUTS/UTS runs. Setting different parameters allows running GUTS, UTS and
    some things in-between.
"""

from dataclasses import dataclass

@dataclass
class Filter2DParams:
    Q_c: float
    Q_s: float
    Q_v: float 
    Q_ds: float
    Q_a: float
    Q_cov: float 
    Q_scov: float 
    R_c: float 
    R_s: float 
    P_factor: float 

def default_2D_params():
    return Filter2DParams(P_factor=49.52171924503543, Q_c=6.460793439870788, 
                          Q_s=4.9395752751071615, Q_v=2.4488558325967986, 
                          Q_ds=0.15796337004620242, Q_a=0.8568526854499889, 
                          Q_cov=0.02289871529864152, 
                          Q_scov=-1.27706334208637e-05,
                          R_c=2.820997964208509e-05, R_s=0.004983898051863616)

def default_3D_params_uts():
    return Filter3DParams(kappa=-9.669064547789426e-06,
                          P_factor=25.275582820768996,
                          Q_c=0.0010016784515698928, Q_s=0.07482293180724456, 
                          Q_phi=3.220211523871328e-05, Q_v=0.009977850753654992, 
                          Q_omega=0.3600739666009156, 
                          Q_cov=3.0633001821224415e-05,
                          R_c=3.220211523871329e-05, R_s=3.220211523871329e-05,
                          R_phi=1.3778588512103429e-05)

def default_3D_params_guts():
    return Filter3DParams(kappa=1.0014337906108299e-05, 
                          P_factor=25.22326672420832, 
                          Q_c=0.0004324632512897495, 
                          Q_s=0.04563285339851997, 
                          Q_phi=0.00024522766809503734, 
                          Q_v=0.008140234349826567,
                          Q_omega=0.7595342100371544, 
                          Q_cov=0.00025346794250902054, 
                          R_c=0.00010007178284241517, 
                          R_s=5.6529561945114e-07, 
                          R_phi=0.006584379489301031)

@dataclass
class Filter3DParams:
    kappa: float
    P_factor: float
    Q_c: float 
    Q_s: float
    Q_phi: float
    Q_v: float 
    Q_omega: float
    Q_cov: float
    R_c: float 
    R_s: float
    R_phi: float

@dataclass
class Options:
    detector:str
    detector_cache:bool
    detector_conf_thresh:float
    flat_ground:bool
    classes:list 
    system3D:str
    tracks2D:bool 
    hungarian2D:bool 
    dataset:str 
    z_dir:float
    ground_alpha:float 
    im_shape:tuple 
    frame_rate:float
    models3D:str 
    max_association_cost_factor:float # compared to the size of object
    time_without_detections:float # after this many seconds, track is discarded
    min_time:float # number of seconds of a track to be included in output
    activecorners_onecam:bool 
    activecorners_spw:float
    activecorners_minv:float
    params2D: Filter2DParams
    params3D: Filter3DParams
    max_dist_from_camera: float
    min_iou_for_match: float 
    saom_thresh: float 
    samhnet_opt_iters: int
    pixel_height_limit: int
    samhnet_input: str # number of characters must correspond to each channel
    samhnet_name: str # which samhnet training run to use by default 
    samhnet_output_phi: bool
    min_dist_for_phi: float # in metres, only used by GUTS
    utocs_root: str 
    phi_smoothing_factor: float # between 0 and 1, set negative to not use
    significant_motion_distance: float # in metres
    min_v_for_rotate: float # happens inside Filter3D
    pedestrian_suppression_range: float # in metres
    bicyclist_thresh: float # relative to AABB size

def default_options():
    o = Options(utocs_root='/media/disk/utocs',
                detector='detectron2',
                detector_cache=True,
                detector_conf_thresh=0.6670066157263224,
                flat_ground=False,
                classes=['car', 'truck', 'bus', 'pedestrian', 'bicyclist'],
                system3D='seg2pose',
                tracks2D=False,
                hungarian2D=False,
                dataset='carla',
                z_dir=1.0,
                ground_alpha=0.3,
                im_shape=(720,1280,3),
                frame_rate=25.0,
                models3D='custom',
                time_without_detections=0.5284208600081997,
                min_time=1.9102987280546078, 
                max_association_cost_factor=49.91834072329573,
                activecorners_onecam=True,
                activecorners_spw=502.7993398419448,
                activecorners_minv=3.0754123168893615,
                params2D=default_2D_params(),
                params3D=default_3D_params_uts(),
                max_dist_from_camera=50.0,
                min_iou_for_match=0.25,
                saom_thresh=0.3010190614277815,
                samhnet_opt_iters=4,
                pixel_height_limit=15,
                samhnet_input='rgbs',
                samhnet_name='new',
                samhnet_output_phi=False,
                phi_smoothing_factor=0.9,
                significant_motion_distance=2.0,
                min_v_for_rotate=0.1,
                min_dist_for_phi=0.02,
                pedestrian_suppression_range=0.0,
                bicyclist_thresh=0.2)
    
    return o  

def profile_options(profile:str):
    o = default_options()
    
    if profile == 'uts':
        o.detector = 'yolo'
        o.flat_ground = True 
        o.classes = ['car', 'truck', 'bus']
        o.system3D = 'activecorners'
        o.tracks2D = True 
        o.hungarian2D = True 
        o.models3D = 'uts'
        o.samhnet_opt_iters = 0
    elif profile == 'guts':
        o.detector = 'detectron2_and_samhnet'
        o.flat_ground = False 
        o.classes = ['car', 'truck', 'bus', 'pedestrian', 'bicyclist']
        o.system3D = 'samhnet'
        o.tracks2D = False 
        o.hungarian2D = False 
        o.models3D = 'basic'
        o.params2D = None 
        o = set_guts_options(o)
    else: 
        raise ValueError(f"Unknown options profile {profile}")

    return o 

def set_guts_options(o:Options):
    o.pedestrian_suppression_range = 1.000482230116181
    o.params3D = default_3D_params_guts()
    o.detector_conf_thresh=0.7710445087071358
    o.max_association_cost_factor=50.77336453804649
    o.time_without_detections=0.35750949419964617
    o.min_time=1.5416698285512676
    o.phi_smoothing_factor=0.8491843301457738
    o.min_dist_for_phi=0.008120603297540732
    return o 