# GUTS: Generalized Urban Traffic Surveillance

This repository contains the source code to GUTS, as well as our reimplementation of [UTS](https://ieeexplore.ieee.org/document/9575140). Because these methods share so many components, they share one codebase. Instances of the `Options` class specify which method is used, including possible superpositions between the two methods, such as running UTS with Mask R-CNN as its detector.

If you find this project useful, please cite our paper: 
```
bibtex coming soon, paper not yet published
```

This method can be used alongside the [UTOCS dataset](https://github.com/ahrnbom/utocs), in addition to custom datasets. To run on your own data, you can either create a dataset object similar to the `UTOCS` class in `utocs.py`, or look at `guts.py` and apply the method to images you load manually. 

Both GUTS and our implementation of UTS makes the following assumptions on your data:
1. The camera is fully calibrated with a 4x3 projection matrix, and radial distortion is removed from the images.
2. Some sampled ground points in world coordinates exists that are visible in the scene. For UTS, these are only used to define a plane perpendicular to the z-axis, at the z-level that is the average of the points.
3. GUTS supports cars, trucks, buses, pedestrians and bicyclists. UTS supports cars, trucks and buses.

## Installation
You will need the following dependencies:
1. A powerful, modern Linux computer with an Nvidia GPU
2. `git`
3. `singularity` (we use version 3.8.1)

### Instructions
```
git clone https://github.com/ahrnbom/guts
cd guts
./run_singularity.sh bash

# Inside the Singularity image now

python -m pytest .
```

After following these steps, visit [this link in a web browser](https://lunduniversityo365-my.sharepoint.com/:u:/g/personal/ma7467ah_lu_se/EdD5AExNahhChwPhPqrBRQgBj-X4FwopdwEwMTOLg1yEpA?e=ajPSYh) to download the file and place it under `samhnet_training/new`, in order to use pre-trained SAMHNet (trained on the UTOCS dataset). If you don't do this, you will have to train SAMHNet yourself in order to use it.

When inside the Singularity container, press `Ctrl`+`D` to exit to your own shell.
