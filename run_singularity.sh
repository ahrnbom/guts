#!/bin/bash

# Copyright (C) 2022 Martin Ahrnbom
# Released under MIT License. See the file LICENSE for details.
#
# This script builds the Singularity image and starts it. 
# You can run it with the 'build' parameter to build the image, and
# with the 'bash' parameter to start bash inside the built container

SING_FILE=sing.sif
PARAM=$1
SHOULD_BUILD=false
    
if [ "$1" == "build" ]; then
    SHOULD_BUILD=true
    PARAM=$2
else
    if [ -f "$SING_FILE" ]; then
        echo "Singularity image $SING_FILE already exist"
        SHOULD_BUILD=false
    else
        SHOULD_BUILD=true
    fi
fi

if [ "$SHOULD_BUILD" == true ]; then
    sudo singularity build "$SING_FILE" sing.def
fi

singularity run --nv -B /run,/media "$SING_FILE" $PARAM

echo "Done with Singularity!"
