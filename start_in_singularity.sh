# Copyright (C) 2022 Martin Ahrnbom
# Released under MIT License. See the file LICENSE for details.
#
# This script is the entry-point for Singularity, responsible for launching bash
# Don't ask me why the if statement is necessary

if [ "a" = "a" ]; then
    echo "> Booting into bash inside Singularity container..."
    export PROMPT_COMMAND="echo -n \[\ Singularity \]\ "
    exec bash
fi