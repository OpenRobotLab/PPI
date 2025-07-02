#!/bin/bash
source ~/.bashrc
conda activate ppi
# Xvfb :99 -screen 0 1024x768x16 &
export DISPLAY=:99
export COPPELIASIM_ROOT=YOUR_COPPELIASIM_ROOT
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

cd /your/path/to/RLBench/tools

python dataset_generator_bimanual.py \
    --tasks=bimanual_pick_laptop \
    --save_path=/your/path/to/data/dir \
    --image-size=256x256 \
    --episodes_per_task=1 \
    --headless \