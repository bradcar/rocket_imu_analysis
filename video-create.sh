#!/bin/bash
# run in analysis.py directory

# Vpython puts scenes to browser and can only put frame images in download due to security
mkdir -p frames && mv ~/Downloads/frame_*.png video/

# Video name
VIDEO_NAME="rocket-trajectory-poc.mp4"
# ffmpeg can assemble these frames to video
ffmpeg -framerate 2 -i video/frame_%04d.png -c:v libx264 -pix_fmt yuv420p video/$VIDEO_NAME

# Mac open defaults to Quicktime  (current for Brad's Mac)
open video/$VIDEO_NAME

# if need to clean up frames in download
# rm frame_0*.png
