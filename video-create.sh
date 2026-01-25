#!/bin/bash
# run in analysis.py directory

# Input Paths
VIDEO_DIR="video"
# Debug for 9 DOF
#VIDEO_DIR="video-9-dof"
FRAME_SOURCE_DIR="$HOME/Downloads"
FRAME_PATTERN="frame_%04d.png"

# Output path
VIDEO_NAME="rocket-trajectory-poc.mp4"

# Ensure video directory exists
mkdir -p "$VIDEO_DIR"

# Move frames from source to video directory
mv "$FRAME_SOURCE_DIR"/frame_*.png "$VIDEO_DIR"/

# ffmpeg can assemble these frames to video
ffmpeg -framerate 2 \
       -i "$VIDEO_DIR/$FRAME_PATTERN" \
       -c:v libx264 \
       -pix_fmt yuv420p \
       "$VIDEO_DIR/$VIDEO_NAME"

# Open video (QuickTime on macOS)
open "$VIDEO_DIR/$VIDEO_NAME"

# Optional cleanup
# rm "$FRAME_SOURCE_DIR"/frame_0*.png
# rm ~/Downloads/frame_0*.png
