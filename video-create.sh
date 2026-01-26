#!/bin/bash
# run in analysis.py directory

# Input Paths
VIDEO_DIR="video"
# Debug for 9 DOF
#VIDEO_DIR="video-9-dof"
FRAME_SOURCE_DIR="$HOME/Downloads"
FRAME_PREFIX="frame"
FRAME_PATTERN="frame_%04d.png"

# Output path
VIDEO_NAME="rocket-trajectory-poc.mp4"

# Ensure video directory exists
mkdir -p "$VIDEO_DIR"

# Remove old frames , then Move frames from source to video directory
if ls "$FRAME_SOURCE_DIR"/${FRAME_PREFIX}_*.png 1> /dev/null 2>&1; then
    echo "Moving frames from ~/Downloads to ./$VIDEO_DIR"
    # Clear old frames in target first
    rm -f "$VIDEO_DIR"/${FRAME_PREFIX}_*.png
    # Move new frames
    mv "$FRAME_SOURCE_DIR"/${FRAME_PREFIX}_*.png "$VIDEO_DIR"/
else
    echo "Error: No frames found in ./$FRAME_SOURCE_DIR matching ${FRAME_PREFIX}_*.png"
    exit 1
fi

# ffmpeg can assemble these frames to video
ffmpeg -framerate 2 \
       -i "$VIDEO_DIR/$FRAME_PATTERN" \
       -c:v libx264 \
       -pix_fmt yuv420p \
       "$VIDEO_DIR/$VIDEO_NAME"

# Open video (QuickTime on macOS)
open "$VIDEO_DIR/$VIDEO_NAME"

# Optional remove frames after video creation
if [ -f "$VIDEO_DIR/$VIDEO_NAME" ]; then
    echo "Success: Opening video..."
    open "$VIDEO_DIR/$VIDEO_NAME"

    # Optional: only remove if you are sure you don't need to re-run ffmpeg
        echo "ffmpeg failed to create video."
    rm "$VIDEO_DIR"/${FRAME_PREFIX}_*.png
    echo "removing frames from ./$VIDEO_DIR that were used to create video"
    # Clear old frames in target first
    rm -f "$VIDEO_DIR"/${FRAME_PREFIX}_*.png
else
    echo "ffmpeg failed to create video."
fi

# by hand
# rm ~/Documents/GitHub/rocket_analysis/video/frame_0*.png
# rm ~/Downloads/frame_0*.png
