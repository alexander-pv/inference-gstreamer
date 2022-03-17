#!/bin/bash

# Streaming videos in a sequential manner
# $1-$# - filepaths

FPS=30
VCODEC="libx264"                # libx264, libx265, mpeg4, libxvid
STREAM_ADDRESS="127.0.0.1:8554"

iteration=1
for filepath in "$@"; do
    echo -e "\n\nStreaming video [$iteration/$#]: $filepath with pid $$...\n"
    echo -e "Stream address rtsp://$STREAM_ADDRESS/stream\n"
    ffmpeg -re -i $filepath -filter:v fps=fps=$FPS -vcodec $VCODEC -f rtsp -rtsp_transport tcp "rtsp://$STREAM_ADDRESS/stream"
    ((iteration++))
done