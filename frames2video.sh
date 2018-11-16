#!/bin/bash
dir="$1"
out="$2"
if [[ -z $dir || -z $out || ! -e $dir ]]; then
    echo "Usage: frames2video.sh dir_of_jpgs/ output.mp4"
    exit 1
fi
ffmpeg -pattern_type glob -r 30 -i "$dir/*.jpg" -vf scale=640:480,setsar=1:1 "$out"
