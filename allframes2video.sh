#!/bin/bash
dir="record"
for i in "$dir"/*/; do
    out="${i:0:(-1)}.mp4"
    if [[ ! -e "$out" ]]; then
        ./frames2video.sh "$i" "$out"
    fi
done
