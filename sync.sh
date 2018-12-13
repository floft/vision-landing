#!/bin/bash
#
# Sync files to RPi Zero
#
rsync -Pahuv \
    --exclude="record" \
    --exclude="calibration*" \
    --exclude="exported_models*" \
    --exclude="__pycache__" \
    --exclude="*.npy*" \
    --exclude="capture" \
    ./ rpiz:vision-landing/
