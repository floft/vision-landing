#!/bin/bash
#
# Sync files to RPi Zero
#
rsync -Pahuv --exclude="record" ./ rpiz:vision-landing/
