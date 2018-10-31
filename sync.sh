#!/bin/bash
#
# Sync files to RPi Zero
#
rsync -Pahuv ./ rpiz:vision-landing/
