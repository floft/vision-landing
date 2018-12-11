#!/bin/bash
# Run on laptop
echo "--"
echo "Don't forget to enable the hotspot!"
echo "--"
./object_detector.py --remote --debug --record=record --gst
#./object_detector.py --remote --debug --record=record --gst --host=10.42.0.183
