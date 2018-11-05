#!/bin/bash
#
# Download the TF Lite schema and create the Python code to parse a FlatBuffer
#
wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema.fbs
flatc -p schema.fbs
