"""
Commonly used functions for images
"""
import os
import numpy as np

def load_image_into_numpy_array(image):
    """
    Helper function from: models/research/object_detection/
    """
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def find_files(folder, prefix="", extension=".jpg"):
    """
    Find all files recursively in specified folder with a particular
    prefix and extension
    """
    files = []

    for dirname, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if filename.startswith(prefix) and filename.endswith(extension):
                files += [(dirname, filename)]

    return files
