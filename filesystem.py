"""
Figure out what folders, filenames, etc. to use
"""
import os
import re
import pathlib

def latest_index(dir_name, glob="*", folders=False):
    """
    Looks in dir_name at all files matching glob and returns highest number.

    For folders (folders=True): you can leave glob="*"
    For files (folders=False): probably should specify something like glob="*.jpg"
    """
    # Get list of files/folders
    files = pathlib.Path(dir_name).glob(glob)

    # Get number from filenames or folders
    regex = re.compile(r'\d+')
    numbers = []

    for f in files:
        if not folders or os.path.isdir(str(f)):
            f_numbers = [int(x) for x in regex.findall(str(f.name))]
            # look at last number
            assert len(f_numbers) > 0, "Should be at least 1 number in file/folder"
            numbers.append(f_numbers[-1])

    # If there is a highest number, get it. Otherwise, start at zero.
    if len(numbers) > 0:
        return sorted(numbers)[-1] # Highest number
    else:
        return 0

def get_record_dir(path, ext="jpg", str_format="%05d"):
    """
    Directory for recording, e.g. if "record" and there are directories
    record/00000 and record/00001, then we'll start next at record record/00002
    """
    # Find previous index
    record_index = latest_index(path, folders=True)
    # If no images in that directory, use it. If images, increment
    record_dir_prev = os.path.join(path, str_format%record_index)
    img_index = latest_index(record_dir_prev, "*."+ext)

    if img_index == 0:
        record_dir = record_dir_prev
    else:
        record_index += 1
        record_dir = os.path.join(path, str_format%record_index)

    return record_dir
