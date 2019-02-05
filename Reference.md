# Reference

Probably should use the information in *README.md*. But, here's some
not-fully-functional information:

## Camera Calibration (maybe?)
Print
[checkerboard](http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration?action=AttachFile&do=view&target=check-108.pdf).
Then on the Pi record images at 1 Hz holding the checkerboard (on some flat
object like a clipboard) at varying positions in front of the camera.

    ./camera_calibration_record.py

Copy back all the files in *calibration/* and delete the bad (e.g. blurry, out
of frame, etc.) ones. You need at least maybe 10 good ones. Then generate the
calibration file on any computer with Python's OpenCV (here using version
3.4.3):

    ./camera_calibration_compute.py

If you get an assertion fail, then take more pictures or less or near the edges
of your frame or ... (see
[this discussion](http://answers.opencv.org/question/28438/undistortion-at-far-edges-of-image/)).

## Running Object Detection on RPi
I will assume you've aliased your RPi Zero to be "rpiz" in your *.ssh/config* file.
First, copy your *detect_quantized.tflite* or whatever model you want to use to your
Raspberry Pi.

    sudo apt install python3-matplotlib python3-pil libxml2-dev libxslt-dev
    sudo pip3 install tensorflow pymavlink flatbuffers
    rsync -Pahuv ./ rpiz:vision-landing/

    ssh rpiz
    cd ./vision-landing
    ./object_detector.py --live

**Note:** at present this method gives incredibly low frame rates.
