Vision Landing
==============
Use the model learned from the
[Detect Frying Pan](https://github.com/floft/detect-frying-pan)
code and now run that on live RPi Zero camera input. We want to get the
drone to land on the frying pan.

**Warning:** this code is still in development and not fully functional.

# Camera Calibration (maybe?)
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

# Running Object Detection on RPi
I will assume you've aliased your RPi Zero to be "rpiz" in your *.ssh/config* file.
First, copy your *detect_quantized.tflite* or whatever model you want to use to your
Raspberry Pi.

    sudo apt install python3-matplotlib python3-pil libxml2-dev libxslt-dev
    sudo pip3 install tensorflow pymavlink
    rsync -Pahuv ./ rpiz:vision-landing/

    ssh rpiz
    cd ./vision-landing
    ./object_detection.py --live

# Running Object Detection on another computer
Since the Zero is really slow, I'll stream to another computer to do processing
for now. Though, at least one person has used the GPU on the RPi Zero to get
[~8 fps on face detection](https://www.youtube.com/watch?v=A3BDg13DX3M). So, it
is possible, though he hasn't shared his code. I will return to this problem
later.

    sudo apt install python3-zmq

On Pi:

    cd ./vision-landing
    ./stream.py

On laptop (and outputting debug info, saving images to record/, displaying with
GStreamer):

    sudo pacman -S gst-python
    cd ./vision-landing
    ./object_detection.py --remote --host rpi-zero --debug --record=record --gst

Then map a switch on your R/C controller to channel 6. For low PPM value it'll
do nothing, for higher it'll stream, and for even higher it'll shut down the
Raspberry Pi (for exact values, see script).

Or, if you wish to always run on boot (running */home/pi/vision-landing/stream.py*
as user *pi* and group *dialout* for access to */dev/ttyAMA0*):

    sudo cp stream.service /etc/systemd/system/
    sudo systemctl enable stream
    sudo systemctl start stream
