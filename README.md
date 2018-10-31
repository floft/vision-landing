Vision Landing
==============
Use the model learned from the
[Detect Frying Pan](https://github.com/floft/detect-frying-pan)
code and now run that on live RPi Zero camera input. We want to get the
drone to land on the frying pan.

# Running Object Detection on RPi
I will assume you've aliased your RPi Zero to be "rpiz" in your *.ssh/config* file:

    sudo apt install python3-matplotlib python3-pil
    sudo pip3 install tensorflow
    rsync -Pahuv ./ rpiz:vision-landing/

    ssh rpiz
    ./vision-landing/object_detection.py

# Running Object Detection on another computer
Since the Zero is really slow, I'll stream to another computer to do processing
for now. Though, at least one person has used the GPU on the RPi Zero to get
[~8 fps on face detection](https://www.youtube.com/watch?v=A3BDg13DX3M). So, it
is possible, though he hasn't shared his code. I will return to this problem
later.

    sudo apt install python3-zmq

On Pi:

    ./vision-landing/stream.py


On laptop:

    ./vision-landing/object_detection.py
