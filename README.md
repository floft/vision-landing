Vision Landing
==============
Use the model learned from the
[Detect Frying Pan](https://github.com/floft/detect-frying-pan)
code and now run that on live RPi Zero camera input. We want to get the
drone to land on the frying pan.

# Raspberry Pi Zero Setup
I will assume you've aliased your RPi Zero to be "rpiz" in your *.ssh/config* file:

    sudo apt install python3-matplotlib python3-pil
    sudo pip3 install tensorflow
    rsync -Pahuv ./ rpiz:vision-landing/

    ssh rpiz
    python3 vision-landing/object_detection.py
