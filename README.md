Vision Landing
==============
Use the model learned from the
[Detect Frying Pan](https://github.com/floft/detect-frying-pan)
code and now run that on live RPi Zero camera input. We want to get the
drone to land on the frying pan.

See the [demo video on Youtube](https://www.youtube.com/watch?v=9iIUZG7x9K0).

# Running Object Detection on another computer

## Wiring the Raspberry Pi to the Pixhawk

Connect RPi UART/serial pins to the Serial 4 of the Pixhawk looking at the
pinouts of TELEM2 or Serial 4/5 on
[this](https://docs.px4.io/en/flight_controller/pixhawk.html) or
[this](http://ardupilot.org/copter/docs/common-pixhawk-overview.html). Basically,
just follow [this](http://ardupilot.org/dev/docs/raspberry-pi-via-mavlink.html)
(i.e. swapping TX and RX between above and RPi) and the
[RPi diagram](https://pinout.xyz/pinout/pin8_gpio14) (Note: the square pin on
the RPi is pin 1). Note: no need for level converter since Pixhawk TX/RX says
it's also 3.3V just like the RPi

In APM Planner, under Config/Tuning -> Full Parameter List -> ... set:

    SERIAL4_PROTOCOL=1 # default is 5, but 1 is MAVLINK1 and 2 is MAVLINK2
                       # See: http://ardupilot.org/copter/docs/parameters.html#serial1-protocol
	SERIAL4_BAUD=115 # i.e. 115200 which is probably fast enough
	                 # See: http://ardupilot.org/copter/docs/common-telemetry-port-setup-for-apm-px4-and-pixhawk.html

Set up the Pi:

	sudo raspi-config # Interfacing options -> Serial -> No login screen but Yes Serial Enabled

Edit */boot/config.txt*, adding "dtoverlay=pi3-miniuart-bt" to the bottom
([src](https://www.raspberrypi.org/documentation/configuration/uart.md)).

Reboot to apply the above boot configuration, then install dependencies:

    sudo systemctl reboot
    sudo apt install screen python-wxgtk2.8 python-matplotlib python-opencv \
        python-pip python-numpy python-dev libxml2-dev libxslt-dev python-lxml
    sudo pip install future pymavlink mavproxy

## Raspberry Pi Setup
Since the Zero is really slow, I'll stream to another computer to do processing
for now. Though, at least one person has used the GPU on the RPi Zero to get
[~8 fps on face detection](https://www.youtube.com/watch?v=A3BDg13DX3M). So, it
is possible, though he hasn't shared his code. I will return to this problem
later.

    sudo apt install python3-zmq

Also you need to build [v4l2rtspserver](https://github.com/mpromonet/v4l2rtspserver.git)
and [mavlink-router](https://github.com/intel/mavlink-router) and set up a
wireless access point.

### v4l2rtspserver
Download and install:

    git clone --recursive https://github.com/mpromonet/v4l2rtspserver.git
    cd v4l2rtspserver/
    vim CMakeLists.txt # replace ONLY_CMAKE_FIND_ROOT_PATH with /usr/lib/arm-linux-gnueabihf /usr/local/lib
    cmake . # for a total clean before: rm -rf CMakeCache.txt *.a CMakeFiles
    make
    sudo make install

### mavlink-router
Download and install:

    git clone --recursive https://github.com/intel/mavlink-router
    cd mavlink-router
    ./autogen.sh && ./configure CFLAGS='-g -O2' --sysconfdir=/etc --localstatedir=/var/local --libdir=/usr/local/lib --prefix=/usr/local
    make
    sudo make install

Create the config file */etc/mavlink-router/main.conf*:

    [General]
    TcpServerPort=5760
    ReportStats=false
    MavlinkDialect=auto

    [UdpEndpoint groundstation]
    Mode = Eavesdropping
    Address = 0.0.0.0
    Port = 14550

    [UartEndpoint pixhawk]
    Device = /dev/ttyAMA0
    Baud = 115200

### Wireless Access Point
I bought one (actually 3 because I broke 2) of the RT5370 2.4GHz wireless adapters
that had built-in Linux drivers (e.g. on Amazon
[here](https://www.amazon.com/gp/product/B073J3HXZH/)
or [here](https://www.amazon.com/gp/product/B00H95C0A2/)).

#### Install
Update and install dependencies:

    sudo apt update
    sudo apt install dnsmasq hostapd

#### Configure
Edit */etc/dhcpcd.conf* (assuming your wifi adapter shows up as *wlan1* in *ip link*):

    interface wlan1
        static ip_address=192.168.4.1/24
        nohook wpa_supplicant

Edit */etc/dnsmasq.conf*:

    interface=wlan1      # Use the require wireless interface
        dhcp-range=192.168.4.2,192.168.4.100,255.255.255.0,24h

Edit */etc/hostapd/hostapd.conf*:

    interface=wlan1
    driver=nl80211
    ssid=rz
    hw_mode=g
    channel=1 # whatever channel you want
    country_code=US # so it always has the right max power, channels, etc.
    ieee80211n=1
    wmm_enabled=1
    ht_capab=[HT40+][SHORT-GI-20][SHORT-GI-40]
    macaddr_acl=0
    auth_algs=1
    ignore_broadcast_ssid=0
    wpa=2
    wpa_passphrase=password # be sure to set this to something else
    wpa_key_mgmt=WPA-PSK
    wpa_pairwise=TKIP
    rsn_pairwise=CCMP

Edit */etc/default/hostapd*:

    DAEMON_CONF="/etc/hostapd/hostapd.conf"
    #if debugging: DAEMON_OPTS="-dd -t -f /home/pi/hostapd.log"

Edit */etc/network/interfaces.d/wlan1*:

    allow-hotplug wlan1

#### Enable and Run
Then, enable everything:

    sudo systemctl disable wpa_supplicant@wlan1
    sudo systemctl stop wpa_supplicant@wlan1
    sudo systemctl restart dhcpcd dnsmasq hostapd
    sudo systemctl enable dhcpcd dnsmasq hostapd

#### Update Firmware
And, if you wish to reduce *dmesg* errors
([src](https://www.raspberrypi.org/forums/viewtopic.php?t=22623#p324659)), then
download "Mac" version (actually Linux and the Linux is a Mac .dmg file) from
[https://www.mediatek.com/products/broadbandWifi/rt5572](https://www.mediatek.com/products/broadbandWifi/rt5572)
get the *rt2870.bin* file from *common/*.

    mv /lib/firmware/rt2870.bin{,.bak}
    mv rt2870.bin /lib/firmware
    reboot

#### Force 40 MHz Mode
And, if you wish to force *hostapd* to use 40 MHz mode:

    wget https://w1.fi/releases/hostapd-2.7.tar.gz
    tar xavf hostapd-2.7.tar.gz
    cd hostapd-2.7

    sudo apt remove hostapd
    sudo apt install build-essential libnl-3-dev iw crda libssl-dev libnl-genl-3-200 libnl-3-200 libnl-genl-3-dev

Then, edit *src/ap/hw_features.c* and search for search for *wpa_scan_results_free*
or *oper40* and set *oper40=1* right below the checks
([src](https://patchwork.ozlabs.org/patch/144477/). Then:

    cd hostapd
    cp defconfig .config # in .config uncomment CONFIG_IEEE80211N=y and CONFIG_IEEE80211AC=y and CONFIG_ACS=y
    make
    sudo make install # installs /usr/local/bin/hostapd{,_cli}

Then, create */etc/systemd/system/hostapd-custom.service*:

    [Unit]
    Description=Hostapd IEEE 802.11 AP, IEEE 802.1X/WPA/WPA2/EAP/RADIUS Authenticator
    After=network.target

    [Service]
    ExecStart=/usr/local/bin/hostapd /etc/hostapd/hostapd.conf -d
    # -d -t -f /home/pi/hostapd.log
    ExecReload=/bin/kill -HUP $MAINPID

    [Install]
    WantedBy=multi-user.target

And switch to using the custom hostapd:

    sudo systemctl disable hostapd
    sudo systemctl enable hostapd-custom
    sudo systemctl daemon-reload
    sudo systemctl restart dhcpcd dnsmasq hostapd-custom

## Pixhawk Setup
Set these options
([yaw reference](https://docs.px4.io/en/config/flight_controller_orientation.html)):

    PLND_ENABLED=1 # enable always land
    PLND_TYPE=1 # companion computer
    PLND_EST_TYPE=1 # for EKF or raw, see which works best for you

    # Note: this would be ideal, but it's actually cdeg, so we'd have to set
    # this to 27000, but it won't let me set values outside of [0,360]
    # thus, we rotate in code
    #PLND_YAW_ALIGN=270 # depending on how you mounted the camera on the drone

Then map a switch on your R/C controller to channel 6. For low PPM value it'll
do nothing, for higher it'll stream, and for even higher it'll shut down the
Raspberry Pi (for exact values, see script).

Optionally, if you wish to enable precision loiter on some R/C channel:

    CH7_OPT=39

## Running
On Pi:

    cd ./vision-landing
    ./control.py # --device=tcp:127.0.0.1:5760 or --device=/dev/ttyAMA0 etc.

Or, if you wish to always run on boot (running */home/pi/vision-landing/control.py*
as user *pi* and group *dialout* for access to */dev/ttyAMA0*):

    sudo cp control.service /etc/systemd/system/
    sudo systemctl enable control mavlink-router
    sudo systemctl start control mavlink-router

On laptop (and outputting debug info, saving images to record/, displaying with
GStreamer):

    sudo pacman -S gst-python
    cd ./vision-landing
    ./object_detector.py --record record
