#!/usr/bin/env python3
"""
Send fake bounding boxes for debugging
"""
import time
import zmq

class SendDetections:
    def __init__(self, host="192.168.4.1", send_port=5555):
        self.host = host
        self.send_port = send_port
        self.socket = None
        self.exiting = False
        self.send_connect()

    def run(self):
        while not self.exiting:
            try:
                self.send_detections()
                time.sleep(0.1)
            except KeyboardInterrupt:
                self.exiting = True

    def send_connect(self):
        if not self.socket:
            context = zmq.Context()
            self.socket = context.socket(zmq.PUB)
            self.socket.setsockopt(zmq.SNDHWM, 1)
            self.socket.setsockopt(zmq.RCVHWM, 1)
            self.socket.setsockopt(zmq.CONFLATE, 1) # Only get last message
            self.socket.connect("tcp://"+self.host+":"+str(self.send_port))

    def send_detections(self):
        best = {
            "score": str(1.0),
            "xmin": 300,
            "ymin": 300,
            "xmax": 300,
            "ymax": 300,
        }

        # If the socket has closed, try reconnecting
        if not self.socket:
            self.send_connect()

        # If we successfully connected, send over the socket
        if self.socket:
            self.socket.send_json(best)

if __name__ == "__main__":
    s = SendDetections()
    s.run()
