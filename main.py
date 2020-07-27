"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


# MQTT server environment variables
import os
import sys
import time
import socket
import json
import cv2
import numpy as np
import logging as log
import paho.mqtt.client as mqtt
from argparse import ArgumentParser
from inference import Network
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def process_frame(frame, net_input_shape, infer_network, prob_threshold):
    ### Pre-process the image as needed ###
    width = frame.shape[1]
    height = frame.shape[0]
    p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
    p_frame = p_frame.transpose((2, 0, 1))
    p_frame = p_frame.reshape(1, *p_frame.shape)
    ### Start asynchronous inference for specified request ###
    infer_network.exec_net(p_frame)
    ### Wait for the result ###
    if infer_network.wait() == 0:
        ### Get the results of the inference request ###
        outputs = infer_network.get_output()
        coords = []
        for box in outputs[0][0]:
            conf = box[2]
            if conf >= prob_threshold and box[1] == 1:  # 1 is for 'person'
                cv2.rectangle(frame, (int(box[3]*width), int(box[4]*height)),
                              (int(box[5]*width), int(box[6]*height)), (0, 0, 255), 1)
        return frame


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network(args.model, args.device)
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### Load the model through `infer_network` ###
    infer_network.load_model(args.cpu_extension)
    ### Handle the input stream ###
    img = np.array([])
    if args.input.endswith(('.mp4', '.avi')):
        cap = cv2.VideoCapture(args.input)
    elif args.input.endswith(('.jpg', '.png', '.bmp', '.gif')):
        img = cv2.imread(args.input)
    elif args.input == 'cam':
        cap = cv2.VideoCapture(0)
        log.info("Camera resolution: ( %d x %d )", cap.get(3),
                 cap.get(4))
    else:
        raise ValueError('Given input is unsupported')

    # Grab the shape of the input
    net_input_shape = infer_network.get_input_shape()

    ### Write an output image if `single_image_mode` ###
    if img.any():
        frame = process_frame(img, net_input_shape,
                              infer_network, prob_threshold)
        cv2.imwrite('output_image.jpg', frame)
        return

    while cap.isOpened():

        ### Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        frame = process_frame(frame, net_input_shape,
                              infer_network, prob_threshold)
        ### Calculate and send relevant information on ###
        ### current_count, total_count and duration to the MQTT server ###
        ### Topic "person": keys of "count" and "total" ###
        ### Topic "person/duration": key of "duration" ###

        ### Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()


def main():
    """
    Load the network and parse the output.

    :return: None
    """

    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    log.basicConfig(
        level=log.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            log.FileHandler("app.log"),
            log.StreamHandler()
        ])
    main()
