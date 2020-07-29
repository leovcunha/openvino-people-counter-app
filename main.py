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
    start = time.time()
    infer_network.exec_net(p_frame)
    ### Wait for the result ###
    if infer_network.wait() == 0:
        ### Get the results of the inference request ###
        outputs = infer_network.get_output()
        infer_time = time.time() - start
        cv2.putText(frame, "Inference Time: {:.3f}ms".format(infer_time * 1000), (20, 20),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5,  (255, 0, 0), 1)
        coords = []
        current_count = 0
        lowscore_counts = 0
        for box in outputs[0][0]:
            conf = box[2]
            if conf >= prob_threshold and box[1] == 1:  # 1 is for 'person'
                # log.info(box)
                cv2.rectangle(frame, (int(box[3]*width), int(box[4]*height)),
                              (int(box[5]*width), int(box[6]*height)), (0, 0, 255), 1)
                current_count += 1
            elif conf < prob_threshold and box[1] == 1:
                # log.info(box)
                lowscore_counts += 1
                cv2.rectangle(frame, (int(box[3]*width), int(box[4]*height)),
                              (int(box[5]*width), int(box[6]*height)), (255, 0, 0), 1)
                cv2.putText(frame, 'Person? {:.2f}%'.format(conf*100), ((int(box[5]*width - 50), int(box[6]*height)-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 12), 1)
        return frame, infer_time, current_count, lowscore_counts


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
        frame, infer_time, count, lc = process_frame(img, net_input_shape,
                                                     infer_network, prob_threshold)
        cv2.imwrite('output_image.jpg', frame)
        return

    total_people = 0
    prev_count = 0
    duration = 0
    start = 0
    box_missing_timer = 0

    while cap.isOpened():

        ### Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        frame, infer_time, current_count, lowscore_counts = process_frame(frame, net_input_shape,
                                                                          infer_network, prob_threshold)
        ### Calculate and send relevant information on ###
        ### current_count, total_count and duration to the MQTT server ###
        ### Topic "person": keys of "count" and "total" ###
        ### Topic "person/duration": key of "duration" ###

        if current_count > prev_count:
            start = time.time()
            total_people += current_count - prev_count
            log.info("{:d} new people in the frame".format(
                current_count-prev_count))
        # Write out information
            client.publish("person", json.dumps(
                {"count": current_count, "total": total_people}))
        # Calc Person Duration

        elif box_missing_timer > 0 and (current_count+lowscore_counts) == prev_count:
            box_missing_timer = 0
            log.info("missing box found in lower scores . box_missing_timer=0")
            current_count = prev_count

        elif current_count < prev_count:
            if box_missing_timer == 0:
                box_missing_timer = time.time()
                log.info(
                    "some people box missing, starting box missing timer and adjusting count to " + str(prev_count))

            elif (time.time() - box_missing_timer) >= 1.2:  # empirically defined
                log.info("{:d} people definitely left the frame".format(
                    prev_count-current_count))
                duration = int(time.time() - start)
                client.publish("person/duration",
                               json.dumps({"duration": duration}))
                client.publish("person", json.dumps(
                    {"count": current_count, "total": total_people}))
                box_missing_timer = 0
                prev_count = current_count
                continue

            current_count = prev_count

        prev_count = current_count
        ### Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()


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
