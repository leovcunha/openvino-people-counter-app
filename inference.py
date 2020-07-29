#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self, model_name, device):
        ###  Initialize any class variables ###
        self.plugin = None
        self.model = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None
        self.model_name = model_name
        self.device = 'CPU'

    def load_model(self, cpu_extension=None):
        ### Load the model ###
        try:
            self.plugin = IECore()
            self.model = IENetwork(
                model=self.model_name+'.xml', weights=self.model_name+'.bin')

        except Exception as e:
            raise ValueError(
                "Could not Initialise the network. Have you enterred the correct model path?")
        ### Add any necessary extensions ###
        if cpu_extension and "CPU" in self.device:
            self.plugin.add_extension(cpu_extension, self.device)

        ### Check for supported layers ###
        self.check_model()
        ### Return the loaded inference plugin ###
        self.exec_network = self.plugin.load_network(
            network=self.model, device_name=self.device)
        # Get the input layer
        self.input_blob = next(iter(self.model.inputs))
        self.output_blob = next(iter(self.model.outputs))

        return

    def check_model(self):
        #raise NotImplementedError
        supported_layers = self.plugin.query_network(self.model, self.device)
        not_supported_layers = [
            l for l in self.model.layers.keys() if l not in supported_layers]
        if len(not_supported_layers):
            log.error("The following layers are not supported "
                      "by the IECore for the specified device {}:\n {}"
                      .format(self.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions ")
            raise NotImplementedError(
                "Some layers are not supported on the device")
        return True

    def get_input_shape(self):
        ### Return the shape of the input layer ###
        return self.model.inputs[self.input_blob].shape

    def exec_net(self, image):
        ### Start an asynchronous request ###
        ### Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        self.exec_network.start_async(request_id=0,
                                      inputs={self.input_blob: image})
        return

    def wait(self):
        ### Wait for the request to be complete. ###
        ### Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[0].wait(-1)

    def get_output(self):
        # Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[0].outputs[self.output_blob]
