import cv2
import logging as log
import time
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True,
                        type=str, help="Path to image or video file.")
    return parser


def main(args):
    image = cv2.imread(args.input)
    image_expanded = np.expand_dims(image, axis=0)
    graph_detection = tf.Graph()

    with tf.compat.v1.Session(graph=graph_detection) as session:
        meta_graph = tf.compat.v1.train.import_meta_graph(args.model + '.meta')
        meta_graph.restore(session, args.model)
        image_tensor = graph_detection.get_tensor_by_name('image_tensor:0')
        boxes = graph_detection.get_tensor_by_name(
            'detection_boxes:0')
        scores = graph_detection.get_tensor_by_name(
            'detection_scores:0')
        classes = graph_detection.get_tensor_by_name('detection_classes:0')
        num_detections = graph_detection.get_tensor_by_name('num_detections:0')
        start = time.time()
        (boxes, scores, classes, num_detections) = session.run(
            [boxes, scores, classes, num_detections], feed_dict={image_tensor: image_expanded})
    inf_time = time.time() - start
    log.info('Inference Time for 1 frame: {} ms'.format(inf_time / 100 * 1000))


if __name__ == '__main__':
    log.basicConfig(
        level=log.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            log.StreamHandler()
        ])
    args = build_argparser().parse_args()
    main(args)
