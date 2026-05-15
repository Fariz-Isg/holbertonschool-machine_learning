#!/usr/bin/env python3
"""
Module to initialize Yolo class and process outputs
"""
import tensorflow as tf
import numpy as np


class Yolo:
    """
    Yolo class for object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes Yolo instance
        """
        self.model = tf.keras.models.load_model(model_path, compile=False)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes Darknet outputs into boundary boxes, confidences,
        and class probabilities
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]
        input_dim = np.sqrt(input_h * input_w)

        img_h, img_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, nb_anchors, _ = output.shape

            box_confidences.append(1 / (1 + np.exp(-output[..., 4:5])))
            box_class_probs.append(1 / (1 + np.exp(-output[..., 5:])))

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            grid_y, grid_x = np.indices((grid_h, grid_w))
            grid_x = grid_x.reshape((grid_h, grid_w, 1))
            grid_y = grid_y.reshape((grid_h, grid_w, 1))

            b_x = (1 / (1 + np.exp(-t_x)) + grid_x) / grid_w
            b_y = (1 / (1 + np.exp(-t_y)) + grid_y) / grid_h

            anchors_w = self.anchors[i, :, 0].reshape((1, 1, nb_anchors))
            anchors_h = self.anchors[i, :, 1].reshape((1, 1, nb_anchors))

            b_w = (anchors_w * np.exp(t_w)) / input_dim
            b_h = (anchors_h * np.exp(t_h)) / input_dim

            x1 = (b_x - b_w / 2) * img_w
            y1 = (b_y - b_h / 2) * img_h
            x2 = (b_x + b_w / 2) * img_w
            y2 = (b_y + b_h / 2) * img_h

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))

        return boxes, box_confidences, box_class_probs
