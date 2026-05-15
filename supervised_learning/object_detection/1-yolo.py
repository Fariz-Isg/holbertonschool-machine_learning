#!/usr/bin/env python3
"""
Module to initialize Yolo class
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
        Processes Darknet outputs
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        img_h, img_w = image_size
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            # Sigmoid for confidence and class probabilities
            conf = 1 / (1 + np.exp(-output[..., 4:5]))
            probs = 1 / (1 + np.exp(-output[..., 5:]))

            box_confidences.append(conf)
            box_class_probs.append(probs)

            # Create grid of (c_x, c_y)
            grid_y, grid_x = np.indices((grid_h, grid_w))
            grid_x = grid_x.reshape((grid_h, grid_w, 1))
            grid_y = grid_y.reshape((grid_h, grid_w, 1))

            # Extract components and apply sigmoid where needed
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # Calculate normalized centers and sizes
            b_x = (1 / (1 + np.exp(-t_x)) + grid_x) / grid_w
            b_y = (1 / (1 + np.exp(-t_y)) + grid_y) / grid_h

            anchors_w = self.anchors[i, :, 0].reshape((1, 1, anchor_boxes))
            anchors_h = self.anchors[i, :, 1].reshape((1, 1, anchor_boxes))

            b_w = (anchors_w * np.exp(t_w)) / input_w
            b_h = (anchors_h * np.exp(t_h)) / input_h

            # Convert to x1, y1, x2, y2 relative to original image
            x1 = (b_x - b_w / 2) * img_w
            y1 = (b_y - b_h / 2) * img_h
            x2 = (b_x + b_w / 2) * img_w
            y2 = (b_y + b_h / 2) * img_h

            # Stack into (grid_h, grid_w, anchor_boxes, 4)
            processed_boxes = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(processed_boxes)

        return boxes, box_confidences, box_class_probs
