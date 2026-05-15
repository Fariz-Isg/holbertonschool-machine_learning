#!/usr/bin/env python3
"""Yolo v3 object detection model."""
import numpy as np
import tensorflow.keras as K
import os
import cv2


class Yolo:
    '''yolo class'''
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        '''
        -model_path is the path to where a Darknet Keras model is stored
        -classes_path is the path to where the list of class names used
        for the Darknet model, listed in order of index, can be found
        -class_t is a float representing the box score threshold for the
        initial filtering step
        -nms_t is a float representing the IOU threshold for non-max
        suppression
        -anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
        containing all of the anchor boxes
        '''
        self.model = K.models.load_model(model_path)
        with open(classes_path, "r") as f:
            self.class_names = f.read().split()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process Darknet model outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        image_h = image_size[0]
        image_w = image_size[1]

        for i, output in enumerate(outputs):
            grid_h = output.shape[0]
            grid_w = output.shape[1]
            anchor_boxes = output.shape[2]

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            c_x = np.arange(grid_w).reshape(1, grid_w, 1)
            c_x = np.tile(c_x, (grid_h, 1, anchor_boxes))

            c_y = np.arange(grid_h).reshape(grid_h, 1, 1)
            c_y = np.tile(c_y, (1, grid_w, anchor_boxes))

            b_x = (1 / (1 + np.exp(-t_x)) + c_x) / grid_w
            b_y = (1 / (1 + np.exp(-t_y)) + c_y) / grid_h

            anchor_w = self.anchors[i, :, 0].reshape((1, 1, anchor_boxes))
            anchor_h = self.anchors[i, :, 1].reshape((1, 1, anchor_boxes))

            b_w = (np.exp(t_w) * anchor_w) / input_w
            b_h = (np.exp(t_h) * anchor_h) / input_h

            x1 = (b_x - (b_w / 2)) * image_w
            y1 = (b_y - (b_h / 2)) * image_h
            x2 = (b_x + (b_w / 2)) * image_w
            y2 = (b_y + (b_h / 2)) * image_h

            box = np.stack((x1, y1, x2, y2), axis=-1)
            boxes.append(box)

            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(box_confidence)

            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes with object threshold."""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            box_confidence = box_confidences[i]
            box_class_prob = box_class_probs[i]

            box_score = box_confidence * box_class_prob
            box_class = np.argmax(box_score, axis=-1)
            box_score = np.max(box_score, axis=-1)
            pos = np.where(box_score >= self.class_t)
            filtered_boxes.append(boxes[i][pos])
            box_classes.append(box_class[pos])
            box_scores.append(box_score[pos])
        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)
        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Apply non-max suppression to filtered boxes."""
        nms_boxes = []
        nms_classes = []
        nms_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            idxs = np.where(box_classes == cls)
            cls_boxes = filtered_boxes[idxs]
            cls_box_scores = box_scores[idxs]

            sorted_idxs = np.argsort(cls_box_scores)[::-1]
            cls_boxes = cls_boxes[sorted_idxs]
            cls_box_scores = cls_box_scores[sorted_idxs]

            while len(cls_boxes) > 0:
                nms_boxes.append(cls_boxes[0])
                nms_classes.append(cls)
                nms_scores.append(cls_box_scores[0])

                if len(cls_boxes) == 1:
                    break

                ious = self.iou(cls_boxes[0], cls_boxes[1:])
                idxs_to_keep = np.where(ious < self.nms_t)[0] + 1
                cls_boxes = cls_boxes[idxs_to_keep]
                cls_box_scores = cls_box_scores[idxs_to_keep]

        return np.array(nms_boxes), np.array(nms_classes), np.array(nms_scores)

    def iou(self, box1, boxes):
        """Calculate Intersection over Union (IoU) manually.
        Args:
            box1: single box with shape (4,) containing [x1, y1, x2, y2]
            boxes: multiple boxes with shape (N, 4)
            where each is [x1, y1, x2, y2]
        Returns:
            ious: array of IoU values with shape (N,)
        """
        x1_1 = box1[0]
        y1_1 = box1[1]
        x2_1 = box1[2]
        y2_1 = box1[3]
        x1_2 = boxes[:, 0]
        y1_2 = boxes[:, 1]
        x2_2 = boxes[:, 2]
        y2_2 = boxes[:, 3]
        x1_i = np.maximum(x1_1, x1_2)
        y1_i = np.maximum(y1_1, y1_2)
        x2_i = np.minimum(x2_1, x2_2)
        y2_i = np.minimum(y2_1, y2_2)
        w_i = np.maximum(0, x2_i - x1_i)
        h_i = np.maximum(0, y2_i - y1_i)
        area_i = w_i * h_i
        area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area_2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        area_u = area_1 + area_2 - area_i
        ious = area_i / area_u
        return ious

    def preprocess_images(self, images):
        """Resize and rescale images for YOLO input"""
        pimages = []
        image_shapes = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for image in images:
            image_shapes.append(image.shape[:2])

            resized = cv2.resize(
                image,
                (input_h, input_w),
                interpolation=cv2.INTER_CUBIC
            )

            pimages.append(resized / 255.0)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    @staticmethod
    def load_images(folder_path):
        """Load all images from a folder"""
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            if os.path.isfile(path):
                image = cv2.imread(path)
                if image is not None:
                    images.append(image)
                    image_paths.append(path)

        return images, image_paths

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
            """Display image with boundary boxes, class names, and box scores"""
            image_copy = image.copy()
    
            for i, box in enumerate(boxes):
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
    
                cv2.rectangle(
                    image_copy,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )
    
                class_name = self.class_names[box_classes[i]]
                score = box_scores[i]
                text = "{} {:.2f}".format(class_name, score)
    
                cv2.putText(
                    image_copy,
                    text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )
    
            cv2.imshow(file_name, image_copy)
            key = cv2.waitKey(0)
    
            if key == ord('s'):
                if not os.path.isdir("detections"):
                    os.makedirs("detections")
    
                save_path = os.path.join("detections", os.path.basename(file_name))
                cv2.imwrite(save_path, image_copy)
    
            cv2.destroyAllWindows()

    def predict(self, folder_path):
        """Predict objects in all images inside a folder"""
        predictions = []
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)

        outputs = self.model.predict(pimages, verbose=0)

        for i, image in enumerate(images):
            image_outputs = [output[i] for output in outputs]

            boxes, box_confidences, box_class_probs = self.process_outputs(
                image_outputs,
                image_shapes[i]
            )

            filtered_boxes, box_classes, box_scores = self.filter_boxes(
                boxes,
                box_confidences,
                box_class_probs
            )

            box_predictions, predicted_box_classes, predicted_box_scores = (
                self.non_max_suppression(
                    filtered_boxes,
                    box_classes,
                    box_scores
                )
            )

            predictions.append(
                (
                    box_predictions,
                    predicted_box_classes,
                    predicted_box_scores
                )
            )

            self.show_boxes(
                image,
                box_predictions,
                predicted_box_classes,
                predicted_box_scores,
                os.path.basename(image_paths[i])
            )

        return predictions, image_paths
