import numpy as np
import os
import cv2
import tensorflow as tf

THRESHOLD = 0.35
CLASS_NAMES = list(map(lambda x: x.replace('\n', ''), open('ms_coco_classnames.txt').readlines()))


class SSD(object):

    def __init__(self, pbfilepath):

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(pbfilepath, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.sess = tf.Session(graph=self.detection_graph)

    
    def detect(self, image):

        detections = []

        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: np.expand_dims(image, axis=0)})

        for (box, score, label) in zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)):

            if score > THRESHOLD:

                y1 = int(box[0] * image.shape[0])
                x1 = int(box[1] * image.shape[1])
                y2 = int(box[2] * image.shape[0])
                x2 = int(box[3] * image.shape[1])

                if int(label) >= len(CLASS_NAMES):
                    continue

                if CLASS_NAMES[int(label)] in ['bicycle', 'motorcycle', 'truck', 'bus', 'car', 'train', 'boat', 'fire hydrant']:
                    detections.append([x1, y1, x2, y2])

        return detections