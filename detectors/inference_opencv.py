#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains the code for OpenCV inference for YOLOv4
Supported model frameworks:
        {
            ['OpenCV','.weight'],
        }
"""
__author__ = "Tharindu Ekanayake"
__copyright__ = "Copyright 2021, pAIx"
__version__ = "0.0.1"
__maintainer__ = "tharindu326"
__email__ = "tharindu@zoomi.ca"
__status__ = "Staging"


import cv2
import os
from config import cfg
import yaml


class ModelFileOpenCV:
    """
    A class to represent the files associated with OpenCV inference.
    """

    def __init__(self, model_directory):
        self.labelsPath = os.path.join(model_directory, cfg.infer.labels_file)
        self.configPath = os.path.join(model_directory, cfg.OpenCV.config_file)
        self.weightsPath = os.path.join(model_directory, cfg.OpenCV.weight_file)
        self.labels = yaml.safe_load(open(self.labelsPath, 'rb').read())['names']


class InferenceOpenCV:
    def __init__(self, model_file, target_gpu_id, overlay=False):
        self.overlay = overlay
        self.model_file = model_file

        self.objectness_confidance = cfg.OpenCV.objectness_confidance
        self.nms_threshold = cfg.OpenCV.nms_threshold
        self.gpu_id = target_gpu_id

        self.net = cv2.dnn_DetectionModel(self.model_file.configPath, self.model_file.weightsPath)

        # initialize a list of colors to represent each possible class label
        self.COLORS = {'green': [64, 255, 64],
                       'blue': [255, 128, 0],
                       'coral': [0, 128, 255],
                       'yellow': [0, 255, 255],
                       'gray': [169, 169, 169],
                       'cyan': [255, 255, 0],
                       'magenta': [255, 0, 255],
                       'white': [255, 255, 255],
                       'red': [64, 0, 255]
                       }

        # OpenCV GPU on CUDA support
        try:
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            print("[INFO] GPU device count", device_count)
            cv2.cuda.setDevice(self.gpu_id)
            print(f"DNN_TARGET_CUDA set to GPU id {self.gpu_id}")

            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            print("[INFO] You are using DNN_TARGET_CUDA_FP16 backend to increase the FPS. "
                  "Please make sure your GPU supports floating point 16, or change it back to DNN_TARGET_CUDA. "
                  "Ref: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions")
        except Exception as e:
            print(e)
            print("[INFO] Please build OpenCV with GPU support in order to use DNN_BACKEND_CUDA: "
                  "https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-"
                  "gpus-cuda-and-cudnn/")
            print("[INFO] Shifting back to DNN_TARGET_CPU")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            pass

        print("[INFO] OpenCV version:", cv2.__version__)
        self.version = cv2.__version__
        if self.version < '4.5.4':
            print(f"[INFO] Your OpenCV version is {self.version} and it does not support the Scaled-YOLO models:"
                  f"yolov4-csp, yolov4-csp-swish, yolov4-p5, yolov4-p6. Please install OpenCV-4.5.3 or later")

        net_width, net_height = self.get_networksize()
        print(f'[INFO] Network size is {net_width} * {net_height}')

        self.net.setInputParams(size=(int(net_width), int(net_height)), scale=1 / 255, swapRB=True, crop=False)

    def infer(self, image_original):
        # start_time = time.monotonic()
        bboxes_ = []
        scores_ = []
        names_ = []
        classes_ = []  # numeric class id

        classes, scores, boxes = self.net.detect(image_original, self.objectness_confidance, self.nms_threshold)
        for (classid, score, box) in zip(classes, scores, boxes):
            if self.model_file.labels[classid] not in cfg.infer.filter_classes:
                x = box[0]
                y = box[1]
                w = int(box[2])
                h = int(box[3])
                if self.overlay:
                    color = self.COLORS[list(self.COLORS)[classid % len(self.COLORS)]]
                    label = "{}: {:.4f}".format(self.model_file.labels[classid], score)
                    cv2.rectangle(image_original, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image_original, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                bboxes_.append([x, y, w, h])
                scores_.append(score)
                names_.append(self.model_file.labels[classid])
                classes_.append(classid)

        # end_time = time.monotonic()
        # print('FPS_calc:', 1/(end_time-start_time))
        # print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        # print("[INFO] approx. FPS: {:.2f}".format(1/(end_time-start_time)))
        # print("=================================================================================================")
        # print("[INFO] Inference FPS, with no IPC overhead: {:.2f}".format(1/(end_time-start_time)) )

        return bboxes_, scores_, names_, classes_

    def get_networksize(self):
        # default net size will be set as 416 if the network size is not detected from the cfg file
        net_width = None
        net_height = None
        for path in open(self.model_file.configPath).read().split("\n"):
            if path.split("=")[0] == 'width':
                net_width = path.split("=")[1]
            if path.split("=")[0] == 'height':
                net_height = path.split("=")[1]
        return net_width, net_height



if __name__ == '__main__':
    model_directory = './model_data'
    model_files = ModelFileOpenCV(model_directory)
    target_gpu_device = 0
    detector = InferenceOpenCV(target_gpu_device, model_files, overlay=False)

    path = 'C:/Users/HP-PC/PycharmProjects/pAIx/face_detection'
    os.chdir('C:/Users/HP-PC/PycharmProjects/pAIx/result/2/')
    for image_ in os.listdir(path):
        image = os.path.join(path, image_)
        img = cv2.imread(image)
        box, name, score, classes = detector.infer(cv2.imread(image))
        for bboxes in box:
            x, y, w, h = bboxes[0], bboxes[1], bboxes[2], bboxes[3]
            cv2.rectangle(img,(x,y), (x+w, y+h), (200,0,0),2)
        cv2.imshow('detect', img)
        cv2.imwrite('aa'+image_, img)
        cv2.waitKey(0)

