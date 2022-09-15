#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains the code for pytorch inference YOLOv5
Supported model frameworks:
        {
            ['PyTorch','.pt'],
            ['TorchScript', '.torchscript'],
            ['ONNX', '.onnx'],
            ['OpenVINO', '_openvino_model'],
            ['TensorRT', '.engine'],
            ['CoreML', 'coreml', '.mlmodel'],
            ['TensorFlow SavedModel', '_saved_model'],
            ['TensorFlow GraphDef', 'pb', '.pb'],
            ['TensorFlow Lite', 'tflite', '.tflite'],
            ['TensorFlow Edge TPU', '_edgetpu.tflite'],
            ['TensorFlow.js', '_web_model']
        }
"""
__author__ = "Tharindu Ekanayake"
__copyright__ = "Copyright 2021, pAIx"
__version__ = "0.0.1"
__maintainer__ = "tharindu326"
__email__ = "tharindu@zoomi.ca"
__status__ = "Staging"

import cv2
import numpy as np
import hashlib
import os
import yaml
from config import cfg
import torch
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_coords
import torch.backends.cudnn as cudnn


class ModelFilePyTorch:
    """
    A class to represent the files associated with a model.
    """
    def __init__(self, model_directory):
        self.labelsPath = os.path.join(model_directory, cfg.infer.labels_file)
        self.weightsPath = os.path.join(model_directory, cfg.PyTorch.weight_file)
        self.labels = yaml.safe_load(open(self.labelsPath, 'rb').read())['names']

        md5 = hashlib.md5()
        for i in self.labelsPath, self.weightsPath:
            md5.update(open(i, "rb").read())
        print(f"Loaded model {os.path.abspath(model_directory)}, md5 {md5.hexdigest()}")


class InferencePyTorch:
    def __init__(self, model_file, target_gpu_id, overlay=False):
        self.overlay = overlay
        self.model_file = model_file

        self.objectness_confidance = cfg.OpenCV.objectness_confidance
        self.nms_threshold = cfg.OpenCV.nms_threshold
        self.gpu_id = target_gpu_id

        self.weight_file = model_file.weightsPath  # model.pt path(s)
        self.data = model_file.labelsPath  # dataset.yaml path
        self.confidence_threshold = cfg.PyTorch.conf_thres  # confidence threshold
        self.nms_iou_threshold = cfg.PyTorch.iou_thres  # NMS IOU threshold
        self.max_detections = cfg.PyTorch.max_det  # maximum detections per image
        self.classes = cfg.PyTorch.classes  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = cfg.PyTorch.agnostic_nms  # class-agnostic NMS
        self.hide_labels = cfg.PyTorch.hide_labels  # hide labels
        self.hide_confidence = cfg.PyTorch.hide_conf  # hide confidences
        self.FP16_half_precision = cfg.PyTorch.half  # use FP16 half-precision inference
        self.dnn = cfg.PyTorch.dnn  # use OpenCV DNN for ONNX inference

        self.colors = {'green': [64, 255, 64],
                       'blue': [255, 128, 0],
                       'coral': [0, 128, 255],
                       'yellow': [0, 255, 255],
                       'gray': [169, 169, 169],
                       'cyan': [255, 255, 0],
                       'magenta': [255, 0, 255],
                       'white': [255, 255, 255],
                       'red': [64, 0, 255]
                       }
        # Load model
        self.device = select_device(self.gpu_id)
        self.model = DetectMultiBackend(self.weight_file, device=self.device, dnn=self.dnn, data=self.data,
                                        fp16=self.FP16_half_precision)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        # imgsz=(640, 640)
        # cudnn.benchmark = True
        # self.model.warmup(imgsz=(1 if self.pt else 1, 3, *imgsz))  # warmup

    @torch.no_grad()
    def infer(self, img0):
        # Convert
        img = img0.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        # Run inference
        im = torch.from_numpy(img).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = self.model(im)
        # NMS
        pred = non_max_suppression(pred, self.confidence_threshold, self.nms_iou_threshold, self.classes,
                                   self.agnostic_nms, max_det=self.max_detections)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        bboxes = []
        scores = []
        class_names = []
        class_ids = []

        # Process predictions
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    if self.names[int(cls)] not in cfg.infer.filter_classes:
                        scores.append(float(f'{conf}'))
                        box = xyxy
                        x = int(box[0])
                        y = int(box[1])
                        w = int(box[2]) - x
                        h = int(box[3]) - y
                        bboxes.append([x, y, w, h])
                        c = int(cls)  # integer class
                        class_ids.append(c)
                        label = None if self.hide_labels else (self.names[c] if self.hide_confidence else f'{self.names[c]} {conf:.2f}')
                        class_names.append(self.names[c])
                        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                        color = self.colors[list(self.colors)[c % len(self.colors)]]
                        lw = 3 or max(round(sum(img0.shape) / 2 * 0.003), 2)  # line width
                        if label and self.overlay:
                            cv2.rectangle(img0, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
                            tf = max(lw - 1, 1)  # font thickness
                            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
                            outside = p1[1] - h >= 3
                            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                            cv2.rectangle(img0, p1, p2, color, -1, cv2.LINE_AA)  # filled
                            cv2.putText(img0,
                                        label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                                        0,
                                        lw / 3,
                                        (255, 255, 255),
                                        thickness=tf,
                                        lineType=cv2.LINE_AA)
        if self.overlay:
            cv2.imwrite('out.jpg', img0)

        return bboxes, scores, class_names, class_ids

    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

