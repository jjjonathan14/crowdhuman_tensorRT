#!/usr/bin/env python3

"""
This file contains the code for the main application for inference service (detector process)
usage:
python inference_service.py -p 10031 --detector -m ./model_data -g 0 --conflate --inference-engine OpenCV
"""

__author__ = "Tharindu Ekanayake"
__copyright__ = "Copyright 2021, pAIx"
__version__ = "0.0.1"
__maintainer__ = "tharindu326/Nuwan1654"
__email__ = "tharindu@zoomi.ca/dinusha@zoomi.ca"
__status__ = "Staging"

from config import cfg

from detectors.inference_tensorrt import ModelFileTensorRT, InferenceTensorRT
from utils.logger import get_debug_logger
from utils.fps import FPS
from utils.tensorrt_util import *
import os, glob
import json
import cv2

Inference = InferenceTensorRT()

image = cv2.imread('/home/zoomi2022/jonathan/people_detection_inference/chloe_grace_low.png')

bboxes, scores, classes_name, class_ids = Inference.infer(image)

for j, box in enumerate(bboxes):
    plot_one_box(
        box,
        image,
        (0,0,255),
        label="{}:{:.2f}".format(classes_name[j], scores[j]),
        #label='Hand With Object',
    )


cv2.imwrite('test.jpg', image)



