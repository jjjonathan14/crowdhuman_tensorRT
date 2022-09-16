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

Inference = InferenceTensorRT()


def infer(video_path,  save_name):
    
    anno = {

    }
    i = 1
    capture = cv2.VideoCapture(video_path)
    while True:
        ret, frame = capture.read()

        if not ret:
            print('finished video')
            break

        else:
            bboxes, scores, classes_name, class_ids = Inference.infer(frame)
            m = 0
            print(classes_name, class_ids, bboxes)
            child_dict = {

            }
            for j, box in enumerate(bboxes):
                child_dict[str(m)] = {'bboxe' : [str(x) for x in bboxes[j]],
                                       'score': str(scores[j]),
                                       'name':classes_name
                                       }
                m += 1

            _name = 'frame id'+str(i) 
            anno[_name] = child_dict
         
            
            print(i)
            i += 1

    os.chdir('/home/zoomi2022/jonathan/people_detection_inference/output_json')
    with open(save_name, 'w') as f:
        json.dump(anno, f, indent=4)


for video in ['/home/zoomi2022/jonathan/inference/Camera3-173012-173105.mp4', '/home/zoomi2022/jonathan/inference/Camera4-173014-173105.mp4', '/home/zoomi2022/jonathan/inference/Camera5-173013-173105.mp4', '/home/zoomi2022/jonathan/inference/Camera6-173013-173105.mp4']:
    video_name = video.split('/')[-1]
    model_name = 'crowdhuman'
    save_name = video_name.split('.')[0] +model_name + '.json'
    infer(video, save_name)
