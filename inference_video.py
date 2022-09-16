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
    os.chdir('/home/zoomi2022/jonathan/people_detection_inference/output_video')
    fourcc = cv2.VideoWriter_fourcc(*cfg.video.FOURCC)
    writer = cv2.VideoWriter(save_name, fourcc, cfg.video.video_writer_fps,(640, 640), True)

    while True:
        ret, frame = capture.read()
        

        if not ret:
            print('finished video')
            break

        else:
            
            frame = cv2.resize(frame, (640, 640), interpolation = cv2.INTER_AREA)
       
            bboxes, scores, classes_name, class_ids = Inference.infer(frame)

            _name = 'frame id'+str(i) 
            for j, box in enumerate(bboxes):
                plot_one_box(
                    box,
                    frame,
                    (0,0,255),
                    label="{}:{:.2f}".format(classes_name[j], scores[j]),
                    #label='Hand With Object',
                )
            print(bboxes)
        

            cv2.putText(
                frame,
                _name,
                (300, 400),
                0,
                2,
                [0, 0, 255],
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            writer.write(frame)

            
            print(i)
            i += 1



for i, video in enumerate(['/home/zoomi2022/jonathan/inference/Camera3-173012-173105.mp4', '/home/zoomi2022/jonathan/inference/Camera4-173014-173105.mp4', '/home/zoomi2022/jonathan/inference/Camera5-173013-173105.mp4', '/home/zoomi2022/jonathan/inference/Camera6-173013-173105.mp4']):
    video_name = video.split('/')[-1]
    print('video name', video_name)
    model_name = '_crowdhuman'
    save_name = video_name.split('.')[0] +model_name + '.mp4'

    infer(video, save_name)
