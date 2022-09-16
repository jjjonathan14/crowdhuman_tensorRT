#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C

# Inference options
__C.infer = edict()
__C.infer.batch_size = 1  # this has fixed to 1. so cannot use batch operations
__C.infer.engines = ['OpenCV', 'PyTorch', 'TensorRT']
__C.infer.model_directory = './model_data'
__C.infer.labels_file = "coco128.yaml"
__C.infer.filter_classes = ['Free Hand']  # ['Free Hand', 'Hand With Object', 'Both Hands With Object']

# OpenCV
__C.OpenCV = edict()
__C.OpenCV.objectness_confidance = 0.24  # threshold for filter the higher detection scores lager than 0.24
__C.OpenCV.nms_threshold = 0.4  # iou threshold when executing the NMS process to remove, when there are more than a single bbox with various detection scores for a single object.
__C.OpenCV.weight_file = "yolo-obj_best.weights"  # weight/model file to load the trained weights
__C.OpenCV.config_file = "yolo-obj.cfg"  # configuration file which includes the network architecture configs

# PyTorch
__C.PyTorch = edict()
__C.PyTorch.weight_file = "20220728_yolov5m_3_augmentation.pt"  # model.pt path(s)
__C.PyTorch.conf_thres = 0.25  # confidence threshold
__C.PyTorch.iou_thres = 0.45  # NMS IOU threshold
__C.PyTorch.max_det = 1000  # maximum detections per image
__C.PyTorch.classes = None  # filter by class: --class 0, or --class 0 2 3
__C.PyTorch.agnostic_nms = False  # class-agnostic NMS
__C.PyTorch.hide_labels = False  # hide labels
__C.PyTorch.hide_conf = False  # hide confidences
__C.PyTorch.half = False  # use FP16 half-precision inference
__C.PyTorch.dnn = False  # use OpenCV DNN for ONNX inference

# TensorRT
__C.TensorRT = edict()
# DEMO 22-08-19 models
__C.TensorRT.weight_file = 'crowdhuman_latest_yolov5m_static.engine'
__C.TensorRT.plugin_lib = "/home/zoomi2022/dinusha/converted_models/20220817/libmyplugins.so"
__C.TensorRT.model_input_shape = (640, 640)
__C.TensorRT.batch = False
__C.TensorRT.batch_size = 1

# OLD models
# __C.TensorRT.weight_file = 'yolov5m_two_class.engine'
# __C.TensorRT.plugin_lib = "/home/zoomi2022/dinusha/converted_models/20220802-2/libmyplugins.so"
# __C.TensorRT.model_input_shape = (640, 640)

__C.TensorRT.conf_thres = 0.2  # confidence threshold
__C.TensorRT.iou_thres = 0.1  # NMS IOU threshold

# Tracking options

'''
All the tracking related configuration values 
'''
__C.tracker = edict()

# Byte Tracker
__C.tracker.track_thresh = 0.2  # if the confidence_score> track_thresh + det_tresh_gap then initialize a new track otherwise it will only match tracklets where confidance_score> track_thresh
__C.tracker.track_buffer = 30  # length of maximum frames where can a lost tracklet be. if the tracklet not appear within 30 frames track will be deleted., else track will rebirth
__C.tracker.match_thresh = 0.9  # linear assignment threshold where it uses Jonker-Volgenant algorithm. when this is lower no tracklets age. this can be also defined as cost of assignment of the Jonker-Volgenant algorithm. maximum error that allow for the linear assignment.
__C.tracker.det_tresh_gap = 0.2  # if the confidence_score> track_thresh + det_tresh_gap then initialize a new track for untrack detection/new detection in first association; tracklets matching in higher score region (confidence_score> track_thresh) 

'''
following configs are not effect for a significant change on the tracking output
'''
__C.tracker.track_thresh_lower = 0.1  # lower bound for the detected objects which has lower confidance score. so the confidance values between track_thresh_lower and the track_thresh will be considered for second association when matching the tracklets with detections
__C.tracker.match_thresh_second_association = 0.5  # the iou cost that allowed in the linear assignment when second association is happening.
__C.tracker.match_thresh_unconfirmed = 0.7  # linear assignment threshold when matching the unconfirmed tracklets with detections. unconfirmed tracklets: the tracklets which were not activated. usually tracks with only one beginning frame.

__C.tracker.frame_paring_threshold = 0.008
__C.tracker.triangulation_threshold = 0.15

# ByteTrack 3D

__C.tracker_3d = edict()

__C.tracker_3d.F_mat_path = './camera_metrics/F_mat_cam1cam2_8_16.npy'
__C.tracker_3d.R_mat_path = './camera_metrics/R_mat_cam1cam2_8_16.npy'
__C.tracker_3d.T_vec_path = './camera_metrics/T_vec_cam1cam2_8_16.npy'
__C.tracker_3d.cam1_mtx_path = './camera_metrics/cam1_mtx_8_16.npy'
__C.tracker_3d.cam2_mtx_path = './camera_metrics/cam2_mtx_8_16.npy'

# 3d semantic edges definition
__C.tracker.corner_points = './camera_metrics/corner_data_co_aug17.pkl'

__C.tracker_3d.camera_mapping_pairs = [['camera1', 'camera2']]

__C.tracker_3d.regions = ['scan', 'bag']
__C.tracker_3d.broadcast_frame_resize = (960, 540)
# websocket
__C.websocket = edict()
__C.websocket.ip = "localhost"
__C.websocket.checkout_manager_websocket_ip = '0.0.0.0'
__C.websocket.broadcast_manager_websocket_ip = '0.0.0.0'

# memcache
__C.memcache = edict()
__C.memcache.port = 11211
__C.memcache.ip = "127.0.0.1"

# video save option

__C.video = edict()
__C.video.output_folder = './output/'
__C.video.video_writer_fps = 25
__C.video.FOURCC = 'MP4V'  # 4-byte code used to specify the video codec
__C.video.save_labels = f'{__C.TensorRT.weight_file.split(".")[0]}-2022-07-21'  # anything you wish to add on save file name
__C.video.extension = 'mp4'


# camera service option

__C.camera = edict()
__C.camera.initial_flush_time = 30
__C.camera.frame_number_reset_time = 5  # in minutes
