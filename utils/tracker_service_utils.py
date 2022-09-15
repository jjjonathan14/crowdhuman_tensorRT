import os.path
import pickle
import cv2
from config import cfg
import numpy as np
from nvjpeg import NvJpeg
nj = NvJpeg()


def assign_color(class_name):
    # make sure classes names are like this ['Free Hand', 'Hand With Object', 'Both Hands With Object']
    colors = {'green': [64, 255, 64],  # both hand with object
              'blue': [255, 128, 0],  # single hand with object
              'yellow': [0, 255, 255],  # fee hand
              'red': [64, 0, 255]  # barcode with hand
              }
    if class_name == 'Free Hand':
        color = colors['yellow']
    elif class_name == 'Hand With Object':
        color = colors['blue']
    elif class_name == 'Both Hands With Object':
        color = colors['green']
    else:
        color = colors['red']
    return color


def get_camera_data(cam1_buffer_cp, cam2_buffer_cp, is_paired_):
    for i, cam1_data in enumerate(list(cam1_buffer_cp)):
        frame_pairs_dif = []
        frame_pairs = []
        for j, cam2_data in enumerate(list(cam2_buffer_cp)):
            if abs(cam1_data['frameNumber'] - cam2_data['frameNumber']) < cfg.tracker.frame_paring_threshold:
                frame_pair_ = [cam1_data, cam2_data]
                frame_pairs.append(frame_pair_)
                frame_pairs_dif.append(abs(cam1_data['frameNumber'] - cam2_data['frameNumber']))
                is_paired_ = True
        if is_paired_:
            frame_pair_filtered = frame_pairs[frame_pairs_dif.index(min(frame_pairs_dif))]
            return is_paired_, frame_pair_filtered

    frame_pair_filtered = []
    return is_paired_, frame_pair_filtered


def prepare_data(frame_pair):
    """
    This function is to repair the data inside the frame_pair and arrange them accordingly such a way it will be compatible with 3d trackers
    EX: resize the frame detection data to the original frame size
        include the first camera data in the first position in the 3d feeder

    sample input to 3d trackers
    {
        'bboxes': [[3035.0404858299594, 937.3481781376518, 204.4129554655874, 217.53036437246965], [2731.1538461538457, 1091.4777327935221, 346.51821862348197, 162.8744939271254], [1434.7165991902834, 1029.17004048583, 232.8340080971659, 165.0607287449394], [2299.9190283400812, 1188.2186234817814, 148.66396761133592, 155.22267206477727], [2172.0242914979754, 825.8502024291497, 155.22267206477727, 138.825910931174], [1120.4453441295545, 888.1578947368421, 166.1538461538462, 121.33603238866397]],
        'scores': [1, 1, 1, 1, 1, 1],
        'class_label': ['Sponge', 'Sponge', 'Sponge', 'Sponge', 'Sponge', 'Sponge'],
    }
    """
    target_size_cam1 = [frame_pair[0]['frameOriginalWidth'], frame_pair[0]['frameOriginalHeight']]
    current_size_cam1 = [frame_pair[0]['frame_width'], frame_pair[0]['frame_height']]
    detections_cam1 = frame_pair[0]['detection_boxes']

    target_size_cam2 = [frame_pair[1]['frameOriginalWidth'], frame_pair[1]['frameOriginalHeight']]
    current_size_cam2 = [frame_pair[1]['frame_width'], frame_pair[1]['frame_height']]
    detections_cam2 = frame_pair[1]['detection_boxes']

    resized_bboxes_cam1 = resize_detections(target_size_cam1, current_size_cam1, detections_cam1)
    resized_bboxes_cam2 = resize_detections(target_size_cam2, current_size_cam2, detections_cam2)

    bboxes_cam1 = get_centroid_detections(resized_bboxes_cam1)
    bboxes_cam2 = get_centroid_detections(resized_bboxes_cam2)

    # arranging the order of the cameras as shown in the camera_mapping_pairs [camera1 first and camera2 second]
    if frame_pair[0]['cameraID'] == cfg.tracker_3d.camera_mapping_pairs[0][0]:
        camera1_index = 0
        bbox_data_frame_ax1 = {
            'bboxes': bboxes_cam1,
            'scores': frame_pair[0]['detection_scores'],
            'class_label': frame_pair[0]['detection_names']
        }
    else:
        camera1_index = 1
        bbox_data_frame_ax1 = {
            'bboxes': bboxes_cam2,
            'scores': frame_pair[1]['detection_scores'],
            'class_label': frame_pair[1]['detection_names']
        }

    if frame_pair[1]['cameraID'] == cfg.tracker_3d.camera_mapping_pairs[0][1]:
        camera2_index = 1
        bbox_data_frame_ax2 = {
            'bboxes': bboxes_cam2,
            'scores': frame_pair[1]['detection_scores'],
            'class_label': frame_pair[1]['detection_names']
        }

    else:
        camera2_index = 0
        bbox_data_frame_ax2 = {
            'bboxes': bboxes_cam1,
            'scores': frame_pair[0]['detection_scores'],
            'class_label': frame_pair[0]['detection_names']
        }

    frame_pair = [frame_pair[camera1_index], frame_pair[camera2_index]]
    return bbox_data_frame_ax1, bbox_data_frame_ax2, frame_pair


def get_pipeline_output_data(data, frame_pair, boxes, camera_axis1, camera_axis2):
    with open('out_data.txt', "a") as text_file:
        text_file.write(f'paired:{[frame_pair[0]["frameNumber"], frame_pair[1]["frameNumber"]]} \n')
        text_file.write(f'paired_monotonicTime:{[frame_pair[0]["frameTime"], frame_pair[1]["frameTime"]]} \n')
        text_file.write(
            f'detection_boxes_{frame_pair[0]["cameraID"]}: {frame_pair[0]["detection_boxes"]} \n')
        text_file.write(
            f'detection_boxes_{frame_pair[1]["cameraID"]}: {frame_pair[1]["detection_boxes"]} \n')
        text_file.write(f'cam1_input_{frame_pair[0]["frameNumber"]}: {camera_axis1} \n')
        text_file.write(f'cam2_input_{frame_pair[1]["frameNumber"]}: {camera_axis2} \n')
        text_file.write(f'trangulated_boxes: {boxes} \n')
        text_file.write(f'tracking_trackids_3d: {data["tracking_trackids_3d"]} \n')
        text_file.write(f'tracking_centroids_3d: {data["tracking_centroids_3d"]} \n')
        text_file.write(f'tracking_boxes_3d: {data["tracking_boxes_3d"]} \n')
        # text_file.write(f'barcode_Reader_data: {mc.get("data", [])} \n')
        text_file.write(f'\n')
        text_file.write(f'\n')


def json_transform(o):
    """From <https://stackoverflow.com/a/50577730>"""
    if isinstance(o, np.int64):
        return int(o)
    if isinstance(o, np.int32):
        return int(o)
    if isinstance(o, np.float32):
        return float(o)
    raise TypeError


def overlay_broadcast_frames(data_3dtracker, data_camera_1, data_camera_2, writers, args, encode=True):
    for cam in range(len(data_3dtracker['frame_data'])):
        if data_3dtracker['frame_data'][cam]['cameraID'] == 'camera1':
            camera_id = 'Camera1'
            frame = nj.decode(data_camera_1['frame_UI'])
        elif data_3dtracker['frame_data'][cam]['cameraID'] == 'camera2':
            camera_id = 'Camera2'
            frame = nj.decode(data_camera_2['frame_UI'])
        else:
            raise ValueError(f"[ERROR] cameraID: {data_3dtracker['frame_data'][cam]['cameraID']} not recognized. CameraID can be either camera1 or camera2")
        
        detections = data_3dtracker['frame_data'][cam].get('detection_boxes', [])
        class_names = data_3dtracker['frame_data'][cam].get('detection_names', [])
        current_size = [data_3dtracker['frame_data'][cam].get('frame_width', []), data_3dtracker['frame_data'][cam].get('frame_height', [])]
        target_size = [cfg.tracker_3d.broadcast_frame_resize[0], cfg.tracker_3d.broadcast_frame_resize[1]]
        resized_det_boxes = resize_detections(target_size, current_size, detections)

        for i, bbox in enumerate(resized_det_boxes):
            class_name = class_names[i]
            color = assign_color(class_name)
            w = int(bbox[2])
            h = int(bbox[3])
            x = int(bbox[0])
            y = int(bbox[1])
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        if args.save_video:
            writers[cam] = video_save(frame, writers[cam], camera_id)

        if encode:
            encode_frame = nj.encode(frame)
        else:
            encode_frame = frame
        data_3dtracker['frame_data'][cam]['UI_frame_3d'] = encode_frame
    
    return data_3dtracker, writers


def video_save(frame, writer, camera_id):
    # initialize our video writer
    if not os.path.exists(cfg.video.output_folder):
        os.mkdir(cfg.video.output_folder)
    if writer is None:
        output_vid = f'{cfg.video.output_folder}{camera_id}_{cfg.video.save_labels}.{cfg.video.extension}'
        fourcc = cv2.VideoWriter_fourcc(*cfg.video.FOURCC)
        writer = cv2.VideoWriter(output_vid, fourcc, cfg.video.video_writer_fps,
                                 (frame.shape[1], frame.shape[0]), True)
    writer.write(frame)
    return writer


def resize_detections(target_size, current_size, detections):
    """
    target_size: [w, h]
    current_size: [w, h]
    """
    bboxes = []
    h_current = current_size[1]
    w_current = current_size[0]
    h_target = target_size[1]
    w_target = target_size[0]

    for t_box in detections:
        x = (t_box[0] / w_current) * w_target
        y = (t_box[1] / h_current) * h_target
        w = (t_box[2] / w_current) * w_target
        h = (t_box[3] / h_current) * h_target
        bboxes.append([x, y, w, h])

    return bboxes


def get_centroid_detections(boxes):
    converted_boxes = []
    for box in boxes:
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        x_center = x + w / 2
        y_center = y + h / 2
        converted_boxes.append([x_center, y_center, w, h])
    return converted_boxes


def get_corner_points():
    corner_data = pickle.load(open(cfg.tracker.corner_points, "rb"))
    corner_point_list = []
    for data in corner_data:
        corner_points_temp = {'X3d_tbl_tr': data['tr'], 'X3d_tbl_br': data['br'], 'X3d_tbl_tl': data['tl'],
                              'X3d_tbl_bl': data['bl']}
        corner_point_list.append(corner_points_temp)
    return corner_point_list


