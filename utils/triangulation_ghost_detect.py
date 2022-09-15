import numpy as np
import cv2 as cv
from config import cfg


def triangulatePoints(P1, P2, x1, x2):
    X = cv.triangulatePoints(P1[:3], P2[:3], x1[:2], x2[:2])
    return X / X[3]  # Remember to divide out the 4th row. Make it homogeneous


def find_obj(img_obj_lista, img_obj_listb, fundamental_mat):
    obj_indx_list = []
    img_pts_list = []
    # or index, (key, value) in enumerate(your_dict.items()):
    boxes_a = img_obj_lista['bboxes']
    # track_ids_a = img_obj_lista['track_ids']
    boxes_b = img_obj_listb['bboxes']
    # track_ids_b = img_obj_listb['track_ids']
    # obj_det_mat = np.empty([len(boxes_a), len(boxes_b)])
    obj_det_matrix = np.zeros([len(boxes_a), len(boxes_b)])  # assign 1 if there is a corresponding object is detected
    det_a_3d = np.zeros(len(boxes_a))  # check each 2d box has a 3d bbox
    det_b_3d = np.zeros(len(boxes_b))
    for ia, bbox_a in enumerate(boxes_a):
        for ib, bbox_b in enumerate(boxes_b):

            # bbox_a = obj_a['bboxes']
            bbox_xy_a = np.array([bbox_a[0], bbox_a[1], 1]).astype(float)
            # bbox_b = obj_b['bboxes']
            bbox_xy_b = np.array([bbox_b[0], bbox_b[1], 1]).astype(float)
            temp_ = np.dot(bbox_xy_b, fundamental_mat)
            out_ = np.dot(temp_, bbox_xy_a)
            # obj_det_mat[ia, ib] = out_
            if abs(out_) < cfg.tracker.triangulation_threshold:
                obj_indx = [ia, ib, out_]
                obj_indx_list.append(obj_indx)
                img_pts = [bbox_xy_a, bbox_xy_b]
                img_pts_list.append(img_pts)
                obj_det_matrix[ia, ib] = 1
                det_a_3d[ia] = 1
                det_b_3d[ib] = 1

    return obj_indx_list, img_pts_list


class DetectTriangulate:
    def __init__(self, proj_mat_a, proj_mat_b, fundamental_mat):
        self.proj_mat_a = proj_mat_a
        self.proj_mat_b = proj_mat_b
        self.fundamental_mat = fundamental_mat

    def dimension_obj(self, bbox_a, bbox_b, obj_xyz):
        """
        :param bbox_a: bbox data of image a
        :param bbox_b: bbox data of image b
        :param obj_xyz: centroid of the corresponding object
        :return: object width and height (to determine the 3d bbox boundary)
        """
        x_a, y_a, w_a, h_a = bbox_a[0], bbox_a[1], bbox_a[2], bbox_a[3]
        x_b, y_b, w_b, h_b = bbox_b[0], bbox_b[1], bbox_b[2], bbox_b[3]
        w_pt_a = np.array([x_a + (w_a / 2), y_a, 1]).astype(float)
        h_pt_a = np.array([x_a, y_a + (h_a / 2), 1]).astype(float)
        w_pt_b = np.array([x_b + (w_b / 2), y_b, 1]).astype(float)
        h_pt_b = np.array([x_b, y_b + (h_b / 2), 1]).astype(float)
        obj_pt_width = triangulatePoints(self.proj_mat_a, self.proj_mat_b, w_pt_a, w_pt_b)
        obj_width = 2 * np.linalg.norm(obj_pt_width - obj_xyz)
        obj_pt_height = triangulatePoints(self.proj_mat_a, self.proj_mat_b, h_pt_a, h_pt_b)
        obj_height = 2 * np.linalg.norm(obj_pt_height - obj_xyz)

        return obj_width, obj_height

    def object_3d(self, img_obj_lista, img_obj_listb):

        obj_indx_list, img_pts_list = find_obj(img_obj_lista, img_obj_listb, self.fundamental_mat)

        obj_bbox_list = []
        ghost_obj = []
        track_id_list = []

        for i_p, img_pts in enumerate(img_pts_list):  # getting object points for all pairs of image points
            ghost_obj.append([])
            a_pt = img_pts[0]  # image a coordinates for detected object
            b_pt = img_pts[1]  # image b coordinates for detected object
            ia = obj_indx_list[i_p][0]
            ib = obj_indx_list[i_p][1]
            bbox_a = img_obj_lista['bboxes'][ia]
            bbox_b = img_obj_listb['bboxes'][ib]
            len_search = i_p
            while len_search > 0:
                len_search = len_search - 1
                if obj_indx_list[i_p][0] == obj_indx_list[len_search][0] or obj_indx_list[i_p][1] == obj_indx_list[len_search][1]:
                    ghost_obj[i_p].append(len_search)
                    ghost_obj[len_search].append(i_p)

            if len(a_pt) > 0 and len(b_pt) > 0:
                obj_xyz = triangulatePoints(self.proj_mat_a, self.proj_mat_b, a_pt, b_pt)
            obj_bbox = obj_xyz[:3]
            w_obj, h_obj = self.dimension_obj(bbox_a, bbox_b, obj_xyz)
            obj_bbox = np.concatenate((obj_bbox, [w_obj, h_obj]), axis=None)
            obj_bbox_list.append(obj_bbox)

        return obj_bbox_list, ghost_obj
