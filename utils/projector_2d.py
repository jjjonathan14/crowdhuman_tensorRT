import numpy as np


def get_rot_mat(phi, theta, si):
    Rx_phi = np.array([[1, 0, 0], [0, np.cos(phi), -1 * np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    Ry_theta = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-1 * np.sin(theta), 0, np.cos(theta)]])
    Rz_si = np.array([[np.cos(si), -1 * np.sin(si), 0], [np.sin(si), np.cos(si), 0], [0, 0, 1]])

    return Rx_phi * Ry_theta * Rz_si


class DataProj2d:
    def __init__(self, proj_angles, proj_loc, k1):
        tvec_p = proj_loc
        phi = proj_angles[0] * np.pi / 180  # Fig2
        theta = proj_angles[1] * np.pi / 180  # Fig2
        si = proj_angles[2] * np.pi / 180  # Fig2

        rot_mat_p = get_rot_mat(phi, theta, si)
        rot_t_p = np.hstack((rot_mat_p, tvec_p))
        k_p = k1
        k_p[0, 2] = 0
        k_p[1, 2] = 0
        self.proj_mat = np.dot(k_p, rot_t_p)

    def projection2d_func(self, loc_3d):
        loc_3d_h = np.hstack((loc_3d, 1))
        loc_2d_h = np.dot(self.proj_mat, loc_3d_h.reshape(4, 1))
        loc_2d = loc_2d_h[:2] / loc_2d_h[2]
        return loc_2d

    def projected_data2d(self, active_tracks, frame_id):
        frame_data = []
        for track in active_tracks:
            loc_3d = track.centrd_wh[:3]
            loc_2d = self.projection2d_func(loc_3d)
            temp_dict_ = {'track_id': track.track_id,
                          'track_data': {'frame_id': frame_id, 'data_xy': [int(loc_2d[0]), int(loc_2d[1])]}}
            # temp_dict_ = {'track_id': track.track_id, 'data_xy': [int(loc_2d[0]), int(loc_2d[1])]}
            frame_data.append(temp_dict_)

        return frame_data
