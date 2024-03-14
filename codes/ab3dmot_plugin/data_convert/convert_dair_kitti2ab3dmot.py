import os
import json
import numpy as np
import math
from rich.progress import track


def range2box(box_range):
    # [x0, y0, z0, x1, y1, z1]
    box_range = np.array(box_range)
    indexs = [
        [0, 1, 2],
        [3, 1, 2],
        [3, 4, 2],
        [0, 4, 2],
        [0, 1, 5],
        [3, 1, 5],
        [3, 4, 5],
        [0, 4, 5],
    ]
    return np.array([[box_range[index] for index in indexs]])

def dot_product(p1, p2):
    return p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]


def GetCross(x1, y1, x2, y2, x, y):
    a = (x2 - x1, y2 - y1)
    b = (x - x1, y - y1)
    return a[0] * b[1] - a[1] * b[0]


def cross_product(p1, p2):
    return [
        p1[1] * p2[2] - p1[2] * p2[1],
        p1[2] * p2[0] - p1[0] * p2[2],
        p1[0] * p2[1] - p2[0] * p1[1],
    ]


def isInSide(x1, y1, x2, y2, x3, y3, x4, y4, x, y):
    return (
        GetCross(x1, y1, x2, y2, x, y) * GetCross(x3, y3, x4, y4, x, y) >= 0
        and GetCross(x2, y2, x3, y3, x, y) * GetCross(x4, y4, x1, y1, x, y) >= 0
    )


def above_plane(point, plane):
    # ax + by + cz = d
    norm = cross_product(plane[1] - plane[0], plane[2] - plane[0])  # [a, b, c]
    d = dot_product(plane[0], norm)
    z_intersec = (d - norm[0] * point[0] - norm[1] * point[1]) / norm[2]
    # https://www.cnblogs.com/nobodyzhou/p/6145030.html
    t = (norm[0] * point[0] + norm[1] * point[1] + norm[2] * point[2] - d) / (
        norm[0] ** 2 + norm[1] ** 2 + norm[2] ** 2
    )
    point_x = point[0] - norm[0] * t
    point_y = point[1] - norm[1] * t
    if z_intersec <= point[2] and isInSide(
        plane[0][0],
        plane[0][1],
        plane[1][0],
        plane[1][1],
        plane[2][0],
        plane[2][1],
        plane[3][0],
        plane[3][1],
        point_x,
        point_y,
    ):
        # if z_intersec <= point[2] and point_in_matrix([point_x,point_y], plane[:, :2]):
        return 1
    else:
        return 0


def point_in_box(point, box):
    return above_plane(point, box[:4]) + above_plane(point, box[4:]) == 1


def get_lidar_3d_8points(label_3d_dimensions, lidar_3d_location, rotation_z):
    lidar_rotation = np.matrix(
        [
            [math.cos(rotation_z), -math.sin(rotation_z), 0],
            [math.sin(rotation_z), math.cos(rotation_z), 0],
            [0, 0, 1]
        ]
    )
    l, w, h = label_3d_dimensions
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
        ]
    )
    lidar_3d_8points = lidar_rotation * corners_3d_lidar + np.matrix(lidar_3d_location).T
    return lidar_3d_8points.T.tolist()


class Filter(object):
    def __init__(self):
        pass

    def __call__(self, **args):
        return True


class RectFilter(Filter):
    def __init__(self, bbox):
        super().__init__()
        self.bbox = bbox

    def __call__(self, box, **args):
        for corner in box:
            if point_in_box(corner, self.bbox):
                return True
        return False


def convert_gt_label(kitti_track_tvt_path, ab3dmot_path, extended_range=None):
    """
    v2x/AB3DMOT_plugin/scripts/KITTI/evaluate_tracking.seqmap.val
    eval_label_data/full/validation_gt_label -> scripts/KITTI/label/
        Args:
            input_path: "cooperative_gt_label/validation_gt_label"
            kitti_track_tvt_path: "../../data/V2X-Seq-SPD-KITTI/cooperative/validation"
            ab3dmot_path: "./AB3DMOT_plugin"
            output_path: "eval_label_data/full/validation_gt_label"
        Returns:
            dict
    """
    if extended_range is None:
        extended_range = [0, -39.68, -3, 100, 39.68, 1]
    bbox_filter = RectFilter(range2box(np.array(extended_range))[0])

    seqmap_file_path = os.path.join(ab3dmot_path, 'scripts/KITTI/evaluate_tracking.seqmap.val')
    output_path = os.path.join(ab3dmot_path, 'scripts/KITTI/label')

    os.system(f'rm {seqmap_file_path}')
    os.makedirs(output_path, exist_ok=True)
    os.system(f'rm -rf {output_path}/*')

    dic_data_json = {}
    list_sequence_files = sorted(os.listdir(kitti_track_tvt_path))
    with open(seqmap_file_path, 'w') as a:
        for sequence_file in track(list_sequence_files):
            b = [sequence_file.split('.')[0], "empty", "000000"]
            input_seq_file_path = f'{kitti_track_tvt_path}/{sequence_file}/label_02/{sequence_file}.txt'
            output_seq_file_path = f'{output_path}/{sequence_file}.txt'
            with open(input_seq_file_path, 'r') as read_f, open(output_seq_file_path, "w") as write_f:
                list_lines = read_f.readlines()
                list_lines_filt = []
                for line in list_lines:
                    line = line.replace("Truck", "Car")
                    line = line.replace("Van", "Car")
                    line = line.replace("Bus", "Car")
                    i = line.strip().split(' ')
                    corners = get_lidar_3d_8points([float(i[12]), float(i[11]), float(i[10])], [float(i[17]), float(i[18]), float(i[19])],
                                                   float(i[20]))
                    if bbox_filter(corners):
                        list_lines_filt.append(line)

                list_frame = []
                for i in list_lines_filt:
                    list_i = i.strip("\n").split(" ")
                    if list_i[0] not in list_frame:
                        list_frame.append(list_i[0])
                
                dic_frame2id = {k: str(j) for j, k in enumerate(list_frame)}
                dic_data_json[b[0]] = dic_frame2id
                len_frame = len(list_frame)
                b.append(f"{len_frame:06d}\n")
                a.write(' '.join(b))
                for line in list_lines_filt:
                    list_line = line.strip("\n").split(" ")
                    list_line[0] = dic_frame2id[list_line[0]]
                    list_line_output = [list_line[0], "{:.0f}".format(float(list_line[2])), list_line[1], list_line[3], list_line[4],
                                        list_line[5], list_line[6], list_line[7], list_line[8], list_line[9], list_line[10], list_line[11],
                                        list_line[12], list_line[13], list_line[14], list_line[15], list_line[16]]
                    str_line = ' '.join(list_line_output) + '\n'
                    write_f.write(str_line)
    return dic_data_json


def convert_track_label(track_results_path, output_path, dic, extended_range=None):
    """
        Args:
            track_results_path: "../output/${EVAL_RESULT_NAME}/tracking_results_to_kitti/${SUB_OUTPUT_PATH_DTC}_H1/data_0"
            output_path: ./AB3DMOT_plugin/results/KITTI/imvoxelnet_Car_val_H1/data_0
            dic: dict from val_dic_new_frame2id_frame.json
            extended_range: [0, -39.68, -3, 100, 39.68, 1]
        Returns:
            None
    """
    if extended_range is None:
        extended_range = [0, -39.68, -3, 100, 39.68, 1]
    bbox_filter = RectFilter(range2box(np.array(extended_range))[0])
    os.makedirs(output_path, exist_ok=True)
    os.system(f'rm -rf {output_path}/*')
    list_sequence_files = sorted(os.listdir(track_results_path))
    for sequence_file in track(list_sequence_files):
        dic_frame2id = dic[sequence_file.split('.')[0]]
        input_seq_file_path = os.path.join(track_results_path, sequence_file)
        output_seq_file_path = os.path.join(output_path, sequence_file)
        with open(input_seq_file_path, 'r') as read_f, open(output_seq_file_path, "w") as write_f:
            list_lines = read_f.readlines()
            list_lines_filt = []
            for line in list_lines:
                line = line.replace("Truck", "Car")
                line = line.replace("Van", "Car")
                line = line.replace("Bus", "Car")
                i = line.strip().split(' ')
                corners = get_lidar_3d_8points([float(i[12]), float(i[11]), float(i[10])], [float(i[17]), float(i[18]), float(i[19])], float(i[20]))
                if bbox_filter(corners):
                    list_lines_filt.append(line)

            for line in list_lines_filt:
                list_line = line.strip("\n").split(" ")
                if list_line[0] not in dic_frame2id.keys():
                    # print(sequence_file, list_line[0])
                    continue
                list_line[0] = dic_frame2id[list_line[0]]
                list_line_output = [list_line[0], str(int(list_line[2])), list_line[1], list_line[3], list_line[4], list_line[5],
                                    list_line[6], list_line[7], list_line[8], list_line[9], list_line[10], list_line[11], list_line[12],
                                    list_line[13], list_line[14], list_line[15], list_line[16], list_line[23]]
                str_line = ' '.join(list_line_output) + '\n'
                write_f.write(str_line)
