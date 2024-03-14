import os
import argparse
from rich.progress import track
import math
import json
import errno
import numpy as np
import open3d as o3d
from pypcd import pypcd


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_files_path(path_my_dir, extention=".json"):
    path_list = []
    for (dirpath, dirnames, filenames) in os.walk(path_my_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == extention:
                path_list.append(os.path.join(dirpath, filename))
    return path_list


def write_txt(path, file):
    with open(path, "w") as f:
        f.write(file)


def read_json(path):
    with open(path, "r") as f:
        my_json = json.load(f)
        return my_json


def write_json(path_json, new_dict):
    with open(path_json, "w") as f:
        json.dump(new_dict, f)


def read_pcd(path_pcd):
    pointpillar = o3d.io.read_point_cloud(path_pcd)
    points = np.asarray(pointpillar.points)
    points = points.tolist()
    return points


def write_pcd(path_pcd, new_points, path_save):
    pc = pypcd.PointCloud.from_path(path_pcd)
    pc.pc_data['x'] = np.array([a[0] for a in new_points])
    pc.pc_data['y'] = np.array([a[1] for a in new_points])
    pc.pc_data['z'] = np.array([a[2] for a in new_points])
    pc.save_pcd(path_save, compression='binary_compressed')


def show_pcd(path_pcd):
    pcd = read_pcd(path_pcd)
    o3d.visualization.draw_geometries([pcd])


def pcd2bin(pcd_file_path, bin_file_path):
    pc = pypcd.PointCloud.from_path(pcd_file_path)

    np_x = (np.array(pc.pc_data["x"], dtype=np.float32)).astype(np.float32)
    np_y = (np.array(pc.pc_data["y"], dtype=np.float32)).astype(np.float32)
    np_z = (np.array(pc.pc_data["z"], dtype=np.float32)).astype(np.float32)
    np_i = (np.array(pc.pc_data["intensity"], dtype=np.float32)).astype(np.float32) / 255

    points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))
    points_32.tofile(bin_file_path)


def inverse_matrix(R):
    R = np.matrix(R)
    rev_R = R.I
    rev_R = np.array(rev_R)
    return rev_R


def trans_point(input_point, rotation, translation=None):
    if translation is None:
        translation = [0.0, 0.0, 0.0]
    input_point = np.array(input_point).reshape(3, 1)
    translation = np.array(translation).reshape(3, 1)
    rotation = np.array(rotation).reshape(3, 3)
    output_point = np.dot(rotation, input_point).reshape(3, 1) + np.array(translation).reshape(3, 1)
    output_point = output_point.reshape(1, 3).tolist()
    return output_point[0]


def trans(input_point, rotation, translation):
    input_point = np.array(input_point).reshape(3, -1)
    translation = np.array(translation).reshape(3, 1)
    rotation = np.array(rotation).reshape(3, 3)
    output_point = np.dot(rotation, input_point).reshape(3, -1) + np.array(translation).reshape(3, 1)
    return output_point


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


def get_label_lidar_rotation(lidar_3d_8_points):
    """
    计算 lidar 坐标系下的偏航角 rotation_z
        目标 3D 框示意图:
          4 -------- 5
         /|         /|
        7 -------- 6 .
        | |        | |
        . 0 -------- 1
        |/         |/
        3 -------- 2
        Args:
            lidar_3d_8_points: 八个角点组成的矩阵[[x,y,z],...]
        Returns:
            rotation_z: Lidar坐标系下的偏航角rotation_z (-pi,pi) rad
    """
    x0, y0 = lidar_3d_8_points[0][0], lidar_3d_8_points[0][1]
    x3, y3 = lidar_3d_8_points[3][0], lidar_3d_8_points[3][1]
    dx, dy = x0 - x3, y0 - y3
    rotation_z = math.atan2(dy, dx)  # Lidar坐标系xyz下的偏航角yaw绕z轴与x轴夹角，方向符合右手规则，所以用(dy,dx)
    return rotation_z


def get_camera_3d_8points(label_3d_dimensions, camera_3d_location, rotation_y):
    camera_rotation = np.matrix(
        [
            [math.cos(rotation_y), 0, math.sin(rotation_y)],
            [0, 1, 0],
            [-math.sin(rotation_y), 0, math.cos(rotation_y)]
        ]
    )
    l, w, h = label_3d_dimensions
    corners_3d_camera = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [0, 0, 0, 0, -h, -h, -h, -h],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        ]
    )
    camera_3d_8points = camera_rotation * corners_3d_camera + np.matrix(camera_3d_location).T
    return camera_3d_8points.T.tolist()


def get_camera_3d_alpha_rotation(camera_3d_8_points, camera_3d_location):
    x0, z0 = camera_3d_8_points[0][0], camera_3d_8_points[0][2]
    x3, z3 = camera_3d_8_points[3][0], camera_3d_8_points[3][2]
    dx, dz = x0 - x3, z0 - z3
    rotation_y = -math.atan2(dz, dx)  # 相机坐标系xyz下的偏航角yaw绕y轴与x轴夹角，方向符合右手规则，所以用(-dz,dx)
    # alpha = rotation_y - math.atan2(center_in_cam[0], center_in_cam[2])
    alpha = rotation_y - (-math.atan2(-camera_3d_location[2], -camera_3d_location[0])) + math.pi / 2  # yzw
    # add transfer
    if alpha > math.pi:
        alpha = alpha - 2.0 * math.pi
    if alpha <= (-1 * math.pi):
        alpha = alpha + 2.0 * math.pi
    return alpha, rotation_y


def get_cam_calib_intrinsic(calib_path):
    my_json = read_json(calib_path)
    cam_K = my_json["cam_K"]
    calib = np.zeros([3, 4])
    calib[:3, :3] = np.array(cam_K).reshape([3, 3], order="C")
    return calib


def get_lidar2camera(path_lidar2camera):
    lidar2camera = read_json(path_lidar2camera)
    rotation = lidar2camera['rotation']
    translation = lidar2camera['translation']
    rotation = np.array(rotation).reshape(3, 3)
    translation = np.array(translation).reshape(3, 1)
    return rotation, translation


def get_lidar2novatel(path_lidar2novatel):
    lidar2novatel = read_json(path_lidar2novatel)
    rotation = lidar2novatel['transform']['rotation']
    translation = lidar2novatel['transform']['translation']
    rotation = np.array(rotation).reshape(3, 3)
    translation = np.array(translation).reshape(3, 1)
    return rotation, translation


def get_novatel2world(path_novatel2world):
    novatel2world = read_json(path_novatel2world)
    rotation = novatel2world['rotation']
    translation = novatel2world['translation']
    rotation = np.array(rotation).reshape(3, 3)
    translation = np.array(translation).reshape(3, 1)
    return rotation, translation


def get_virtuallidar2world(path_virtuallidar2world):
    virtuallidar2world = read_json(path_virtuallidar2world)
    rotation = virtuallidar2world['rotation']
    translation = virtuallidar2world['translation']
    rotation = np.array(rotation).reshape(3, 3)
    translation = np.array(translation).reshape(3, 1)
    return rotation, translation

parser = argparse.ArgumentParser("Generate the Kitti Format Data")
parser.add_argument("--source-root", type=str, default="data/V2X-Seq-SPD",
                    help="Raw data root about SPD")
parser.add_argument("--target-root", type=str, default="data/KITTI-Track/cooperative",
                    help="The data root where the data with kitti-Track format is generated")
parser.add_argument("--split-path", type=str, default="data/split_datas/cooperative-split-data-spd.json",
                    help="Json file to split the data into training/validation/testing.")
parser.add_argument("--no-classmerge", action="store_true",
                    help="Not to merge the four classes [Car, Truck, Van, Bus] into one class [Car]")
parser.add_argument("--temp-root", type=str, default="./tmp_file", help="Temporary intermediate file root.")


def concat_txt(input_path, output_path, output_file_name):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    path_file_output = output_path + '/' + output_file_name + '.txt'
    write_f = open(path_file_output, 'w')
    list_files = os.listdir(input_path)
    list_files.sort()
    for file in list_files:
        path_file_input = input_path + '/' + file
        with open(path_file_input, 'r') as read_f:
            list_lines = read_f.readlines()
            for line in list_lines:
                write_f.writelines(line)
    write_f.close()


def label_dair2kiiti_by_frame(dair_label_file_path, kitti_label_file_path, rotation, translation, frame, pointcloud_timestamp,
                              no_classmerge):
    save_file = open(kitti_label_file_path, 'w')
    list_labels = read_json(dair_label_file_path)
    for label in list_labels:
        # if not no_classmerge:
        #     label["type"] = label["type"].replace("Truck", "Car")
        #     label["type"] = label["type"].replace("Van", "Car")
        #     label["type"] = label["type"].replace("Bus", "Car")
        label["type"] = "Car"
        label_3d_dimensions = [float(label["3d_dimensions"]["l"]), float(label["3d_dimensions"]["w"]),
                               float(label["3d_dimensions"]["h"])]
        lidar_3d_location = [float(label["3d_location"]["x"]), float(label["3d_location"]["y"]),
                             float(label["3d_location"]["z"])]
        rotation_z = float(label["rotation"])
        lidar_3d_8_points = get_lidar_3d_8points(label_3d_dimensions, lidar_3d_location, rotation_z)

        lidar_3d_bottom_location = [float(label["3d_location"]["x"]), float(label["3d_location"]["y"]),
                                    float(label["3d_location"]["z"]) - float(label["3d_dimensions"]["h"]) / 2]
        camera_3d_location = trans_point(lidar_3d_bottom_location, rotation, translation)
        camera_3d_8_points = []
        for lidar_point in lidar_3d_8_points:
            camera_point = trans_point(lidar_point, rotation, translation)
            camera_3d_8_points.append(camera_point)

        alpha, rotation_y = get_camera_3d_alpha_rotation(camera_3d_8_points, camera_3d_location)

        list_item = [frame, str(label["type"]), str(label["track_id"]),
                     str(label["truncated_state"]), str(label["occluded_state"]), str(alpha),
                     str(100.0), str(200.0),
                     str(300.0), str(400.0),
                     str(label_3d_dimensions[2]), str(label_3d_dimensions[1]), str(label_3d_dimensions[0]), str(camera_3d_location[0]),
                     str(camera_3d_location[1]), str(camera_3d_location[2]), str(rotation_y), str(lidar_3d_location[0]),
                     str(lidar_3d_location[1]), str(lidar_3d_location[2]), str(rotation_z), pointcloud_timestamp, "1", "1",
                     str(label["token"])]
        str_item = ' '.join(list_item) + '\n'
        save_file.writelines(str_item)
    save_file.close()


def label_dair2kitti(source_root, target_root, temp_root, dict_sequence2tvt, frame_info, no_classmerge):
    for i in track(frame_info):
        calib_lidar_to_camera_path = f'{source_root}/vehicle-side/calib/lidar_to_camera/{i["vehicle_frame"]}.json'
        rotation, translation = get_lidar2camera(calib_lidar_to_camera_path)
        source_label_path = f'{source_root}/cooperative/label/{i["vehicle_frame"]}.json'
        temp_label_path = f'{temp_root}/{dict_sequence2tvt[i["vehicle_sequence"]]}/{i["vehicle_sequence"]}/label_02_split'
        os.makedirs(temp_label_path, exist_ok=True)
        temp_label_file_path = f'{temp_label_path}/{i["vehicle_frame"]}.txt'        
        label_dair2kiiti_by_frame(source_label_path, temp_label_file_path, rotation, translation, i["vehicle_frame"], "-1", no_classmerge)

    list_tvt = os.listdir(temp_root)
    for tvt in list_tvt:
        temp_tvt_path = f'{temp_root}/{tvt}'
        list_seqs = os.listdir(temp_tvt_path)
        for seq in list_seqs:
            temp_label_path = f'{temp_tvt_path}/{seq}/label_02_split'
            target_label_path = f'{target_root}/{tvt}/{seq}/label_02'
            concat_txt(temp_label_path, target_label_path, seq)


if __name__ == "__main__":
    args = parser.parse_args()
    spd_data_root = args.source_root
    target_root = args.target_root
    split_path = args.split_path
    split_info = read_json(split_path)
    dict_sequence2tvt = {}
    for tvt in [["train", "training"], ["val", "validation"], ["test", "testing"]]:
        for seq in split_info["batch_split"][tvt[0]]:
            dict_sequence2tvt[seq] = tvt[1]
    frame_info = read_json(f'{spd_data_root}/cooperative/data_info.json')
    temp_root = args.temp_root
    no_classmerge = args.no_classmerge
    if os.path.exists(temp_root):
        os.system("rm -rf %s" % temp_root)
    os.system("mkdir -p %s" % temp_root)
    label_dair2kitti(spd_data_root, target_root, temp_root, dict_sequence2tvt, frame_info, no_classmerge)
    os.system("cp -r %s/* %s/" % (temp_root, target_root))
    os.system("rm -rf %s" % temp_root)
