import os
import math
import json
import numpy as np
from tqdm import tqdm
import argparse
import errno
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


type2id = {
    "Car": 2,
    "Van": 2,
    "Truck": 2,
    "Bus": 2,
    "Cyclist": 1,
    "Tricyclist": 3,
    "Motorcyclist": 3,
    "Barrow": 3,
    "Barrowlist": 3,
    "Pedestrian": 0,
    "Trafficcone": 3,
    "Pedestrianignore": 3,
    "Carignore": 3,
    "otherignore": 3,
    "unknowns_unmovable": 3,
    "unknowns_movable": 3,
    "unknown_unmovable": 3,
    "unknown_movable": 3,
}

id2type = {
    0: "Pedestrian",
    1: "Cyclist",
    2: "Car",
    3: "Motorcyclist"
}


def get_sequence_id(frame, data_info):
    for obj in data_info:
        if int(frame) == int(obj["image_path"].split('/')[-1].split('.')[0]):
            sequence_id = obj["sequence_id"]
            return sequence_id


def trans_points_cam2img(camera_3d_8points, calib_intrinsic, with_depth=False):
    """
        Transform points from camera coordinates to image coordinates.
        Args:
            camera_3d_8points: list(8, 3)
            calib_intrinsic: np.array(3, 4)
        Returns:
            list(8, 2)
    """
    camera_3d_8points = np.array(camera_3d_8points)
    points_shape = np.array([8, 1])
    points_4 = np.concatenate((camera_3d_8points, np.ones(points_shape)), axis=-1)
    point_2d = np.dot(calib_intrinsic, points_4.T)
    point_2d = point_2d.T
    point_2d_res = point_2d[:, :2] / point_2d[:, 2:3]
    if with_depth:
        return np.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)
    return point_2d_res.tolist()


def label_det_result2kitti(input_file_path, output_dir_path, ori_path, split_data_path):
    """
        Convert detection results from mmdetection3d_kitti format to KITTI format.
        Args:
            input_file_path: mmdetection3d_kitti results pickle file path
            output_dir_path: converted kitti format file directory
            ori_path: path to ori dataset
    """
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    with open(input_file_path, 'rb') as load_f:
        det_result_data = json.load(load_f)

    with open(split_data_path,'r') as fp:
        split_data = json.load(fp)
    
    index = int(input_file_path.split('/')[-1].split('_')[0])
    real_frame_id = int(split_data["vehicle_split"]['val'][index])
    
    veh_frame = str(real_frame_id)
    lidar2camera_path = f'{ori_path}/vehicle-side/calib/lidar_to_camera/{veh_frame.zfill(6)}.json'
    camera2image_path = f'{ori_path}/vehicle-side/calib/camera_intrinsic/{veh_frame.zfill(6)}.json'
    rotation, translation = get_lidar2camera(lidar2camera_path)
    calib_intrinsic = get_cam_calib_intrinsic(camera2image_path)
    output_file_path = output_dir_path + '/' + veh_frame + '.txt'
    if os.path.exists(output_file_path):
        print("veh_frame", veh_frame, "det_result_name", input_file_path.split('/')[-1].split('.')[0])
        save_file = open(output_file_path, 'a')
    else:
        save_file = open(output_file_path, 'w')
    num_boxes = len(det_result_data["score"])
    for i in range(num_boxes):
        lidar_3d_8points_det_result = det_result_data["bbox"][i]
        # lidar_3d_8points = [lidar_3d_8points_det_result[3], lidar_3d_8points_det_result[0], lidar_3d_8points_det_result[4],
        #                     lidar_3d_8points_det_result[7], lidar_3d_8points_det_result[2], lidar_3d_8points_det_result[1],
        #                     lidar_3d_8points_det_result[5], lidar_3d_8points_det_result[6]]
        lidar_3d_8points = lidar_3d_8points_det_result

        # calculate l, w, h, x, y, z in LiDAR coordinate system
        lidar_xy0, lidar_xy3, lidar_xy1 = lidar_3d_8points[0][0:2], lidar_3d_8points[3][0:2], lidar_3d_8points[1][0:2]
        lidar_z4, lidar_z0 = lidar_3d_8points[4][2], lidar_3d_8points[0][2]
        l = math.sqrt((lidar_xy0[0] - lidar_xy3[0]) ** 2 + (lidar_xy0[1] - lidar_xy3[1]) ** 2)
        w = math.sqrt((lidar_xy0[0] - lidar_xy1[0]) ** 2 + (lidar_xy0[1] - lidar_xy1[1]) ** 2)
        h = lidar_z4 - lidar_z0
        lidar_x0, lidar_y0 = lidar_3d_8points[0][0], lidar_3d_8points[0][1]
        lidar_x2, lidar_y2 = lidar_3d_8points[2][0], lidar_3d_8points[2][1]
        lidar_x = (lidar_x0 + lidar_x2) / 2
        lidar_y = (lidar_y0 + lidar_y2) / 2
        lidar_z = (lidar_z0 + lidar_z4) / 2
        
        obj_type = id2type[2]
        score = det_result_data["score"][i]

        camera_3d_8points = []
        for lidar_point in lidar_3d_8points:
            camera_point = trans_point(lidar_point, rotation, translation)
            camera_3d_8points.append(camera_point)

        # generate the yaw angle of the object in the lidar coordinate system at the vehicle-side.
        lidar_rotation = get_label_lidar_rotation(lidar_3d_8points)
        # generate the alpha and yaw angle of the object in the camera coordinate system at the vehicle-side
        camera_x0, camera_z0 = camera_3d_8points[0][0], camera_3d_8points[0][2]
        camera_x2, camera_z2 = camera_3d_8points[2][0], camera_3d_8points[2][2]
        camera_x = (camera_x0 + camera_x2) / 2
        camera_y = camera_3d_8points[0][1]
        camera_z = (camera_z0 + camera_z2) / 2
        camera_3d_location = [camera_x, camera_y, camera_z]

        image_8points_2d = trans_points_cam2img(camera_3d_8points, calib_intrinsic)
        x_max = max(image_8points_2d[:][0])
        x_min = min(image_8points_2d[:][0])
        y_max = max(image_8points_2d[:][1])
        y_min = min(image_8points_2d[:][1])

        alpha, camera_rotation = get_camera_3d_alpha_rotation(camera_3d_8points, camera_3d_location)

        str_item = str(veh_frame) + ' ' + str(obj_type) + ' ' + '-1' + ' ' + '-1' + ' ' + '-1' + ' ' + str(alpha) + ' ' + str(
            x_min) + ' ' + str(y_min) + ' ' + str(x_max) + ' ' + str(y_max) + ' ' + str(h) + ' ' + str(w) + ' ' + str(l) + ' ' + str(
            camera_x) + ' ' + str(camera_y) + ' ' + str(camera_z) + ' ' + str(camera_rotation) + ' ' + str(lidar_x) + ' ' + str(
            lidar_y) + ' ' + str(lidar_z) + ' ' + str(lidar_rotation) + ' ' + '-1' + ' ' + str(score) + ' ' + '-1' + ' ' + '-1' + '\n'
        save_file.writelines(str_item)
    save_file.close()


def gen_kitti_result(input_dir_path, output_dir_path, ori_path, split_data_path):
    """
        Convert detection results from mmdetection3d_kitti format to KITTI format for all files in input_dir_path.
        Args:
            input_dir_path: directory containing mmdetection3d_kitti results pickle files
            output_dir_path: directory to save converted KITTI format files
            ori_path: path to ori dataset
    """
    if os.path.exists(output_dir_path):
        os.system('rm -rf %s' % output_dir_path)
    os.makedirs(output_dir_path)
    for file in tqdm(os.listdir(input_dir_path)):
        path_file = input_dir_path + '/' + file
        label_det_result2kitti(path_file, output_dir_path, ori_path, split_data_path)


def gen_kitti_seq_result(input_dir_path, output_dir_path, ori_path):
    """
        Convert detection results from mmdetection3d_kitti format to KITTI format and group them by sequence.
        Args:
            input_dir_path: directory containing mmdetection3d_kitti results pickle files
            output_dir_path: directory to save converted KITTI format files grouped by sequence
            ori_path: path to ori dataset
    """
    data_info = read_json(f'{ori_path}/vehicle-side/data_info.json')
    list_input_files = os.listdir(input_dir_path)
    if os.path.exists(output_dir_path):
        os.system('rm -rf %s' % output_dir_path)
    os.makedirs(output_dir_path)
    for input_file in tqdm(list_input_files):
        input_file_path = input_dir_path + '/' + input_file
        index = int(input_file.split('.')[0])
        sequence_id = get_sequence_id(index, data_info)
        sequence_path = output_dir_path + '/' + sequence_id
        if not os.path.exists(sequence_path):
            os.makedirs(sequence_path)
        os.system('cp %s %s/' % (input_file_path, sequence_path))


def gen_kitti_seq_txt(input_dir_path, output_dir_path):
    """
        Group converted KITTI format files by sequence and write them into one txt file per sequence.
        Args:
            input_dir_path: directory containing KITTI format files grouped by sequence
            output_dir_path: directory to save txt files grouped by sequence
    """
    if os.path.exists(output_dir_path):
        os.system('rm -rf %s' % output_dir_path)
    os.makedirs(output_dir_path)
    list_dir_sequences = os.listdir(input_dir_path)
    for dir_sequence in tqdm(list_dir_sequences):
        path_seq_input = input_dir_path + '/' + dir_sequence
        file_output = output_dir_path + '/' + dir_sequence + '.txt'
        save_file = open(file_output, 'w')
        list_files = os.listdir(path_seq_input)
        list_files.sort()
        for file in list_files:
            path_file = path_seq_input + '/' + file
            with open(path_file, "r") as read_f:
                data_txt = read_f.readlines()
                for item in data_txt:
                    save_file.writelines(item)
        save_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert detection results to KITTI format')
    parser.add_argument('--input-dir-path', type=str, help='Directory containing mmdetection3d_kitti results pickle files',
                        default='/data/daiyingru/projects/OpenCOOD/opencood/logs/point_pillar_fcooper_corridor_full/json')
    parser.add_argument('--output-dir-path', type=str, help='Directory to save converted KITTI format files',
                        default='/data/haoruiyang/HOPE/DAIR-V2X/output-full/corridor_fcooper/detection_results_to_kitti/opencood_Car_val')
    parser.add_argument('--ori-path', type=str, help='Path to ori dataset',
                        default='/data/ad_sharing/datasets/RCOOPER-DAIR/corridor')
    parser.add_argument('--split-data-path', type=str, help='Directory to split data',
                        default='/data/ad_sharing/datasets/RCOOPER-DAIR/corridor/split-data.json')
    args = parser.parse_args()

    input_dir_path = args.input_dir_path
    ori_path = args.ori_path
    split_data_path = args.split_data_path

    output_dir_path = os.path.join(args.output_dir_path, 'label')
    output_dir_path_seq = os.path.join(args.output_dir_path, 'label_seq')
    output_dir_path_track = os.path.join(args.output_dir_path + 'label_track')
    # Convert detection results from mmdetection3d_kitti format to KITTI format for all files in input_dir_path
    gen_kitti_result(input_dir_path, output_dir_path, ori_path, split_data_path)
    # Group converted KITTI format files by sequence
    gen_kitti_seq_result(output_dir_path, output_dir_path_seq, ori_path)
    # Group converted KITTI format files by sequence and write them into one txt file per sequence
    gen_kitti_seq_txt(output_dir_path_seq, output_dir_path_track)

    os.system("cp %s/* %s/"%(output_dir_path_track,args.output_dir_path))
    os.system("rm -rf %s"%(output_dir_path))
    os.system("rm -rf %s"%(output_dir_path_seq))
    os.system("rm -rf %s"%(output_dir_path_track))
