import os
import shutil
import json
from tqdm import tqdm

from converter_config import ConfigRCooper2DAIR


class RCOOPER2DAIR():
    """
    Convert RCOOPER to DAIR
    """
    def __init__(self, src_path_data, src_path_label, src_path_calib, dst_path):
        """
        src_path_data: str, path to data source
        src_path_label: str, path to label source
        src_path_calib: str, path to calib source
        dst_path: str, path to destination
        """
        self.src_path_data = src_path_data
        self.src_path_label = src_path_label
        self.src_path_calib = src_path_calib
        self.dst_path = dst_path
    
    def get_datainfo(self):
        """
        Get datainfo
        """
        # load the existing datainfo or initialize a new
        try:
            with open(os.path.join(self.dst_path, 'vehicle-side', 'data_info.json'), 'r') as f:
                veh_info = json.load(f)
        except:
            veh_info = []
        
        try:
            with open(os.path.join(self.dst_path, 'infrastructure-side', 'data_info.json'), 'r') as f:
                inf_info = json.load(f)
        except:
            inf_info = []
        
        return veh_info, inf_info
    
    def write_datainfo_single(self, veh_info, inf_info):
        """
        Write the datainfo for single side
        """
        with open(os.path.join(self.dst_path, 'vehicle-side', 'data_info.json'), 'w') as f:
            veh_info = json.dump(veh_info, f, indent=4)
        
        with open(os.path.join(self.dst_path, 'infrastructure-side', 'data_info.json'), 'w') as f:
            inf_info = json.dump(inf_info, f, indent=4)

    def gen_data(self, scenes=["116-115"], agent_pairs=[['116','115']], sensor_pairs=[['cam-0','cam-1']], valid_scene_seqs=None):
        """
        Convert data

        scenes: list(str), scenes list, e.g., ['116-115']
        agent_pairs: list(list(str)), which agents should be converted, e.g., [['116','115']]
        sensor_pairs: list(list(str)), which sensor of the agent should be convert, e.g., [['cam-0','cam-1']]
        valid_scene_seqs: dict(list(int)), which sequences should be convert, e.g., 
                            VALID_SCENE_SEQS = {
                                '136-137-138-139': [0,4]
                            }
        """

        print ("########## Generate Data ##########")

        # Get data info
        veh_info, inf_info = self.get_datainfo()
        veh_cnt = 0
        inf_cnt = 0

        processed_tokens = {}
        # Scene
        for scene_idx, scene_name in enumerate(scenes):
            print ("-", scene_name)
            # Agent
            for agent_id in os.listdir(os.path.join(self.src_path_data, scene_name)):
                # Only convert the agent that need to be converted
                if agent_id not in agent_pairs[scene_idx]:
                    continue
                else:
                    agent_idx = agent_pairs[scene_idx].index(agent_id)

                print ("--", agent_id)
                # Sequence
                seq_names = os.listdir(os.path.join(self.src_path_data, scene_name, agent_id))
                seq_names.sort(key=lambda x:int(x.split('-')[1]))
                for seq_id in seq_names:
                    # Only convert the sequence that need to be converted
                    if valid_scene_seqs and int(seq_id.split('-')[1]) not in valid_scene_seqs[scene_name]:
                        continue

                    print ('---', seq_id)
                    # Sensor
                    for sensor_id in tqdm(os.listdir(os.path.join(self.src_path_data, scene_name, agent_id, seq_id))):
                        # Only convert the sensor data that need to be converted
                        if sensor_id != sensor_pairs[scene_idx][agent_idx] and sensor_id != 'lidar':
                            continue
                        
                        filenames = os.listdir(os.path.join(self.src_path_data, scene_name, agent_id, seq_id, sensor_id))
                        filenames.sort(key=lambda x:float(x[:-4]))

                        v2i_mapping = 'infrastructure-side' if agent_idx else 'vehicle-side'
                        sensor_mapping = 'image' if 'cam' in sensor_id else 'velodyne'
                        dir_dst = os.path.join(self.dst_path, v2i_mapping, sensor_mapping)
                        if not os.path.exists(dir_dst):
                            os.makedirs(dir_dst)
                        
                        # Files
                        for frame_ids, filename in enumerate(filenames):
                            dir_src = os.path.join(self.src_path_data, scene_name, agent_id, seq_id, sensor_id, filename)
                            file_dst = ".jpg" if 'cam' in sensor_id else '.pcd'

                            file_dst_idx = f'{veh_cnt:0>6}' if 'veh' in v2i_mapping else f'{inf_cnt:0>6}'
                            processed_token = scene_name+'-'+str(agent_id)+'-'+str(seq_id)+'-'+str(frame_ids)
                            file_dst_idx = f'{processed_tokens[processed_token]:0>6}' if processed_token in processed_tokens.keys() else file_dst_idx
                            file_dst = file_dst_idx + file_dst
                            shutil.copy(dir_src, os.path.join(dir_dst, file_dst))

                            # Generate DataInfo
                            if 'cam' in sensor_id:
                                sensor_info = {
                                    "camera_ip": "",
                                    "camera_id": agent_id + '-' + sensor_id,
                                    "image_timestamp": filename[:-4],
                                    "image_path": f"image/{file_dst_idx}.jpg"
                                }
                            else:
                                sensor_info = {
                                    "lidar_id": agent_id + '-' + sensor_id,
                                    "pointcloud_timestamp": filename[:-4],
                                    "pointcloud_path": f"velodyne/{file_dst_idx}.pcd"
                                }

                            if 'inf' in v2i_mapping:
                                # For inf-side
                                if processed_token not in processed_tokens.keys():
                                    # Init obj token, if it is not processed
                                    processed_tokens[processed_token] = inf_cnt
                                    # Init infos
                                    temp = {
                                        "intersection_loc": scene_name,
                                        "calib_camera_intrinsic_path": f"calib/camera_intrinsic/{file_dst_idx}.json",
                                        "calib_virtuallidar_to_camera_path": f"calib/virtuallidar_to_camera/{file_dst_idx}.json",
                                        "calib_virtuallidar_to_world_path": f"calib/virtuallidar_to_world/{file_dst_idx}.json",
                                        "label_camera_std_path": f"label/virtuallidar/{file_dst_idx}.json", 
                                        "label_lidar_std_path": f"label/virtuallidar/{file_dst_idx}.json", 
                                        "frame_id": f'{frame_ids:0>6}', 
                                        "valid_frames_splits": [{"start_frame_id": "000000", "end_frame_id": f"{len(filenames)-1:0>6}"}], 
                                        "num_frames": str(len(filenames)), 
                                        "sequence_id": seq_id
                                    }
                                    temp.update(sensor_info)
                                    inf_info.append(temp)
                                    inf_cnt += 1
                                else:
                                    # Update obj token, if it is already processed
                                    inf_info[processed_tokens[processed_token]].update(sensor_info)
                            else:
                                # For veh-side
                                if processed_token not in processed_tokens.keys():
                                    # Init obj token, if it is not processed
                                    processed_tokens[processed_token] = veh_cnt
                                    # Init infos
                                    temp = {
                                        "intersection_loc": scene_name,
                                        "calib_camera_intrinsic_path": f"calib/camera_intrinsic/{file_dst_idx}.json", 
                                        "calib_lidar_to_camera_path": f"calib/lidar_to_camera/{file_dst_idx}.json", 
                                        "calib_lidar_to_novatel_path": f"calib/lidar_to_novatel/{file_dst_idx}.json", 
                                        "calib_novatel_to_world_path": f"calib/novatel_to_world/{file_dst_idx}.json", 
                                        "label_camera_std_path": f"label/lidar/{file_dst_idx}.json", 
                                        "label_lidar_std_path": f"label/lidar/{file_dst_idx}.json", 
                                        "frame_id": f'{frame_ids:0>6}', 
                                        "start_frame_id": "000000", 
                                        "end_frame_id": f"{len(filenames)-1:0>6}", 
                                        "num_frames": str(len(filenames)), 
                                        "sequence_id": seq_id
                                    }
                                    temp.update(sensor_info)
                                    veh_info.append(temp)
                                    veh_cnt += 1
                                else:
                                    # Update obj token, if it is already processed
                                    veh_info[processed_tokens[processed_token]].update(sensor_info)
        # Save  
        self.write_datainfo_single(veh_info, inf_info)
    
    def gen_calib(self):
        """
        Convert Calibration
        """
        print ("########## Generate Calib ##########")
        # For veh-side
        print ('--- vehicle-side')
        with open(os.path.join(self.dst_path, 'vehicle-side', 'data_info.json'), 'r') as f:
            # load info files
            infos = json.load(f)
            # Init directory
            dirs = ["calib", 'calib/camera_intrinsic', 'calib/lidar_to_camera', 'calib/lidar_to_novatel', 'calib/novatel_to_world']
            for dir in dirs:
                dir_path = os.path.join(self.dst_path, 'vehicle-side', dir)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

            for info in tqdm(infos):
                agent_id = info['lidar_id'].split('-')[0]

                # Cam
                cam_id = info['camera_id'].split('-')[-1]
                with open(os.path.join(self.src_path_calib, 'lidar2cam', agent_id+".json"), 'r') as f_calib:
                    mat = json.load(f_calib)
                    cam_int = mat["cam_"+cam_id]["intrinsic"]
                    cam_ext_mat = mat["cam_"+cam_id]["extrinsic"]
                cam_K = []
                for i in cam_int:
                    cam_K.extend(i)
                cam_int = {
                    "cam_K": cam_K
                }
                cam_ext = {
                    "translation": [[cam_ext_mat[0][3]], [cam_ext_mat[1][3]], [cam_ext_mat[2][3]],],
                    "rotation": [cam_ext_mat[0][:3], cam_ext_mat[1][:3], cam_ext_mat[2][:3]]
                }
                with open(os.path.join(self.dst_path, 'vehicle-side', info["calib_camera_intrinsic_path"]), 'w') as f_calib:
                    json.dump(cam_int, f_calib, indent=4)
                with open(os.path.join(self.dst_path, 'vehicle-side', info["calib_lidar_to_camera_path"]), 'w') as f_calib:
                    json.dump(cam_ext, f_calib, indent=4)
                
                # Lidar2novatel
                with open(os.path.join(self.dst_path, 'vehicle-side', info["calib_lidar_to_novatel_path"]), 'w') as f_calib:
                    temp = {
                        "transform":{
                            "rotation": [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
                            "translation": [[0.0], [0.0], [0.0]]
                        }
                    }
                    json.dump(temp, f_calib, indent=4)
                
                # Novatel2World
                with open(os.path.join(self.src_path_calib, 'lidar2world', agent_id+".json"), 'r') as f_calib:
                    mat = json.load(f_calib)
                with open(os.path.join(self.dst_path, 'vehicle-side', info["calib_novatel_to_world_path"]), 'w') as f_calib:
                    temp = {
                        "transform":{
                            "rotation": mat['rotation'],
                            "translation": [[mat['translation'][0]], [mat['translation'][1]], [mat['translation'][2]]]
                        }
                    }
                    json.dump(temp, f_calib, indent=4)
        
        # For inf-side
        print ('--- infrastructure-side')
        with open(os.path.join(self.dst_path, 'infrastructure-side', 'data_info.json'), 'r') as f:
            # load info files
            infos = json.load(f)
            # Init directory
            dirs = ["calib", 'calib/camera_intrinsic', 'calib/virtuallidar_to_camera', 'calib/virtuallidar_to_world']
            for dir in dirs:
                dir_path = os.path.join(self.dst_path, 'infrastructure-side', dir)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

            for info in tqdm(infos):
                agent_id = info['lidar_id'].split('-')[0]

                # Cam
                cam_id = info['camera_id'].split('-')[-1]
                with open(os.path.join(self.src_path_calib, 'lidar2cam', agent_id+".json"), 'r') as f_calib:
                    mat = json.load(f_calib)
                    cam_int = mat["cam_"+cam_id]["intrinsic"]
                    cam_ext_mat = mat["cam_"+cam_id]["extrinsic"]
                cam_K = []
                for i in cam_int:
                    cam_K.extend(i)
                cam_int = {
                    "cam_K": cam_K
                }
                cam_ext = {
                    "translation": [[cam_ext_mat[0][3]], [cam_ext_mat[1][3]], [cam_ext_mat[2][3]],],
                    "rotation": [cam_ext_mat[0][:3], cam_ext_mat[1][:3], cam_ext_mat[2][:3]]
                }
                with open(os.path.join(self.dst_path, 'infrastructure-side', info["calib_camera_intrinsic_path"]), 'w') as f_calib:
                    json.dump(cam_int, f_calib, indent=4)
                with open(os.path.join(self.dst_path, 'infrastructure-side', info["calib_virtuallidar_to_camera_path"]), 'w') as f_calib:
                    json.dump(cam_ext, f_calib, indent=4)
                
                # Lidar2world           
                with open(os.path.join(self.src_path_calib, 'lidar2world', agent_id+".json"), 'r') as f_calib:
                    mat = json.load(f_calib)
                with open(os.path.join(self.dst_path, 'infrastructure-side', info["calib_virtuallidar_to_world_path"]), 'w') as f_calib:
                    temp = {
                        "transform":{
                            "rotation": mat['rotation'],
                            "translation": [[mat['translation'][0]], [mat['translation'][1]], [mat['translation'][2]]]
                        }
                    }
                    json.dump(temp, f_calib, indent=4)

    def update_info(self, info_path):
        """
        data info updating

        info_path: path to data info
        """
        # load the raw data info
        with open(info_path, 'r') as f:
            infos = json.load(f)

        # init
        start_id = -1
        end_id = -1
        seq_id = None

        # id update
        for info in infos:
            if info["sequence_id"] != seq_id:
                start_id = end_id + 1
                end_id = start_id + int(info["num_frames"]) - 1
                seq_id = info["sequence_id"]
            
            info["frame_id"] = info["image_path"].split("/")[1].split('.')[0]
            info["start_frame_id"] = f'{start_id:0>6}'
            info["end_frame_id"] = f'{end_id:0>6}'
            info["sequence_id"] = "0" + f"{seq_id.split('-')[-1]:0>4}"[1:] if info["intersection_loc"] in ['106-105', '117-118-120-119'] else "1" + f"{seq_id.split('-')[-1]:0>4}"[1:]
        
        with open(info_path, 'w') as f:
            json.dump(infos, f, indent=4)
    
    def update_scene_infos(self):
        """
        Update the scene info
        """

        agent_names = ['infrastructure-side', 'vehicle-side']

        for agent_name in agent_names:
            info_path = os.path.join(self.dst_path, agent_name, 'data_info.json')
            self.update_info(info_path)

        # specific operation for infrastructure-side info
        with open(os.path.join(self.dst_path, 'infrastructure-side', 'data_info.json'), 'r') as f:
            infos = json.load(f)
        
        for info in infos:
            info["valid_frames_splits"][0]["start_frame_id"] = info["start_frame_id"]
            info["valid_frames_splits"][0]["end_frame_id"] = info["end_frame_id"]
            info.pop("start_frame_id")
            info.pop("end_frame_id")
        
        with open(os.path.join(self.dst_path, 'infrastructure-side', 'data_info.json'), 'w') as f:
            json.dump(infos, f, indent=4)
        
    def gen_label(self, val_seqs):
        """
        Convert labels
        """

        print ("########## Generate Labels ##########")
        # for veh-side
        print ('--- vehicle-side')
        with open(os.path.join(self.dst_path, 'vehicle-side', 'data_info.json'), 'r') as f:
            # load infos
            infos = json.load(f)
            # init directory
            dirs = ["label", 'label/lidar']
            for dir in dirs:
                dir_path = os.path.join(self.dst_path, 'vehicle-side', dir)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

            for info in tqdm(infos):
                scene_id = info['intersection_loc']
                agent_id = info['lidar_id'].split('-')[0]
                seq_id = info['sequence_id']
                frame_id = info['frame_id']
                filename = info['pointcloud_timestamp'] + ".json"

                dst_filename = info['label_lidar_std_path']

                with open(os.path.join(self.src_path_label, scene_id, agent_id, seq_id, filename), 'r') as f_label:
                    annos = json.load(f_label)

                annos_dair = []
                for anno in annos:

                    anno_dair = {
                        "token": scene_id+'-'+seq_id+'-'+frame_id+'-'+agent_id+'-'+str(anno['track_id']),
                        "type": anno['type'],
                        'track_id': str(anno['track_id']),
                        'truncated_state': anno['truncated_state'],
                        'occluded_state': anno['occluded_state'],
                        'alpha': -1,
                        '2d_box': None,
                        '3d_dimensions': anno['3d_dimensions'],
                        '3d_location': anno['3d_location'],
                        'rotation': anno['rotation']
                    }

                    annos_dair.append(anno_dair)
                    
                with open(os.path.join(self.dst_path, 'vehicle-side', dst_filename), 'w') as f_label:
                    json.dump(annos_dair, f_label, indent=4)
        
        # for inf-side
        print ('--- infrastructure-side')
        with open(os.path.join(self.dst_path, 'infrastructure-side', 'data_info.json'), 'r') as f:
            # load infos
            infos = json.load(f)
            # init directory
            dirs = ["label", 'label/virtuallidar']
            for dir in dirs:
                dir_path = os.path.join(self.dst_path, 'infrastructure-side', dir)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

            for info in tqdm(infos):
                scene_id = info['intersection_loc']
                agent_id = info['lidar_id'].split('-')[0]
                seq_id = info['sequence_id']
                frame_id = info['frame_id']
                filename = info['pointcloud_timestamp'] + ".json"

                dst_filename = info['label_lidar_std_path']

                with open(os.path.join(self.src_path_label, scene_id, agent_id, seq_id, filename), 'r') as f_label:
                    annos = json.load(f_label)

                annos_dair = []
                for anno in annos:

                    anno_dair = {
                        "token": scene_id+'-'+seq_id+'-'+frame_id+'-'+agent_id+'-'+str(anno['track_id']),
                        "type": anno['type'],
                        'track_id': str(anno['track_id']),
                        'truncated_state': anno['truncated_state'],
                        'occluded_state': anno['occluded_state'],
                        'alpha': -1,
                        '2d_box': None,
                        '3d_dimensions': anno['3d_dimensions'],
                        '3d_location': anno['3d_location'],
                        'rotation': anno['rotation']
                    }

                    annos_dair.append(anno_dair)
                    
                with open(os.path.join(self.dst_path, 'infrastructure-side', dst_filename), 'w') as f_label:
                    json.dump(annos_dair, f_label, indent=4)
        
        # for cooperative
        print ('--- cooperative')
        with open(os.path.join(self.dst_path, 'vehicle-side', 'data_info.json'), 'r') as f:
            # load infos
            infos = json.load(f)
            # init directory
            dirs = ["cooperative", "cooperative/label"]
            for dir in dirs:
                dir_path = os.path.join(self.dst_path, dir)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

            data_infos = []
            split_infos = dict(batch_split=dict(train=list(),val=list(),test=list()),\
                               vehicle_split=dict(train=list(),val=list(),test=list()),\
                               infrastructure_split=dict(train=list(),val=list(),test=list()))

            for info in tqdm(infos):
                scene_id = info['intersection_loc']
                seq_id = info['sequence_id']
                frame_id = info['frame_id']
                filename = info['pointcloud_timestamp'] + ".json"

                dst_filename = info['label_lidar_std_path'].split('/')[-1]

                with open(os.path.join(self.src_path_label, scene_id, 'coop', seq_id, filename), 'r') as f_label:
                    annos = json.load(f_label)

                annos_dair = []
                for anno in annos:

                    anno_dair = {
                        "token": scene_id+'-'+seq_id+'-'+frame_id+'-coop-'+str(anno['track_id']),
                        "type": anno['type'],
                        'track_id': str(anno['track_id']),
                        'truncated_state': anno['truncated_state'],
                        'occluded_state': anno['occluded_state'],
                        'alpha': -1,
                        '2d_box': None,
                        '3d_dimensions': anno['3d_dimensions'],
                        '3d_location': anno['3d_location'],
                        'rotation': anno['rotation']
                    }

                    annos_dair.append(anno_dair)
                    
                with open(os.path.join(self.dst_path, 'cooperative/label', dst_filename), 'w') as f_label:
                    json.dump(annos_dair, f_label, indent=4)

                # Cooperative data_info
                data_id = dst_filename.split('.')[0]
                seq_id = f"{seq_id.split('-')[-1]:0>4}"
                seq_id_w_scene = "0" + seq_id[1:] if scene_id in ['106-105', '117-118-120-119'] else "1" + seq_id[1:]
                data_info = {
                    "vehicle_frame": data_id,
                    "infrastructure_frame": data_id,
                    "vehicle_sequence": seq_id_w_scene, 
                    "infrastructure_sequence": seq_id_w_scene,
                    "system_error_offset": {"delta_x": 0.0, "delta_y": 0.0}
                }
                data_infos.append(data_info)

                # Split infos
                split = "val" if int(seq_id) in val_seqs[scene_id] else "train"
                if seq_id_w_scene not in split_infos['batch_split'][split]:
                    split_infos['batch_split'][split].append(seq_id_w_scene)
                
                split_infos['vehicle_split'][split].append(data_id)
                split_infos['infrastructure_split'][split].append(data_id)
        
        with open(os.path.join(self.dst_path, 'cooperative/data_info.json'), 'w') as f_info:
            json.dump(data_infos, f_info, indent=4)
        
        split_infos['batch_split']['train'] = sorted(split_infos['batch_split']['train'])
        split_infos['batch_split']['val'] = sorted(split_infos['batch_split']['val'])
        split_infos['vehicle_split']['train'] = sorted(split_infos['vehicle_split']['train'])
        split_infos['vehicle_split']['val'] = sorted(split_infos['vehicle_split']['val'])
        split_infos['infrastructure_split']['train'] = sorted(split_infos['infrastructure_split']['train'])
        split_infos['infrastructure_split']['val'] = sorted(split_infos['infrastructure_split']['val'])

        with open(os.path.join(self.dst_path, 'split-data.json'), 'w') as f_split:
            json.dump(split_infos, f_split, indent=4)
        

###########################################################################
#   SOURCE INFORMATION
###########################################################################
SRC_PATH = ConfigRCooper2DAIR['src_path']['data']
SRC_PATH_LABEL = ConfigRCooper2DAIR['src_path']['label']
SRC_PATH_CALIB = ConfigRCooper2DAIR['src_path']['calib']

# ###########################################################################
# #   CORRIDOR SCENES
# ###########################################################################
DST_PATH_DATA = ConfigRCooper2DAIR['dst_path']['corridor']
SCENES = ['106-105', '116-115']
AGENT_PAIRS = [['106', '105'], ['116','115']]
SENSOR_PAIRS = [['cam-1','cam-0'], ['cam-0','cam-1']]
VALID_SCENE_SEQS = {
    '106-105': list(range(75)),
    '116-115': list(range(171))
}
VAL_SPLIT = {
    '106-105': [1,6,9,20,22,27,31,33,40,44,51,56,65,68],
    '116-115': [1,3,8,11,12,17,20,23,26,28,31,45,61,69,70,75,83,84,89,91,96,101,107,117,118,124,130,140,141,155,159,165,166,170]
}

convertor = RCOOPER2DAIR(SRC_PATH, SRC_PATH_LABEL, SRC_PATH_CALIB, DST_PATH_DATA)
convertor.gen_data(SCENES, AGENT_PAIRS, SENSOR_PAIRS, VALID_SCENE_SEQS)
convertor.gen_calib()
convertor.gen_label(VAL_SPLIT)
convertor.update_scene_infos()


###########################################################################
#   INTERSECTION SCENES
###########################################################################
DST_PATH_DATA = ConfigRCooper2DAIR['dst_path']['intersection']
SCENES = ['117-118-120-119', '136-137-138-139']
AGENT_PAIRS = [['117','120'], ['136', '139']]
SENSOR_PAIRS = [['cam-0','cam-0'], ['cam-0','cam-0']]
VALID_SCENE_SEQS = {
    '117-118-120-119': list(range(34)),
    '136-137-138-139': list(range(11))
}
VAL_SPLIT = {
    '117-118-120-119': [2,5,12,15,21,31,32],
    '136-137-138-139': [0,4]
}

convertor = RCOOPER2DAIR(SRC_PATH, SRC_PATH_LABEL, SRC_PATH_CALIB, DST_PATH_DATA)
convertor.gen_data(SCENES, AGENT_PAIRS, SENSOR_PAIRS, VALID_SCENE_SEQS)
convertor.gen_calib()
convertor.gen_label(VAL_SPLIT)
convertor.update_scene_infos()
