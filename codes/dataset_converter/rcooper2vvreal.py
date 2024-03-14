import os
import shutil
import json
import yaml
import numpy as np
from tqdm import tqdm

from converter_config import ConfigRCooper2VVReal

class RCOOPER2VVREAL():
    """
    Convert RCOOPER to V2V4Real
    """
    def __init__(self, src_path_data, src_path_label, src_path_calib, dst_path_data):
        """
        src_path_data: str, path to data source
        src_path_label: str, path to label source
        src_path_calib: str, path to calib source
        dst_path_data: str, path to data destination
        """
        self.src_path_data = src_path_data
        self.src_path_label = src_path_label
        self.src_path_calib = src_path_calib
        self.dst_path_data = dst_path_data

        self.sensor_id_mapping = {
            'lidar': ".pcd",
            'cam-0': "_camera0.jpg",
            'cam-1': "_camera1.jpg"
        }

    def gen_data(self, scenes):
        """
        Convert data

        scenes: list(str), scenes list, e.g., ['116-115', '106-105', '117-118-120-119', '136-137-138-139']
        """

        print ("########## Generate Data ##########")
        # Scene
        for scene_name in scenes:
            print ("-", scene_name)
            # Agent
            for agent_id in os.listdir(os.path.join(self.src_path_data, scene_name)):
                print ("--", agent_id)
                # Sequence
                for seq_id in os.listdir(os.path.join(self.src_path_data, scene_name, agent_id)):
                    print ('---', seq_id)
                    # Sensor
                    for sensor_id in tqdm(os.listdir(os.path.join(self.src_path_data, scene_name, agent_id, seq_id))):
                        # Get filenames
                        filenames = os.listdir(os.path.join(self.src_path_data, scene_name, agent_id, seq_id, sensor_id))
                        filenames.sort(key=lambda x:float(x[:-4]))

                        dir_dst = os.path.join(self.dst_path_data, scene_name+'_'+seq_id, str(scene_name.split('-').index(agent_id)))
                        if not os.path.exists(dir_dst):
                            os.makedirs(dir_dst)
                        
                        # files copy
                        for idx, filename in enumerate(filenames):

                            name_dst = f'{idx:0>5}'+self.sensor_id_mapping[sensor_id]
                            dir_src = os.path.join(self.src_path_data, scene_name, agent_id, seq_id, sensor_id, filename)
                            shutil.copy(dir_src, os.path.join(dir_dst, name_dst))
    
    def json2yaml(self, json_path, yaml_path, agent_id):
        """
        Convert json to yaml

        json_path: path to json file
        yaml_path: path to yaml file
        agent_id: id of the agent
        """
        # Get calibration
        calib_file = os.path.join(self.src_path_calib, 'lidar2world', agent_id) + ".json"
        with open(calib_file, 'r') as f:
            calib = json.load(f)
        l2w = np.eye(4)
        l2w[:3, :3], l2w[:3, 3] = calib['rotation'], calib['translation']

        # Yaml initialization
        r_yaml = {
            'ego_speed': 0,
            'gps': [],
            'lidar_pose': np.array(l2w),
            'true_ego_pose': np.array(l2w),
            'vehicles': {}
        }

        # Get annotations
        with open(json_path, 'r') as f:
            annos = json.load(f)

        # Convert annotations
        for anno in annos:
            r_yaml['vehicles'][anno['track_id']] = {
                'angle': [0.0, anno['rotation']*180/np.pi, 0.0],
                'center': [0, 0, 0],
                'extent': [anno['3d_dimensions']['l']/2, anno['3d_dimensions']['w']/2, anno['3d_dimensions']['h']/2],
                'location': [anno['3d_location']['x'], anno['3d_location']['y'], anno['3d_location']['z']],
                'object_type': anno['type']
            }
        
        # Save Yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(r_yaml, f, default_flow_style=False)
    
    def gen_label(self, scenes):
        """
        Convert Labels

        scenes: list(str), scenes list, e.g., ['116-115', '106-105', '117-118-120-119', '136-137-138-139']
        """

        print ("########## Generate Label ##########")
        # Scene
        for scene_name in scenes:
            print ("-", scene_name)
            dir_label = os.path.join(self.src_path_label, scene_name, 'coop')
            # Sequence
            for seq_id in os.listdir(dir_label):
                print ('--', seq_id)
                filenames = os.listdir(os.path.join(dir_label, seq_id))
                filenames.sort(key=lambda x:float(x[:-5]))

                # Agent
                agent_ids = scene_name.split('-')
                for agent_id in agent_ids:
                    print ("---", agent_id)

                    # get source files
                    dir_label = os.path.join(self.src_path_label, scene_name, agent_id)
                    filenames = os.listdir(os.path.join(dir_label, seq_id))
                    filenames.sort(key=lambda x:float(x[:-5]))

                    dir_dst = os.path.join(self.dst_path_data, scene_name+'_'+seq_id, str(scene_name.split('-').index(agent_id)))
                    if not os.path.exists(dir_dst):
                        os.makedirs(dir_dst)
                    
                    # convert labels to yaml format
                    for idx, filename in tqdm(enumerate(filenames)):

                        name_dst = f'{idx:0>5}.yaml'
                        json_path = os.path.join(dir_label, seq_id, filename)

                        self.json2yaml(json_path, os.path.join(dir_dst, name_dst), agent_id)
    

    def gen_subset(self, val_split):
        """
        Generate subset

        val_split: dict, split info, e.g.,
            VAL_SPLIT = {
                "106-105": [1,6,9,20,22,27,31,33,40,44,51,56,65,68],
                "116-115": [1,3,8,11,12,17,20,23,26,28,31,45,61,69,70,75,83,84,89,91,96,101,107,117,118,124,130,140,141,155,159,165,166,170],
                "117-118-120-119": [2,5,12,15,21,31,32],
                "136-137-138-139": [0,4]
            }
        """

        val_seqs = []
        for scene in val_split.keys():
            seqs = val_split[scene]
            for seq in seqs:
                val_seqs.append(scene+'_seq-'+str(seq))

        for i in os.listdir(self.dst_path_data):
            # if i in ['intersection', 'corridor']: continue
            i_w_0 = i.split('_')[0] + '_seq-' + f"{int(i.split('-')[-1]):0>3}"

            if i in val_seqs:
                if len(i.split('_')[0].split('-')) > 3:
                    shutil.move(os.path.join(self.dst_path_data, i), os.path.join(self.dst_path_data, 'intersection', 'val', i_w_0))
                else:
                    shutil.move(os.path.join(self.dst_path_data, i), os.path.join(self.dst_path_data, 'corridor', 'val', i_w_0))
            else:
                if len(i.split('_')[0].split('-')) > 3:
                    shutil.move(os.path.join(self.dst_path_data, i), os.path.join(self.dst_path_data, 'intersection', 'train', i_w_0))
                else:
                    shutil.move(os.path.join(self.dst_path_data, i), os.path.join(self.dst_path_data, 'corridor', 'train', i_w_0))


###########################################################################
#   SRC INFORMATION
###########################################################################
SRC_PATH = ConfigRCooper2VVReal['src_path']['data']
SRC_PATH_LABEL = ConfigRCooper2VVReal['src_path']['label']
SRC_PATH_CALIB = ConfigRCooper2VVReal['src_path']['calib']

###########################################################################
#   CONVERTOR INFORMATION
###########################################################################
DST_PATH_DATA = ConfigRCooper2VVReal['dst_path']

SCENES = ['116-115', '106-105', '117-118-120-119', '136-137-138-139']
VAL_SPLIT = {
    "106-105": [1,6,9,20,22,27,31,33,40,44,51,56,65,68],
    "116-115": [1,3,8,11,12,17,20,23,26,28,31,45,61,69,70,75,83,84,89,91,96,101,107,117,118,124,130,140,141,155,159,165,166,170],
    "117-118-120-119": [2,5,12,15,21,31,32],
    "136-137-138-139": [0,4]
}

convertor = RCOOPER2VVREAL(SRC_PATH, SRC_PATH_LABEL, SRC_PATH_CALIB, DST_PATH_DATA)
convertor.gen_data(SCENES)
convertor.gen_label(SCENES)
convertor.gen_subset(VAL_SPLIT)