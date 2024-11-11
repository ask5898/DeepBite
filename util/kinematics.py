import pandas as pd
import numpy as np
import os
from vame.util.vame_utils import *
from vame.util.auxiliary import read_config
from pathlib import Path
import glob
import tqdm
import dlc2kinematics as d2k
from .info import *

def get_file_paths(config, mode) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    path_to_file = cfg['project_path']
    files = cfg['video_sets']
    #read_path = os.path.join(path_to_file, 'annotatedTracks')
    file_paths = [f for file in files for f in glob.glob(os.path.join(path_to_file, 'videos', 'pose_estimation', file+'.'+mode))]
    #file_paths = [file for file in glob.glob(os.path.join(read_path, '*.h5'))]

    return file_paths

def dummy2(config) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    path_to_file = cfg['project_path']
    cols = get_cols()[:-3]
    data = pd.read_csv('/mnt/DATA1/ali/BitoScopeVAME_new/BiteScopeWithVAME-Mar10-2023/testing/annotationData2-full.csv', dtype=object)
    data = data.values[2:,1:]
 
    df = pd.DataFrame(data=data, columns=cols)
    print(df)
    df.to_hdf('/mnt/DATA1/ali/BitoScopeVAME_new/BiteScopeWithVAME-Mar10-2023/testing/annotationData2-full.h5', "df_with_missing",format="table",mode="w")


def saving_csv_as_hdf(config) :
    file_paths = get_file_paths(config, mode='csv') 
    cols = get_cols()[:-3]
    smoothing_window = 5
    for file_path in file_paths :
        data = pd.read_hdf(file_path)
        data = np.asarray(data)
        #print(data)
        #frames = data[:,:1]
        #data = data[:,1:]
        pose_list = []
        for i in range(int(data.shape[1]/3)):
            pose_list.append(data[:,i*3:(i+1)*3])
        
        for i in pose_list:
            for j in i:
                if j[2] <= confidence:
                    j[0],j[1] = np.nan, np.nan        

        for i in pose_list:
                i = interpol(i)
 
        position = np.concatenate(pose_list, axis=1)
        smoothed_position = np.zeros((position.shape[1], position.shape[0]-(smoothing_window-1)))
        idx = 0
        position = position.T
        for i in range(int(position.shape[0]/3)) :
            smoothed_position[idx] = moving_average(position[idx])
            smoothed_position[idx+1] = moving_average(position[idx+1])
            smoothed_position[idx+2] = position[idx+2][: -(smoothing_window-1)]
            idx+=3

        smoothed_position = smoothed_position.T
        df = pd.DataFrame(data=smoothed_position,columns=cols)
        #df_frame = pd.DataFrame(data = np.squeeze(frames[:-(smoothing_window-1)]), columns=['Frame'])
      
        print('Saving H5 files for %s'%Path(file_path).resolve())
        df.to_hdf(file_path[:-4]+'.h5', "df_with_missing",format="table",mode="w")
        #df_frame.to_hdf(file_path[:-4]+'-frames.h5', "df_with_missing",format="table",mode="w")
        
def get_cols() :
    a = pd.read_hdf('/mnt/DATA1/ali/BitoScopeVAME_new/BiteScopeWithVAME-Mar10-2023/h5_output/201105_KPPTN_ctrl1_2DLC_dlcrnetms5_aedesJun22shuffle1_72500_full/201105_KPPTN_ctrl1_2DLC_dlcrnetms5_aedesJun22shuffle1_72500_el_00057.h5')
    return a.columns.droplevel(1)

def get_velocity(config) :
    file_paths = get_file_paths(config,mode='h5')
    #file_paths = ['/mnt/DATA1/ali/BitoScopeVAME_new/BiteScopeWithVAME-Mar10-2023/testing/annotationData2-full.h5']
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    test_frac = cfg['test_fraction']
    velocity = []
    bodyparts = ['thorax']
    #train = np.load(os.path.join(cfg['project_path'], 'data', 'train', 'train_seq.npy'))
    #test = np.load(os.path.join(cfg['project_path'], 'data', 'train', 'test_seq.npy'))
    #num_frames = train.shape[1] + test.shape[1]
    for file_path in file_paths :
        df, _, _ = d2k.load_data(file_path)
        print(df)
        print(type(df))
        file = file_path.split('/')[-1][:-3]
        remove = np.load(os.path.join(cfg['project_path'], 'marked videos', file+'_to_remove.npy'))
        df_vel = d2k.compute_speed(df, bodyparts=bodyparts)
        vel_mat = np.asarray(df_vel).T[: - len(bodyparts)]
        rem = []
        for i in remove :
            rem.extend(range(i[0], i[1]))

        vel_mat = np.delete(vel_mat, rem, axis=1)
        velocity.append(vel_mat)

    return 0
    velocity = np.concatenate(velocity, axis=1)
    frac = int(velocity.shape[1] * test_frac)
    np.save(os.path.join(cfg['project_path'],"data", 'train','test_velocity_seq.npy'), velocity[:, :frac])
    np.save(os.path.join(cfg['project_path'],"data", 'train','train_velocity_seq.npy'), velocity[:, frac:])
    print( velocity[:, frac:].shape)
    return velocity
    
def get_joint_angular_velocity(config) :
    file_paths = get_file_paths(config,mode='h5') 
    angular_velocity = []
    arm_ang = []
    joints_dict = {}
    joints_dict_ang = {}
    joints_dict['FR-leg'] = ['thorax','rightForeLeg3','rightForeLeg1']
    joints_dict['FL-leg'] = ['thorax','leftForeLeg3','leftForeLeg1']
    joints_dict_ang['Left-leg'] = ['head', 'thorax', 'leftForeLeg1']
    joints_dict_ang['Right-leg'] = ['head', 'thorax', 'rightForeLeg1']
    for file_path in file_paths :
        df, _ ,_ = d2k.load_data(file_path)
        #print(df)
        joint_angles_fr = d2k.compute_joint_angles(df, joints_dict, save=False)
        joint_angles_arms = np.asarray(d2k.compute_joint_angles(df, joints_dict_ang, save=False))
        joint_angles_arms = interpol(joint_angles_arms)
        joint_speed = d2k.compute_joint_velocity(joint_angles_fr)
        joint_speed = np.asarray(joint_speed)
        joint_speed = interpol(joint_speed)
        angular_velocity.append(joint_speed)
        arm_ang.append(joint_angles_arms)
    return angular_velocity, arm_ang

def add_kinematics_to_egocentric(config, path=False) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    path_to_file = cfg['project_path']
    files = cfg['video_sets']
    if path :
        file_paths = [f for file in files for f in glob.glob(os.path.join(path_to_file,'data', file, '*-PE-seq.npy'))]
    else :
         file_paths = [f for file in files for f in glob.glob(os.path.join(path_to_file,'annotatedTracks', '*.h5'))]
        
    velocity = get_velocity(config)
    ref = get_joint_angular_velocity(config)
    belly_size = get_length_between_bodyparts(config, bodyparts=['abdomenL','abdomenR'])
    body_length = get_length_between_bodyparts(config, bodyparts=['head','bottom'])
    proboscis_length = get_length_between_bodyparts(config, bodyparts=['proboscis1','proboscis3'])
    leg_distance = get_length_between_bodyparts(config, bodyparts=['rightForeLeg1','leftForeLeg1'])
    angular_vel = ref[0]
    arm_ang = ref[1]
    for idx,file_path in enumerate(file_paths) :
        #data = np.load(file_path)
        #print(data, data.shape)

        ang_vel_FR = angular_vel[idx][:,:1]
        ang_vel_FL = angular_vel[idx][:,1:]
        ang_FR = arm_ang[idx][:,:1]
        ang_FL = arm_ang[idx][:,1:]
        avg_ang = (ang_FL+ang_FR)/2
        vel_thorax = np.expand_dims(velocity[idx][0], axis=1)
        vel_rfl = np.expand_dims(velocity[idx][1], axis=1)
        vel_lfl = np.expand_dims(velocity[idx][2], axis=1)
        avg_vel = (vel_lfl+vel_rfl)/2
        belly_s = np.expand_dims(belly_size[idx], axis=1)
        prob_len = np.expand_dims(proboscis_length[idx], axis=1)
        body_len = np.expand_dims(body_length[idx], axis=1)
        leg_d = np.expand_dims(leg_distance[idx], axis=1)
        final_data = np.concatenate([vel_thorax, avg_vel, ang_FR, ang_FL, belly_s, body_len],axis=1)

        np.save(file_path[:-3],final_data.T)

    


def get_length_between_bodyparts(config, bodyparts) :
    file_paths = get_file_paths(config,mode='h5')
    if len(bodyparts) != 2 :
        raise Exception('Please enter only two bodyparts')
    
    length = []
    for file_path in file_paths :
        data = pd.read_hdf(file_path)
        coords = []
        for bodypart in bodyparts :
            bpt_pos = [idx for idx,bpt in enumerate(list(data.columns.get_level_values('bodyparts'))) if bpt == bodypart]
            coords.append(data.values[:,bpt_pos[0]:bpt_pos[1]+1].T)

        x_diff = coords[0][0] - coords[1][0]
        y_diff = coords[0][1] - coords[1][1]
        dist = np.sqrt(x_diff**2 + y_diff**2)
        length.append(dist)

    return np.asarray(length)