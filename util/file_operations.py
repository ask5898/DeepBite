import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from pathlib import Path
import os
import re
import copy
import shutil
import glob
from vame.util.auxiliary import read_config
from .vame_utils import *
from .info import *


def get_exclude_pos() :
    pos = []
    for exc in exclude :
        pos.extend(bodyparts[exc])

    return pos

def append_csv(config) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    path_to_file = cfg['project_path']
    path_to_csv = os.path.join(path_to_file, 'csv_folder')
    to_exclude = get_exclude_pos()
    file_exclude = exclude_short_file_length(config)
    for folder in os.listdir(path_to_csv) :
        concat_files = csv_file_append = pd.DataFrame()
        len_sum = 0
        path_name = os.path.join(path_to_csv,str(folder))
        file_len = len(os.listdir(path_name))
        files = os.listdir(path_name)
        files.sort(key=lambda x : int(x.split('.')[0][-5:]))
        for i,file in enumerate(files) :
            if not file.startswith('.') and file.endswith('.csv') and str(Path(file).resolve()).split('.')[0] not in file_exclude:
                csv_file = pd.read_csv(os.path.join(path_to_csv,str(folder),file),dtype='unicode',skiprows=[1], usecols=[i for i in range(0,col_len) if i not in to_exclude])
                len_sum = len_sum + int(csv_file.shape[0])	
                if i == 0 :
                    concat_files = csv_file
                else :
                    csv_file_append = csv_file[2:]	
                    
                concat_files = pd.concat([concat_files, csv_file_append], ignore_index=False)
        assert len(concat_files) == len_sum - (file_len-1)*2 , 'Not all values were merged for' + str(files[:-12] + 'full')
        concat_files.to_csv(os.path.join(path_to_file,'videos','pose_estimation',str(file[:-12]) +'full.csv'),index=False)


def h5tocsv(config) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    path_to_file = cfg['project_path']
    path_to_h5 = os.path.join(path_to_file, 'h5_output')

    for folder in os.listdir(path_to_h5) :
        for files in os.listdir(os.path.join(path_to_h5,str(folder))) :
            if files.endswith('.h5') :
                df = pd.read_hdf(os.path.join(path_to_h5,str(folder),files))
                save_path = str(files[:-3]) + '.csv'
                df.to_csv(os.path.join(path_to_file,'csv_folder',str(folder),save_path))
	


def save_individual_tracklets(pkl, min_length, **kwargs):
    saveDir = kwargs.get('saveDir', None)
    allTracklets = pd.read_pickle(pkl)
    header = allTracklets.pop('header')
    scorer = header.get_level_values("scorer").unique().to_list()
    bpts = header.get_level_values("bodyparts").unique().to_list()
    coords = header.get_level_values("coords").unique().to_list()
    animal_names = ['ind1']

    columns = pd.MultiIndex.from_product(
                [scorer, animal_names, bpts, coords],
                names=["scorer", "individuals", "bodyparts", "coords"],
            )

    get_frame_ind = lambda s: int(re.findall(r"\d+", s)[0])

    for k, tracklet in allTracklets.items():
        if len(tracklet) > min_length - 1:
            inds, temp = zip(*[(get_frame_ind(k), v) for k, v in tracklet.items()])
            data = np.asarray(temp, dtype=np.float16)
            flatData = data[:, :, 0:3].reshape((np.shape(data)[0], -1))
            df = pd.DataFrame(flatData, columns=columns, index=inds)
            if saveDir:
                output_name = saveDir + os.path.split(pkl)[1][:-7] + '_' + str('%05d' % k) + '.h5'
            else:
                output_name = pkl[:-7] + '_' + str('%05d' % k) + '.h5'
            df.to_hdf(output_name, "df_with_missing", format="table", mode="w")

def sort_community_videos(config) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    path_to_file = cfg['project_path']
    videos = cfg['video_sets'][:8]

    for video in videos :
        path_to_vid = os.path.join(path_to_file, 'community videos', video)
        community_videos = os.listdir(path_to_vid)
        for c in community_videos :
            suffix_len = len(str(Path(c).resolve().suffix))
            community_label = c.split('community_')[1][ : -suffix_len]
            src = os.path.join(path_to_vid,c)
            dest = os.path.join(path_to_file,'Communities sorted','community'+str(community_label),c)
            if not os.path.exists(os.path.join(path_to_file, 'Communities sorted', 'community'+str(community_label))) :
                os.makedirs(os.path.join(path_to_file, 'Communities sorted', 'community'+str(community_label)))

            shutil.copyfile(src,dest)


def exclude_short_file_length(config) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    path_to_file = cfg['project_path']
    videos = cfg['video_sets']
    video_path = [glob.glob(os.path.join(path_to_file,'csv_folder',video,'*.csv')) for video in videos ]
    remove_file = []
    for path in video_path :
        experiment_file = path[0].split('/')[-2]
        #print('\nFor file %s\n'%experiment_file)
        length = dict()
        for v_path in path :
            sub_file = v_path.split('/')[-1][:-4] 
            data = pd.read_csv(v_path, skiprows=4)
            length[sub_file] = len(data)

        sorted_by_file_len = dict(sorted(length.items(),key = lambda k : k[1]))
        write_file = [str(length[f]) + ' = ' + f for f in sorted_by_file_len]
        for file, l in sorted_by_file_len.items() :
            if l < 1000 :
                remove_file.append(file)

    return remove_file

def get_annotated_videos() :
        annotated_path = '/mnt/DATA1/ali/BitoScopeVAME_new/annotatedTracks'
        files = []
        for folder in os.listdir(annotated_path) :
            folder = os.path.join(annotated_path, folder)
            for file in os.listdir(folder) :
                
                if file.endswith('.csv') :
                    file = file[:-15] + '.h5'
                    f = copy.deepcopy(file)
                    file = os.path.join(folder, file)
                    data = pd.read_hdf(file)
                    print(f)
                    frames = list(data.index)
               
                    cols = data.columns.droplevel(1)[:105]
                    data = np.asarray(data)
                    
                    data = data[:,:106]
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
                    output_name = '/mnt/DATA1/ali/BitoScopeVAME_new/BiteScopeWithVAME-Mar10-2023/annotatedTracks'
                    df = pd.DataFrame(data=position, columns=cols)
                    df_frame = pd.DataFrame(data=frames, columns=['frames'])
                    df.to_hdf(os.path.join(output_name, f), "df_with_missing", format="table", mode="w")
                    df_frame.to_csv(os.path.join(output_name, f[:-3]+'_frames.csv'), index=False)
                    shutil.copyfile(file[:-3]+'_annotation.csv', os.path.join(output_name, f[:-3]+'_annotation.csv'))
                    