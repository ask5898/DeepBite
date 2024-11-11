import pandas as pd
import os
import numpy as np
from numpy.fft import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from .info import *
from pathlib import Path
import random
import scipy
import tqdm
from scipy.signal import convolve2d
from pairing import pair, depair
import sklearn
import glob
import matplotlib.pyplot as plt
import cv2
import pywt
import igraph
from sklearn.cluster import FeatureAgglomeration, KMeans
from scipy.spatial.distance import cdist
import umap
from cdlib import algorithms, readwrite
from communities.algorithms import louvain_method
import networkx as nx
import scipy.cluster.hierarchy as spc
from scipy.spatial import distance
from vame.util.auxiliary import read_config

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0] 

def interpol(arr):
        
    y = np.transpose(arr)
     
    nans, x = nan_helper(y[0])
    y[0][nans]= np.interp(x(nans), x(~nans), y[0][~nans])   
    nans, x = nan_helper(y[1])
    y[1][nans]= np.interp(x(nans), x(~nans), y[1][~nans])
    arr = np.transpose(y)
    return arr

def filter_signal(signal, threshold=1e3):
    fourier = rfft(signal)
    frequencies = rfftfreq(signal.size, d=1e-5)
    fourier[frequencies > threshold] = 0
    return irfft(fourier)

def moving_average(arr, window=5) :
    moving_averages = []
    for idx in range(0,len(arr) - window + 1):
        window_average = np.mean(arr[idx:idx+window])
        moving_averages.append(window_average)
    
    return np.asarray(moving_averages)

def filter_trajectories(pose_list) :   
    new_pos = []
    if pose_list.shape[0] == 1 :
        new_pos = filter_signal(np.squeeze(pose_list))

    else :    
        for i in pose_list :
            #i = scipy.signal.savgol_filter(i, 11, 3)
            i = filter_signal(i)
            #if len(i) != len(conf) :
            #    for _ in range(len(conf) - len(i)) :
            #        i = np.concatenate([i,[np.nan]])
            new_pos.append(i)
    return np.asarray(new_pos)
    #return pos

def get_bodypart_position(config, file, bodypart, num_points, start=0) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    path_to_file = cfg['project_path']
    temp_win = cfg['time_window']
    data = pd.read_hdf(os.path.join(path_to_file, 'videos','pose_estimation', file+'.h5'))
    frames = pd.read_hdf(os.path.join(path_to_file, 'videos','pose_estimation', file+'-frames.h5'))
    remove = np.load(os.path.join(cfg['project_path'], 'marked videos', file+'_to_remove.npy'))
    brkpnts = np.load(os.path.join(path_to_file, 'results', file, file+'_breakpoints.npy'))
    if len(brkpnts[0]) > 1 :
            brkpnts = np.squeeze(brkpnts)
    else :
            brkpnts = [np.squeeze(brkpnts)]
    rem = []
    remove_idx = []
    for i in remove :
            rem.extend(range(i[0], i[1]))

    for brkpnt in brkpnts :
            if brkpnt > temp_win*2 :
                temp = list(range(brkpnt-temp_win*2, brkpnt+1))
            else :
                temp = list(range(0, brkpnt+1))
            remove_idx.extend(temp)
    #rem = np.concatenate([rem, remove_idx], axis=0)
    #remove_idx = np.squeeze(remove_idx)
    data.drop(rem, axis=0, inplace=True)
    data = data.reset_index(drop=True)
    print(remove_idx)
    data.drop(remove_idx, axis=0, inplace=True)
    frames.drop(rem, axis=0, inplace=True)
    frames = frames.reset_index(drop=True)
    frames.drop(remove_idx, axis=0, inplace=True)
    frames = frames.reset_index(drop=True)

    bpt_pos = [idx for idx,bpt in enumerate(list(data.columns.get_level_values('bodyparts'))) if bpt == bodypart]
    coords = data.values[:,bpt_pos[0]:bpt_pos[1]+1]

    coords = interpol(coords)
    if num_points > coords.shape[0] :
        num_points = coords.shape[0]
    coords = coords[start:start+num_points,:]
    return zip(np.squeeze(frames.values)[start:start+num_points].T, coords)


def get_coordinate_pair(trainset) :
    idx = 0
    final_mat = []
    for i in range(int(trainset.shape[0]/2)) :
        pairs =[]
        for x, y in zip(trainset[idx],trainset[idx+1]) :
            x = (-2*x - 1) if x < 0 else 2*x
            y = (-2*y - 1) if y < 0 else 2*y
            pairs.append(pair(int(x), int(y)))
        final_mat.append(pairs)
        idx += 2  
    final_mat = np.asarray(final_mat)
    assert trainset.shape[0] == final_mat.shape[0]*2
    assert trainset.shape[1] == final_mat.shape[1]
    return final_mat


def get_track_position(config, save) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    path_to_file = cfg['project_path']
    skiprows = 3
    path_to_csv = os.path.join(path_to_file,'videos','pose_estimation')
    usecols = [i for i in range(col_len-3)]

    for file in os.listdir(path_to_csv) :
        if file.endswith('.csv') :
            data = pd.read_csv(os.path.join(path_to_csv,file),skiprows=skiprows, header=None, usecols=usecols)
            data = np.asarray(data)
            for i in data :
                for j in i :
                        j[0],j[1] = np.nan, np.nan        
    
            for i in pose_list:
                    i = interpol(i)

            position = np.concatenate(pose_list, axis=1)
            final_positions = np.zeros((position.shape[0], int(position.shape[1]/3)*2))
            jdx = 0
            idx = 0
            for i in range(int(position.shape[1]/3)):
                final_positions[:,idx:idx+2] = position[:,jdx:jdx+2]
                jdx += 3
                idx += 2

           
            print('Saved %s'%file)
            np.save(os.path.join(path_to_file,'data',file[:-4], file[:-4]+'-orig.npy'), final_positions.T)

def get_cluster_center(cfg,videos, model_name, parameterization, n_cluster) :
    cluster_center = []
    for video in videos :
        save_data = os.path.join(cfg['project_path'],"results",video,model_name,parameterization+'-'+str(n_cluster),"")
        data = np.load(os.path.join(save_data,'cluster_center_'+video+'.npy'))
        cluster_center.append(data)

    return cluster_center[0]
    np.save(os.path.join(cfg['project_path'], 'results', 'latent_space_vector.npy'), latents)

def get_adjacency_matrix(clust_center, n_cluster, thresh=0.4) :
    adjacency_matrix = np.zeros((n_cluster, n_cluster))
    distt = []
    for idx in range(n_cluster) :
        for jdx in range(n_cluster) :
            print(clust_center[idx])
            dist = cdist([clust_center[idx]], [clust_center[jdx]], metric='cosine')
            distt.append(dist)
            if dist > thresh :
                adjacency_matrix[idx, jdx] = 1
            else :
                adjacency_matrix[idx, jdx] = 0

    print(np.mean(distt))     
    return adjacency_matrix
              
def get_communities(config, reduction, validation=False) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    path_to_proj = cfg['project_path']
    random_state = cfg['random_state_kmeans']
    videos = cfg['video_sets']
    parameterization = cfg['parameterization']
    model_name = cfg['model_name']
    n_init = cfg['n_init_kmeans']
    n_cluster = cfg['n_cluster']
    covar_mat_full = np.load(os.path.join(path_to_proj, 'results','covariance.npy'))
    clust_center = get_cluster_center(cfg,videos, model_name, parameterization, n_cluster)
    if validation :
        latent_space = np.load('/mnt/DATA1/ali/BitoScopeVAME_new/BiteScopeWithVAME-Mar10-2023/results/annotation_latent_space.npy')
    else :
        latent_space = np.load('/mnt/DATA1/ali/BitoScopeVAME_new/BiteScopeWithVAME-Mar10-2023/results/latent_space_vector.npy')

    print(latent_space.shape)
    sim_mat = np.zeros((n_cluster,n_cluster))
    graph = nx.Graph()
   

    #sim_mat = np.linalg.inv(sim_mat)
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, node_size = 75, alpha = 0.8)
    graph = igraph.Graph.from_networkx(graph)
    plt.savefig('graph.png')

    sim_mat = get_adjacency_matrix(clust_center, n_cluster)
    print(sim_mat)
    if reduction == 'louvain' :
        comm, _ = louvain_method(sim_mat)
        labels = louvain_to_community_label(comm, n_cluster)
        print(np.unique(labels))

    if reduction == 'flat' :
        linkage = spc.linkage(sim_mat, method='complete')    
        spc.dendrogram(linkage)
        plt.axhline(4, color='k', ls='--')
        plt.savefig('ref.png')
        labels = spc.fcluster(linkage, 4, 'maxclust')

    if reduction == 'agglo' :
        print(latent_space.shape)
        sc = FeatureAgglomeration(n_clusters=5, linkage='ward', connectivity=sim_mat)
        clusters = sc.fit(latent_space.T)
        labels = clusters.labels_ 
        print(labels) 

    if validation :
        np.save(os.path.join(cfg['project_path'], 'results', 'output_annot.npy'), labels)

    return cfg, labels

def louvain_to_community_label(community, n_cluster) :
    community_label = []
    label = 0
    community = list(community)
    for idx in range(n_cluster) :
        for i in community :
            if idx in i :
                label = community.index(i)
        community_label.append(label)
    return community_label
        


def get_community_label(config, reduction, validation) :
    cfg, labels = get_communities(config, reduction=reduction, validation=validation)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    parameterization = cfg['parameterization']
    videos = cfg['video_sets']
    for video in videos :
        path_to_file = os.path.join(cfg['project_path'],"results",video,"",model_name,"",parameterization+'-'+str(n_cluster))
        motif_label = np.load(os.path.join(path_to_file,"",str(n_cluster)+'_km_label_'+video+'.npy'))
        community_label = [labels[i] for i in motif_label]
        #print(np.unique(community_label, return_counts=True))
        np.save(os.path.join(path_to_file,"",str(n_cluster)+'_community_label_'+video+'.npy'), community_label)


def clean_community_label(config, thresh=25) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    videos = cfg['video_sets']
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    parameterization = cfg['parameterization']
    for video in videos :
        path_to_file = os.path.join(cfg['project_path'],"results",video,"",model_name,"",parameterization+'-'+str(n_cluster))
        label = np.load(os.path.join(path_to_file,"",str(n_cluster)+'_community_label_'+video+'.npy'))
        cons = np.split(label, np.where(np.diff(label)!=0)[0]+1)
        count = 0 
        for i in range(1,len(cons)-1) :
            first = cons[i+1]
            last = cons[i-1]
            fill_val = 0
            if len(cons[i]) < thresh :
                if first[0] == last[-1]:
                    fill_val = first[0]
                    cons[i] = [fill_val for _ in range(len(cons[i]))]
                

        cons = np.concatenate(cons, axis=0)    
        np.save(os.path.join(path_to_file,"",str(n_cluster)+'_community_label_'+video+'.npy'), cons)


def create_cluster_videos(config, mode, thresh=25) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    videos = cfg['video_sets']
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    parameterization = cfg['parameterization']
    for video in videos :
        path_to_file = os.path.join(cfg['project_path'],"results",video,"",model_name,"",parameterization+'-'+str(n_cluster))
        if mode == 'community' :
            label = np.load(os.path.join(path_to_file,"",str(n_cluster)+'_community_label_'+video+'.npy'))
            
        else :
            label = np.load(os.path.join(path_to_file,"",str(n_cluster)+'_km_label_'+video+'.npy'))
        print(len(label))
        community = {str(i) : [] for i in np.unique(label)}
        cons_condition = np.where(np.diff(label)!=0)[0]
        cons = np.split(label, cons_condition+1)
        idx = -1
        for comm in cons :
            if idx == -1 :
                community[str(np.squeeze(np.unique(comm)))].append([0,cons_condition[0]+1])
            else :
                if np.squeeze(np.diff(cons_condition[idx:idx+2])) > thresh :
                    community[str(np.squeeze(np.unique(comm)))].append(cons_condition[idx:idx+2]+1)
            idx+=1
            
        
        capture = cv2.VideoCapture(os.path.join(cfg['project_path'],"videos",video+'.mp4'))
        if capture.isOpened():
            width  = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = 25

        if not os.path.exists(os.path.join(cfg['project_path'], 'community videos', video)) :
            os.makedirs(os.path.join(cfg['project_path'], 'community videos', video))

        print('Creating community videos for %s\n'%video)
        for key,val in community.items() :
               
            print('Community-{}'.format(key))
            if len(val) < 1 :
                print('Community {} not found in {}'.format(key, video))
                continue

            output_path = os.path.join(cfg['project_path'], 'community videos', video, video+'_community_'+key+'.avi')
            output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (int(width), int(height)))
            for idx,k in enumerate(val) :
                #if idx > 30 :
                #    break
                start = k[0]
                num_points = np.squeeze(np.diff(k))
                if num_points < 1 :
                    continue
                if num_points > 1000 :
                    num_points = 1000
                bpts = list(get_bodypart_position(config, video, 'thorax', num_points=num_points, start=start))
                vid_frame, positions = zip(*bpts)
                
                for i in tqdm.tqdm(range(num_points), leave=False, desc=str(idx)) :
                    capture.set(1,int(vid_frame[i]))
              
                    marker_coord = int(positions[i][0]), int(positions[i][1])
                    ret, frame = capture.read()
                    cv2.circle(frame, marker_coord, radius=8, color=(0,0,255), thickness=7)
                    output.write(frame)

            output.release()
        capture.release()


def sharpen_image(config, iterations=5) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    image_path = glob.glob(os.path.join(cfg['project_path'], 'mosquito_crop','*.png'))
    extract_rgb = lambda x : (x[:,:,0], x[:,:,1], x[:,:,2])
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])

    fig,ax = plt.subplots(2,2, figsize=(20,20))
    ax[0,0].set_title('Normal')
    ax[0,1].set_title('Normal sharpened')
    ax[1,0].set_title('Grayscale')
    ax[1,1].set_title('Grayscale sharpened')

    for i in range(0,iterations) :
        for img in image_path :
            img = cv2.imread(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_b, img_g, img_r = extract_rgb(img)
            img_b_sharp = convolve2d(img_b, sharpen_kernel, 'full', boundary = 'fill',fillvalue = 0)
            img_g_sharp = convolve2d(img_g, sharpen_kernel, 'full', boundary = 'fill',fillvalue = 0)
            img_r_sharp = convolve2d(img_r, sharpen_kernel, 'full', boundary = 'fill',fillvalue = 0)
            img_grey_sharp = convolve2d(gray, sharpen_kernel, 'full', boundary = 'fill',fillvalue = 0)/255
            img_sharp = np.dstack((np.rint(abs(img_b_sharp)), np.rint(abs(img_g_sharp)), np.rint(abs(img_r_sharp))))/255
            break
        ax[0,0].imshow(img)
        ax[1,0].imshow(gray)
        ax[0,1].imshow(img_sharp)
        ax[1,1].imshow(img_grey_sharp)
        plt.show()


def image_threshold(config) :
    data_path = '/mnt/DATA1/ali/aedes-f-2022-06-22_221019/labeled-data'
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    #image_path = glob.glob(os.path.join(cfg['project_path'], 'mosquito_crop','*.png'))[:10]
    image_path = [f for folder in os.listdir(data_path) for f in glob.glob(os.path.join(data_path, folder, '*.png'))]
    #print(image_path)
    #kernel_erode = np.ones((3, 3), np.uint8)
    #kernel_dilate = np.ones((2,2), np.uint8)
    #fig,ax = plt.subplots(1,3, figsize=(20,20))
    #ax[0].set_title('Normal')
    #ax[1].set_title('Adaptive Mean Threshold+Denoising')
    #ax[2].set_title('Adaptive Gaussian Threshold+Denoising')

    for idx, img_path in enumerate(image_path) :
        img = cv2.imread(img_path)
        #ax[0].imshow(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh_mean = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        thresh_mean = cv2.cvtColor(thresh_mean, cv2.COLOR_GRAY2BGR)
        thresh_mean = cv2.fastNlMeansDenoisingColored(thresh_mean,None,90,90,7,101)

        #thresh_gaus = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        #thresh_gaus = cv2.cvtColor(thresh_gaus, cv2.COLOR_GRAY2BGR)
        #thresh_gaus = cv2.fastNlMeansDenoisingColored(thresh_gaus,None,100,100,7,101)
        #thresh_mean_erode = cv2.erode(thresh_mean, kernel_erode, iterations=1)
        #ax[1].imshow(thresh_mean)
        #ax[2].imshow(thresh_gaus)
        cv2.imwrite(img_path, thresh_mean)
        #plt.savefig(str(idx)+'.png')
        #plt.clf()

def apply_wavelet_transform(config) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    path_to_proj = cfg['project_path']
    videos = cfg['video_sets']

    #videos = [file for file in glob.glob(os.path.join(path_to_proj, 'annotatedTracks','*.npy'))]
    for video in videos :
        data = np.load(os.path.join(path_to_proj,'data', video, video+'-PE-seq.npy'))
        #data = np.load(video)
        sig = []
        fig, ax = plt.subplots(10, 2, figsize=(10,10))
        fig, axx = plt.subplots(2,figsize=(6,6))
        #axx[0].set_title('Origingal Signal')
        #axx[1].set_title('Transformed Signal')
        
        for feature in data :
            resample_size = len(feature)
            coeff = [0.0]*resample_size
            feat = feature.copy()
            for idx in range(10) :
                (feat, coeff_d) = pywt.dwt(feat, 'db5')
                coeff = coeff + scipy.signal.resample(coeff_d, resample_size)
              

                ax[idx,0].plot(feature,'r')
                ax[idx,1].plot(coeff_d,'g')
                ax[idx,0].set_ylabel('Level {}'.format(idx+1), rotation=90)
                ax[idx,0].set_yticklabels([])
                ax[idx,1].set_yticklabels([])
                if idx == 0 :
                    ax[idx, 0].set_title('Approximation')
                    ax[idx,1].set_title('Detailed')
            
            trans_sig = scipy.signal.resample(feat, resample_size) + coeff
            axx[1].plot(trans_sig)
            sig.append(trans_sig)
            
            
        plt.savefig('WT.png')
        np.save(os.path.join(path_to_proj,'data', video, video+'-PE-seq-wt.npy'),sig)
        #np.save(video, sig)
        plt.close()

def discretize_egocentric(config, block_size=16, test=False) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    project_path = cfg['project_path']
    videos = cfg['video_sets']
    if test :
        data = np.load(os.path.join(project_path, 'testing', 'egocentric', 'annotationData-full.-EGOCENTRIC.npy'))
        for idx in range(int(data.shape[0])) :
            data[idx] = [np.round(i) for i in data[idx]/block_size]
        return data
        #np.save(os.path.join(project_path, 'testing', 'egocentric', 'annotationData-disc.npy'), data)

    else :
        for video in videos :
            data = np.load(os.path.join(project_path, 'data', video, video+'-PE-seq.npy'))
   
            for idx in range(int(data.shape[0])) :
                data[idx] = [np.round(i) for i in data[idx]/block_size]
   
            np.save(os.path.join(project_path, 'data', video, video+'-PE-seq-disc.npy'), data)

def get_latent_vector(config) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    videos = cfg['video_sets']
    model_name = cfg['model_name']
    parameterization = cfg['parameterization']
    n_cluster = cfg['n_cluster']
    latents = []
    for video in videos :
        save_data = os.path.join(cfg['project_path'],"results",video,model_name,parameterization+'-'+str(n_cluster),"")
        data = np.load(os.path.join(save_data,'latent_vector_'+video+'.npy'))
        latents.append(data)
    
    latents = np.concatenate(latents, axis=0)
    print(latents.shape)
    np.save(os.path.join(cfg['project_path'], 'results', 'latent_space_vector.npy'), latents)

def create_marked_videos(config) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    videos = cfg['video_sets']
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    parameterization = cfg['parameterization']
    for video in videos :
        file = np.load(os.path.join(cfg['project_path'],"data", video,video+'-PE-seq.npy'))
        totalframecount = file.shape[1] - 4
        capture = cv2.VideoCapture(os.path.join(cfg['project_path'],"videos",video+'.mp4'))
        if capture.isOpened():
            width  = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = 25
        
        if not os.path.exists(os.path.join(cfg['project_path'], 'marked videos')) :
            os.makedirs(os.path.join(cfg['project_path'], 'marked videos'))

        output_path = os.path.join(cfg['project_path'], 'marked videos', video+'.mp4')
        output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (int(width), int(height)))

        bpts = list(get_bodypart_position(config, video, 'thorax', num_points=totalframecount, start=0))
        vid_frame, positions = zip(*bpts)
        print('Marking ',video)
        for i in tqdm.tqdm(range(totalframecount)) :
            capture.set(1,int(vid_frame[i]))
            marker_coord = int(positions[i][0]), int(positions[i][1])
            ret, frame = capture.read()
            cv2.circle(frame, marker_coord, radius=8, color=(0,0,255), thickness=7)
            output.write(frame)

        output.release()
    capture.release()


def get_breakpoint(config) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    project_path = cfg['project_path']
    videos = cfg['video_sets']
    frames = []
    for video in videos :
        frame = pd.read_hdf(os.path.join(project_path,'videos','pose_estimation',video+'-frames.h5'))
        frame = [int(fr) for fr in frame.values]
        remove = np.load(os.path.join(cfg['project_path'], 'marked videos', video+'_to_remove.npy'))
        rem = []
        for i in remove :
            rem.extend(range(i[0], i[1]))
        frame = np.delete(frame, rem, axis=0)
        file_brkpnts = np.where(np.diff(frame)!=1)
        np.save(os.path.join(project_path, 'results', video, video+'_breakpoints.npy'), file_brkpnts)
        frames.extend(frame)
        
    split = int(cfg['test_fraction'] * len(frames))

    test_frames = frames[:split]
    train_frames = frames[split:]
    train_brkpnt = np.where(np.diff(train_frames) != 1)
    test_brkpnt = np.where(np.diff(test_frames) != 1)
    train_brkpnt = np.squeeze(train_brkpnt)
    test_brkpnt = np.squeeze(test_brkpnt)
    np.save(os.path.join(project_path, 'results', 'train_breakpoints.npy'), train_brkpnt)
    np.save(os.path.join(project_path, 'results', 'test_breakpoints.npy'), test_brkpnt)

def cluster_umap(config) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    from hmmlearn import hmm
    umap_projection = np.load(os.path.join(cfg['project_path'], 'results', 'latent_umap.npy'))
    hmm_model = hmm.GaussianHMM(n_components=5, covariance_type="full", n_iter=100)
    hmm_model.fit(umap_projection)
    label = hmm_model.predict(umap_projection)
    idx = 0

    for video in cfg['video_sets'] :
        save_data = os.path.join(cfg['project_path'],"results",video,cfg['model_name'],cfg['parameterization']+'-'+str(cfg['n_cluster']),"")
        data = np.load(os.path.join(save_data,'latent_vector_'+video+'.npy'))
        print(data.shape)
        np.save(os.path.join(save_data,"",str(cfg['n_cluster'])+'_community_label_'+video+'.npy'), label[idx:idx+data.shape[0]])
        print(label[idx:idx+data.shape[0]].shape)
        
        idx += data.shape[0]

        
    scatter = plt.scatter(umap_projection[:,0], umap_projection[:,1], c=label, cmap='Spectral', s=2, alpha=.7)
    #plt.legend(handles=scatter.legend_elements()[0], labels=np.unique(labels))

    plt.gca().set_aspect('equal', 'datalim')
    plt.grid(False)     
    plt.title('UMAP Reduced Latent Space')
    

    plt.savefig('test_umap.png')
        
