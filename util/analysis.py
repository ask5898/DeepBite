import pandas as pd
import os
import torch
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import cv2
import matplotlib.pyplot as plt
import tqdm
from pathlib import Path
import io
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from mpl_toolkits.mplot3d import Axes3D
import pickle
import umap.plot 
from hmmviz import TransGraph
import glob
import umap
import functools
import plotly.express as px
import random
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from vame.util.auxiliary import read_config
from .vame_utils import *
from vame.analysis.umap_visualization import umap_label_vis
from .info import *
import numba
from vame.model.rnn_model import RNN_VAE


plt.rcParams['figure.figsize'] = [20,10]
plt.rcParams["figure.autolayout"] = False

def get_correlation_matrix(config) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    path_to_file = cfg['project_path']
    path_to_trans = os.path.join(path_to_file, 'results','latent_space_vector.npy')
    trans_mat = np.load(path_to_trans).T
    #trans_mat = np.mean(trans_mat, axis=0)
    coeff = np.corrcoef(trans_mat)
    print(coeff)
    pose = list(range(30))

    figure = plt.figure()
    axes = figure.add_subplot(111)
    caxes = axes.matshow(coeff, interpolation ='nearest')
    axes.yaxis.set_major_locator(MultipleLocator(1))
    axes.xaxis.set_major_locator(MultipleLocator(1))
    figure.colorbar(caxes)
    axes.set_xticklabels(['']+pose)
    axes.set_yticklabels(['']+pose)
    plt.savefig('abc.png')
    
def get_motif_transition(config) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    path_to_file = cfg['project_path']
    n_clusters = cfg['n_cluster']
    cols = ['motif-' + str(i) for i in range(n_clusters)]
    nodes = dict(zip(list(range(n_clusters)),cols))
    color = {i : f'C{i}' for i in range(n_clusters)}
    path_to_trans = os.path.join(path_to_file, 'results','transition_mat.npy')
    trans_mat = pd.DataFrame(np.load(path_to_trans))
    print(trans_mat)
    graph = TransGraph(trans_mat)
    graph.draw(nodelabels=nodes, nodecolors=color, edgecolors=color, edgelabels=False, edgewidths=0.5)
    plt.show()

def create_reference_video(config, num=1) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    path_to_file = cfg['project_path']
    files = cfg['video_sets']
    files = np.random.choice(files, size=num, replace=False)
    fps = 25
    width = 1400
    height = 1400
    for file in files :

        output_path = os.path.join(path_to_file, 'egocentric_animation',file+'EGOANI.avi')
        output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps,(width, height))
        figures = get_egocentric_plots(config, file, cleaned=True)
        data = pd.read_csv(os.path.join(path_to_file,'videos','pose_estimation',file+'.csv'), skiprows=3, header=None)
        data = pd.DataFrame.to_numpy(data)
        frame = np.squeeze(data[:,:1])
        position = data[:,1:]
        pose_list = []
        for i in range(int(position.shape[1]/3)):
            pose_list.append(position[:,i*3:(i+1)*3])
         
        for i in pose_list:
            for j in i:
                if j[2] <= confidence:
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

        print('For file %s'%file)
        cropped_frames = get_cropped_frames(path_to_file, file, final_positions, frame)
    
        for fig, cf in tqdm.tqdm(zip(figures, cropped_frames), total = 2000) :
            fig = convert_to_image(fig)
            fig = cv2.cvtColor(fig, cv2.COLOR_RGB2BGR)
            #cf = cv2.flip(cf,1)
                
            #print(fig.shape, cf.shape)
            h = cf.shape[0] + fig.shape[0]
            w = cf.shape[1] + fig.shape[1]
            if h == 0 or w == 0 :
                print('Crop unsuccessfull')
                os.remove(output_path)
                break
                
            combined = np.zeros((h,w,3), dtype=np.uint8)
            #combined = np.concatenate((cf, fig), axis=1)
            idx = 0
            for im in [fig, cf] :
                combined[:im.shape[0] ,idx:im.shape[1] + idx, :] = im
                idx += im.shape[1]

            #print(combined.shape[:2])
            combined = cv2.resize(combined, (width,height))
            output.write(combined)

        output.release()



def plot_train(config,seq_length=1000) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    path_to_file = cfg['project_path']
    path_to_train = os.path.join(path_to_file, 'data','train', 'train_seq.npy')
    trainset = np.load(path_to_train)
    start = 0
    #for i in range(100) :
    for pos in trainset :#[:,start:start+seq_length] :
            plt.plot(pos)
            #plt.xticks(ticks=s, labels=['00016','02251','00000','00654'], rotation=60)
            #plt.savefig(os.path.join(path_to_file,'traindata_plot','test_.png'))
            #start += seq_length
    plt.savefig('trin.png')
    plt.close()

def plot_positions(config) :
    #positions = csv_to_numpy(config, usage=False)
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    path_to_file = cfg['project_path']
    videos = cfg['video_sets']
    plt.plot(range(30))
    plt.savefig('dsdsd.png')
    path_to_exp = [glob.glob(os.path.join(path_to_file, 'data', video,'*-orig.npy')) for video in videos ]
    for path in path_to_exp :
        positions = np.load(path[0])
        for position in positions :
            plt.plot(position)
        print(str(Path(path[0])).split('/')[-1][:-4])
        plt.savefig(str(str(Path(path[0])).split('/')[-1][:-4])+'.png')
        plt.close()

def get_egocentric_plots(config, experiment, cleaned=False, scaled=False) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    path_to_file = cfg['project_path']
    path_to_data = os.path.join(path_to_file, 'data', experiment)
    head = filtered_bodyparts['head']
    bottom = filtered_bodyparts['bottom']
    pbs = filtered_bodyparts['proboscis1']
    abdL = filtered_bodyparts['abdomenL']
    abdR = filtered_bodyparts['abdomenR']
    rfl1 = filtered_bodyparts['rightForeLeg1']
    lfl1 = filtered_bodyparts['leftForeLeg1']
    rml1 = filtered_bodyparts['rightMidleg1']
    lml1 = filtered_bodyparts[ 'leftMidleg1']
    rhl1 = filtered_bodyparts['rightHindleg1']
    lhl1 = filtered_bodyparts['leftHindleg1']
    rfl3 = filtered_bodyparts['rightForeLeg3']
    lfl3 = filtered_bodyparts['leftForeLeg3']
    rml3 = filtered_bodyparts['rightMidleg3']
    lml3 = filtered_bodyparts[ 'leftMidleg3']

    figures = []
    fig = plt.figure()
    if not os.path.exists('egocentric_plots') :
        os.mkdir('egocentric_plots')
    import re
    for file in os.listdir(path_to_data) :
        if  not scaled :
            if re.match(r'^[a-zA-Z0-9_.-]*PE-seq.npy\b', file) :
                pose = np.load(os.path.join(path_to_data,file))
        elif scaled :
            if re.match(r'^[a-zA-Z0-9_.-]*PE-seq-clean.npy\b', file) :
                pose = np.load(os.path.join(path_to_data,file))
    
    if cleaned :
        new_pos = []
        for p in pose :
            new_pos.append(filter_signal(p))
        pose = np.asarray(new_pos)

    for frame in range(10000, 12000) :
        x = []
        y = []
        for idx, p in enumerate(pose[:,frame]) :
            if idx%2 == 0 :
                x.append(p)
            else :
                y.append(p)

        head_pbs_x = int(x[int(head[0]/2)]), int(x[int(pbs[0]/2)])
        head_pbs_y = int(y[int(head[1]/2)]), int(y[int(pbs[1]/2)])
        head_abdl_x = int(x[int(head[0]/2)]), int(x[int(abdL[0]/2)])
        head_abdl_y = int(y[int(head[1]/2)]), int(y[int(abdL[1]/2)])
        head_abdr_x = int(x[int(head[0]/2)]), int(x[int(abdR[0]/2)])
        head_abdr_y = int(y[int(head[1]/2)]), int(y[int(abdR[1]/2)])
        lfl3_abdl_x =  int(x[int(lfl3[0]/2)]), int(x[int(abdL[0]/2)])
        lfl3_abdl_y = int(y[int(lfl3[1]/2)]), int(y[int(abdL[1]/2)])
        lml3_abdl_x =  int(x[int(lml3[0]/2)]), int(x[int(abdL[0]/2)])
        lml3_abdl_y = int(y[int(lml3[1]/2)]), int(y[int(abdL[1]/2)])
        lhl_abdl_x =  int(x[int(lhl1[0]/2)]), int(x[int(abdL[0]/2)])
        lhl_abdl_y = int(y[int(lhl1[1]/2)]), int(y[int(abdL[1]/2)])
        rfl3_abdr_x =  int(x[int(rfl3[0]/2)]), int(x[int(abdR[0]/2)])
        rfl3_abdr_y = int(y[int(rfl3[1]/2)]), int(y[int(abdR[1]/2)])
        rml3_abdr_x =  int(x[int(rml3[0]/2)]), int(x[int(abdR[0]/2)])
        rml3_abdr_y = int(y[int(rml3[1]/2)]), int(y[int(abdR[1]/2)])
        rhl_abdr_x =  int(x[int(rhl1[0]/2)]), int(x[int(abdR[0]/2)])
        rhl_abdr_y = int(y[int(rhl1[1]/2)]), int(y[int(abdR[1]/2)])
        lfl3_lfl1_x = int(x[int(lfl3[0]/2)]), int(x[int(lfl1[0]/2)])
        lfl3_lfl1_y = int(y[int(lfl3[0]/2)]), int(y[int(lfl1[0]/2)])
        lml3_lml1_x = int(x[int(lml3[0]/2)]), int(x[int(lml1[0]/2)])
        lml3_lml1_y = int(y[int(lml3[0]/2)]), int(y[int(lml1[0]/2)])
        rfl3_rfl1_x = int(x[int(rfl3[0]/2)]), int(x[int(rfl1[0]/2)])
        rfl3_rfl1_y = int(y[int(rfl3[0]/2)]), int(y[int(rfl1[0]/2)])
        rml3_rml1_x = int(x[int(rml3[0]/2)]), int(x[int(rml1[0]/2)])
        rml3_rml1_y = int(y[int(rml3[0]/2)]), int(y[int(rml1[0]/2)])
        abdr_btm_x =  int(x[int(abdR[0]/2)]), int(x[int(bottom[0]/2)])
        abdr_btm_y =  int(y[int(abdR[1]/2)]), int(y[int(bottom[1]/2)])
        abdl_btm_x =  int(x[int(abdL[0]/2)]), int(x[int(bottom[0]/2)])
        abdl_btm_y =  int(y[int(abdL[1]/2)]), int(y[int(bottom[1]/2)])
        body_x = [head_pbs_x,head_abdl_x,head_abdr_x,lfl3_abdl_x, lml3_abdl_x,  lhl_abdl_x, rfl3_abdr_x, rfl3_abdr_x, rml3_abdr_x, rhl_abdr_x,lfl3_lfl1_x,lml3_lml1_x,rfl3_rfl1_x,rml3_rml1_x,abdr_btm_x,abdl_btm_x]
        body_y = [head_pbs_y,head_abdl_y,head_abdr_y,lfl3_abdl_y, lml3_abdl_y,  lhl_abdl_y, rfl3_abdr_y, rfl3_abdr_y, rml3_abdr_y, rhl_abdr_y,lfl3_lfl1_y,lml3_lml1_y,rfl3_rfl1_y,rml3_rml1_y,abdr_btm_y,abdl_btm_y]
        body_x_y = zip(body_x, body_y)
        plt.scatter(x,y)
        plt.xlim(-100, 500)
        plt.ylim(-100, 500)
        for xx, yy in body_x_y :
            plt.plot(xx, yy)
        
        fig = plt.gcf()
        #plt.savefig(os.path.join(path_to_file,'egocentric_plots', experiment+str(frame)+'.png'))
        figures.append(fig)
        plt.close()

    return figures
        

def convert_to_image(fig) :
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    #print(img.size)
    return np.asarray(img)
    
def plot_animation(config, experiment, cleaned=False) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    path_to_file = cfg['project_path']
    figures = get_egocentric_plots(config, experiment, cleaned)
    width = 1000
    height = 1000
    fps = 25
    output_path = os.path.join(path_to_file,'egocentric_animation')
    if not os.path.exists(output_path) :
        os.mkdir(output_path)
    
    if cleaned :
        exp_path = os.path.join(experiment+'-clean.mp4')
    else :
        exp_path = os.path.join(output_path,experiment+'.mp4')

    video = cv2.VideoWriter(exp_path,cv2.VideoWriter_fourcc('m','p','4','v'), fps, (int(width), int(height)))
    for fig in tqdm.tqdm(figures, total=500) :
        image = convert_to_image(fig)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(image)

    video.release()

def latent_space_exploration(config,file, mode, start=0) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    parameterization = cfg['parameterization']
    files = cfg['video_sets']
    #file = random.choice(files)
    thresh = 12
    #for file in files[:5] :
    path_to_file = os.path.join(cfg['project_path'],"results",file,"",model_name,"",parameterization+'-'+str(n_cluster))
    path_to_umap = os.path.join(path_to_file,"community","umap_embedding_"+file+'.npy')

    umap_projection = np.load(path_to_umap)
    print(umap_projection.shape)
    writer = writers['ffmpeg'](fps=25)
    num_points = cfg['num_points']
    if num_points > umap_projection.shape[0]:
        num_points = umap_projection.shape[0]
        
    #if len(os.listdir(os.path.join(cfg['project_path'],'mosquito_crop'))) > 0:
        
    get_crop_of_mosquito(config, file, num_points+thresh,start=start, size=200)
    image_path = glob.glob(os.path.join(cfg['project_path'], 'mosquito_crop','*.png'))
    image_path.sort(key=lambda x : int(str(Path(x).resolve()).split('/')[-1].split('-')[1][:-4]))
    fig,ax = plt.subplots(1,2)
    if mode == 'motif' :
        motif_label = np.load(os.path.join(path_to_file,"",str(n_cluster)+'_km_label_'+file+'.npy'))
        label = motif_label[start:start+num_points+thresh]

    if mode == 'community' :
        community_label = np.load(os.path.join(path_to_file,"",str(n_cluster)+'_community_label_'+file+'.npy'),allow_pickle=True)
        label = community_label[start:start+num_points+thresh]

    print(label)
    ax[0].scatter(umap_projection[start:start+num_points+thresh,0], umap_projection[start:start+num_points+thresh,1], c=label, cmap='Spectral', s=2, alpha=.7)
    tracker, = ax[0].plot([],[],'x-', markevery=[-1], color='black')
    plt.gca().set_aspect('equal', 'datalim')
    plt.grid(False)
    def animate(i,thresh,start) :
        ax[1].clear()
        j = i+thresh
        comm = label[j]
        img = cv2.imread(image_path[j])
        img = cv2.putText(img,str(mode)+'-'+str(comm), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        tracker.set_data(umap_projection[start+i:start+j , 0], umap_projection[start+i:start+j , 1])
        ax[1].imshow(img)
        del img
        return fig                          
    
    anim = FuncAnimation(fig,functools.partial(animate, thresh=thresh, start=start), frames=num_points-1, interval=1,repeat=False)
    #plt.show()
    #anim.ipython_display(fps=25, autoplay=True)
    anim.save(file+'.mp4', writer=writer)


def get_crop_of_mosquito(config, file, num_points, start=0, size=200) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    capture = cv2.VideoCapture(os.path.join(cfg['project_path'],'videos', file+'.mp4'))
    if capture.isOpened() :
        width  = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    thorax_frame_and_pos = list(get_bodypart_position(config, file, 'thorax', num_points, start))
    print(len(thorax_frame_and_pos))
    crop_frames = []
    pos_size = len(list(thorax_frame_and_pos))
    print('Saving cropped frames for %s'%file)
    for idx,(frame, pos) in tqdm.tqdm(enumerate(thorax_frame_and_pos), total=pos_size) :
        ref = int(frame)
        capture.set(1, ref)
        _,fr = capture.read()
        
        x,y = int(pos[0]), int(pos[1])
        x_upper = x+size
        x_lower = x-size
        y_upper = y+size
        y_lower = y-size
        if x_upper > width :
            x_upper = int(width)
        if x_lower < 0 :
            x_lower = 0
        if y_upper > height :
            y_upper = int(height)
        if y_lower < 0 :
            y_lower = 0

        cv2.imwrite(os.path.join(cfg['project_path'],'mosquito_crop','crop-'+str(idx)+'.png'), fr[y_lower:y_upper, x_lower:x_upper])
        #crop_frames.append(fr[y_lower:y_upper, x_lower:x_upper])

    capture.release()
    #return crop_frames

def view_clustered_latent_space(config, mode)  :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    parameterization = cfg['parameterization']
    videos = cfg['video_sets']
    num_points = cfg['num_points']
    latent_vectors = []
    #split, latent_umap = get_eval_split(config)
    for video in videos :
        path_to_file = os.path.join(cfg['project_path'],"results",video,"",model_name,"",parameterization+'-'+str(n_cluster))
        if mode == 'community' :
            community_label = np.load(os.path.join(path_to_file,"",str(n_cluster)+'_community_label_'+video+'.npy'),allow_pickle=True)
        if mode == 'motif' :
            community_label = np.load(os.path.join(path_to_file,"",str(n_cluster)+'_km_label_'+video+'.npy'),allow_pickle=True)
            
        labels = ['Community-'+str(label) for label in np.unique(community_label)]
        print(labels)
        path_to_umap = os.path.join(path_to_file,"community","umap_embedding_"+video+'.npy')
        umap_projection = np.load(path_to_umap)
        if num_points > umap_projection.shape[0]:
            num_points = umap_projection.shape[0]
        num = len(np.unique(community_label))
        scatter = plt.scatter(umap_projection[:num_points,0], umap_projection[:num_points,1], c=community_label[:num_points], cmap='Spectral', s=2, alpha=.7)
        plt.legend(handles=scatter.legend_elements()[0], labels=labels)
        plt.gca().set_aspect('equal', 'datalim')
        plt.grid(False)  
        plt.savefig(os.path.join(cfg['project_path'],'Community-UMAP',video,video+'.png'))
        plt.close()
        
def plot_reduced_latents(config, mode, reduction, dim=3) :
    config_file = Path(config).resolve()
    #numba.config.NUMBA_NUM_THREADS = 5
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    parameterization = cfg['parameterization']
    videos = cfg['video_sets']
    num_points = cfg['num_points']
    community = []

    if not os.path.exists(os.path.join(cfg['project_path'], 'UMAPs', str(model_name))) :
       os.makedirs(os.path.join(cfg['project_path'], 'UMAPs', str(model_name))) 

    for video in videos :
        path_to_file = os.path.join(cfg['project_path'],"results",video,"",model_name,"",parameterization+'-'+str(n_cluster))
        if mode == 'community' :
            community_label = np.load(os.path.join(path_to_file,"",str(n_cluster)+'_community_label_'+video+'.npy'),allow_pickle=True)
        elif mode == 'motif' :
            community_label = np.load(os.path.join(path_to_file,"",str(n_cluster)+'_km_label_'+video+'.npy'),allow_pickle=True)
        community.append(community_label)

    community_label = np.concatenate(community, axis=0)


    path_to_umap = os.path.join(cfg['project_path'],'results','latent_space_vector.npy')
    umap_projection = np.load(path_to_umap)
    print(umap_projection.shape)
    #umap_projection = umap_projection

    pipe = StandardScaler()
    umap_projection = pipe.fit_transform(umap_projection.copy())

    if reduction == 'TSNE' :
        reducer = TSNE(n_components=dim, learning_rate='auto', init='random', perplexity=200, random_state=42)
    if reduction == 'UMAP' :
        reducer = umap.UMAP(n_components=dim, metric='euclidean' ,min_dist=cfg['min_dist'], n_neighbors=cfg['n_neighbors'], random_state=cfg['random_state'])
    umap_projection = reducer.fit_transform(umap_projection)
    np.save(os.path.join(os.path.join(cfg['project_path'], 'results', 'latent_umap.npy')), umap_projection)
    num = len(np.unique(community_label))
    labels = ['Community-'+str(label) for label in np.unique(community_label)]
    
    if dim == 3 :
        fig = plt.figure()
        ax = Axes3D(fig)
        aspect = 'auto'
        scatter = ax.scatter(umap_projection[:,0], umap_projection[:,1], umap_projection[:,2], c=community_label, cmap='Paired', s=0.5, alpha=.7)

    elif dim == 2 :
         # pip install umap-learn[plot]

        #umap.plot.points(umap_projection, labels=community_label, theme="fire")
        aspect = 'equal'
        scatter = plt.scatter(umap_projection[:,0], umap_projection[:,1], c=community_label, cmap='Spectral', s=2, alpha=.7)
        
    else :
        raise RuntimeError('Only 2 and 3 dimmensional visualisation is supported')
   
    plt.legend(handles=scatter.legend_elements()[0], labels=labels)

    plt.gca().set_aspect(aspect, 'datalim')
    plt.grid(False)     
    plt.title(str(reduction)+' Reduced Latent Space')
    

    plt.savefig(os.path.join(cfg['project_path'], 'UMAPs', str(model_name),str(mode)+str(reduction)+'-reducer.png'))
    if dim == 3 :
        pickle.dump(fig, open('FigureObject_louvain.fig.pickle', 'wb')) 
    #plt.show()
    plt.close()

def view_community_plot(config, mode,thresh=12) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    parameterization = cfg['parameterization']
    #features = ['mosquito velocity','Avg. forelegs velocity', 'right leg angle', 'left leg angle', 'belly size', 'body length']
    for video in cfg['video_sets'] :
        data = np.load(os.path.join(cfg['project_path'], 'data', video, video+'-PE-seq.npy'))
        print(data.shape)
        path_to_file = os.path.join(cfg['project_path'],"results",video,"",model_name,"",parameterization+'-'+str(n_cluster))
        if mode == 'community' :
            community_label = np.load(os.path.join(path_to_file,"",str(n_cluster)+'_community_label_'+video+'.npy'),allow_pickle=True)

        if mode == 'motif' :
            community_label = np.load(os.path.join(path_to_file,"",str(n_cluster)+'_km_label_'+video+'.npy'),allow_pickle=True)



        community = {str(i) : [] for i in np.unique(community_label)}
        print(community)
        cons_condition = np.where(np.diff(community_label)!=0)[0]
        cons = np.split(community_label, cons_condition+1)
        idx = -1
        fig,ax = plt.subplots(len(np.unique(community_label)),10, figsize=(20,20))
        print(len(np.unique(community_label)))
        fig.suptitle('Behaviour Community Kinematics', fontsize=16)
        for id, i in enumerate(np.unique(community_label)) :
            ax[id,0].set_ylabel('Community-'+str(id), rotation=90)
        #ax[3,0].set_ylabel('Community-4', rotation=90)
        #ax[4,0].set_ylabel('Commnuity-5', rotation=90)
        #ax[1]
        for comm in cons :
            if idx == -1 :
                community[str(np.squeeze(np.unique(comm)))].append([0,cons_condition[0]+1])
            else :
                if np.squeeze(np.diff(cons_condition[idx:idx+2])) > thresh :
                    community[str(np.squeeze(np.unique(comm)))].append(cons_condition[idx:idx+2]+1)
            idx+=1

        for idx,(key,val) in enumerate(community.items()) :
            if len(val) < 1 :
                continue

            jdx=0
            feat = 0
            for k in val :
                if jdx == 10:
                    break
                start = k[0]
                num_points = np.squeeze(np.diff(k))
                if num_points < 1 :
                    continue
         
                ax[int(key)-1,jdx].plot(data[:,start:start+num_points].T)
                jdx+=1
                
        ax[0,0].legend(loc='upper right')
        plt.savefig(os.path.join(cfg['project_path'], 'Community-Plots', video+'-community-plot.png'))
        plt.close()
        

def plot_annonations(config) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    path_to_file = cfg['project_path']
    files = cfg['video_sets']
    path_to_data = [file for file in glob.glob(os.path.join(path_to_file, 'annotatedTracks','*.npy'))]
    all_data=[]
    features = ['mosquito velocity','Avg. forelegs velocity', 'Avg. foreleg angle', 'bell size', 'body length']
    for file in files:
        path_to_file = os.path.join(cfg['project_path'],"data", file, file+'-PE-seq.npy')
        data = np.load(path_to_file)
      
        all_data.append(data)

    all_data = np.concatenate(all_data, axis=1)

    x_mean = []
    x_std = []
    for x in all_data :
            x_mean.append(np.mean(x,axis=None))
            x_std.append(np.std(x, axis=None))
            

    for file in path_to_data :
        data = np.load(file)
        print(data.shape)
        X_z = []
        i = 0
        for x in data :

            X_z.append((x - x_mean[i])/x_std[i])
            i+=1

        X_z = np.concatenate([X_z], axis=1).T
        data = X_z.copy()
        annot = pd.read_csv(file[:-4]+'_annotation.csv')
        annot = annot.values
        annot = np.squeeze(np.delete(annot, 0, axis=1))
        split = {i:[] for i in np.unique(annot)}
        temp = []
        for idx in range(len(annot)-1) :
            key = annot[idx]
            if annot[idx] == annot[idx+1] :
                temp.append(idx)

            else :
                split[key].append(temp)
                temp = []
        
        fig, ax = plt.subplots(len(np.unique(annot)), 3, figsize=(10,10))
        for idx, (key, val) in enumerate(split.items()) :
            jdx = 0
            ax[idx,0].set_ylabel(str(key), rotation=90)
            for v in val :
                if jdx == 3 :
                    break
                ax[idx,jdx-1].plot(data[v,:], label=features)
                jdx+=1
        ax[0,0].legend(loc='upper right')
        plt.savefig(file.split('/')[-1]+'-WT.png')
        plt.close()    
        #print(data)


def load_model(cfg, model_name, fixed = False):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        pass
    else:
        torch.device("cpu")
    
    # load Model
    ZDIMS = cfg['zdims']
    FUTURE_DECODER = cfg['prediction_decoder']
    TEMPORAL_WINDOW = cfg['time_window']*2
    FUTURE_STEPS = cfg['prediction_steps']
    NUM_FEATURES = cfg['num_features']
    if fixed == False:
        NUM_FEATURES = NUM_FEATURES - 2
    hidden_size_layer_1 = cfg['hidden_size_layer_1']
    hidden_size_layer_2 = cfg['hidden_size_layer_2']
    hidden_size_rec = cfg['hidden_size_rec']
    hidden_size_pred = cfg['hidden_size_pred']
    dropout_encoder = cfg['dropout_encoder']
    dropout_rec = cfg['dropout_rec']
    dropout_pred = cfg['dropout_pred']
    softplus = cfg['softplus']
     

    if use_gpu:
        model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1, 
                                hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder, 
                                dropout_rec, dropout_pred, softplus).cuda()
    else:
        model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1, 
                                hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder, 
                                dropout_rec, dropout_pred, softplus).to()
    
    model.load_state_dict(torch.load(os.path.join(cfg['project_path'],'model','best_model',model_name+'_'+cfg['Project']+'.pkl')))
    model.eval()
    
    return model


def get_eval_split(config, use_latent=True) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    path_to_file = cfg['project_path']
    model_name = cfg['model_name']
    temp_win = cfg['time_window']
    num_features = cfg['num_features']
    model = load_model(cfg, model_name)
    path_to_file = cfg['project_path']
    files = cfg['video_sets']
    #path_to_data = ['/mnt/DATA1/ali/BitoScopeVAME_new/BiteScopeWithVAME-Mar10-2023/testing/egocentric/annotationData-full.-EGOCENTRIC.npy']

    all_data=[]

    for file in files:
        path_to_dat = os.path.join(cfg['project_path'],"data", file, file+'-PE-seq.npy')
        data = np.load(path_to_dat)
      
        all_data.append(data)

    all_data = np.concatenate(all_data, axis=1)

    x_mean = []
    x_std = []
    for x in all_data :
            x_mean.append(np.mean(x,axis=None))
            x_std.append(np.std(x, axis=None))

    split = {i:[] for i in ['engorge','explore','probe','rest','undefined','walk']}
    data = discretize_egocentric(config, block_size=8, test=True)
    data = np.delete(data,[3,5], axis=0)
    #velocity = np.load( os.path.join(cfg['project_path'],"testing",'annotationData2-velocity.npy'), allow_pickle=True)
    #x_mean.append(np.mean(velocity,axis=None))
    #x_std.append(np.std(velocity, axis=None))
    #velocity = np.concatenate([velocit]], axis=1)
    #data = np.concatenate([data, velocity], axis=0)
    X_z = []
    i = 0
    for x in data :
        X_z.append((x - x_mean[i])/x_std[i])
        i+=1

    X_z = np.concatenate([X_z], axis=1)
    data = X_z.copy()
    data = scipy.signal.savgol_filter(data, cfg['savgol_length'], cfg['savgol_order'])

    annot = pd.read_csv(os.path.join(path_to_file, 'testing', 'annotation-full.csv'))
    annot = annot.values
    annot = np.squeeze(np.delete(annot, 0, axis=1))
    annot_size = annot.shape[0]
    if use_latent :
        latent_vector_list = []
        with torch.no_grad(): 
            for i in tqdm.tqdm(range(data.shape[1] - temp_win)):
                data_sample_np = data[:,i:temp_win+i].T
                data_sample_np = np.reshape(data_sample_np, (1, temp_win, num_features-2))
                h_n = model.encoder(torch.from_numpy(data_sample_np).type('torch.FloatTensor').cuda())
                mu, _, _ = model.lmbda(h_n)
                latent_vector_list.append(mu.cpu().data.numpy())

        latent_vector = np.concatenate(latent_vector_list, axis=0)[:annot_size-temp_win]
        annotations = annot[:-temp_win]
    else :
        latent_vector = data.T[:annot_size]
        annotations = annot

 
    reducer = umap.UMAP(n_components=2, min_dist=cfg['min_dist'], n_neighbors=cfg['n_neighbors'], random_state=cfg['random_state'])
    latents = reducer.fit_transform(latent_vector)
    

    for i in np.unique(annot) :
        if not os.path.exists(os.path.join(path_to_file, 'Annotation-UMAP', i)) :
            os.makedirs(os.path.join(path_to_file, 'Annotation-UMAP', i))

    #latents = np.concatenate(latents, axis=0)
    np.save(os.path.join(cfg['project_path'], 'results', 'annotation_latent_space.npy'), latent_vector)
    np.save(os.path.join(cfg['project_path'], 'results', 'annotations.npy'), annotations)
    return annotations, latents
 

def annotation_latents(config, use_latent=False) :
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    parameterization = cfg['parameterization']
    videos = cfg['video_sets']
    num_points = cfg['num_points']
    latent_vectors = []
    labels = ['engorge', 'explore', 'probe', 'rest', 'undefined', 'walk']
    
    community = {'engorge':0,'explore':1, 'probe':2, 'rest':3, 'undefined':4, 'walk':5}
    annotations, latent_umap = get_eval_split(config, use_latent=use_latent)
    print(latent_umap.shape)
    #annotations = np.load('/mnt/DATA1/ali/BitoScopeVAME_new/BiteScopeWithVAME-Mar10-2023/results/output_annot.npy', allow_pickle=True)
    color_code = []
    for annot in annotations :
        color_code.append(community[annot])

    #print(color_code)

    scatter = plt.scatter(latent_umap[:,0], latent_umap[:,1], c=color_code, cmap='Spectral', s=2, alpha=.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    plt.gca().set_aspect('equal', 'datalim')
    plt.grid(False)
    plt.savefig(os.path.join(cfg['project_path'], 'UMAPs', 'annotatedTrackPose-UMAP.png'))
    plt.close()


def explained_variance(config) :
    config_file = Path(config).resolve() 
    cfg = read_config(config_file)
    project_path = cfg['project_path']
    latent = np.load(os.path.join(project_path, 'data', 'train', 'train_seq.npy'))
    pca = PCA(n_components=2)
    latent_reduced = pca.fit_transform(latent.T)
    print(np.sum(pca.explained_variance_ratio_))


def silhouette_analysis(config) :
    config_file = Path(config).resolve() 
    cfg = read_config(config_file)
    project_path = cfg['project_path']
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    parameterization = cfg['parameterization']
    videos = cfg['video_sets']
    latent_space = np.load(os.path.join(project_path, 'results', 'latent_space_vector.npy'))
    #latent_space = latent_space.T
    labels = []
    for video in videos :
        path_to_file = os.path.join(cfg['project_path'],"results",video,"",model_name,"",parameterization+'-'+str(n_cluster))
        label = np.load(os.path.join(path_to_file,"",str(n_cluster)+'_km_label_'+video+'.npy'),allow_pickle=True)
        labels.append(label)
    
    labels = np.concatenate(labels, axis=0)
    sil_coef = silhouette_score(latent_space, labels, sample_size=100000)
    print(sil_coef)