import napari
from napari_video.napari_video import VideoReaderNP
import numpy as np
import yaml
import os
import pandas as pd
import sys

VIDEO_PATH = '/mnt/DATA1/ali/BitoScopeVAME_new/BiteScopeWithVAME-Mar10-2023/marked videos/210715_KPPTN_denv3_4DLC_dlcrnetms5_aedesJun22shuffle1_72500_full.mp4'
output_name = VIDEO_PATH[:-4] + "_to_remove.npy"
vr = VideoReaderNP(VIDEO_PATH)
remove = []

# global vars
flag_exist = False
start_frame = 0
end_frame = 0

viewer = napari.Viewer()


@viewer.bind_key('s')
def set_start_flag(event=None):
    global flag_exist
    global start_frame
    global end_frame

    if flag_exist:
        print('flag already exist!!')
    else:
        start_frame = image_layer._slice_indices[0]
        flag_exist = True
        print("start frame set to frame:", image_layer._slice_indices[0])

@viewer.bind_key('e')
def annotate_A(event=None):
    global flag_exist
    global start_frame
    global end_frame
    end_frame = image_layer._slice_indices[0]
    if flag_exist:
        if start_frame > end_frame:
            print("go to the later frame than the start flag.")
            show_globals()
        else:
            print(start_frame, "to", end_frame, "are removed")
            remove.append([start_frame,end_frame])
            flag_exist = False
    else:
        print("need to set the start flag.")


@viewer.bind_key('v')
def show_globals(event=None):
    global start_frame
    global flag_exist
    global end_frame
    print("flag:", flag_exist)
    print("pos_start:", start_frame)
    print("pos_end:", start_frame)

@viewer.bind_key('1')
def delete_flag(event=None):
    global flag_exist
    print("delete_flag")
    flag_exist = False

@viewer.bind_key('0')
def save_label(event=None):
    print("save!")
    np.save(output_name, remove)

image_layer = viewer.add_image(vr, rgb=True)

napari.run()