import cv2
import json
import numpy as np
from tqdm import tqdm
import os
import multiprocessing
from functools import partial
imgpath = "data/AIC21_Track5_NL_Retrieval/"
with open("data/test-tracks.json") as f:
    tracks_test = json.load(f)
with open("data/test-tracks.json") as f:
    tracks_train = json.load(f)
all_tracks = tracks_train
for track in tracks_test:
    all_tracks[track] = tracks_test[track]
n_worker = 12
import glob
save_bk_dir = "data/bk_map"
os.makedirs(save_bk_dir,exist_ok = True)
save_mo_dir = "data/motion_map"
os.makedirs(save_mo_dir,exist_ok = True)
def get_bk_map(info):
    path,save_name = info
    # print(path)
    img = glob.glob(path+"/img1/*")
    img.sort()
    interval = min(5,max(1,int(len(img)/200)))
    img = img[::interval][:1000]
    imgs=[]
    for name in img:
        imgs.append(cv2.imread(name))
    avg_img = np.mean(np.stack(imgs),0)
    avg_img = avg_img.astype(np.int)
    cv2.imwrite(save_bk_dir+"/%s.jpg"%save_name,avg_img)
    return path,avg_img.shape,name


def get_motion_map(info):
    track,track_id = info
    for i in range(len(track["frames"])):
        frame_path = track["frames"][i]
        frame_path = os.path.join(imgpath, frame_path)
        frame = cv2.imread(frame_path)
        box = track["boxes"][i]
        if i ==0:
            example = np.zeros(frame.shape,np.int)
        if i%7==1:
            example[box[1]:box[1] + box[3], box[0]: box[0] + box[2], :] = frame[box[1]:box[1] + box[3], box[0]: box[0] + box[2], :]

    words = track["frames"][0].split('/')
    avg_img = cv2.imread("data/bk_map/"+words[-4]+'_'+words[-3]+'.jpg').astype(np.int)
    postions = (example[:,:,0]==0)&(example[:,:,1]==0)&(example[:,:,2]==0)
    example[postions] = avg_img[postions]
    cv2.imwrite(save_mo_dir+"/%s.jpg"%track_id,example)



root = "data/AIC21_Track5_NL_Retrieval/"
paths = ["train/S01","train/S03","train/S04","validation/S02","validation/S05"]
files =[]
for path in paths:
    seq_list = os.listdir(root+path)
    for seq in seq_list:
        files.append((os.path.join(root+path,seq),path[-3:]+'_'+seq))
with multiprocessing.Pool(n_worker) as pool:
     for imgs in tqdm(pool.imap_unordered(get_bk_map, files)):
         pass

all_tracks_ids = list(all_tracks.keys())
files = []
for track_id in all_tracks:
    files.append((all_tracks[track_id],track_id))

with multiprocessing.Pool(n_worker) as pool:
    for imgs in tqdm(pool.imap_unordered(get_motion_map, files)):
        pass
 
