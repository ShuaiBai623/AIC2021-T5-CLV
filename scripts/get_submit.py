import json
import cv2
import os 
import pickle
import numpy as np
import torch
from numpy import linalg as LA
import torch
import torch.nn.functional as F

def get_mean_feats1(img_feat, tacks_ids):
    mean_gallery = []
    for k in tacks_ids:
        tmp = []
        for fid in  img_feat[k]:
            tmp.append(img_feat[k][fid])
        tmp = np.vstack(tmp)
        tmp = np.mean(tmp,0)
        mean_gallery.append(tmp)
    mean_gallery = np.vstack(mean_gallery)
    return mean_gallery

def get_mean_feats2(img_feat, tacks_ids):
    mean_gallery = []
    for k in tacks_ids:
       mean_gallery.append(img_feat[(k,)])
    mean_gallery = np.vstack(mean_gallery)
    mean_gallery = torch.from_numpy(mean_gallery)
    mean_gallery = F.normalize(mean_gallery, p = 2, dim = -1).numpy()
    return mean_gallery



used_models1 = ["motion_effb3_NOCLS_nlpaug_320","motion_effb2_1CLS_nlpaug_288","motion_SE_3CLS_nonlpaug_288","motion_SE_NOCLS_nlpaug_288", "motion_SE_NOCLS_nonlpaug_288"]
used_models2 = ["b48_lr1_w20_clip_e120_dd","b36_lr1_w20_clip_e100_dd_drop0.9_CROP256_nseg3_pad0.2","b48_lr1_w20_clip_e100_dd_drop0.75_CROP320_pad0.2"]
merge_weights1 = [4.,1.,1.,1.,1.]
merge_weights2 = [1.,1.,1.]



with open("data/test-queries.json") as f:
    queries = json.load(f)
with open("data/test-tracks.json") as f:
    tracks = json.load(f)
query_ids = list(queries.keys())
tacks_ids = list(tracks.keys())
print(len(tacks_ids),len(query_ids))

img_feats1 = []
img_feats2 = []
nlp_feats1 = []
nlp_feats2 = []
for i,m in enumerate(used_models1):
    img_feats1.append(get_mean_feats1(pickle.load(open("output/img_feat_%s.pkl"%m,'rb')),tacks_ids))
    nlp_feats1.append(pickle.load(open("output/lang_feat_%s.pkl"%m,'rb')))
for i,m in enumerate(used_models2):
    img_feats2.append(get_mean_feats2(pickle.load(open("output/submit_result_%s_gallery.pkl"%m,'rb')),tacks_ids))
    nlp_feats2.append(pickle.load(open("output/submit_result_%s_query.pkl"%m,'rb')))


results = dict()

for query in query_ids:
    score = 0.
    for i in range(len(nlp_feats1)):
        q = nlp_feats1[i][query]
        score += merge_weights1[i]*np.mean(np.matmul(q,img_feats1[i].T),0)

    for i in range(len(nlp_feats2)):
        q = nlp_feats2[i][query]
        q = torch.from_numpy(q).view(1,-1)
        q = F.normalize(q, p = 2, dim = -1).numpy()
        score += merge_weights2[i]*np.matmul(q,img_feats2[i].T)[0]

    index = np.argsort(score)[::-1]
    results[query]=[]
    for i in index:
        results[query].append(tacks_ids[i])
with open("results_mergefinal.json", "w") as f:
    json.dump(results, f,indent=4)
