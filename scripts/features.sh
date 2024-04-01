#!/bin/sh
gpu_id=0

clip_len=30
rel_feat=True
mask_feat=False
lan_feat=False
v2d_feat=False
mot_feat=True
v3d_feat=False
clip_feat=True
intern_feat=False
bbox_feat=True
dataset=vidvrd

CUDA_VISIBLE_DEVICES=$gpu_id python features.py\
	--clip_len ${clip_len}\
    --rel_feat ${rel_feat}\
    --mask_feat ${mask_feat}\
    --lan_feat ${lan_feat}\
    --v2d_feat ${v2d_feat}\
    --mot_feat ${mot_feat}\
    --v3d_feat ${v3d_feat}\
    --clip_feat ${clip_feat}\
    --intern_feat ${intern_feat}\
    --bbox_feat ${bbox_feat}\
    --dataset ${dataset}
    #--use_gt_traj

 


