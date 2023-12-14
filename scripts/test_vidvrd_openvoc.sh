#!/bin/sh
gpu_id=0
batch_size_eval=1
ckpt='../dataset/vidvrd/model/model_stage2.pth'
CUDA_VISIBLE_DEVICES=$gpu_id python test_openvoc.py\
    --batch_size_eval ${batch_size_eval}\
    --ckpt ${ckpt}
