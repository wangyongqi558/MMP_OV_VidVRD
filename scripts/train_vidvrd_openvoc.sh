#!/bin/sh
gpu_id=0
lr=1e-3
dropout=0.5
clip_emb_dim=512
temp_out_dim=512
temp_model=none
num_layers=0
max_epoch=30
batch_size=1
clip_len=30
ckpt_path='../dataset/vidvrd/model/model_stage1.pth'
ps='stage2'
mask_feat=False
lan_feat=False
rel_feat=True
v2d_feat=False
mot_feat=True
v3d_feat=False
clip_feat=True
intern_feat=False
train_traj=gt
test_traj=meta
dataset=vidvrd
ptm_mode=vision_text
src_split=base
tgt_split=all
obj_loss_weight=0.2
int_loss_weight=0.1

CUDA_VISIBLE_DEVICES=$gpu_id python train.py "$@"\
	--lr ${lr} \
	--dropout ${dropout} \
    --clip_emb_dim ${clip_emb_dim} \
    --temp_out_dim ${temp_out_dim} \
    --ckpt_path ${ckpt_path} \
    --num_layers ${num_layers} \
    --temp_model ${temp_model} \
	--batch_size ${batch_size}\
	--ps ${ps}\
	--max_epoch ${max_epoch}\
    --clip_len ${clip_len}\
	--rel_feat ${rel_feat}\
    --mask_feat ${mask_feat}\
    --lan_feat ${lan_feat}\
    --v2d_feat ${v2d_feat}\
    --mot_feat ${mot_feat}\
    --v3d_feat ${v3d_feat}\
    --clip_feat ${clip_feat}\
    --intern_feat ${intern_feat}\
    --train_traj ${train_traj}\
    --test_traj ${test_traj}\
    --dataset ${dataset}\
    --ptm_mode ${ptm_mode}\
    --src_split ${src_split}\
    --tgt_split ${tgt_split}\
    --obj_loss_weight ${obj_loss_weight}\
    --int_loss_weight ${int_loss_weight}\
    --print_freq 10
