from collections import defaultdict
from os.path import join
import copy
import json
from math import log

import torch
from torch.nn.modules.activation import Sigmoid
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset import Dataset, padding_collate_fn
from dataset_openvoc import Dataset as DatasetOpenVoc
from utils.parser_func import parse_args
from utils.video_relation_detection import evaluate
from utils.video_relation_detection_openvoc import eval_relation_detection_openvoc
from utils.utils import get_feat_types, AverageMeter, get_logger, relation_filter, cal_entropy, print_results
from utils.post_process import process_pred, association, format_

from model_zoo.model_stage2 import Model

if __name__ == '__main__':
    
    test_args = parse_args()
    ckpt = torch.load(test_args.ckpt_path)
    print(ckpt['map'])
    args = ckpt['config']
    args.batch_size_eval = test_args.batch_size_eval

    feat_types = get_feat_types(args)
    feat_config = "_"
    for type_ in feat_types:
        feat_config += type_.split("_")[0] + "_"
    env_config = \
        args.dataset+ \
        "_bs"+str(args.batch_size)+ \
        "_lr"+str(args.lr)+ \
        "_drop"+str(args.dropout)+ \
        "_dim"+str(args.clip_emb_dim)+ \
        feat_config+args.ps
    logger = get_logger(join('.','log',env_config+'_eval.log'))
    logger.info('Experiment Config: {}'.format(args))

    args.test_traj = 'meta'
    val_dataset_det = Dataset(args, "test")
    val_loader_det = DataLoader(val_dataset_det, batch_size=args.batch_size_eval, shuffle=False, collate_fn=padding_collate_fn, num_workers=8)
    
    args.test_traj = 'gt'
    val_dataset_gt = Dataset(args, "test")
    val_loader_gt = DataLoader(val_dataset_gt, batch_size=args.batch_size_eval, shuffle=False, collate_fn=padding_collate_fn, num_workers=8)
    
    model = Model(args).cuda()
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    model.tgt_split = 'all'
    pred_rels = defaultdict(list)
    with torch.no_grad():
        for data, seq_lens in tqdm(val_loader_det):
            vids = data['vid']
            pair_data = data['pair_data']
            feats = {}
            for k in data:
                if k not in ['vid', 'pair_data']:
                    feats[k] = data[k]
            pre_preds, sbj_preds, obj_preds = model(feats, seq_lens)
            for seq_id, seq_len in enumerate(seq_lens):
                clip_rels = process_pred(args, val_dataset_det.id2pre, val_dataset_det.obj2id, val_dataset_det.prior, pre_preds[seq_id][:seq_len], pair_data[seq_id])
                pred_rels[vids[seq_id]].extend(association(clip_rels))
    for vid in pred_rels:
        pred_rels[vid] = format_(args, pred_rels[vid])
    logger.info("==============Evaluation for SGDet and All split==============")
    mean_ap, rec_at_n = eval_relation_detection_openvoc(target_split_pred='all', prediction_results=pred_rels)
    logger.info("mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(mean_ap*100, rec_at_n[50]*100, rec_at_n[100]*100))

    model.tgt_split = 'novel'
    pred_rels = defaultdict(list)
    with torch.no_grad():
        for data, seq_lens in tqdm(val_loader_det):
            vids = data['vid']
            pair_data = data['pair_data']
            feats = {}
            for k in data:
                if k not in ['vid', 'pair_data']:
                    feats[k] = data[k]
            pre_preds, sbj_preds, obj_preds = model(feats, seq_lens)
            for seq_id, seq_len in enumerate(seq_lens):
                clip_rels = process_pred(args, val_dataset_det.id2pre, val_dataset_det.obj2id, val_dataset_det.prior, pre_preds[seq_id][:seq_len], pair_data[seq_id])
                pred_rels[vids[seq_id]].extend(association(clip_rels))
    for vid in pred_rels:
        pred_rels[vid] = format_(args, pred_rels[vid])
    logger.info("==============Evaluation for SGDet and Novel split==============")
    mean_ap, rec_at_n = eval_relation_detection_openvoc(target_split_pred='novel', prediction_results=pred_rels)
    logger.info("mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(mean_ap*100, rec_at_n[50]*100, rec_at_n[100]*100))

    model.tgt_split = 'all'
    pred_rels = defaultdict(list)
    with torch.no_grad():
        for data, seq_lens in tqdm(val_loader_gt):
            vids = data['vid']
            pair_data = data['pair_data']
            feats = {}
            for k in data:
                if k not in ['vid', 'pair_data']:
                    feats[k] = data[k]
            pre_preds, sbj_preds, obj_preds = model(feats, seq_lens)
            for seq_id, seq_len in enumerate(seq_lens):
                clip_rels = process_pred(args, val_dataset_gt.id2pre, val_dataset_gt.obj2id, val_dataset_gt.prior, pre_preds[seq_id][:seq_len], pair_data[seq_id])
                pred_rels[vids[seq_id]].extend(association(clip_rels))
    for vid in pred_rels:
        pred_rels[vid] = format_(args, pred_rels[vid])
    logger.info("==============Evaluation for PredCls and All split==============")
    mean_ap, rec_at_n = eval_relation_detection_openvoc(target_split_pred='all', prediction_results=pred_rels)
    logger.info("mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(mean_ap*100, rec_at_n[50]*100, rec_at_n[100]*100))

    model.tgt_split = 'novel'
    pred_rels = defaultdict(list)
    with torch.no_grad():
        for data, seq_lens in tqdm(val_loader_gt):
            vids = data['vid']
            pair_data = data['pair_data']
            feats = {}
            for k in data:
                if k not in ['vid', 'pair_data']:
                    feats[k] = data[k]
            pre_preds, sbj_preds, obj_preds = model(feats, seq_lens)
            for seq_id, seq_len in enumerate(seq_lens):
                clip_rels = process_pred(args, val_dataset_gt.id2pre, val_dataset_gt.obj2id, val_dataset_gt.prior, pre_preds[seq_id][:seq_len], pair_data[seq_id])
                pred_rels[vids[seq_id]].extend(association(clip_rels))
    for vid in pred_rels:
        pred_rels[vid] = format_(args, pred_rels[vid])
    logger.info("==============Evaluation for PredCls and Novel split==============")
    mean_ap, rec_at_n = eval_relation_detection_openvoc(target_split_pred='novel', prediction_results=pred_rels)
    logger.info("mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(mean_ap*100, rec_at_n[50]*100, rec_at_n[100]*100))