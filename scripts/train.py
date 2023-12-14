import numpy as np
from collections import defaultdict
from os.path import join

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from dataset import Dataset, padding_collate_fn
from utils.parser_func import parse_args
from utils.utils import get_feat_types, AverageMeter, get_logger, print_results
from utils.post_process import process_pred, association, format_
from utils.video_relation_detection_openvoc import eval_relation_detection_openvoc


def seed_everything(seed = 3047):
    import random, os
    import numpy as np
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    
    seed_everything(3047)

    args = parse_args()

    from model_zoo.model_stage2 import Model
    model = Model(args).cuda()
    best_mmap = 0
    if args.resume or args.stage2:
        ckpt = torch.load(args.ckpt_path)
        if args.resume:
            args = ckpt['config']
            args.start_epoch = ckpt['epoch']
            best_mmap = sum(ckpt['map'])/4
        pretrained_dict = ckpt['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    feat_types = get_feat_types(args)
    feat_config = "_"
    for type_ in feat_types:
        feat_config += type_.split("_")[0] + "_"
    env_config = args.dataset+ \
        "_bs"+str(args.batch_size)+ \
        "_lr"+str(args.lr)+ \
        "_dim"+str(args.clip_emb_dim)+ \
        "_"+str(args.temp_model)+ \
        feat_config+args.ps
    
    logger = get_logger(join('.','log',env_config+'_train.log'))
    logger.info('Experiment Config: {}'.format(args))

    logger.info("Preparing data from %s..."%args.dataset)
    train_dataset = Dataset(args, "train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=padding_collate_fn, num_workers=8)
    args.test_traj = 'meta'
    val_dataset_det = Dataset(args, "test")
    val_loader_det = DataLoader(val_dataset_det, batch_size=args.batch_size, shuffle=False, collate_fn=padding_collate_fn, num_workers=8)
    args.test_traj = 'gt'
    val_dataset_gt = Dataset(args, "test")
    val_loader_gt = DataLoader(val_dataset_gt, batch_size=args.batch_size, shuffle=False, collate_fn=padding_collate_fn, num_workers=8)
    
    for name, param in model.named_parameters():
        if "text_encoder" in name:
            param.requires_grad_(False)
        elif "featEmbedding" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20, 25], gamma=0.1)
    
    epoch_loss = AverageMeter()
    
    for epoch in range(args.start_epoch+1, args.max_epoch+1):      
 
        model.train()
        logger.info("Training data from %s..."%args.dataset)
        batch_loss = AverageMeter()
        for idx, (data, seq_lens) in enumerate(tqdm(train_loader)):
            
            labels = {
                'pre_label': data['pre_label'].cuda(),
                'sbj_label': data['sbj_label'].cuda(),
                'obj_label': data['obj_label'].cuda()
            }
            feats = {}
            for k in data:
                if 'feat' in k:
                    feats[k] = data[k]
            
            loss = model(feats, seq_lens, labels)
            loss.requires_grad_(True)
            loss.backward()
            if ((idx+1)%32)==0:
                optimizer.step()
                optimizer.zero_grad()

            batch_loss.update(loss.item()/args.batch_size, args.batch_size)
            epoch_loss.update(loss.item()/args.batch_size, args.batch_size)
        scheduler.step()
        logger.info('Epoch: [{0}] \t LR: [{1}] \t Avg Train Loss:  {loss.avg:.4f}'.\
            format(epoch, optimizer.state_dict()['param_groups'][0]['lr'], loss=epoch_loss))  
        epoch_loss.reset()

        model.eval()
        map_list = []
        
        model.tgt_split = 'all'
        pred_rels = defaultdict(list)
        with torch.no_grad():
            for data, seq_lens in val_loader_det:
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
        mean_ap, rec_at_n = eval_relation_detection_openvoc(target_split_pred='all', prediction_results=pred_rels, rt_hit_infos=True)
        logger.info("SGDet and All split     | mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(mean_ap*100, rec_at_n[50]*100, rec_at_n[100]*100))
        map_list.append(mean_ap*100)

        model.tgt_split = 'novel'
        pred_rels = defaultdict(list)
        with torch.no_grad():
            for data, seq_lens in val_loader_det:
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
        mean_ap, rec_at_n = eval_relation_detection_openvoc(target_split_pred='novel', prediction_results=pred_rels, rt_hit_infos = True)
        logger.info("SGDet and Novel split   | mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(mean_ap*100, rec_at_n[50]*100, rec_at_n[100]*100))
        map_list.append(mean_ap*100)

        model.tgt_split = 'all'
        pred_rels = defaultdict(list)
        with torch.no_grad():
            for data, seq_lens in val_loader_gt:
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
        mean_ap, rec_at_n = eval_relation_detection_openvoc(target_split_pred='all', prediction_results=pred_rels)
        logger.info("PredCls and All split   | mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(mean_ap*100, rec_at_n[50]*100, rec_at_n[100]*100))
        map_list.append(mean_ap*100)

        model.tgt_split = 'novel'
        pred_rels = defaultdict(list)
        with torch.no_grad():
            for data, seq_lens in val_loader_gt:
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
        mean_ap, rec_at_n = eval_relation_detection_openvoc(target_split_pred='novel', prediction_results=pred_rels)
        logger.info("PredCls and Novel split | mAP:{:.2f}, Recall@50:{:.2f}, Recall@100:{:.2f}".format(mean_ap*100, rec_at_n[50]*100, rec_at_n[100]*100))
        map_list.append(mean_ap*100)

        mmap = sum(map_list)/4

        logger.info(f"Mean mAP: {mmap}, Best Mean mAP: {best_mmap}")
        if mmap > best_mmap:
            best_mmap = mmap
            state = {
                'map': map_list,
                'epoch': epoch,
                'config': args,
                'state_dict': model.state_dict()}
            ckpt_path = join('..', 'dataset', args.dataset, 'model', env_config+".pth")
            torch.save(state, ckpt_path)
    
    print("================================Final Results======================================")
    logger.info('Best Epoch: {}, mAP List: {}'.format(state['epoch'], str(state['map'])))



