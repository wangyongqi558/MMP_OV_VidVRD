
import json
import math
from math import log, e
import logging
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import numpy as np
from copy import deepcopy


class FocalWithLogitsLoss(nn.Module):

    def __init__(self, gamma=2, alpha=0.25):
        super(FocalWithLogitsLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        assert input.shape == target.shape

        logpt = - F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(logpt)
        # alpha_t = target*self.alpha + (1-target)*(1-self.alpha)
        focal_loss = -( (1-pt)**self.gamma ) * logpt
        focal_loss = torch.mean(focal_loss)
        return focal_loss

class FocalWithLogitsLossAlpha(nn.Module):

    def __init__(self, gamma=2, alpha=0.25):
        super(FocalWithLogitsLossAlpha, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        assert input.shape == target.shape

        logpt = - F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(logpt)
        alpha_t = target*self.alpha + (1-target)*(1-self.alpha)
        focal_loss = -( (1-pt)**self.gamma ) * logpt * alpha_t
        focal_loss = torch.mean(focal_loss)
        return focal_loss

def cal_weights(types, cat_values, id2pre):

    categories = {cat:[] for cat in id2pre}
    for cat_id, cat in enumerate(categories):
        for t in types:
            categories[cat].append(cat_values[t][cat_id].item())
    
    data = np.array(list(categories.values()))
    cat_weights = torch.from_numpy(data)
    cat_weights = cat_weights/cat_weights.sum(dim=-1).view(-1,1).repeat(1,len(types))
    return cat_weights



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_feat_types(args):
    feat_types = []
    args = args.__dict__
    for k in args:
        if ('_feat' in k) and (args[k] == True):
            feat_types.append(k)
    return feat_types

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, val=0):
        self.reset(val)

    def reset(self, val=0):
        self.avg = val
        self.count = 0

    def update(self, val, n=1):
        assert n > 0
        self.avg = self.avg * (self.count / (self.count + n)) + val * (n / (self.count + n))
        self.count += n
        

def _convert_bbox(bbox):
    x = (bbox[0]+bbox[2])/2
    y = (bbox[1]+bbox[3])/2
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    return x, y, w, h

def vtranse_ext_loc_feat(subj_bbox, obj_bbox):
    subj_x, subj_y, subj_w, subj_h = _convert_bbox(subj_bbox)
    obj_x, obj_y, obj_w, obj_h = _convert_bbox(obj_bbox)
    rx = (subj_x-obj_x)/obj_w
    ry = (subj_y-obj_y)/obj_h
    log_subj_w, log_subj_h = np.log(subj_w), np.log(subj_h)
    log_obj_w, log_obj_h = np.log(obj_w), np.log(obj_h)
    rw = log_subj_w-log_obj_w
    rh = log_subj_h-log_obj_h
    return np.asarray([rx, ry, rw, rh])

def _convert_bbox_relative(bbox, img_h, img_w):
    x1 = bbox[0] / img_w
    x2 = bbox[2] / img_w
    y1 = bbox[1] / img_h
    y2 = bbox[3] / img_h

    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    return [x1, y1, x2, y2, x, y, w, h]

def ext_bbox_loc_feat(s_bbox, o_bbox, u_bbox, img_h, img_w):
    s_relative = np.asarray(_convert_bbox_relative(s_bbox, img_h, img_w))
    o_relative = np.asarray(_convert_bbox_relative(o_bbox, img_h, img_w))
    u_relative = np.asarray(_convert_bbox_relative(u_bbox, img_h, img_w))
    t_relative = np.asarray([0, 0, 1, 1, 0.5, 0.5, 1, 1])
    return s_relative, o_relative, u_relative, t_relative

def aaai18_ext_mask_feat(sbj_box, obj_box, img_h, img_w):
    '''
    # mask location feature calculation referencing
        "[AAAI][18]Visual Relationship Detection with Deep Structural Ranking"
    '''
    rh = 32.0 / img_h
    rw = 32.0 / img_w

    sBB = sbj_box
    x1 = max(0, int(math.floor(sBB[0] * rw)))
    x2 = min(32, int(math.ceil(sBB[2] * rw)))
    y1 = max(0, int(math.floor(sBB[1] * rh)))
    y2 = min(32, int(math.ceil(sBB[3] * rh)))
    mask = np.zeros((32, 32))
    mask[y1 : y2, x1 : x2] = 1
    assert(mask.sum() == (y2 - y1) * (x2 - x1))
    feat = mask
    
    oBB = obj_box
    x1 = max(0, int(math.floor(oBB[0] * rw)))
    x2 = min(32, int(math.ceil(oBB[2] * rw)))
    y1 = max(0, int(math.floor(oBB[1] * rh)))
    y2 = min(32, int(math.ceil(oBB[3] * rh)))
    mask = np.zeros((32, 32))
    mask[y1 : y2, x1 : x2] = 1
    assert(mask.sum() == (y2 - y1) * (x2 - x1))

    feat = np.array([feat,mask])
    return feat

def vru19_ext_loc_feat(sbj_box, obj_box, img_h, img_w):
    '''
    # relative location feature calculation referencing VRU'19 top-1
    '''
    sbj_box = {
        'xmin':sbj_box[0],
        'ymin':sbj_box[1],
        'xmax':sbj_box[2],
        'ymax':sbj_box[3]
    }
    obj_box = {
        'xmin':obj_box[0],
        'ymin':obj_box[1],
        'xmax':obj_box[2],
        'ymax':obj_box[3]
    }
    sbj_h = sbj_box['ymax'] - sbj_box['ymin'] + 1
    sbj_w = sbj_box['xmax'] - sbj_box['xmin'] + 1
    obj_h = obj_box['ymax'] - obj_box['ymin'] + 1
    obj_w = obj_box['xmax'] - obj_box['xmin'] + 1
    spatial_feat = [
        # subject location and size in image
        sbj_box['xmin'] * 1.0 / img_w, 
        sbj_box['ymin'] * 1.0 / img_h,
        sbj_box['xmax'] * 1.0 / img_w, 
        sbj_box['ymax'] * 1.0 / img_h,
        (sbj_h * sbj_w * 1.0) / (img_h * img_w),
        # object location and size in image
        obj_box['xmin'] * 1.0 / img_w, 
        obj_box['ymin'] * 1.0 / img_h,
        obj_box['xmax'] * 1.0 / img_w, 
        obj_box['ymax'] * 1.0 / img_h,
        (obj_h * obj_w * 1.0) / (img_h * img_w),
        # relative location and size of subject and object
        (sbj_box['xmin'] - obj_box['xmin'] + 1) / (obj_w * 1.0),
        (sbj_box['ymin'] - obj_box['ymin'] + 1) / (obj_h * 1.0),
        log(sbj_w * 1.0 / obj_w, e),
        log(sbj_h * 1.0 / obj_h, e)]
    spatial_feat = np.array(spatial_feat)
    return spatial_feat

def adjust_lr(optimizer, args, epoch):
    lr_curr = args.lr * (args.lr_decay ** int((epoch+1) / args.lr_adjust_freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_curr
    return lr_curr

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def relation_filter(rels, filter):
    filtered_rels = defaultdict(list)
    for vid in rels:
        for rel in rels[vid]:
            if rel['triplet'][1] in filter:
                filtered_rels[vid].append(rel)
    return filtered_rels

def gen_union_bbox(sbbox, obbox):
    xmin = min(sbbox[0],obbox[0])
    ymin = min(sbbox[1],obbox[1])
    xmax = max(sbbox[2],obbox[2])
    ymax = max(sbbox[3],obbox[3])
    return [xmin, ymin, xmax, ymax]

def cal_entropy(pred):
    ents = []
    for clip_id in range(len(pred)):
        ent = -torch.sum(pred[clip_id]*torch.log2(pred[clip_id]))
        ents.append(ent)
    return torch.tensor(ents).cuda()

def gen_padding_mask(batch_size, max_len, seq_lens):
    return torch.BoolTensor([[j>=seq_lens[i] for j in range(max_len)] for i in range(batch_size)])

def print_results(logger, results, verbose=False):
    logger.info('detection mean AP (used in challenge): {}'.format(results["mean_ap"]))
    logger.info('detection recall@50:  {}'.format(results["rec_at_n"][50]))
    logger.info('detection recall@100: {}'.format(results["rec_at_n"][100]))
    logger.info('tagging precision@1:  {}'.format(results["mprec_at_n"][1]))
    logger.info('tagging precision@5:  {}'.format(results["mprec_at_n"][5]))
    logger.info('tagging precision@10: {}'.format(results["mprec_at_n"][10]))
    logger.info('predicate detection mean AP : {}'.format(results["pre_mean_ap"]))
    logger.info('predicate detection recall@50: {}'.format(results["pre_mrec_at_n"][50]))
    logger.info('predicate detection recall@100: {}'.format(results["pre_mrec_at_n"][100]))
    # logger.info('predicate tagging precision: {}'.format(results["pre_mprec"]))
    # logger.info('predicate tagging recall: {}'.format(results["pre_mrecall"]))


    if verbose:
        logger.info('predicate detection mean AP : {}'.format(results["pre_mean_ap"]))
        logger.info('-----------detection AP of predicates----------')
        for pre in results["pre_ap"]:
            logger.info('[{}] : {}'.format(pre.center(20,' '), results["pre_ap"][pre]))
        logger.info('-----------detection AP of predicates----------')

        logger.info('predicate detection recall@50: {}'.format(results["pre_mrec_at_n"][50]))
        logger.info('-----------detection recall@50 of predicates----------')
        for pre in results["pre_rec_at_n"][50]:
            logger.info('[{}] : {}'.format(pre.center(20,' '), results["pre_rec_at_n"][50][pre]))
        logger.info('-----------detection recall@50 of predicates----------')

        logger.info('predicate detection recall@100: {}'.format(results["pre_mrec_at_n"][100]))
        logger.info('-----------detection recall@100 of predicates----------')
        for pre in results["pre_rec_at_n"][100]:
            logger.info('[{}] : {}'.format(pre.center(20,' '), results["pre_rec_at_n"][100][pre]))
        logger.info('-----------detection recall@100 of predicates----------')

        # logger.info('predicate tagging precision@1: {}'.format(results["pre_mprec_at_n"][1]))
        # logger.info('-----------tagging precision@1 of predicates----------')
        # for pre in results["pre_prec_at_n"][1]:
        #     logger.info('[{}] : {}'.format(pre.center(20,' '), results["pre_prec_at_n"][1][pre]))
        # logger.info('-----------tagging precision@1 of predicates----------')

        # logger.info('predicate tagging precision@5: {}'.format(results["pre_mprec_at_n"][5]))
        # logger.info('-----------tagging precision@5 of predicates----------')
        # for pre in results["pre_prec_at_n"][5]:
        #     logger.info('[{}] : {}'.format(pre.center(20,' '), results["pre_prec_at_n"][5][pre]))
        # logger.info('-----------tagging precision@5 of predicates----------')

        # logger.info('predicate tagging precision@10: {}'.format(results["pre_mprec_at_n"][10]))
        # logger.info('-----------tagging precision@10 of predicates----------')
        # for pre in results["pre_prec_at_n"][10]:
        #     logger.info('[{}] : {}'.format(pre.center(20,' '), results["pre_prec_at_n"][10][pre]))
        # logger.info('-----------tagging precision@10 of predicates----------')
    
