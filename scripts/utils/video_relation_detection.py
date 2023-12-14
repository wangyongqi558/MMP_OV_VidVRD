from collections import defaultdict
import re
from tqdm import tqdm
import numpy as np
from copy import deepcopy

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).

    Adopted from https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def iou(bbox_1, bbox_2):
    """
    Get IoU value of two bboxes
    :param bbox_1:
    :param bbox_2:
    :return: IoU
    """
    w_1 = bbox_1[2] - bbox_1[0] + 1
    h_1 = bbox_1[3] - bbox_1[1] + 1
    w_2 = bbox_2[2] - bbox_2[0] + 1
    h_2 = bbox_2[3] - bbox_2[1] + 1
    area_1 = w_1 * h_1
    area_2 = w_2 * h_2

    overlap_bbox = (max(bbox_1[0], bbox_2[0]), max(bbox_1[1], bbox_2[1]),
                    min(bbox_1[2], bbox_2[2]), min(bbox_1[3], bbox_2[3]))
    overlap_w = max(0, (overlap_bbox[2] - overlap_bbox[0] + 1))
    overlap_h = max(0, (overlap_bbox[3] - overlap_bbox[1] + 1))

    overlap_area = overlap_w * overlap_h
    union_area = area_1 + area_2 - overlap_area
    IoU = overlap_area * 1.0 / union_area
    return IoU


def viou(traj_1, duration_1, traj_2, duration_2):
    """ compute the voluminal Intersection over Union
    for two trajectories, each of which is represented
    by a duration [fstart, fend) and a list of bounding
    boxes (i.e. traj) within the duration.
    """
    if duration_1[0] >= duration_2[1] or duration_1[1] <= duration_2[0]:
        return 0.
    elif duration_1[0] <= duration_2[0]:
        head_1 = duration_2[0] - duration_1[0]
        head_2 = 0
        if duration_1[1] < duration_2[1]:
            tail_1 = duration_1[1] - duration_1[0]
            tail_2 = duration_1[1] - duration_2[0]
        else:
            tail_1 = duration_2[1] - duration_1[0]
            tail_2 = duration_2[1] - duration_2[0]
    else:
        head_1 = 0
        head_2 = duration_1[0] - duration_2[0]
        if duration_1[1] < duration_2[1]:
            tail_1 = duration_1[1] - duration_1[0]
            tail_2 = duration_1[1] - duration_2[0]
        else:
            tail_1 = duration_2[1] - duration_1[0]
            tail_2 = duration_2[1] - duration_2[0]
    v_overlap = 0
    for i in range(tail_1 - head_1):
        roi_1 = traj_1[head_1 + i]
        roi_2 = traj_2[head_2 + i]
        left = max(roi_1[0], roi_2[0])
        top = max(roi_1[1], roi_2[1])
        right = min(roi_1[2], roi_2[2])
        bottom = min(roi_1[3], roi_2[3])
        v_overlap += max(0, right - left + 1) * max(0, bottom - top + 1)
    v1 = 0
    for i in range(len(traj_1)):
        v1 += (traj_1[i][2] - traj_1[i][0] + 1) * (traj_1[i][3] - traj_1[i][1] + 1)
    v2 = 0
    for i in range(len(traj_2)):
        v2 += (traj_2[i][2] - traj_2[i][0] + 1) * (traj_2[i][3] - traj_2[i][1] + 1)
    return float(v_overlap) / (v1 + v2 - v_overlap)

def eval_detection_scores(gt_relations, pred_relations, viou_threshold, top_returns=None):
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    if top_returns is not None:
        pred_relations = pred_relations[:top_returns]
    gt_detected = np.zeros((len(gt_relations),), dtype=bool)
    hit_scores = np.ones((len(pred_relations))) * -np.inf
    classes = []
    scores = []
    for pred_idx, pred_relation in enumerate(pred_relations):
        pred_relation['hit_gt'] = False
        classes.append(pred_relation['triplet'][1])
        scores.append(pred_relation['score'])
        ov_max = -float('Inf')
        k_max = -1
        for gt_idx, gt_relation in enumerate(gt_relations):
            if not gt_detected[gt_idx]\
                    and tuple(pred_relation['triplet']) == tuple(gt_relation['triplet']):
                s_iou = viou(pred_relation['sub_traj'], pred_relation['duration'],
                        gt_relation['sub_traj'], gt_relation['duration'])
                o_iou = viou(pred_relation['obj_traj'], pred_relation['duration'],
                        gt_relation['obj_traj'], gt_relation['duration'])
                ov = min(s_iou, o_iou)
                if ov >= viou_threshold and ov > ov_max:
                    ov_max = ov
                    k_max = gt_idx
        if k_max >= 0:
            hit_scores[pred_idx] = pred_relation['score']
            pred_relation['hit_gt'] = True
            gt_detected[k_max] = True
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_relations), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    assert len(tp) == len(classes)
    return prec, rec, tp, classes, scores

def eval_tagging_scores(gt_relations, pred_relations, top_returns=None):
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    if top_returns is not None:
        pred_relations = pred_relations[:top_returns]

    # ignore trajectories
    gt_triplets = set(tuple(r['triplet']) for r in gt_relations)
    pred_triplets = []
    hit_scores = []
    classes = []
    for r in pred_relations:
        triplet = tuple(r['triplet'])
        if not triplet in pred_triplets:
            pred_triplets.append(triplet)
            hit_scores.append(r['score'])
            classes.append(triplet[1])
    hit_scores = np.asarray(hit_scores)
    for i, t in enumerate(pred_triplets):
        if not t in gt_triplets:
            hit_scores[i] = -np.inf
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_triplets), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    assert len(tp) == len(classes)
    return prec, rec, tp, classes


def evaluate(groundtruth, prediction, pre_list, top_returns=200, viou_threshold=0.5, det_recall_at_n=[50, 100], tag_precision_at_n=[1, 5, 10]):
    """ evaluate visual relation detection and visual
    relation tagging.
    """
    video_ap = dict()
    tot_tps = defaultdict(list)
    prec_at_n = defaultdict(list)
    tot_gt_relations = 0

    video_pre_ap = defaultdict(dict)
    video_pre_tps = defaultdict(list)
    video_pre_scores = defaultdict(list)

    pre_tps = defaultdict(list)
    pre_scores = defaultdict(list)
    meta_dict = {}
    for pre in pre_list:
        meta_dict[pre] = []
    pre_det_tps = {nre:deepcopy(meta_dict) for nre in det_recall_at_n}
    pre_gt_relations = defaultdict(int)

    print('Computing average precision AP over {} videos...'.format(len(groundtruth)))
    print('Actual detection results of {} videos...'.format(len(prediction)))
    
    for vid, gt_relations in groundtruth.items():
        if len(gt_relations)==0:
            continue
        tot_gt_relations += len(gt_relations)
        for rel in gt_relations:
            pre_gt_relations[rel["triplet"][1]] += 1

        vid = vid.split('/')[-1]
        predict_relations = prediction.get(vid, [])
        # compute average precision and recalls in detection setting

        det_prec, det_rec, det_tps, det_classes, det_scores = eval_detection_scores(
                gt_relations, predict_relations, viou_threshold, top_returns)
        
        
        video_ap[vid] = voc_ap(det_rec, det_prec)
        # collect scores and tp/fp results for predicate mean ap in detection
        for det_id, det_class in enumerate(det_classes):
            pre_scores[det_class].append(det_scores[det_id])
            pre_tps[det_class].append(det_tps[det_id])

        for nre in det_recall_at_n:
            cut_off = min(nre, len(det_scores))
            tot_tps[nre].append(det_tps[:cut_off])

            # collect tp/fp results for predicate mean recall in detection
            for det_id, det_class in enumerate(det_classes[:cut_off]):
                pre_det_tps[nre][det_class].append(det_tps[det_id])

        # compute precisions in tagging setting
        tag_prec, _, tag_tps, tag_classes = eval_tagging_scores(gt_relations, predict_relations, top_returns)
        for nre in tag_precision_at_n:
            cut_off = min(nre, len(tag_prec))
            if cut_off > 0:
                prec_at_n[nre].append(tag_prec[cut_off - 1])
            else:
                prec_at_n[nre].append(0.)


    # calculate video mean ap for detection
    mean_ap = round(float(np.mean(list(video_ap.values()))), 4)
    # calculate predicate mean ap for detection
    pre_ap = dict()
    for pre in pre_list:
        scores = np.asarray(pre_scores[pre])
        if len(scores) == 0:
            pre_ap[pre] = 0.0
            continue
        sort_indices = np.argsort(scores)[::-1]
        tp = np.asarray(pre_tps[pre])
        tp = tp[sort_indices]
        fp = ~tp
        cum_tp = np.cumsum(tp).astype(np.float32)
        cum_fp = np.cumsum(fp).astype(np.float32)
        rec = cum_tp / np.maximum(pre_gt_relations[pre], np.finfo(np.float32).eps)
        prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
        pre_ap[pre] = voc_ap(rec, prec)
    pre_mean_ap = round(float(np.mean(list(pre_ap.values()))), 4)

    # calculate recall and predicate mean recall for detection
    rec_at_n = dict()
    pre_rec_at_n = defaultdict(dict)
    pre_mrec_at_n = dict()
    for nre in det_recall_at_n:
        tps = np.concatenate(tot_tps[nre])
        cum_tp = np.cumsum(tps).astype(np.float32)
        rec = cum_tp / np.maximum(tot_gt_relations, np.finfo(np.float32).eps)
        if len(rec) > 0:
            rec_at_n[nre] = round(float(rec[-1]), 4)
        else:
            rec_at_n[nre] = 0.
        
        for pre in pre_list:
            tp = np.asarray(pre_det_tps[nre][pre])
            if len(tp) == 0:
                pre_rec_at_n[nre][pre] = 0.0
                continue
            cum_tp = np.cumsum(tp).astype(np.float32)
            rec = cum_tp / np.maximum(pre_gt_relations[pre], np.finfo(np.float32).eps)
            pre_rec_at_n[nre][pre] = round(float(rec[-1]), 4)
        pre_mrec_at_n[nre] = round(float(np.mean(list(pre_rec_at_n[nre].values()))), 4)     
            
    # calculate mean precision for tagging
    mprec_at_n = dict()
    for nre in tag_precision_at_n:
        mprec_at_n[nre] = round(float(np.mean(prec_at_n[nre])), 4)
    

    return {
        "mean_ap":mean_ap, 
        "rec_at_n":rec_at_n, 
        "mprec_at_n":mprec_at_n, 
        "pre_mean_ap":pre_mean_ap, 
        "pre_mrec_at_n":pre_mrec_at_n,
        "pre_ap":pre_ap,
        "pre_rec_at_n":pre_rec_at_n}