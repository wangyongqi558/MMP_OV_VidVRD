import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm

def load_json(path):
    with open(path,'r') as f:
        x = json.load(f)
    return x

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



def eval_detection_scores(gt_relations, pred_relations, viou_threshold):
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    gt_detected = np.zeros((len(gt_relations),), dtype=bool)
    hit_scores = np.ones((len(pred_relations))) * -np.inf
    for pred_idx, pred_relation in enumerate(pred_relations):
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
            gt_detected[k_max] = True
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_relations), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    return prec, rec, hit_scores


def eval_tagging_scores(gt_relations, pred_relations):
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    # ignore trajectories
    gt_triplets = set(tuple(r['triplet']) for r in gt_relations)
    pred_triplets = []
    hit_scores = []
    for r in pred_relations:
        triplet = tuple(r['triplet'])
        if not triplet in pred_triplets:
            pred_triplets.append(triplet)
            hit_scores.append(r['score'])
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
    return prec, rec, hit_scores


def evaluate(groundtruth, prediction, viou_threshold=0.5,
        det_nreturns=[50, 100], tag_nreturns=[1, 5, 10]):
    """ evaluate visual relation detection and visual 
    relation tagging.
    """
    video_ap = dict()
    tot_scores = defaultdict(list)
    tot_tp = defaultdict(list)
    prec_at_n = defaultdict(list)
    tot_gt_relations = 0            # tot represents total
    # print('Computing average precision AP over {} videos...'.format(len(groundtruth)))
    # for vid, gt_relations in tqdm(groundtruth.items()):
    for vid, gt_relations in groundtruth.items():
        if len(gt_relations)==0:
            continue
        tot_gt_relations += len(gt_relations)
        predict_relations = prediction.get(vid, [])
        # compute average precision and recalls in detection setting
        det_prec, det_rec, det_scores = eval_detection_scores(
                gt_relations, predict_relations, viou_threshold)
        video_ap[vid] = voc_ap(det_rec, det_prec)
        tp = np.isfinite(det_scores)
        for nre in det_nreturns:
            cut_off = min(nre, det_scores.size)
            tot_scores[nre].append(det_scores[:cut_off])
            tot_tp[nre].append(tp[:cut_off])
        # compute precisions in tagging setting
        tag_prec, _, _ = eval_tagging_scores(gt_relations, predict_relations)
        for nre in tag_nreturns:
            cut_off = min(nre, tag_prec.size)
            if cut_off > 0:
                prec_at_n[nre].append(tag_prec[cut_off - 1])
            else:
                prec_at_n[nre].append(0.)
    # calculate mean ap for detection
    mean_ap = np.mean(list(video_ap.values()))
    # calculate recall for detection
    rec_at_n = dict()
    for nre in det_nreturns:
        scores = np.concatenate(tot_scores[nre])
        tps = np.concatenate(tot_tp[nre])
        sort_indices = np.argsort(scores)[::-1]
        tps = tps[sort_indices]
        cum_tp = np.cumsum(tps).astype(np.float32)
        rec = cum_tp / np.maximum(tot_gt_relations, np.finfo(np.float32).eps)
        rec_at_n[nre] = rec[-1]
    # calculate mean precision for tagging
    mprec_at_n = dict()
    for nre in tag_nreturns:
        mprec_at_n[nre] = np.mean(prec_at_n[nre])
    # print scores
    # print('detection mean AP (used in challenge): {}'.format(mean_ap))
    # print('detection recall@50: {}'.format(rec_at_n[50]))
    # print('detection recall@100: {}'.format(rec_at_n[100]))
    # print('tagging precision@1: {}'.format(mprec_at_n[1]))
    # print('tagging precision@5: {}'.format(mprec_at_n[5]))
    # print('tagging precision@10: {}'.format(mprec_at_n[10]))
    return mean_ap, rec_at_n, mprec_at_n

####### the following func is added by Kaifeng Gao (kite_phone@zju.edu.cn)


def eval_detection_scores_v2(gt_relations, pred_relations, viou_threshold):
    '''this func is modified based on eval_detection_scores, add function to return gt2det_ids'''
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    gt_detected = np.zeros((len(gt_relations),), dtype=bool)
    gt2det_ids = np.ones((len(gt_relations),), dtype=int) * (-1)
    hit_scores = np.ones((len(pred_relations))) * -np.inf
    for pred_idx, pred_relation in enumerate(pred_relations):
        pred_relation['hit_gt'] = False
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
                # print(gt_relation["triplet"],gt_relation["duration"],pred_relation["triplet"],pred_relation["duration"],s_iou,o_iou)
                if ov >= viou_threshold and ov > ov_max:
                    ov_max = ov
                    k_max = gt_idx
        if k_max >= 0:
            hit_scores[pred_idx] = pred_relation['score']
            gt_detected[k_max] = True
            pred_relation['hit_gt'] = True
            gt2det_ids[k_max] = pred_idx
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_relations), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    return prec, rec, hit_scores, gt2det_ids



def evaluate_v2(groundtruth, prediction, viou_threshold=0.5,
        det_nreturns=[50, 100], tag_nreturns=[1, 5, 10]):
    """ evaluate visual relation detection and visual 
    relation tagging.

    this func is modified based on evaluate, add function to return hit info
    """
    video_ap = dict()
    tot_scores = defaultdict(list)
    tot_tp = defaultdict(list)
    prec_at_n = defaultdict(list)
    tot_gt_relations = 0
    print('Computing average precision AP over {} videos...'.format(len(groundtruth)))
    
    det_infos = {}
    # for vid, gt_relations in groundtruth.items():
    for vid, gt_relations in tqdm(groundtruth.items()):
        if len(gt_relations)==0:
            continue
        tot_gt_relations += len(gt_relations)
        predict_relations = prediction.get(vid, [])
        # compute average precision and recalls in detection setting
        det_prec, det_rec, det_scores,gt2det_ids = eval_detection_scores_v2(
                gt_relations, predict_relations, viou_threshold)
        # print(predict_relations,det_scores)
        det_infos[vid] = (det_scores,gt2det_ids)
        video_ap[vid] = voc_ap(det_rec, det_prec)
        tp = np.isfinite(det_scores)
        # print(det_scores)
        for nre in det_nreturns:
            cut_off = min(nre, det_scores.size)
            tot_scores[nre].append(det_scores[:cut_off])
            tot_tp[nre].append(tp[:cut_off])
        # compute precisions in tagging setting
        tag_prec, _, _ = eval_tagging_scores(gt_relations, predict_relations)
        for nre in tag_nreturns:
            cut_off = min(nre, tag_prec.size)
            if cut_off > 0:
                prec_at_n[nre].append(tag_prec[cut_off - 1])
            else:
                prec_at_n[nre].append(0.)
    # calculate mean ap for detection
    mean_ap = np.mean(list(video_ap.values()))
    # calculate recall for detection
    rec_at_n = dict()
    for nre in det_nreturns:
        scores = np.concatenate(tot_scores[nre])
        tps = np.concatenate(tot_tp[nre])
        sort_indices = np.argsort(scores)[::-1]
        tps = tps[sort_indices]
        cum_tp = np.cumsum(tps).astype(np.float32)
        rec = cum_tp / np.maximum(tot_gt_relations, np.finfo(np.float32).eps)
        # print(tps,cum_tp,tot_gt_relations)
        rec_at_n[nre] = rec[-1]
    # calculate mean precision for tagging
    mprec_at_n = dict()
    for nre in tag_nreturns:
        mprec_at_n[nre] = np.mean(prec_at_n[nre])
    # print scores
    # print('detection mean AP (used in challenge): {}'.format(mean_ap))
    # print('detection recall@50: {}'.format(rec_at_n[50]))
    # print('detection recall@100: {}'.format(rec_at_n[100]))
    # print('tagging precision@1: {}'.format(mprec_at_n[1]))
    # print('tagging precision@5: {}'.format(mprec_at_n[5]))
    # print('tagging precision@10: {}'.format(mprec_at_n[10]))
    return mean_ap, rec_at_n, mprec_at_n,det_infos

def eval_relation_detection_openvoc(
    target_split_traj="all",
    target_split_pred="all",
    enti_cls_split_info_path="../dataset/vidvrd/data/openvoc_obj_class_spilt_info.json",
    pred_cls_split_info_path="../dataset/vidvrd/data/openvoc_pred_class_spilt_info.json",
    prediction_results=None,
    json_results_path=None,
    gt_json="../dataset/vidvrd/data/test_relation_gt.json",
    rt_hit_infos = False,
):
    '''
    NOTE this func is only support for VidVRD currently
    '''
    if prediction_results is None:
        print("loading json results from {}".format(json_results_path))
        prediction_results = load_json(json_results_path)
    else:
        assert json_results_path is None


    # print("filter gt triplets with traj split: {}, predicate split: {}".format(target_split_traj, target_split_pred))
    traj_cls_info = load_json(enti_cls_split_info_path)
    pred_cls_info = load_json(pred_cls_split_info_path)
    traj_categories = [c for c,s in traj_cls_info["cls2split"].items() if (s == target_split_traj) or target_split_traj=="all"]
    traj_categories = set([c for c in traj_categories if c != "__background__"])
    pred_categories = [c for c,s in pred_cls_info["cls2split"].items() if (s == target_split_pred) or target_split_pred=="all"]
    pred_categories = set([c for c in pred_categories if c != "__background__"])

    gt_relations = load_json(gt_json)
    gt_relations_ = defaultdict(list)
    for vsig, relations in gt_relations.items(): # same format as prediction results json, refer to `VidVRDhelperEvalAPIs/README.md`
        for rel in relations:
            s,p,o = rel["triplet"]
            if not ((s in traj_categories) and (p in pred_categories) and (o in traj_categories)):
                continue
            gt_relations_[vsig].append(rel)
    gt_relations = gt_relations_

    prediction_results_ = defaultdict(list)
    for vsig, relations in prediction_results.items(): # same format as prediction results json, refer to `VidVRDhelperEvalAPIs/README.md`
        for rel in relations:
            s,p,o = rel["triplet"]
            if not ((s in traj_categories) and (p in pred_categories) and (o in traj_categories)):
                continue
            prediction_results_[vsig].append(rel)
    prediction_results = prediction_results_
    
    if rt_hit_infos:
        mean_ap, rec_at_n, mprec_at_n, hit_infos = evaluate_v2(gt_relations,prediction_results,viou_threshold=0.5)
    else:
        mean_ap, rec_at_n, mprec_at_n = evaluate(gt_relations,prediction_results,viou_threshold=0.5)
    # print(f"mAP:{mean_ap}, Retection Recall:{rec_at_n}, Tagging Precision: {mprec_at_n}")
    # print('detection mean AP (used in challenge): {}'.format(mean_ap))
    # print('detection recall: {}'.format(rec_at_n))
    # print('tagging precision: {}'.format(mprec_at_n))

    # if rt_hit_infos:
    #     return hit_infos
    # else:
    return mean_ap, rec_at_n
