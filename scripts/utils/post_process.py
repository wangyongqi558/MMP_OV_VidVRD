import numpy as np
import torch

def process_pred(args, id2pre, obj2id, prior, preds, pair_data):
    CLIP_LEN = args.clip_len
    CLIP_TOP_N = args.clip_top_n
    scores = preds.cpu().detach().numpy()

    clip_num = len(scores)
    clip_rels = [[] for _ in range(clip_num)]
    begin_fid, end_fid = pair_data['duration']
    if args.use_prior:
        sbj_id = obj2id[pair_data['sbj_cls']]
        obj_id = obj2id[pair_data['obj_cls']]
        scores += prior[sbj_id, obj_id]*torch.ones_like(scores)
    for clip_id in range(clip_num):

        ## top N
        pre_ids = np.argsort(-scores[clip_id])[:CLIP_TOP_N] # for sigmoid  
        pre_scrs = scores[clip_id][pre_ids].tolist()
        # print([round(scr,4) for scr in pre_scrs])
        pre_clss = [id2pre[x] for x in pre_ids]

        ## threshold
        # conf_thres = 0.05
        # pre_ids = np.where(scores[clip_id] > conf_thres)[0]
        # pre_scrs = scores[clip_id][pre_ids].tolist()
        # print([round(scr,4) for scr in pre_scrs])
        # pre_clss = [id2pre[x] for x in pre_ids]

        if clip_id == (clip_num-1):
            clip_sbj_traj = pair_data['sbj_traj'][clip_id*CLIP_LEN:]
            clip_obj_traj = pair_data['obj_traj'][clip_id*CLIP_LEN:]
            duration = [begin_fid + clip_id*CLIP_LEN, end_fid]
        else:
            clip_sbj_traj = pair_data['sbj_traj'][clip_id*CLIP_LEN:(clip_id+1)*CLIP_LEN]
            clip_obj_traj = pair_data['obj_traj'][clip_id*CLIP_LEN:(clip_id+1)*CLIP_LEN]
            duration = [begin_fid + clip_id*CLIP_LEN, begin_fid + (clip_id+1)*CLIP_LEN]

        for idx in range(len(pre_scrs)):
            clip_rels[clip_id].append({
                'pre_cls': pre_clss[idx],
                'pre_scr': pre_scrs[idx],
                'sbj_cls': pair_data['sbj_cls'],
                'sbj_scr': pair_data['sbj_scr'],
                'obj_cls': pair_data['obj_cls'],
                'obj_scr': pair_data['obj_scr'],
                'sbj_traj': clip_sbj_traj.copy(),
                'obj_traj': clip_obj_traj.copy(),
                'duration': duration.copy(),
                'connected': False})
    return clip_rels

def association(cand_clips):

    '''
    code partly referencing VRU'19 Challenge Top1
    '''
    rel_instances = []
    for i in range(len(cand_clips)):
        curr_clips = cand_clips[i]

        for j in range(len(curr_clips)):
            # current clip
            curr_clip = curr_clips[j]
            curr_scores = [curr_clip['pre_scr']]
            if curr_clip['connected']:
                continue
            else:
                curr_clip['connected'] = True

            for p in range(i+1, len(cand_clips)):
                # connect next clip
                next_clips = cand_clips[p]
                success = False
                for q in range(len(next_clips)):

                    next_clip = next_clips[q]
                    if next_clip['connected']:
                        continue

                    if curr_clip['pre_cls'] == next_clip['pre_cls']:
                        # merge trajectories
                        curr_clip['sbj_traj'].extend(next_clip['sbj_traj'])
                        curr_clip['obj_traj'].extend(next_clip['obj_traj'])

                        # record clip predicate scores
                        curr_scores.append(next_clip['pre_scr'])
                        curr_clip['duration'][1] = next_clip['duration'][1]
                        next_clip['connected'] = True
                        success = True
                        break

                if not success:
                    break

            curr_clip['pre_scr'] = sum(curr_scores) / len(curr_scores)
            curr_clip['score'] = curr_clip['sbj_scr'] * curr_clip['obj_scr'] * curr_clip['pre_scr']
            rel_instances.append(curr_clip)
    return rel_instances

def format_(args, rela_cands):

    '''
    code partly referencing VRU'19 Challenge Top1
    '''
    MAX_PER_VIDEO = args.max_per_video
    sorted_cands = sorted(rela_cands, key=lambda rela: rela['score'], reverse=True)
    if len(sorted_cands) > MAX_PER_VIDEO:
        sorted_cands = sorted_cands[:MAX_PER_VIDEO]

    format_relas = []
    for rela in sorted_cands:
        format_rela = dict()
        format_rela['triplet'] = [rela['sbj_cls'], rela['pre_cls'], rela['obj_cls']]
        format_rela['score'] = rela['score']
        assert len(rela['sbj_traj']) == len(rela['obj_traj'])
        assert len(rela['sbj_traj']) == (rela['duration'][1]-rela['duration'][0])
        format_rela['sub_traj'] = rela['sbj_traj']
        format_rela['obj_traj'] = rela['obj_traj']
        format_rela['duration'] = rela['duration']
        format_relas.append(format_rela)
    return format_relas