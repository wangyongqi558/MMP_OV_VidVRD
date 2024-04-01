import os
import math
import pickle
import json
import shutil
from collections import defaultdict
from os.path import join
import time

import numpy as np
import torch
from torch.nn.modules.module import T
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import convert_image_dtype
from tqdm import tqdm

from utils.parser_func import parse_args
from utils.utils import get_feat_types, vru19_ext_loc_feat, aaai18_ext_mask_feat, vtranse_ext_loc_feat, gen_union_bbox, ext_bbox_loc_feat
from utils.i3d import I3D
from utils.resnet import ResNet
from utils.ptm_encoder import CLIPVisualEncoder, InternlVisualEncoder
# from text_encoder import CLIPFeatureEncoder

class FeatExtractor:
    def __init__(self, o2v_path, feat_types):
        self.type2extractor = {
            'rel_feat':self._extract_rel_feat,
            # 'mask_feat':self._extract_mask_feat,
            'lan_feat':self._extract_lan_feat,
            'v2d_feat':self._extract_v2d_feat,
            'mot_feat':self._extract_mot_feat,
            'v3d_feat':self._extract_v3d_feat,
            'clip_feat':self._extract_clip_feat,
            'intern_feat':self._extract_intern_feat,
            'bbox_feat':self._extract_bbox_feat,
            }
        self.feat_types = feat_types
        self.data = None
        with open(o2v_path, 'rb') as f:
            self.obj2vec = pickle.load(f, encoding='latin1')
        
        # self.v2d_model = ResNet().cuda()
        # self.v2d_model.eval()
        # self.v3d_model = I3D().cuda()
        # self.v3d_model.eval()

        self.clip_visual_encoder = CLIPVisualEncoder(backbone="ViT-L/14@336px", gpu_id=0).eval()
        self.intern_visual_encoder = InternlVisualEncoder(gpu_id=0).eval()
        self.clip_feature_encoder = CLIPFeatureEncoder("ViT-L/14@336px").cuda().eval()

    def load_frames(self, data):
        self.num_frames = len(os.listdir(join(ROOT, 'frames', data['video_id'])))
        self.video_height = data["height"]
        self.video_width = data["width"]
        self.frame_paths = {}
        for fid in range(self.num_frames):
            self.frame_paths[fid] = join(ROOT, 'frames', data['video_id'], '%06d.jpg'%(fid+1)) # picture No is named from 1

    def gen_feats(self, clip_data):
        self.data = clip_data
        feats = {}
        for type in self.feat_types:
            feats[type] = self.type2extractor[type]()
        return feats
    
    def _extract_rel_feat(self):
        data = self.data
        assert data != None
        mid_fno = len(data['sbj_traj'])//2
        begin_feat  = vru19_ext_loc_feat(data['sbj_traj'][0], data['obj_traj'][0], self.video_height, self.video_width)
        mid_feat    = vru19_ext_loc_feat(data['sbj_traj'][mid_fno], data['obj_traj'][mid_fno], self.video_height, self.video_width)
        end_feat    = vru19_ext_loc_feat(data['sbj_traj'][-1], data['obj_traj'][-1], self.video_height, self.video_width)
        feat = np.concatenate((begin_feat, mid_feat, end_feat))
        return feat
    def _extract_mot_feat(self):
        data = self.data
        assert data != None
        mid_fno = len(data['sbj_traj'])//2
        begin_feat  = vru19_ext_loc_feat(data['sbj_traj'][0], data['obj_traj'][0], self.video_height, self.video_width)
        mid_feat    = vru19_ext_loc_feat(data['sbj_traj'][mid_fno], data['obj_traj'][mid_fno], self.video_height, self.video_width)
        end_feat    = vru19_ext_loc_feat(data['sbj_traj'][-1], data['obj_traj'][-1], self.video_height, self.video_width)
        bm_mot_feat = mid_feat-begin_feat
        me_mot_feat = end_feat-mid_feat
        be_mot_feat = end_feat-begin_feat
        feat = np.concatenate((bm_mot_feat, me_mot_feat, be_mot_feat))
        return feat

    def _extract_mask_feat(self):
        data = self.data
        assert data != None
        mid_fno = len(data['sbj_traj'])//2
        mid_feat = aaai18_ext_mask_feat(data['sbj_traj'][mid_fno], data['obj_traj'][mid_fno], self.video_height, self.video_width)
        feat = mid_feat
        return feat

    def _extract_lan_feat(self):
        data = self.data
        assert data != None
        return np.concatenate((self.obj2vec[data['sbj_id']],self.obj2vec[data['obj_id']]))

    def _extract_v2d_feat(self):
        with torch.no_grad():
            data = self.data
            assert data != None
        
            # <-------- use all imgs-------- >
            # feats = torch.zeros((0, 2048))
            # clip_len = len(data['sbj_traj'])
            # for fno in range(clip_len):
            #     fid = data['begin_fid'] + fno
            #     image_path = self.frame_paths[fid]
            #     image = convert_image_dtype(read_image(image_path, ImageReadMode.RGB), dtype=torch.float)
            #     image = image.view(1, *image.shape)
            #     bboxes = [torch.tensor([data['sbj_traj'][fno],
            #                             data['obj_traj'][fno]])]
            #     feat = self.v2d_model(image, bboxes).view(1,-1).cpu().detach()
            #     feats = torch.cat((feats, feat), dim=0)
            # feat = torch.mean(feats, dim=0).view(-1)

            # <-------- use the middle img -------- >
            mid_fno = len(data['sbj_traj'])//2
            mid_fid = data['begin_fid'] + mid_fno
            image_path = self.frame_paths[mid_fid]
            image = convert_image_dtype(read_image(image_path, ImageReadMode.RGB), dtype=torch.float)
            image = image.view(1, *image.shape)

            # bboxes = [torch.tensor([data['sbj_traj'][mid_fno],
            #                         data['obj_traj'][mid_fno]])]

            bboxes = [torch.tensor([data['sbj_traj'][mid_fno],
                                    data['obj_traj'][mid_fno], 
                                    gen_union_bbox(data['sbj_traj'][mid_fno], data['obj_traj'][mid_fno])])]

            feat = self.v2d_model(image, bboxes).view(-1).cpu().detach()
        return feat.numpy()
    
    # def _extract_v3d_feat(self):
    #     with torch.no_grad():
    #         data = self.data
    #         assert data != None
    #         image_paths = [self.frame_paths[fid] for fid in range(data['begin_fid'], data['end_fid'])]
    #         images = [convert_image_dtype(read_image(image_path, ImageReadMode.RGB), dtype=torch.float) for image_path in image_paths]
    #         sbj_bboxes = data['sbj_traj']
    #         sbj_feats = self.v3d_model(images, sbj_bboxes)
    #         obj_bboxes = data['obj_traj']
    #         obj_feats = self.v3d_model(images, obj_bboxes)
            
    #         # feat = torch.cat((sbj_feats, obj_feats), dim=0).cpu().detach()

    #         union_bboxes = [gen_union_bbox(sbj_bboxes[i], obj_bboxes[i]) for i in range(len(sbj_bboxes))]
    #         union_feats = self.v3d_model(images, union_bboxes)
    #         feat = torch.cat((sbj_feats, obj_feats, union_feats), dim=0).cpu().detach()
    #     return feat.numpy()
    
    def _extract_v3d_feat(self):
        data = self.data
        assert data != None
        frame_num = len(data['sbj_traj'])
        fnos = [round(x) for x in np.linspace(0, frame_num-1, 30)]
        with torch.no_grad():
            image_paths = [self.frame_paths[fno+data['begin_fid']] for fno in fnos]
            images = [convert_image_dtype(read_image(image_path, ImageReadMode.RGB), dtype=torch.float) for image_path in image_paths]
            sbj_bboxes = [data['sbj_traj'][fno] for fno in fnos]
            sbj_feats = self.v3d_model(images, sbj_bboxes)
            obj_bboxes = [data['obj_traj'][fno] for fno in fnos]
            obj_feats = self.v3d_model(images, obj_bboxes)

            # feat = torch.cat((sbj_feats, obj_feats), dim=0).cpu().detach()

            union_bboxes = [gen_union_bbox(sbj_bboxes[i], obj_bboxes[i]) for i in range(len(sbj_bboxes))]
            union_feats = self.v3d_model(images, union_bboxes)
            feat = torch.cat((sbj_feats, obj_feats, union_feats), dim=0).cpu().detach()
        return feat.numpy()
    
    def _extract_intern_feat(self):
        with torch.no_grad():
            data = self.data
            assert data != None

            frame_num = len(data['sbj_traj'])
            fnos = [round(x) for x in np.linspace(0, frame_num-1, 30)]
            image_paths = [self.frame_paths[fno+data['begin_fid']] for fno in fnos]
            sbj_bboxes = [data['sbj_traj'][fno] for fno in fnos]
            obj_bboxes = [data['obj_traj'][fno] for fno in fnos]
            union_bboxes = [gen_union_bbox(sbj_bboxes[i], obj_bboxes[i]) for i in range(len(image_paths))]        
            trajs = [
                sbj_bboxes,
                obj_bboxes,
                union_bboxes,
                [None]*len(image_paths)
            ]
            sequences = []
            for traj in trajs:
                sequences.append([image_paths, traj])
            feat = self.intern_visual_encoder(sequences)
        return feat

    def _extract_clip_feat(self):
        with torch.no_grad():
            data = self.data
            assert data != None

            mid_fno = len(data['sbj_traj'])//2
            mid_fid = data['begin_fid'] + mid_fno
            img_path = self.frame_paths[mid_fid]

            bboxes = [
                    data['sbj_traj'][mid_fno],
                    data['obj_traj'][mid_fno], 
                    gen_union_bbox(data['sbj_traj'][mid_fno], data['obj_traj'][mid_fno]),
                    None
                    ]
            regions = []
            for bbox in bboxes:
                regions.append([img_path, bbox])
            feat = self.clip_visual_encoder(regions)
            # feat = self.clip_feature_encoder(regions)
        return feat

    def _extract_bbox_feat(self):
        data = self.data
        assert data != None
        head_s, head_o, head_u, head_t = ext_bbox_loc_feat(
                                   data['sbj_traj'][0], 
                                   data['obj_traj'][0], 
                                   gen_union_bbox(data['sbj_traj'][0], data['obj_traj'][0]),
                                   self.video_height, self.video_width)
        tail_s, tail_o, tail_u, tail_t = ext_bbox_loc_feat(
                                   data['sbj_traj'][-1], 
                                   data['obj_traj'][-1], 
                                   gen_union_bbox(data['sbj_traj'][-1], data['obj_traj'][-1]),
                                   self.video_height, self.video_width)
        diff_s = tail_s - head_s
        diff_o = tail_o - head_o
        diff_u = tail_u - head_u
        diff_t = tail_t - head_t
        feat = np.asarray([
                np.concatenate((head_s, tail_s, diff_s)),
                np.concatenate((head_o, tail_o, diff_o)),
                np.concatenate((head_u, tail_u, diff_u)),
                np.concatenate((head_t, tail_t, diff_t))])
        return feat




def gen_label(clip, gt_relations):
    labels = []
    for relation in gt_relations:
        if (clip["sbj_tid"] == relation["subject_tid"]) and (clip["obj_tid"] == relation["object_tid"]):

            # <1> clipment has the relation label if having intersection with the relation duration
            # if (relation["begin_fid"] <= clip["begin_fid"] < relation["end_fid"]) or\
            #     (relation["begin_fid"] < clip["end_fid"] <= relation["end_fid"]):
            #     labels.append(relation["predicate"]) 

            # <2> clipment has the relation label if having intersection larger than threshold
            left = max(clip["begin_fid"], relation["begin_fid"])
            right = min(clip["end_fid"], relation["end_fid"])
            if right - left >= 10:
                labels.append(relation["predicate"])
    labels = list(set(labels))
    return labels  

def gen_gt_trajs(vid_anno):
    gt_trajs = defaultdict(dict)
    for fid, frame in enumerate(vid_anno["trajectories"]):
        for bbox_anno in frame:
            tid = bbox_anno["tid"]
            bbox = bbox_anno["bbox"]
            bbox = [bbox["xmin"],bbox["ymin"],bbox["xmax"],bbox["ymax"]]
            gt_trajs[tid][fid] = bbox
    return gt_trajs
    
def cal_viou(traj_1, traj_2):
    """ compute the voluminal Intersection over Union
    for two trajectories, each of which is represented
    by a duration [fstart, fend) and a list of bounding
    boxes (i.e. traj) within the duration.
    """
    v_overlap = 0
    for i in range(len(traj_1)):
        roi_1 = traj_1[i]
        roi_2 = traj_2[i]
        if roi_2 == None:
            v_overlap += 0
        else:
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
        if traj_2[i] == None:
            v2 += 0
        else:
            v2 += (traj_2[i][2] - traj_2[i][0] + 1) * (traj_2[i][3] - traj_2[i][1] + 1)
    return float(v_overlap) / (v1 + v2 - v_overlap)

def gen_hit_tid(clip, gt_trajs):

    clip_begin_fid = clip['begin_fid']
    clip_end_fid = clip['end_fid']

    sbj_max_viou = -float('Inf')
    sbj_hit_tid = -1 
    obj_max_viou = -float('Inf')
    obj_hit_tid = -1 
    for gt_tid in gt_trajs:
        clip_gt_traj = []
        for fid in range(clip_begin_fid, clip_end_fid):
            if fid in gt_trajs[gt_tid]:
                clip_gt_traj.append(gt_trajs[gt_tid][fid])
            else:
                clip_gt_traj.append(None)
        sbj_viou = cal_viou(clip['sbj_traj'], clip_gt_traj)
        if sbj_viou >= 0.5 and sbj_viou > sbj_max_viou:
            sbj_max_viou = sbj_viou
            sbj_hit_tid = gt_tid
        
        obj_viou = cal_viou(clip['obj_traj'], clip_gt_traj)
        if obj_viou >= 0.5 and obj_viou > obj_max_viou:
            obj_max_viou = obj_viou
            obj_hit_tid = gt_tid
    return sbj_hit_tid, obj_hit_tid
        
def gen_feats(split, traj_source):
    
    feat_path = split + '_' + traj_source + '_' + str(CLIP_LEN)

    if not os.path.exists(join(ROOT, 'feature', feat_path)):
        os.mkdir(join(ROOT, 'feature', feat_path))
    else:
        shutil.rmtree(join(ROOT, 'feature', feat_path))
        os.mkdir(join(ROOT, 'feature', feat_path))

    trajs = json.load(open(join(ROOT, 'data', '{}_object_trajectories_{}.json'.format(split, traj_source))))

    if args.dataset == "vidor":
        pkg_list = os.listdir(join(ROOT, 'anno', split))
        vid_list = []
        for pkg_name in pkg_list:
            pkg_vids = os.listdir(join(ROOT, 'anno', split, pkg_name))
            for vid_name in pkg_vids:
                vid_list.append(join(ROOT, 'anno', split, pkg_name, vid_name))
    elif args.dataset == "vidvrd":
        vid_list = [join(ROOT, 'anno', split, vid_name) for vid_name in os.listdir(join(ROOT, 'anno', split))]
    else: assert "Unknown Dataset!"
    
    print("Extracing feature for {} {} split from {} trajectories:".format(args.dataset, split, traj_source))
    for vid_name in tqdm(vid_list):
        vid_anno = json.load(open(vid_name,'r'))
        if split == 'train' and traj_source != 'gt':
            vid_gt_trajs = gen_gt_trajs(vid_anno)
        feat_extractor.load_frames(vid_anno)
        vid_name = vid_name.split("/")[-1].split(".")[0]
        if vid_name not in trajs:
            continue
        vid_trajs = trajs[vid_name]

        pair_id = 0
        for s_traj_id, s_traj in enumerate(vid_trajs):
            for o_traj_id, o_traj in enumerate(vid_trajs):
                if s_traj_id == o_traj_id: continue
                begin_fid = max(s_traj['begin_fid'], o_traj['begin_fid'])
                end_fid   = min(s_traj['end_fid'], o_traj['end_fid'])
                if (end_fid - begin_fid) < 10: continue

                clip_num = int((end_fid-begin_fid) / CLIP_LEN)
                tail_len = (end_fid-begin_fid) % CLIP_LEN
                if tail_len >= 10:
                    clip_num += 1
                elif 0 < tail_len <10:
                    end_fid = end_fid - tail_len
                
                s_bboxes = []
                o_bboxes = []
                for fid in range(begin_fid, end_fid):
                    s_bboxes.append(s_traj['trajectory'][str(fid)])
                    o_bboxes.append(o_traj['trajectory'][str(fid)])
                
                if split == 'train':
                    pair_labels = []
                else:
                    pair_data = {
                    'sbj_scr': s_traj['score'],
                    'obj_scr': o_traj['score'],
                    'sbj_cls': s_traj['category'],
                    'obj_cls': o_traj['category'],
                    'sbj_traj': s_bboxes,
                    'obj_traj': o_bboxes,
                    'duration': [begin_fid, end_fid]
                    }

                pair_name = vid_name + "_%06d.pkl"%pair_id
                pair_id += 1
                pair_feats = defaultdict(list)
                
                for clip_id in range(clip_num):
                    clip = {} 
                    clip["sbj_id"] = object2id[s_traj['category']]
                    clip["obj_id"] = object2id[o_traj['category']]
                    clip["begin_fid"] = begin_fid + clip_id*CLIP_LEN
                    if clip_id == clip_num-1:
                        clip["sbj_traj"] = s_bboxes[clip_id*CLIP_LEN:]
                        clip["obj_traj"] = o_bboxes[clip_id*CLIP_LEN:]
                        clip["end_fid"] = end_fid
                    else:
                        clip["sbj_traj"] = s_bboxes[clip_id*CLIP_LEN:(clip_id+1)*CLIP_LEN]
                        clip["obj_traj"] = o_bboxes[clip_id*CLIP_LEN:(clip_id+1)*CLIP_LEN]
                        clip["end_fid"] = begin_fid + (clip_id+1)*CLIP_LEN
                    
                    if split == 'train':
                        if traj_source == 'gt':
                            clip["sbj_tid"], clip["obj_tid"] = s_traj['tid'], o_traj['tid']
                        else:
                            clip["sbj_tid"], clip["obj_tid"] = gen_hit_tid(clip, vid_gt_trajs)
                        pair_labels.append([predicate2id[p] for p in gen_label(clip, vid_anno["relation_instances"])])
                    
                    feats = feat_extractor.gen_feats(clip)
                    for type_ in feat_types:
                        pair_feats[type_].append(feats[type_])
                
                if split == 'train':
                    pair_labels = [object2id[s_traj['category']], object2id[o_traj['category']], pair_labels]
                    with open(join(ROOT, 'feature', feat_path, pair_name), 'wb') as f:
                        pickle.dump([pair_feats, pair_labels], f)
                else:
                    with open(join(ROOT, 'feature', feat_path, pair_name), 'wb') as f:
                        pickle.dump([pair_feats, pair_data], f)
    
# def gen_feats(split, traj_source):
    
#     feat_path = split + '_' + traj_source + '_' + str(CLIP_LEN) + '_2stage'

#     if not os.path.exists(join(ROOT, 'feature', feat_path)):
#         os.mkdir(join(ROOT, 'feature', feat_path))
#     else:
#         shutil.rmtree(join(ROOT, 'feature', feat_path))
#         os.mkdir(join(ROOT, 'feature', feat_path))

#     if args.dataset == "vidor":
#         pkg_list = os.listdir(join(ROOT, 'anno', split))
#         vid_list = []
#         for pkg_name in pkg_list:
#             pkg_vids = os.listdir(join(ROOT, 'anno', split, pkg_name))
#             for vid_name in pkg_vids:
#                 vid_list.append(join(ROOT, 'anno', split, pkg_name, vid_name))
#     elif args.dataset == "vidvrd":
#         vid_list = [join(ROOT, 'anno', split, vid_name) for vid_name in os.listdir(join(ROOT, 'anno', split))]
#     else: assert "Unknown Dataset!"
    
#     print("Extracing feature for {} {} split from {} trajectories:".format(args.dataset, split, traj_source))
#     for vid_name in tqdm(vid_list):
#         vid_anno = json.load(open(vid_name,'r'))
#         feat_extractor.load_frames(vid_anno)
#         vid_name = vid_name.split("/")[-1].split(".")[0]

#         pair_id = 0
#         trajs = {}
#         tid2cat = {}
#         for obj in vid_anno["subject/objects"]:
#             trajs[obj['tid']] = {}
#             tid2cat[obj['tid']] = obj['category']

#         for fid, bboxes in enumerate(vid_anno["trajectories"]):
#             for bbox in bboxes:
#                 trajs[bbox['tid']][fid] = [bbox['bbox']['xmin'],bbox['bbox']['ymin'],bbox['bbox']['xmax'],bbox['bbox']['ymax']]

#         for rel in vid_anno["relation_instances"]:
#             s_time = time.time()
#             s_bboxes = []
#             o_bboxes = []
#             for fid in range(rel['begin_fid'], rel['end_fid']):
#                 s_bboxes.append(trajs[rel['subject_tid']][fid])
#                 o_bboxes.append(trajs[rel['object_tid']][fid])
#             assert len(s_bboxes) == rel['end_fid'] - rel['begin_fid']
        
#             pair_data = {
#             'sbj_scr': 1.0,
#             'obj_scr': 1.0,
#             'sbj_id': object2id[tid2cat[rel['subject_tid']]],
#             'obj_id': object2id[tid2cat[rel['object_tid']]],
#             'sbj_cls': tid2cat[rel['subject_tid']],
#             'obj_cls': tid2cat[rel['object_tid']],
#             'sbj_traj': s_bboxes,
#             'obj_traj': o_bboxes,
#             'begin_fid': rel['begin_fid'],
#             'end_fid': rel['end_fid']
#             }
#             pair_label = predicate2id[rel['predicate']]
#             pair_feats = feat_extractor.gen_feats(pair_data)
            
#             pair_name = vid_name + "_%06d.pkl"%pair_id
#             pair_id += 1
#             if split == 'train':
#                 with open(join(ROOT, 'feature', feat_path, pair_name), 'wb') as f:
#                     pickle.dump([pair_feats, pair_label], f)
#             else:
#                 with open(join(ROOT, 'feature', feat_path, pair_name), 'wb') as f:
#                     pickle.dump([pair_feats, pair_data], f)
#             print("pair:",time.time()-s_time)
       
         

if __name__ == '__main__':
    
    args = parse_args()
    CLIP_LEN = args.clip_len
    USE_UNLABELED_PAIR = args.use_unlabeld_pair
    ROOT = join("..","dataset",args.dataset)

    object2id = json.load(open(join(ROOT, 'data', 'object2id.json'),'r'))
    predicate2id = json.load(open(join(ROOT, 'data','predicate2id.json'),'r'))
    feat_types = get_feat_types(args)
    feat_extractor = FeatExtractor(join(ROOT,'data','object_vectors.pkl'), feat_types)
    
    gen_feats("train", traj_source="gt")
    gen_feats("test", traj_source="gt")
    # gen_feats("test", traj_source="openvoc")
    # gen_feats("test", traj_source="0324")
    # gen_feats("test", traj_source="openvoc3")
    # gen_feats("test", traj_source="vidvrd-ii-2")

    

