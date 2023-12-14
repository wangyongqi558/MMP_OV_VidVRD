import os
import numpy as np
import pickle
import _pickle as cPickle
import json
from collections import defaultdict
from os.path import join
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):

    def __init__(self, args, split):
        super().__init__()
        self.dataset = args.dataset
        self.split = split
        if split=='train':
            feat_path = 'train_{}_{}'.format(args.train_traj, args.clip_len)  
        elif split=='val':
            self.gt_rels = json.load(open(
                join('..', 'dataset', self.dataset, 'data', 'val_relation_gt.json'),"r"))
            feat_path = 'val_{}_{}'.format(args.val_traj, args.clip_len)
        elif split=='test':
            self.gt_rels = json.load(open(
                join('..', 'dataset', self.dataset, 'data', 'test_relation_gt.json'),"r"))
            feat_path = 'test_{}_{}'.format(args.test_traj, args.clip_len)

        self.FEAT_ROOT = join('..', 'dataset', self.dataset, 'feature', feat_path)
        self.path_list = [[join(self.FEAT_ROOT, path), join(self.FEAT_ROOT+"_ptm", path), join(self.FEAT_ROOT+"_box", path)] for path in os.listdir(self.FEAT_ROOT)]
        
        self.id2pre = json.load(open(
            join('..', 'dataset', self.dataset, 'data', 'id2predicate.json'),"r"))
        self.pre2id = json.load(open(
            join('..', 'dataset', self.dataset, 'data', 'predicate2id.json'),"r"))
        self.id2obj = json.load(open(
            join('..', 'dataset', self.dataset, 'data', 'id2object.json'),"r"))
        self.obj2id = json.load(open(
            join('..', 'dataset', self.dataset, 'data', 'object2id.json'),"r"))
        self.pre_num = len(self.id2pre)
        self.prior = pickle.load(open(join('..', 'dataset', self.dataset, 'data', 'prior.pkl'),'rb'))

        self.all_pair_data_rel = cPickle.load(open(join(self.FEAT_ROOT+'.pkl'),"rb"))
        self.all_pair_data_ptm = cPickle.load(open(join(self.FEAT_ROOT+"_ptm.pkl"),"rb"))
        self.all_pair_data_box = cPickle.load(open(join(self.FEAT_ROOT+"_box.pkl"),"rb"))

    def __getitem__(self, index):

        pair_path = self.path_list[index]
        pair_data_rel = self.all_pair_data_rel[pair_path[0]]
        pair_data_ptm = self.all_pair_data_ptm[pair_path[1]]
        pair_data_box = self.all_pair_data_box[pair_path[2]]

        pair_feats = {}
        pair_feats.update(pair_data_rel[0])
        pair_feats.update(pair_data_ptm[0])
        pair_feats.update(pair_data_box[0])
        pair_data = pair_data_rel[1]
        item = {}
        for type_ in pair_feats:
            item[type_] = np.array(pair_feats[type_])
        if self.split == 'train':
            pair_label = np.zeros((len(pair_data), self.pre_num),)
            for clip_idx, clip_label in enumerate(pair_data):
                tmp_label = np.zeros(self.pre_num,)
                if len(clip_label) > 0:
                    tmp_label[clip_label] = 1
                pair_label[clip_idx] = tmp_label
            item['pre_label'] = pair_label
            item['sbj_label'] = pair_data_box[1][0]
            item['obj_label'] = pair_data_box[1][1]        
        else:
            item['vid'] = pair_path[0].split('/')[-1].split('.')[0][:-7]
            item['pair_data'] = pair_data
        
        return item

    def __len__(self):
        return len(self.path_list)

def padding_collate_fn(batch):
    seq_lens = torch.LongTensor([len(x['clip_feat']) for x in batch])
    batch_data = {}
    for k in batch[0]:
        if k == 'mask_feat': continue
        batch_data[k] = [x[k] for x in batch]
        if k not in ['vid', 'pair_data']:
            if k == 'sbj_label' or k == 'obj_label':
                batch_data[k] = torch.tensor(batch_data[k]).type(torch.long)
            else:
                batch_data[k] = pad_sequence([torch.from_numpy(x).type(torch.float32) for x in batch_data[k]], batch_first=True)
    return batch_data, seq_lens