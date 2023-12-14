import sys
sys.path.append('') #your scripts path

from os.path import join
import pickle
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch.nn.functional as F
from copy import deepcopy
from utils.utils import get_feat_types
from utils.utils import FocalWithLogitsLoss, FocalWithLogitsLossAlpha
import json
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from collections import OrderedDict

class TemporalDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.intHead = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.transformer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.pos_embeddings = nn.Embedding(40, 512)
        self.relEmb = nn.Linear(512, 512, bias=False)
        
    def _gen_pos_embeddings(self, batch_size, seq_length):
        pos_ids = torch.arange(seq_length, dtype=torch.long).cuda()
        pos_ids = pos_ids.unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.pos_embeddings(pos_ids)
        return F.normalize(pos_embeddings, dim=-1)

    def forward(self, x_clip, seq_lens):
        bs, slen = x_clip.shape[0], x_clip.shape[1]
        x_clip = x_clip.permute(0, 2, 1, 3).reshape(bs*4, slen, -1)

        x_clip = x_clip + self._gen_pos_embeddings(x_clip.shape[0], x_clip.shape[1])
        x_clip = self.transformer(x_clip)
        x_clip = x_clip.reshape(bs, 4, slen, -1).permute(0, 2, 1, 3)

        x_int = self.intHead(x_clip.reshape(bs, slen, -1)).squeeze(-1)
        x_rel = F.normalize(self.relEmb(x_clip), dim=-1)
        return x_rel, x_int
    
class SpatialDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.role_embeddings = nn.Embedding(4, 512)
        self.relEmb = nn.Linear(512, 512, bias=False)
        self.objEmb = nn.Linear(512, 512, bias=False)
        self.boxEmb = nn.Sequential(
            nn.Linear(24, 512),
            nn.ReLU(),
            nn.Linear(512, 512, bias=False))
        self.posEmb = nn.Sequential(
            nn.Linear(42*2, 512),
            nn.ReLU(),
            nn.Linear(512, 512, bias=False))
    
    def _gen_role_embeddings(self, batch_size):
        role_ids = torch.arange(4, dtype=torch.long).cuda()
        role_ids = role_ids.unsqueeze(0).expand(batch_size, -1)
        role_embeddings = self.role_embeddings(role_ids)
        return F.normalize(role_embeddings, dim=-1)

    def forward(self, x_img, x_box=None, x_pos=None):

        x_img = x_img + F.normalize(self.boxEmb(x_box), dim=-1)+ self._gen_role_embeddings(x_img.shape[0])
        x_img = F.normalize(self.transformer(x_img), dim=-1)
        x_obj = F.normalize(self.objEmb(x_img[:, :2]), dim=-1)

        x_pos = F.normalize(self.posEmb(x_pos), dim=-1).unsqueeze(-2).repeat(1, 4, 1)
        x = F.normalize(x_img + x_pos, dim=-1)
        x_rel = x
        return x_rel, x_obj[:, 0], x_obj[:, 1]

class FeatEmbedding(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.feat_types = get_feat_types(args)

        if args.dataset == "vidor":
            self.fusionEmb = nn.Sequential(
                nn.Linear(512*4, args.clip_hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(args.clip_hidden_dim, args.clip_emb_dim),
                nn.ReLU(),
                nn.Dropout(p=args.dropout))
        else:
            self.spatial_decoder = SpatialDecoder()
            self.temporal_decoder = TemporalDecoder()

    def forward(self, inputs, seq_lens):
        bs, slen = inputs['clip_feat'].shape[0], inputs['clip_feat'].shape[1]
        interactiveness = torch.ones((bs, slen)).cuda()

        img_feat = inputs['clip_feat']
        bbox_feat =inputs['bbox_feat']
        pos_feat = torch.cat((inputs['rel_feat'], inputs['mot_feat']), dim=-1)

        pre_embs, sbj_embs, obj_embs = self.spatial_decoder(
            x_img = img_feat.view(bs*slen, 4, -1), 
            x_box = bbox_feat.view(bs*slen, 4, -1), 
            x_pos = pos_feat.view(bs*slen, -1)
        )
        sbj_embs = torch.mean(sbj_embs.view(bs, slen, -1), dim=1)
        obj_embs = torch.mean(obj_embs.view(bs, slen, -1), dim=1)

        pre_embs = pre_embs.view(bs, slen, 4, -1)
        pre_embs, interactiveness = self.temporal_decoder(pre_embs, seq_lens)
        pre_embs = pre_embs.view(bs, slen, -1) / 2
        return pre_embs, sbj_embs, obj_embs, interactiveness, 


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.N_CTX
        ctx_init = cfg.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim*4, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        im_features = torch.mean(im_features,dim=1) #(batch, dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx = ctx.expand(bias.shape[0],ctx.shape[1],ctx.shape[2])
        bias_reshape = bias.expand(bias.shape[0],self.n_ctx,bias.shape[2])
        mask = [[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0]]
        mask = torch.BoolTensor(mask).cuda()
        
        bias_masked = bias_reshape.masked_fill(mask,value=torch.tensor(0))
        ctx_shifted = ctx + bias_masked           # (batch, n_ctx, ctx_dim)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype

    def forward(self,image_features):
        prompts = self.prompt_learner(image_features).cuda()
        
        tokenized_prompts = self.tokenized_prompts
        a = 0
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = torch.unsqueeze(text_features, dim=0)
            if a == 0:
                text_features_resize = text_features
            else:
                text_features_resize = torch.cat((text_features_resize,text_features),0)
            a = a + 1
        
        return text_features_resize

class cfg():
    def __init__(self):
        self.N_CTX = 16
        self.CTX_INIT = ""
        self.CSC = False
        self.CLASS_TOKEN_POSITION = "end"

class ObjectTextEncoder(nn.Module):
    def __init__(self, text_encoder='clip', cls_split_info_path='../dataset/vidvrd/data/openvoc_obj_class_spilt_info.json'):
        super().__init__()

        cls_split_info = json.load(open(cls_split_info_path, 'r'))
        self.id2cls = cls_split_info['id2cls']
        self.cls2id = cls_split_info['cls2id']
        self.cls2split = cls_split_info['cls2split']
        self.base_oids = [self.cls2id[cls] for cls in self.cls2id if self.cls2split[cls] == 'base']
        self.novel_oids = [self.cls2id[cls] for cls in self.cls2id if self.cls2split[cls] == 'novel']
        self.all_oids = list(range(len(self.id2cls)))

        convert_oid_on_base = []
        reorder = 0
        for oid in self.all_oids:
            if oid in self.base_oids:
                convert_oid_on_base.append(reorder)
                reorder += 1
            else:
                convert_oid_on_base.append(-1)
        self.convert_oid_on_base = torch.tensor(convert_oid_on_base).cuda()

        classnames = [self.id2cls[str(id_)] for id_ in range(len(self.id2cls))]
        if text_encoder == 'clip':
            classifier_weights = self.build_clip_fixed_prompts(classnames)
        elif text_encoder == 'intern':
            classifier_weights = self.build_intern_fixed_prompts(classnames)
        self.register_buffer("classifier_weights", classifier_weights, persistent=False)


    def split_classifier_weights(self, classifier_weights, split):
        oids_list = eval(f"self.{split}_oids")
        classifier_weights = classifier_weights[oids_list,:]
        return classifier_weights

    def build_clip_fixed_prompts(self, classnames):
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [f"An image of {name}." for name in classnames]
        prompts = clip.tokenize(prompts).cuda()
        model, _ = clip.load(name='ViT-B/16', device='cpu')
        model = model.cuda().eval()
        with torch.no_grad():
            text_embeddings = model.encode_text(prompts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

        return text_embeddings
    
class PredicateTextEncoder(nn.Module):
    def __init__(self, text_encoder='clip', cls_split_info_path='../dataset/vidvrd/data/openvoc_pred_class_spilt_info.json'):
        super().__init__()

        cls_split_info = json.load(open(cls_split_info_path, 'r'))
        self.id2cls = cls_split_info['id2cls']
        self.cls2id = cls_split_info['cls2id']
        self.cls2split = cls_split_info['cls2split']
        self.base_pids = [self.cls2id[cls] for cls in self.cls2id if self.cls2split[cls] == 'base']
        self.novel_pids = [self.cls2id[cls] for cls in self.cls2id if self.cls2split[cls] == 'novel']
        self.all_pids = list(range(len(self.id2cls)))

        classnames = [self.id2cls[str(id_)] for id_ in range(len(self.id2cls))]
        if text_encoder == 'clip':
            sbj_classifier_weights = self.build_clip_fixed_prompts('subject', classnames)
            obj_classifier_weights = self.build_clip_fixed_prompts('object', classnames)
        self.register_buffer("sbj_classifier_weights", sbj_classifier_weights, persistent=False)
        self.register_buffer("obj_classifier_weights", obj_classifier_weights, persistent=False)

    def split_classifier_weights(self, classifier_weights, split):
        pids_list = eval(f"self.{split}_pids")
        classifier_weights = classifier_weights[pids_list,:]
        return classifier_weights

    def build_clip_fixed_prompts(self, prompt_format, classnames):
        classnames = [name.replace("_", " ") for name in classnames]
        if prompt_format == 'subject':
            prompts = [f"An image of a person or object {name} something." for name in classnames]
        elif prompt_format == 'object':
            prompts = [f"An image of something {name} a person or object." for name in classnames]
        else:
            prompts = [f"An image of the visual relation {name} between two entities." for name in classnames]
        prompts = clip.tokenize(prompts).cuda()
        model, _ = clip.load(name='ViT-B/16', device='cpu')
        model = model.cuda().eval()
        with torch.no_grad():
            text_embeddings = model.encode_text(prompts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

        return text_embeddings
    
      
class Model(nn.Module):

    def __init__(self, args):
        
        super().__init__()
        ds2dim = {"vidor":50, "vidvrd":132}
        self.temp = args.temp_model
        self.ptm_mode = args.ptm_mode
        self.clip_pred_dim = ds2dim[args.dataset]
        self.feat_types = get_feat_types(args)
        self.featEmbedding = FeatEmbedding(args)
        self.int_criterion = FocalWithLogitsLoss()
        self.pre_criterion = FocalWithLogitsLoss()
        self.obj_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)   
 
        self.obj_loss_weight = args.obj_loss_weight
        self.int_loss_weight = args.int_loss_weight
        
        self.pre_text_encoder = PredicateTextEncoder()
        self.obj_text_encoder = ObjectTextEncoder()
        self.src_split = args.src_split
        self.tgt_split = args.tgt_split
        self.temperature = 0.01

        cls_split_info = json.load(open('../dataset/vidvrd/data/openvoc_pred_class_spilt_info.json', 'r'))
        self.id2cls = cls_split_info['id2cls']
        self.cls2id = cls_split_info['cls2id']
        self.cls2split = cls_split_info['cls2split']
        self.base_pids = [self.cls2id[cls] for cls in self.cls2id if self.cls2split[cls] == 'base']
        self.novel_pids = [self.cls2id[cls] for cls in self.cls2id if self.cls2split[cls] == 'novel']
        self.all_pids = list(range(len(self.id2cls)))
        self.cfg = cfg()
        classnames = [self.id2cls[str(id_)] for id_ in range(len(self.id2cls))]
        classnames = [name.replace("_", " ") for name in classnames]

        clip_model, _ = clip.load(name='ViT-B/16', device='cpu')
        clip_model = clip_model.cuda()

        self.pre_classifier = CustomCLIP(self.cfg, classnames, clip_model)
    
    def split_classifier_weights(self, classifier_weights, split):
        pids_list = eval(f"self.{split}_pids")
        if len(classifier_weights.shape) == 3:
            classifier_weights = classifier_weights[:,pids_list,:]
        else:
            classifier_weights = classifier_weights[pids_list,:]
        return classifier_weights
    
    def split_text_embeddings(self, split, pre_classifier_weights):

        # compositional text embeddings
        pre_sbj_text_embeddings = self.pre_text_encoder.split_classifier_weights(
            classifier_weights=self.pre_text_encoder.sbj_classifier_weights,
            split=split).detach()
        pre_obj_text_embeddings = self.pre_text_encoder.split_classifier_weights(
            classifier_weights=self.pre_text_encoder.obj_classifier_weights,
            split=split).detach()
        pre_rel_text_embeddings = self.split_classifier_weights(
            classifier_weights=pre_classifier_weights,
            split=split)
        if len(pre_rel_text_embeddings.shape) > 2:
            bs = pre_rel_text_embeddings.shape[0]
            pre_sbj_text_embeddings = pre_sbj_text_embeddings.unsqueeze(0)
            pre_obj_text_embeddings = pre_obj_text_embeddings.unsqueeze(0)
            pre_sbj_text_embeddings = pre_sbj_text_embeddings.expand(bs,pre_sbj_text_embeddings.shape[1],pre_sbj_text_embeddings.shape[2])
            pre_obj_text_embeddings = pre_obj_text_embeddings.expand(bs,pre_obj_text_embeddings.shape[1],pre_obj_text_embeddings.shape[2])
        pre_text_embeddings = torch.cat([pre_sbj_text_embeddings, pre_obj_text_embeddings, pre_rel_text_embeddings, pre_rel_text_embeddings], dim=-1) / 2
        # pre_text_embeddings = pre_rel_text_embeddings

        obj_text_embeddings = self.obj_text_encoder.split_classifier_weights(
            classifier_weights=self.obj_text_encoder.classifier_weights,
            split=split).detach()
        
        return pre_text_embeddings, obj_text_embeddings


    def forward(self, inputs, seq_lens, labels=None):
        seq_lens = seq_lens.cuda()
        for k in inputs:
            inputs[k] = inputs[k].cuda()
        visual_pre_embeddings, visual_sbj_embeddings, visual_obj_embeddings, interactiveness = self.featEmbedding(inputs, seq_lens)
        visual_pre_embeddings = F.normalize(visual_pre_embeddings,dim=-1)
        visual_sbj_embeddings = F.normalize(visual_sbj_embeddings,dim=-1)
        visual_obj_embeddings = F.normalize(visual_obj_embeddings,dim=-1)

        pre_classifier_weights = self.pre_classifier(visual_pre_embeddings).squeeze(0)

        if not self.training:
            text_pre_embeddings, text_obj_embeddings = self.split_text_embeddings(split=self.tgt_split, pre_classifier_weights=pre_classifier_weights)
            # pre_logits = torch.matmul(visual_pre_embeddings, text_pre_embeddings.t()) / self.temperature
            if len(text_pre_embeddings.shape) == 2:
                text_pre_embeddings = text_pre_embeddings.unsqueeze(0)
            pre_logits = torch.bmm(visual_pre_embeddings, text_pre_embeddings.transpose(1, 2)) / self.temperature
            pre_scores = torch.sigmoid(pre_logits)
            sbj_logits = torch.matmul(visual_sbj_embeddings, text_obj_embeddings.t()) / self.temperature
            sbj_scores = torch.softmax(sbj_logits, dim=-1)
            obj_logits = torch.matmul(visual_obj_embeddings, text_obj_embeddings.t()) / self.temperature
            obj_scores = torch.softmax(obj_logits, dim=-1)
            int_scores = torch.sigmoid(interactiveness)
            if self.tgt_split == 'novel':
                scores_ = torch.zeros([pre_scores.shape[0], pre_scores.shape[1], 132]).cuda()
                scores_[:, :, self.pre_text_encoder.novel_pids] = pre_scores
                pre_scores = scores_

                scores_ = torch.zeros([sbj_scores.shape[0], 35]).cuda()
                scores_[:, self.obj_text_encoder.novel_oids] = sbj_scores
                sbj_scores = scores_

                scores_ = torch.zeros([obj_scores.shape[0], 35]).cuda()
                scores_[:, self.obj_text_encoder.novel_oids] = obj_scores
                obj_scores = scores_

            pre_scores = pre_scores*int_scores.unsqueeze(-1).repeat(1, 1, 132)
            return pre_scores, sbj_scores, obj_scores
            # return pre_scores, None, None
        else:
            assert labels != None

            int_labels = (torch.sum(labels['pre_label'], dim=-1) > 0).float()
            if self.src_split == 'base':
                pre_labels = labels['pre_label'][:, :, self.base_pids]
                sbj_labels = self.obj_text_encoder.convert_oid_on_base[labels['sbj_label']]
                obj_labels = self.obj_text_encoder.convert_oid_on_base[labels['obj_label']]
            else:
                pre_labels = labels['pre_label']
                sbj_labels = labels['sbj_label']
                obj_labels = labels['obj_label']

            text_pre_embeddings, text_obj_embeddings = self.split_text_embeddings(split=self.src_split, pre_classifier_weights=pre_classifier_weights)
            if len(text_pre_embeddings.shape) == 2:
                text_pre_embeddings = text_pre_embeddings.unsqueeze(0)
            pre_logits = torch.bmm(visual_pre_embeddings, text_pre_embeddings.transpose(1, 2)) / self.temperature
            pre_loss = torch.Tensor([0]).cuda()
            int_loss = torch.Tensor([0]).cuda()
            for seq_id, seq_len in enumerate(seq_lens):
                pre_loss += self.pre_criterion(pre_logits[seq_id][:seq_len], pre_labels[seq_id][:seq_len])
                int_loss += self.int_criterion(interactiveness[seq_id][:seq_len], int_labels[seq_id][:seq_len])
            sbj_logits = torch.matmul(visual_sbj_embeddings, text_obj_embeddings.t()) / self.temperature
            obj_logits = torch.matmul(visual_obj_embeddings, text_obj_embeddings.t()) / self.temperature
            if sbj_labels.item() == -1 and obj_labels.item()==-1:
                loss = pre_loss + int_loss*self.int_loss_weight
            elif sbj_labels.item() == -1:
                obj_loss = self.obj_criterion(obj_logits, obj_labels)
                loss = pre_loss + obj_loss*self.obj_loss_weight + int_loss*self.int_loss_weight
            elif obj_labels.item() == -1:
                obj_loss = self.obj_criterion(sbj_logits, sbj_labels)
                loss = pre_loss + obj_loss*self.obj_loss_weight + int_loss*self.int_loss_weight
            else:
                obj_loss = self.obj_criterion(sbj_logits, sbj_labels) + self.obj_criterion(obj_logits, obj_labels) 
                loss = pre_loss + obj_loss*self.obj_loss_weight + int_loss*self.int_loss_weight         
            
            return loss
            # return pre_loss
        

if __name__ == "__main__":
    # from utils.parser_func import parse_args
    # args = parse_args()
    # args.ptm_mode = 'vision_text'
    # Model(args)

    model = SpatialDecoder()
    from torchsummaryX  import summary
    summary(model.cuda(), torch.zeros((32, 4, 512)).cuda())
