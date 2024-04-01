from clip import clip
import InternVideo
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from InternVideo import video_transform

class CLIPVisualEncoder(nn.Module):
    def __init__(self, backbone="ViT-B/16", gpu_id=0):
        super().__init__()
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load(name = backbone, device = self.device)
        self.preprocess = preprocess
        self.visual =  clip_model.visual
        self.dtype = clip_model.dtype

    def forward(self, regions):
        feats = []
        for image_path, bbox in regions:
            if bbox:
                image = Image.open(image_path).crop(bbox)
            else:
                image = Image.open(image_path)
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            feat = self.visual(image.type(self.dtype)).detach().cpu().numpy()[0]
            feats.append(feat)
        return np.asarray(feats)


class InternlVisualEncoder(nn.Module):
    def __init__(self, backbone="ViT-B/16", gpu_id=0):
        super().__init__()
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.model = InternVideo.load_model("/media/sda1/jixf/common_data/ckpts/InternVideo-MM-L-14.ckpt").cuda(device=self.device)
        self.preprocess = transforms.Compose([
        video_transform.TensorToNumpy(),
        video_transform.Resize((224, 224)),
        video_transform.ClipToTensor(channel_nb=3),
        video_transform.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    def forward(self, sequences):
        feats = []
        for sequence in sequences:
            video = []
            for image_path, bbox in zip(sequence[0], sequence[1]):
                if bbox:
                    image = Image.open(image_path).crop(bbox).resize((224,224))
                else:
                    image = Image.open(image_path)
                video.append(torch.from_numpy(np.array(image)).unsqueeze(0))
            video = torch.cat(video, dim=0)
            video = video[np.linspace(0, len(video) - 1, 8).astype(np.int)]
            video = video.permute(3, 0, 1, 2)
            video = self.preprocess(video).unsqueeze(0).to(self.device)
            feats.append(self.model.encode_video(video).detach().cpu().numpy()[0])
        return np.asarray(feats)