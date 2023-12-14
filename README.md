# MMP_OV_VidVRD
This is the implementation for the paper "Multi-modal Prompting for Open-vocabulary Video Visual Relationship Detection"(AAAI2024).
## Prerequisites
- pytorch=2.0.1
- python=3.8.17
- torchvision=0.15.2
- tqdm
- pillow
- ftfy
- regex
  
You can also run the following commands to prepare the conda environment.
```
#preparing environment
bash conda.sh
conda activate MMP_OV_VidVRD
```
## Train & eval for VidVRD dataset
### VidVRD Dataset
**Annotations**  
The frame-annotations we use can be downloaded from [link](https://xdshang.github.io/docs/imagenet-vidvrd.html).  
**Data**  
The category information, trajectory information and gt relation of testing we use are available in the `dataset/vidvrd/data` folder.  
**Features**  
Based on the existing object trajectories, we match objects into pairs and extract features over the duration of these object pairs. We utilize visual features extracted from video frames, object bounding box features, and features extracted by pre-trained model CLIP. All the features we use can be downloaded from [link](https://xdshang.github.io/docs/imagenet-vidvrd.html).
