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
Our focus is on open-vocabulary relationship detection. So we extract the features of object pairs and provide open-vocabulary relationship detection code which conducts training in two stages.  
- In the first stage, we train only the spatio-temporal modeling module for vision, using manually crafted prompts on the text side.
- In the second stage, we freeze the visual part and train the visual-guided prompt module.
### VidVRD Dataset
#### Annotation
The frame-annotations we use can be downloaded from [link](https://xdshang.github.io/docs/imagenet-vidvrd.html). Please save them to the  `dataset/vidvrd/anno` folder. 
#### Data
The category information, trajectory information and gt relation of testing we use are available in the `dataset/vidvrd/data` folder. 
#### Feature
Based on the existing object trajectories, we match objects into pairs and extract features over the duration of these object pairs. We utilize visual features extracted from video frames, object bounding box features, and features extracted by pre-trained model CLIP(ViT-B/16). All the features we use can be downloaded from [link](https://pan.baidu.com/s/1h1A2Qfcj6oEW8VJDYKyRlA?pwd=a8s6). Please save them to the  `dataset/vidvrd/feature` folder. 
#### Model
Our trained models are provided in [link](https://pan.baidu.com/s/1is8cNDm0_Ni3XeQawGQRwg?pwd=9pe2). Please download them to the `dataset/vidvrd/model` folder.  
#### Format
The dataset should be formatted as, e.g.,
```
dataset/
|   vidvrd/
|   |   anno/-----------------------------------------------------------------
|   |   |   train/------------------------------------------------------------
|   |   |   test/-------------------------------------------------------------
|   |   data/-----------------------------------------------------------------
|   |   feature/--------------------------------------------------------------
|   |   |   train_gt_30/------------------------------------------------------
|   |   |   train_gt_30_box/--------------------------------------------------
|   |   |   train_gt_30_ptm/--------------------------------------------------
|   |   |   test_gt_30/-------------------------------------------------------
|   |   |   test_gt_30_box/---------------------------------------------------
|   |   |   test_gt_30_ptm/---------------------------------------------------
|   |   |   test_meta_30/-----------------------------------------------------
|   |   |   test_meta_30_box/-------------------------------------------------
|   |   |   test_meta_30_ptm/-------------------------------------------------
|   |   model/----------------------------------------------------------------
|   vidor/--------------------------------------------------------------------
|   |   ...-------------------------------------------------------------------
```
### Quick Start
Run the following commands for evaluation:
```
#Evaluate for VidVRD
cd scripts
bash test_vidvrd_openvoc.sh
```
Run the following commands for training:
```
#Train for VidVRD
cd scripts
bash train_vidvrd_openvoc.sh --stage2
```
