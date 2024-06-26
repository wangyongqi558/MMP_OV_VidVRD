# MMP_OV_VidVRD
This is the implementation for the paper "Multi-modal Prompting for Open-vocabulary Video Visual Relationship Detection" (AAAI2024).
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
## Quick Start
Our focus is on open-vocabulary relationship detection. So we extract the features of object pairs and provide open-vocabulary relationship detection code which conducts training in two stages.  
- In the first stage, we train only the spatio-temporal modeling module for vision side, using manually crafted prompts on the text side.
- In the second stage, we freeze the vision part and train the vision-guided prompt module.
### VidVRD Dataset
#### Annotation
The frame-annotations we use can be downloaded from [link](https://xdshang.github.io/docs/imagenet-vidvrd.html). Please save them to the  `dataset/vidvrd/anno` folder. 
#### Data
The category information, trajectory information and gt relation of testing we use are available in the `dataset/vidvrd/data` folder. 
#### Feature
Based on the existing object trajectories, we match objects into pairs and extract features over the duration of these object pairs. We utilize visual features extracted from video frames, object bounding box features, and features extracted by pre-trained model CLIP(ViT-B/16). All the features we use can be downloaded from [link](https://pan.baidu.com/s/1h1A2Qfcj6oEW8VJDYKyRlA?pwd=a8s6). Please save them to the  `dataset/vidvrd/feature` folder. 

Recently, we uploaded feature extraction files which can be downloaded via the path `scripts/features.py` and `scripts/features.sh`. You can download relevant data of VidVRD dataset via the path `dataset/vidvrd/data` and relevant data of VidOR dataset from [link](https://pan.baidu.com/s/11SioAmEkWVpbWuAnVmTE-g?pwd=tgn4). Then, use the feature extraction files to extract features.
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
### Test
- Firstly, download the trained models from [link](https://pan.baidu.com/s/1is8cNDm0_Ni3XeQawGQRwg?pwd=9pe2) and save them to the `dataset/vidvrd/model` folder.
- Secondly, run the following commands for test:  
 ```
#Test for VidVRD
cd scripts
bash test_vidvrd_openvoc.sh
```
### Train

Run the following commands for training and evaluation:
```
#Train for VidVRD
cd scripts
bash train_vidvrd_openvoc.sh --stage2
#Evaluate for VidVRD
bash test_vidvrd_openvoc.sh
```
## Citation
```
@inproceedings{yang2024multi,
  title={Multi-Modal Prompting for Open-Vocabulary Video Visual Relationship Detection},
  author={Yang, Shuo and Wang, Yongqi and Ji, Xiaofeng and Wu, Xinxiao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={7},
  pages={6513--6521},
  year={2024}
}

```
