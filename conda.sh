conda create -n MMP_OV_VidVRD python=3.8.17
conda activate MMP_OV_VidVRD
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install tqdm pillow ftfy regex
