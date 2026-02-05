# Run FLAPPE on RoboTwin (Simulation)
> This example demonstrates how to run on the **RoboTwin** simulation benchmark.<br/>


## 1. Environment Setup
First, clone and install the [RoboTwin repo](https://github.com/RoboTwin-Platform/RoboTwin) and required packages. You can follow the guidance in [RoboTwin Document](https://robotwin-platform.github.io/doc/usage/robotwin-install.html#1-dependencies).
```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
conda create -n flappe python=3.10 -y
conda activate flappe

bash script/_install.sh  #Install RoboTwin basic envs and CuRobo

bash script/_download_assets.sh #Download assets (RoboTwin-OD, Texture Library and Embodiments)
```
Then we can continue to set up the environment for environment. 
```bash
# Make sure python version == 3.10
conda activate flappe

# Install pytorch
# Look up https://pytorch.org/get-started/previous-versions/ with your cuda version for a correct command
pip install torch==2.1.0 torchvision==0.16.0  --index-url https://download.pytorch.org/whl/cu121

# Install packaging
pip install packaging==24.0
pip install ninja
# Verify Ninja --> should return exit code "0"
ninja --version; echo $?
# Install flash-attn
pip install flash-attn==2.7.2.post1 --no-build-isolation

# Install other prequisites
pip install -r requirements.txt
```

Then clone our repo as a policy of the RoboTwin, the directory structure will be as below:
```bash
cd policy
git clone xxx
```
```
RoboTwin
    ├── policy
    ·   ├── FLAPPE        
        │
        └── other policys ...
```
## 2. Download Model
Download the pretrained ckpt nad Encoders we will use in the training stage.

```bash
# In the RoboTwin ROOT directory
cd policy
mkdir weights
cd weights
mkdir RDT && cd RDT

# Download the models
huggingface-cli download google/t5-v1_1-xxl --local-dir t5-v1_1-xxl
huggingface-cli download google/siglip-so400m-patch14-384 --local-dir siglip-so400m-patch14-384
huggingface-cli download robotics-diffusion-transformer/rdt-1b --local-dir rdt-1b

# Teacher eocders
#theia
huggingface-cli download theaiinstitute/theia-base-patch16-224-cdiv --local-dir theia-base-patch16-224-cdiv
#clip
huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K --local-dir CLIP-ViT-H-14-laion2B-s32B-b79K 
#vit
huggingface-cli download google/vit-huge-patch14-224-in21k --local-dir vit-huge-patch14-224-in21k
#dinov2
git clone https://github.com/facebookresearch/dinov2.git
cd dinov2-main
mkdir checkpoints && cd checkpoints
huggingface-cli download facebook/dinov2-base 
```
Then update your real paths of teacher encoders (Theia, CLIP, VIT, DINOv2) in the [utils.py](./models/utils.py).

## 3. Training
We offer a training example for our method. It contains two stage training: 
1) **mid-training ([finetune_mid.sh](finetune_mid.sh) & model_config/[mid_train.yml](model_config\mid_train.yml))**  
For [mid_train.yml](model_config\mid_train.yml), you should update the path of the pretrained ckpts and the pretrained_model_name_or_path; 
2) **post-training ([finetune_post.sh](finetune_post.sh) & model_config/[post_train.yml](model_config\post_train.yml))**
For [post_train.yml](model_config\post_train.yml), you should update the path of the mid-train ckpts, the pretrained_model_name_or_path and the teacher encoder paths;

```bash
conda activate flappe
bash finetune_mid.sh # or bash finetune_post.sh
```
The default configurations match the experimental setup in our paper.
## 4. Inference
We offer a inference example for our method ([eval.sh](./eval.sh)). 

**model_name** should be the checkpoint file name  under the `./checkpoints` folders.
```bash
conda activate flappe
bash eval.sh
```
