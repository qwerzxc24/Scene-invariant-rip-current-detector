# Scene-invariant Rip Current Detector

## Getting started

### 1. Installation

Mamba YOLO is developed based on `torch==2.2.1` `pytorch-cuda==12.1` and `CUDA Version==13.6`. 

#### 2.Clone Project 

```bash
git clone https://github.com/qwerzxc24/Scene-invariant-rip-current-detector.git
```

#### 3.Create and activate a conda environment.
```bash
conda create -n siRipdet -y python=3.11
conda activate siRipdet
```

#### 4. Install torch

```bash
pip3 install torch===2.2.1 torchvision torchaudio
```

#### 5. Install Dependencies
```bash
pip install seaborn thop timm einops
cd selective_scan && pip install . && cd ..
pip install -v -e .
```

#### 6. Download RipScape dataset
RipScape dataset: https://huggingface.co/datasets/anonrip/RipScape

Make sure the data is structured as recommended by Ultralytics: https://github.com/ultralytics/ultralytics

#### 7. Train and test
train.py and test_sample.py are provided as templates.

## Acknowledgement

This repo is modified from [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO).
