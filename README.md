# ADAPT

Hard Negative-Aware Multi-Prototype Contrastive Learning for Out-of-Distribution Detection


**Abstract**: Out-of-distribution (OOD) detection is an essential task when deploying trained deep learning models under the closed-world assumption to real-world scenarios. Recently, many researchers use prototypes representing the mean of each in-distribution (ID) class as anchors in contrastive learning. This approach allows samples within a class to cluster more tightly for compact ID data embeddings while enabling dispersed embeddings for OOD data. As studies utilizing prototypes shift from single to multi-prototype approaches, some prototypes positioned farther from the class center become vulnerable to hard negative samples; these samples have similar features to the anchor prototype despite belonging to different classes. To alleviate this issue, we propose ADAPT, a simple yet effective strategy to adapt for hard negative samples on multiprototype contrastive learning. This approach dynamically adjusts the weight for negative samples, enabling the model to robustly handle hard negative samples and leading to more compact intra-class and dispersed inter-class embeddings for ID data. Moreover, to ensure training stability for initial unstable prototypes, we quantify prototype alignment level and adjust the temperature coefficient used for training accordingly. Consequently, ADAPT demonstrates state-of-the-art performance on both standard and more challenging Near-OOD detection benchmarks.

<img width="1881" height="548" alt="image" src="https://github.com/user-attachments/assets/d7f7d890-e877-45a8-88ba-9f964ce7d1e0" />

# Get Started
1. Install Python 3.9.23 Pytorch 2.0.0

2. This experiment requires the data to be downloaded

# Data
SVHN: http://ufldl.stanford.edu/housenumbers/test_32x32.mat

LSUN: [https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)

iSUN: https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz

Tiny-ImageNet: https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz
  
Textures: https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz


The dataset should be located as follows:
```text
datasets
└── CIFAR100
    └── cifar-100-python
└── SVHN
    └── test_32x32
└── iSUN
    └── ...
└── LSUN_resize
    └── ...
└── imagenet_resize
    └── ...
└── texture
    └── ...

```


# Setup
## Clone the repo

```bash
git clone https://github.com/JINHYUKSUH/ADAPT.git
cd ADAPT
conda create -n ADAPT python=3.9
conda activate CoReVAD
pip install -r requirements.txt
```

## Pipeline Overview

This repository provides a simple two-step pipeline for OOD detection:

1. **Train an ADAPT model on in-distribution data**
2. **Extract features and perform OOD inference in a single evaluation script**

---

## 1. Training (`train_adapt.py`)

```bash
python train_adapt.py \
  --in-dataset CIFAR-100 \
  --backbone resnet34 \
  --method ADAPT \
  --epochs 500 \
  --batch-size 512 \
  --save-path ADAPT.pt

```


The trained model is saved as:

ADAPT.pt


## 2. Feature Extraction & OOD Inference (`eval_ood.py`)

This script performs **feature extraction and OOD evaluation in a single run**.

```bash
python eval_ood.py --in_dataset CIFAR-100 --save_path ADAPT.pt

```

