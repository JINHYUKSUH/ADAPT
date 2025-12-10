# ADAPT

Hard negative-aware multi-prototype contrastive learning for Out-of-distribution Detection


Out-of-distribution (OOD) detection is an essential task when we deploy the trained deep learning model with the closed-world assumption to real-world scenarios. Recently, many researchers use prototypes representing the mean of each in-distribution (ID) class as anchors in contrastive learning. This approach allows samples within a class to cluster more tightly for compact ID data embeddings while enabling dispersed embeddings for OOD data. As studies utilizing prototypes shift from single to multi-prototype approaches, some prototypes positioned farther from the class center become vulnerable to hard negative samples; these samples have similar features to the anchor prototype. To alleviate this issue, we propose ADAPT, a novel strategy to adapt for hard negative samples on multi-prototype contrastive learning. This approach dynamically adjusts the weight for negative samples, enabling the model to robustly handle hard negative samples and leading to more compact intra-class and dispersed inter-class embeddings for ID data. Moreover, to ensure training stability for initial unstable prototypes, we quantify prototype alignment level and adjust the temperature coefficient used for training accordingly. Consequently, ADAPT demonstrates state-of-the-art performance on both standard and more challenging Near-OOD detection benchmarks.


[overview.pdf](https://github.com/user-attachments/files/24079070/overview.pdf)


# Get Started
  1. Install Python 3.9.23 Pytorch 2.0.0

  2. This experiment requires the data to be downloaded

  SVHN: http://ufldl.stanford.edu/housenumbers/test_32x32.mat

  LSUN: [https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)

  iSUN: https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz

  Tiny-ImageNet: https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz
  
  Textures: https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz

  3. You can reproduce the experiment result by FALCON.py.

    git clone https://github.com/abb155889/FALCON.git
    cd ADAPT
    # Edit dataset paths in run_experiments.sh before running
    bash run_experiments.sh

# Pretrained weights
Please check out our pretrained weights here

