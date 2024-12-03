# Lightweight-Adaptive-Feature-De-drifting-for-Compressed-Image-Classification 

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2401.01724)  [![Project](https://img.shields.io/badge/Project-Page-blue.svg)](https://arxiv.org/pdf/2411.10798) 


> **Lightweight-Adaptive-Feature-De-drifting-for-Compressed-Image-Classification**<br>
> Long Peng<sup>1</sup>, Yang Cao<sup>1</sup>, Yuejin Sun<sup>1</sup>, Yang Wang<sup>1</sup> <br>
> <sup>1</sup> University of Science and Technology of China

## :bookmark: News!!!
- [x] 2023-12-1 **Accepted by IEEE Transactions on Multimedia.**

## Abstract

JPEG is a widely used compression scheme to efficiently reduce the volume of the transmitted images at the expense of visual perception drop. The artifacts appear among blocks due to the information loss in the compression process, which not only affects the quality of images but also harms the subsequent high-level tasks in terms of feature drifting. High-level vision models trained on high-quality images will suffer performance degradation when dealing with compressed images, especially on mobile devices. In recent years, numerous learning-based JPEG artifact removal methods have been proposed to handle visual artifacts. However, it is not an ideal choice to use these JPEG artifact removal methods as a pre-processing for compressed image classification for the following reasons: 1) These methods are designed for human vision rather than high-level vision models. 2) These methods are not efficient enough to serve as a pre-processing on resource-constrained devices. To address these issues, this paper proposes a novel lightweight adaptive feature de-drifting module (AFD-Module) to boost the performance of pre-trained image classification models when facing compressed images. First, a Feature Drifting Estimation Network (FDE-Net) is devised to generate the spatial-wise Feature Drifting Map (FDM) in the DCT domain. Next, the estimated FDM is transmitted to the Feature Enhancement Network (FE-Net) to generate the mapping relationship between degraded features and corresponding high-quality features. Specially, a simple but effective RepConv block equipped with structural re-parameterization is utilized in FE-Net, which enriches feature representation in the training phase while keeping efficiency in the deployment phase. After training on limited compressed images, the AFD-Module can serve as a “plug-and-play” module for pre-trained classification models to improve their performance on compressed images. Experiments on images compressed once (i.e. ImageNet-C) and multiple times demonstrate that our proposed AFD-Module can comprehensively improve the accuracy of the pre-trained classification models and significantly outperform the existing methods.

## Motivation
![Motivation](TMM1.jpg)


## Performance
![performance-1](performance-1.jpg)
![performance-2](performance-2.jpg)

## Feature Enhancement
![feature_vis](feature_vis.jpg)


## Cite US
Contact email for Long Peng: longp2001@mail.ustc.edu.cn. Please cite us if this work is helpful to you. 
```
@ARTICLE{10400436,
  author={Peng, Long and Cao, Yang and Sun, Yuejin and Wang, Yang},
  journal={IEEE Transactions on Multimedia}, 
  title={Lightweight Adaptive Feature De-Drifting for Compressed Image Classification}, 
  year={2024},
  volume={26},
  number={},
  pages={6424-6436},
  keywords={Image coding;Transform coding;Discrete cosine transforms;Feature extraction;Performance evaluation;Mobile handsets;Image recognition;Feature drifting;feature enhancement;image classification;JPEG compression},
  doi={10.1109/TMM.2024.3350917}}
```
