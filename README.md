# PrismF
# More Perspectives, Stronger Signals: Multi-perspective enhancement and Progressive fusion for Multimodal Knowledge Graph Completion


## Introduction
Learning robust and fine-grained entity representations is essential for enhancing reasoning capabilities in multimodal knowledge graph completion (MMKGC). However, existing methods often struggle to extract discriminative intra-modal semantics and maintain balanced cross-modal interactions, especially under sparse or noisy conditions. To overcome these challenges, we propose PrismF, a novel framework that captures richer multimodal semantics through multi-perspective modeling and adaptive fusion. Specifically, we introduce a Multi-Perspective Enhancement (MuPE) module to extract diverse fine-grained features across different perspectives, and apply a Progressive Modality Fusion (PMF) mechanism to dynamically balance the contribution of each modality based on contextual confidence. Furthermore, we employ a decoupling constraint to preserve the representational diversity across perspectives and a  gating strategy to guide the cross-modal integration. Extensive experiments on three benchmark MMKG datasets validate the effectiveness of PrismF, achieving state-of-the-art performance in complex multimodal reasoning scenarios.


## Overview

<p align="center">
   <img src="HiMod.png" width="900">
</p>

## Dependencies

- torch == 1.12.1
- torch_scatter == 2.0.9
- numpy == 1.21.6
- scipy == 1.10.1

## Data

The datasets consist of WN18RR, FB15K-237, and NELL-995, each dataset was split into four unique training/testing configurations, resulting in 12 experimental settings.

## Train
The full training scripts can be found in [reproduce.sh]. For example, training on `WN18RR v1` dataset:

##### WN18RR-v1 dataset

```
python train.py --data_path ./data/WN18RR_v1
```
