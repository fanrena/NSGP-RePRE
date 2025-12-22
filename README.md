# Demystifying Catastrophic Forgetting in Two-Stage Incremental Object Detector

This repository contains the official PyTorch implementation of the paper **"Demystifying Catastrophic Forgetting in Two-Stage Incremental Object Detector"**, accepted at the **Forty-first International Conference on Machine Learning (ICML 2025)**.

## Overview

**Abstract:** Catastrophic forgetting is a critical chanllenge for incremental object detection (IOD). Most existing methods treat the detector monolithically, relying on instance replay or knowledge distillation without analyzing component-specific forgetting. Through dissection of Faster R-CNN, we reveal a key insight: Catastrophic forgetting is predominantly localized to the RoI Head classifier, while regressors retain robustness across incremental stages. This finding challenges conventional assumptions, motivating us to develop a framework termed NSGP-RePRE. Regional Prototype Replay (RePRE) mitigates classifier forgetting via replay of two types of prototypes: coarse prototypes represent class-wise semantic centers of RoI features, while fine-grained prototypes model intra-class variations. Null Space Gradient Projection (NSGP) is further introduced to eliminate prototype-feature misalignment by updating the feature extractor in directions orthogonal to subspace of old inputs via gradient projection, aligning RePRE with incremental learning dynamics. Our simple yet effective design allows NSGP-RePRE to achieve state-of-the-art performance on the Pascal VOC and MS COCO datasets under various settings. Our work not only advances IOD methodology but also provide pivotal insights for catastrophic forgetting mitigation in IOD. Code is available at [this link](https://github.com/fanrena/NSGP-RePRE).


**TL;DR:** We identify the RoI head classifier as the main source of forgetting in two-stage detectors and propose NSGP-RePRE, which uses Regional Prototype Replay and Null Space Gradient Projection to address it.

**Authors:** Qirui Wu, Shizhou Zhang, De Cheng, Yinghui Xing, Di Xu, Peng Wang, Yanning Zhang  
**Email:** wuqirui@mail.nwpu.edu.cn  
**Paper:** [arXiv:2502.05540](https://arxiv.org/abs/2502.05540)

## Getting Started

### Prerequisites

- Python 3.8.13
- PyTorch 1.12.0+cu113
- [MMDetection 3.3.0](https://github.com/open-mmlab/mmdetection)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/fanrena/NSGP-RePRE.git
    cd NSGP-RePRE
    ```

2.  Install the package in editable mode along with its dependencies:
    ```bash
    pip install -v -e .
    ```
    
    For a more detailed installation guide, please refer to the official [MMDetection documentation](https://mmdetection.readthedocs.io/en/latest/get_started.html).

### Running Experiments

1.  **Prepare Configuration Files:**
    Configure your experiments according to your specific incremental learning tasks. Detailed instructions and examples can be found in [`cl_faster_rcnn_cfgs/README.md`](https://github.com/fanrena/NSGP-RePRE/blob/main/cl_faster_rcnn_cfgs/README.md).

2.  **Execute Training:**
    Add the paths to your configuration files in `train_list.sh`. Then, run the following command to start all experiments sequentially:
    ```bash
    bash train_list.sh
    ```

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{wu2025demystifying,
  title={Demystifying Catastrophic Forgetting in Two-Stage Incremental Object Detector},
  author={Wu, Qirui and Zhang, Shizhou and Cheng, De and Xing, Yinghui and Xu, Di and Wang, Peng and Zhang, Yanning},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```
