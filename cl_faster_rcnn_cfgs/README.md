# Configuration Guide

This document provides a brief guide to configuring the NSGP-RePRE method. The primary model modifications are concentrated in the **RoI Head** section, where we replace it with a custom module that supports prototype replay.

## Quick Customization Guide

You can easily customize your configuration in three steps:
1. **Change the dataset configuration** (modify Line 12 in the `_base_` list).
2. **Change the task ID** (modify Line 16 for `task_id`), ensuring it aligns with your dataset configuration.
3. **Change the task split** (modify Line 17 for `train_task_split`), ensuring it aligns with your dataset configuration.

**Note:** You must also update the corresponding dataset configuration file with the same `task_id` and `train_task_split` values.

### Configuration Naming Convention
Please name your configuration files in the following format: `"any_prefix_{base_class_num}_{incremental_class_num}_{task_id}.py"`. Utility functions within the framework will automatically deduce the current working directory based on this naming convention and the `previous_dir` parameter.

## Configuration Example & Parameter Details

```python
_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/voc_5_5_task1_2007.py', 
    '../_base_/schedules/schedule_1x_sgdnscl.py',
    '../_base_/brnsrunetime.py'
]

# ==================== Core Task Parameters ====================
task_id = 1                      # Current task ID, must match dataset configuration
train_task_split = [0, 5, 10, 15, 20]  # Task split for each class, must match dataset configuration

# ==================== NSGP (Null Space Gradient Projection) Parameters ====================
offset = 0.0                     # Offset to adjust adaptive threshold. See line 98 in `.mmdet/engine/optimizers/SGD_NSCL.py` for details.
ignore_keys = ["rpn", "roi_head"] # Keywords of modules not regulated by Null Space Gradient Projection. Mainly four parts: backbone, neck, rpn, roi_head.

# ==================== Checkpoint Loading Settings ====================
# Two methods to load checkpoints from the previous task:
#   Method 1 (Auto-inference): Combine `previous_dir` and `ckpt_keywords` to automatically detect a checkpoint from the last task directory containing the keywords.
#   Method 2 (Manual): Directly specify the path via `load_from`. This method has higher priority.

previous_dir = 'path/of/your/last/task'  # Previous task directory, also used to store NSGP and RePRE data.
ckpt_keywords = "your keywords"           # Keywords within the checkpoint filename (e.g., "12" in "epoch12.pth"). This checkpoint is used to compute NSGP and RoI features for the current task and should align with the checkpoint loaded in the next task.
load_from = "path/of/your/ckpt/from/last/task"  # Manually specify checkpoint path (takes precedence over auto-inference)

# ==================== Training & Control Parameters ====================
is_trained = False              # If True, compute NSGP and RePRE data only without training. Computation is automatically performed after training.
max_prototype = 10              # Total number of coarse + fine-grained prototypes to retain. Default is 10.
rr_thresh = [0.5, 0.7]          # IoU thresholds for removing duplicate boxes in RPN and RoI head.

# ==================== Model Definition ====================
# Primary modification: Using a custom RoI Head that supports multi-prototype replay.
model = dict(
    type='FasterRCNNRoIReplay',  # Main model with integrated replay mechanism
    data_preprocessor=dict(...),
    backbone=dict(...),
    neck=dict(...),
    rpn_head=dict(...),
    # ===== Core Modification: RoI Head =====
    roi_head=dict(
        type='StandardMultiPrototypeReplayHead',  # Custom multi-prototype replay head
        previous_path=previous_dir,               # Path to previous task directory for reading prototype data
        task_id=task_id,
        task_split=train_task_split,
        max_prototype=max_prototype,
        bbox_roi_extractor=dict(...),
        bbox_head=dict(
            type='Shared2FCBBoxHeadTask',  # Custom classification head for incremental learning
            task_id=task_id,
            task_split=train_task_split,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=20,                # Total number of classes in VOC dataset
            bbox_coder=dict(...),
            reg_class_agnostic=False,
            loss_cls=dict(...),
            loss_bbox=dict(...)
        )
    ),
    train_cfg=dict(...),
    test_cfg=dict(...)
)
```

## Summary of Key Modifications

1. **Model Type (`type`)**: Changed to `FasterRCNNRoIReplay`.
2. **RoI Head**: Replaced with `StandardMultiPrototypeReplayHead`, the core module implementing **Regional Prototype Replay (RePRE)**.
3. **BBox Head**: Changed to `Shared2FCBBoxHeadTask`, making it task-aware and adaptable for incremental learning.
4. **Parameter Passing**: Critical parameters like `task_id`, `task_split`, `previous_path`, and `max_prototype` are passed to the RoI Head to control prototype management and replay logic.
5. **brnsrunetime**: NSGP Runner, the core module implementing **Null Space Gradient Projection(NSGP)**.

With these modifications, this configuration implements the NSGP and RePRE methods proposed in the paper, aiming to mitigate catastrophic forgetting in two-stage incremental object detectors.