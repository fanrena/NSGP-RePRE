# Configs

A brief example of configs. You can easily customize your config by
1. Change the dataset config(Line 12 in this file).
2. Change the task_id and it should be aligned with the dataset config(Line 16 in this file).
3. Change the train_task_split and it should be aligned with the dataset config(Line 17 in this file).  
You should also modify the corresponding dataset config by changing task_id and train_task_split in dataset config to your desired setup.

You should arrange your config's name in the format of f"whatever_you_like_{base_class_num}_{incremental_class_num}_{task_id}.py" as there is a function that deduce current work_dir according to your previous_dir.

```python
_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/voc_5_5_task1_2007.py', 
    '../_base_/schedules/schedule_1x_sgdnscl.py', '../_base_/brnsrunetime.py'
]

task_id = 1
train_task_split = [0, 5, 10, 15, 20]

offset = 0.0 # Offset to adjust adaptive threshold. See .mmdet/engine/optimizers/SGD_NSCL.py line 98 for more details.
ignore_keys = ["rpn", "roi_head"] # Keywords of parts that are not regulated by Null Space Gradient Projection. Mainly four parts: backbone, neck, rpn, roi_head.

# There are two ways to load ckpt from last tasks.
## 1. Combination of previous_dir and ckpt_keywords. This allows you to automatically detect ckpt from last task with ckpt_keywords in its name. For example: "12" in "path/of/your/last/task/epoch12.pth".
previous_dir = 'path/of/your/last/task' # Also the folder you store your NSGP and RePRE data.
ckpt_keywords = "your keywords" # This also controls the Null Space Gradient Projection and RoI computation in current step. It loads ckpt from **Current** work dir. The reason for this design is because not every last ckpt is desired ckpt to compute NSGP and RoI features. This ckpt should be aligned with the loaded ckpt in the next task. 
## 2. Regular load from your path/of/your/ckpt/from/last/task. Note that load_from has higher priority.
load_from = "path/of/your/ckpt/from/last/task"

is_trained = False # Do NSGP and RePRE data computation but do not training. Note that the computation is automatically performed after the training.

max_prototype = 10 # How many coarese+finegrained prototype you want. Default to 10.

rr_thresh = [0.5, 0.7] # IoU thresholds to remove duplicates in RPN and RoI head.
# model settings
model = dict(
    type='FasterRCNNRoIReplay', # 
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint="pretrained/resnet50-0676ba61.pth")),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardMultiPrototypeReplayHead',
		previous_path = previous_dir,
		task_id = task_id,
		task_split=train_task_split,
        max_prototype = max_prototype,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHeadTask',
            task_id = task_id,
			task_split=train_task_split,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=20,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))


```