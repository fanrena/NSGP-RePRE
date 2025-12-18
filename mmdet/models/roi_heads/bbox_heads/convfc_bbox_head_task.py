# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from torch import Tensor

from mmdet.registry import MODELS
from .bbox_head import BBoxHead


@MODELS.register_module()
class ConvFCBBoxHeadTask(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs: int = 0,
                 num_shared_fcs: int = 0,
                 num_cls_convs: int = 0,
                 num_cls_fcs: int = 0,
                 num_reg_convs: int = 0,
                 num_reg_fcs: int = 0,
                 conv_out_channels: int = 256,
                 fc_out_channels: int = 1024,
                 conv_cfg: Optional[Union[dict, ConfigDict]] = None,
                 norm_cfg: Optional[Union[dict, ConfigDict]] = None,
                 init_cfg: Optional[Union[dict, ConfigDict]] = None,
                 task_split: list = [0, 10, 20],
                 task_id: int = 1,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.task_split = task_split
        self.task_id = task_id
        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            extra_channels = cls_channels - self.num_classes
            self.fc_cls = nn.ModuleList()
            # 前景类，各用一个FC层。
            for i in range(1, len(self.task_split)):
                cls_channels = self.task_split[i] - self.task_split[i-1]
                cls_predictor_cfg_ = self.cls_predictor_cfg.copy()
                cls_predictor_cfg_.update(
                    in_features=self.cls_last_dim, out_features=cls_channels)
                self.fc_cls.append(MODELS.build(cls_predictor_cfg_))
            # 背景类
            cls_predictor_cfg_ = self.cls_predictor_cfg.copy()
            cls_predictor_cfg_.update(
                in_features=self.cls_last_dim, out_features=extra_channels)
            self.fc_cls.append(MODELS.build(cls_predictor_cfg_))
            self.background_nums = extra_channels
        if self.with_reg:
            self.fc_reg = nn.ModuleList()
            if self.reg_class_agnostic:
                box_dim = self.bbox_coder.encode_size
                out_dim_reg = box_dim if self.reg_class_agnostic else \
                    box_dim * self.num_classes
                reg_predictor_cfg_ = self.reg_predictor_cfg.copy()
                if isinstance(reg_predictor_cfg_, (dict, ConfigDict)):
                    reg_predictor_cfg_.update(
                        in_features=self.reg_last_dim, out_features=out_dim_reg)
                self.fc_reg.append(MODELS.build(reg_predictor_cfg_))
            else:
                for i in range(1, len(self.task_split)):
                    cls_channels = self.task_split[i] - self.task_split[i-1]
                    box_dim = self.bbox_coder.encode_size
                    out_dim_reg =  box_dim * cls_channels
                    reg_predictor_cfg_ = self.reg_predictor_cfg.copy()
                    if isinstance(reg_predictor_cfg_, (dict, ConfigDict)):
                        reg_predictor_cfg_.update(
                            in_features=self.reg_last_dim, out_features=out_dim_reg)
                    self.fc_reg.append(MODELS.build(reg_predictor_cfg_))
        # 冻结无关分类头、回归头
        for i, module in enumerate(self.fc_cls):
            if i + 1 <= self.task_id:
                module.requires_grad_(True)
            else:
                module.requires_grad_(False)
            if i+1 == len(self.fc_cls):
                module.requires_grad_(True)
       
        for i, module in enumerate(self.fc_reg):
            if i + 1 <= self.task_id:
                module.requires_grad_(True)
            else:
                module.requires_grad_(False)
            if self.reg_class_agnostic:
                module.requires_grad_(True)
                
        if init_cfg is None:
            # when init_cfg is None,
            # It has been set to
            # [[dict(type='Normal', std=0.01, override=dict(name='fc_cls'))],
            #  [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]
            # after `super(ConvFCBBoxHead, self).__init__()`
            # we only need to append additional configuration
            # for `shared_fcs`, `cls_fcs` and `reg_fcs`
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]
        
        self.null_space = False

    def _add_conv_fc_branch(self,
                            num_branch_convs: int,
                            num_branch_fcs: int,
                            in_channels: int,
                            is_shared: bool = False) -> tuple:
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
                print(fc_in_channels)
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x: Tuple[Tensor]) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = None
        bbox_pred = None
        # 用所有分类器进行分类
        if self.with_cls:
            preds = []
            for i, module in enumerate(self.fc_cls):
                if i+1 > self.task_id and i+1 != len(self.task_split):
                    # print(f"task {i + 1} detached")
                    o = module(x_cls.detach())
                else:
                    # print(f"task {i + 1} NOT detached")
                    o = module(x_cls)
                if i+1 > self.task_id and i+1 != len(self.task_split):
                    o = torch.zeros_like(o)
                    o = o.masked_fill(o==0, float('-inf')) # 利用exp(-inf)=0的性质无视未来任务的类别预测。
                preds.append(o)
            cls_score = torch.cat(preds, dim=-1)
        if self.with_reg:
            preds = []
            for i, module in enumerate(self.fc_reg):
                if i+1 > self.task_id and not self.reg_class_agnostic:
                    o = module(x_reg.detach())
                else:
                    o = module(x_reg)
                if i+1 > self.task_id and not self.reg_class_agnostic:
                    o = torch.zeros_like(o)
                preds.append(o)
            bbox_pred = torch.cat(preds, dim=-1)
        return cls_score, bbox_pred
    
    
    def get_relu(self, x: Tuple[Tensor]) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                print("shared convs")
                x = conv(x)
        ret = []
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                print("avg pool")
                x = self.avg_pool(x)

            x = x.flatten(1)

            for i, fc in enumerate(self.shared_fcs):
                # print(f"shared fcs {i} {x.shape}")
                x = self.relu(fc(x))
                ret.append(x)
        # separate branches
        return ret

    def loss_and_target(self,
                        cls_score: Tensor,
                        bbox_pred: Tensor,
                        rois: Tensor,
                        sampling_results,
                        rcnn_train_cfg: ConfigDict,
                        concat: bool = True,
                        reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the features extracted by the bbox head.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch. Defaults to True.
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss and targets components.
                The targets are only used for cascade rcnn.
        """

        cls_reg_targets = self.get_targets(
            sampling_results, rcnn_train_cfg, concat=concat)
        cls_target, cls_weight, bbox_target, bbox_weight = cls_reg_targets
        if self.null_space:
            self.mask = cls_target == 20
        losses = self.loss(
            cls_score,
            bbox_pred,
            rois,
            cls_target,
            cls_weight,
            bbox_target,
            bbox_weight,
            reduction_override=reduction_override)

        # cls_reg_targets is only for cascade rcnn
        return dict(loss_bbox=losses, bbox_targets=cls_reg_targets)
    
    def get_roi_targets(self,
                        sampling_results,
                        rcnn_train_cfg: ConfigDict,
                        concat: bool = True,
                        reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the features extracted by the bbox head.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch. Defaults to True.
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss and targets components.
                The targets are only used for cascade rcnn.
        """

        cls_reg_targets = self.get_targets(
            sampling_results, rcnn_train_cfg, concat=concat)
        cls_target, cls_weight, bbox_target, bbox_weight = cls_reg_targets

        # cls_reg_targets is only for cascade rcnn
        return [cls_target, cls_weight, bbox_target, bbox_weight]
    
    def replay_loss(self,
                        cls_score: Tensor,
                        bbox_pred: Tensor,
                        rois: Tensor,
                        sampling_results,
                        rcnn_train_cfg: ConfigDict,
                        concat: bool = True,
                        reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the features extracted by the bbox head.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch. Defaults to True.
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss and targets components.
                The targets are only used for cascade rcnn.
        """

        # cls_reg_targets = self.get_targets(
        #     sampling_results, rcnn_train_cfg, concat=concat)
        cls_target, cls_weight, bbox_target, bbox_weight = sampling_results
        if self.null_space:
            self.mask = cls_target == 20
        losses = self.loss(
            cls_score,
            bbox_pred,
            rois,
            cls_target,
            cls_weight,
            bbox_target,
            bbox_weight,
            reduction_override=reduction_override)

        # cls_reg_targets is only for cascade rcnn
        keys = losses.keys()
        for key in list(keys):
            losses[f'replay_{key}'] = losses.pop(key)
        return dict(loss_bbox=losses)
    
    
    

@MODELS.register_module()
class Shared2FCBBoxHeadTask(ConvFCBBoxHeadTask):

    def __init__(self, fc_out_channels: int = 1024, *args, **kwargs) -> None:
        super().__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
