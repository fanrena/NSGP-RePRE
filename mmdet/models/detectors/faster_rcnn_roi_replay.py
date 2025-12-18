# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector
from mmdet.structures import SampleList

from mmdet.structures.bbox import BaseBoxes
from torchvision.ops import box_iou
def calculate_iou(box1, box2):
    """
    计算两个矩形框之间的IOU（Intersection over Union）。

    参数:
    box1: 第一个矩形框，格式为(x1, y1, x2, y2)。
    box2: 第二个矩形框，格式为(x1, y1, x2, y2)。

    返回:
    iou: 两个矩形框之间的IOU值。
    """
    # 计算两个矩形框的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算两个矩形框的交集的坐标
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # 计算交集的面积
    if x_right < x_left or y_bottom < y_top:
        intersection_area = 0
    else:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算并返回IOU
    iou = intersection_area / float(area1 + area2 - intersection_area)
    return iou


from mmdet.models.utils import unpack_gt_instances

@MODELS.register_module()
class FasterRCNNRoIReplay(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
        self.rpn_thresh = 0.5
        self.roi_thresh = 0.7

        # for n, p in self.named_parameters():
        #     if "bn" in n:
        #         p.requires_grad_(False)


    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)
        
        rpn_data_samples = None
        
        with torch.no_grad():
            if hasattr(self, "teacher_model"):
                self.teacher_model.eval()
                self.teacher_model.roi_head.bbox_head.task_id = self.teacher_model.roi_head.bbox_head.task_id - 1
                result_ori = self.teacher_model.predict(batch_inputs, copy.deepcopy(batch_data_samples), rescale=False)
                
                rpn_data_samples = copy.deepcopy(batch_data_samples)
                
                less_reliable_data_samples = copy.deepcopy(batch_data_samples)

                for result, batch_data_sample, rpn_data_sample, less_reliable_data_sample in zip(result_ori, batch_data_samples, rpn_data_samples, less_reliable_data_samples):
                    for bbox_index, bbox in enumerate(result.pred_instances):
                        # print(self.rpn_thresh, self.roi_thresh)
                        max_iou = 0
                        for gt_bboxes in batch_data_sample.gt_instances:
                            iou = box_iou(bbox.bboxes, gt_bboxes.bboxes)
                            if max_iou < iou:
                                max_iou = iou
                        if max_iou > 0.7:
                            continue
                            # less_reliable_data_sample.gt_instances = less_reliable_data_sample.gt_instances.cat([bbox])
                            
                        scores = bbox['scores']
                        bbox.__delattr__('scores')
                        if scores > self.rpn_thresh:
                            rpn_data_sample.gt_instances = rpn_data_sample.gt_instances.cat(
                                [rpn_data_sample.gt_instances, bbox])
                            
                        if scores > self.roi_thresh:
                            batch_data_sample.gt_instances = batch_data_sample.gt_instances.cat(
                                [batch_data_sample.gt_instances, bbox])
                self.teacher_model.roi_head.bbox_head.task_id = self.teacher_model.roi_head.bbox_head.task_id + 1
        losses = dict()
        # RPN forward and loss

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            
            rpn_data_samples = rpn_data_samples if rpn_data_samples else copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            
            # less_reliable_losses, less_reliable_rpn_results_list = self.rpn_head.loss_and_predict(
            #     x, less_reliable_data_samples, proposal_cfg=proposal_cfg)
            
            # keys = less_reliable_losses.keys()
            # for key in list(keys):
            #     if isinstance(less_reliable_losses[key], list):
            #         less_reliable_losses[key.replace("rpn", "less_reliable_rpn")] = [i*0.0 for i in less_reliable_losses.pop(key)]
            #     # if 'loss' in key and 'rpn' not in key:
            #     else:
            #         less_reliable_losses[key.replace("rpn", "less_reliable_rpn")] = less_reliable_losses.pop(key)*0.0
            # losses.update(less_reliable_losses)
            
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)
        
        # roi_losses = self.roi_head.loss(x, less_reliable_rpn_results_list, less_reliable_data_samples)
        # # print(roi_losses)
        # less_reliable_loss = {"less_reliable_cls_loss": roi_losses["loss_cls"]*0.0, "less_reliable_bbox_loss": roi_losses["loss_bbox"]*0.0}
        # losses.update(less_reliable_loss)
        
        return losses
    
    
    def null_space_loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)
        
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        return losses


    def get_bbox_stuff(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        # print('  ', 2,1)
        x = self.extract_feat(batch_inputs)
        # print('  ', 2,2)
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        # print('  ', 2,3)
        feats_etc = self.roi_head.get_bbox_stuff(x, rpn_results_list,
                                        batch_data_samples)

        return feats_etc


    def forward(self,
                inputs: torch.Tensor,
                data_samples = None,
                mode: str = 'tensor',
                adapt_psudo = False):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        elif mode == 'nullspace':
            return self.null_space_loss(inputs, data_samples) # For NSGP.
        elif mode == 'roi_replay':
            return self.get_bbox_stuff(inputs, data_samples)  # For RePRE Regional feature computation.
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
            
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
        # if batch_data_samples[0].get('proposals', None) is None:
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        for data_sample in rpn_data_samples:
            data_sample.gt_instances.labels = \
                torch.zeros_like(data_sample.gt_instances.labels)
        loss, rpn_results_list = self.rpn_head.loss_and_predict(
            x, batch_data_samples, proposal_cfg=proposal_cfg)

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
    