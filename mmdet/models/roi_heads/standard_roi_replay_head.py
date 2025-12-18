# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList,OptConfigType, OptMultiConfig
from ..task_modules.samplers import SamplingResult
from ..utils import empty_instances, unpack_gt_instances
from .base_roi_head import BaseRoIHead
from .standard_roi_head import StandardRoIHead

import os.path as osp
import os
from sklearn.cluster import KMeans, DBSCAN
from collections import Counter
import torch.nn.functional as F
import torch.distributions as tdist
import numpy as np
import scipy
import scipy.ndimage
# from torch_cluster import dbscan

from mmdet.evaluation.functional.bbox_overlaps import bbox_overlaps
from mmdet.structures.bbox import BaseBoxes
import copy
@MODELS.register_module()
class StandardRoIReplayHead(StandardRoIHead):
    def __init__(self,
                 bbox_roi_extractor: OptMultiConfig = None,
                 bbox_head: OptMultiConfig = None,
                 mask_roi_extractor: OptMultiConfig = None,
                 mask_head: OptMultiConfig = None,
                 shared_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 previous_path = None) -> None:
        super().__init__(bbox_roi_extractor, bbox_head, mask_roi_extractor, mask_head, shared_head, train_cfg, test_cfg, init_cfg)
        self.replay = False
        self.counter = [0]*80
        
        self.pos_neg_counter = {0.2: [0,0], 0.3: [0,0], 0.4: [0,0], 0.5: [0,0], 0.6: [0,0], 0.7: [0,0], 0.8: [0,0], 0.9: [0,0], 1.0: [0,0], 2.0: [0,0]}
    
        if previous_path != None and osp.exists(previous_path):
            self.replay = True
            print("Load previous stuff from ", osp.join(previous_path, "rois_etc.pth"))
            self.bbox_featss, self.cls_targets, self.cls_weights, self.bbox_targets, self.bbox_weights, self.roiss = torch.load(osp.join(previous_path, "rois_etc.pth"))
            
    def loss(self, x: Tuple[Tensor], rpn_results_list, batch_data_samples, replay=True) -> dict:
        losses = super().loss(x, rpn_results_list, batch_data_samples)
        device = next(self.parameters()).device
        if self.replay and replay:
            # do sampling
            mask = torch.randperm(self.bbox_featss.shape[0])[:64].to(self.bbox_featss.device)
            bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss = self.bbox_featss[mask], self.cls_targets[mask], self.cls_weights[mask], self.bbox_targets[mask], self.bbox_weights[mask], self.roiss[mask]
            
            bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss = bbox_featss.to(device), cls_targets.to(device), cls_weights.to(device), bbox_targets.to(device), bbox_weights.to(device), roiss.to(device)
            
            sampling_results = [cls_targets, cls_weights, bbox_targets, bbox_weights]
            replay_results = self.replay_loss(bbox_featss, sampling_results, roiss)
            # print("before update", losses)
            losses.update(replay_results['replay_loss'])
            # print("replay loss", replay_results['replay_loss'])
            # print("after update", losses)
        return losses
        
    def replay_loss(self, bbox_feats: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  rois) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        
        # teacher model
        teacher_cls_score, teacher_bbox_pred = self.teacher_model.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        losses = dict()
        
        losses["replay_loss_cls"] = F.mse_loss(cls_score, teacher_cls_score)

        bbox_results.update(replay_loss=losses)
        return bbox_results

    def get_bbox_stuff(self, x, rpn_results_list, batch_data_samples, extract_gt=False):
        
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals
        if not extract_gt:
            num_imgs = len(batch_data_samples)
            sampling_results = []
            for i in range(num_imgs):
                # rename rpn_results.bboxes to rpn_results.priors
                rpn_results = rpn_results_list[i]
                rpn_results.priors = rpn_results.pop('bboxes')

                assign_result = self.bbox_assigner.assign(
                    rpn_results, batch_gt_instances[i],
                    batch_gt_instances_ignore[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    rpn_results,
                    batch_gt_instances[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
            
            rois = bbox2roi([res.priors for res in sampling_results])
            
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
                
            bbox_feats = self.bbox_head.get_mid_features(bbox_feats)
            
            # print(torch.sort(cls))
            
            cls_target, cls_weight, bbox_target, bbox_weight = self.bbox_head.get_roi_targets(sampling_results=sampling_results,
                rcnn_train_cfg=self.train_cfg)
        
        if extract_gt:
            bboxes = [
                data_sample.gt_instances for data_sample in batch_data_samples
            ]
            bboxes = [img.bboxes for img in bboxes]
            
            rois = bbox2roi(bboxes)
        
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            
            # print(torch.sort(cls))
            
            cls_target, cls_weight, bbox_target, bbox_weight = self.bbox_head.get_roi_targets(sampling_results=sampling_results,
                rcnn_train_cfg=self.train_cfg)
        
        mask = cls_target != 20 if VOC else 80 # Default 20 for VOC, this will raise an error as the variant 'VOC' doesn't exists.
        
        target_count = 5
        
        # 计算当前True的数量
        current_count = torch.sum(mask).item()

        # 确定需要增加或减少的True的数量
        delta = target_count - current_count
        # print(delta)

        if delta > 0:
            # 需要增加True的数量
            num_to_add = delta
            # 获取所有False的索引
            false_indices = torch.where(mask == False)[0]
            # 如果False的数量小于需要增加的数量，就全部设置为True
            if len(false_indices) < num_to_add:
                mask[:] = True
            else:
                # 随机选择一些False的位置并设置为True
                indices_to_add = torch.randperm(len(false_indices))[:num_to_add]
                mask[false_indices[indices_to_add]] = True
        elif delta < 0:
            # 需要减少True的数量
            num_to_remove = -delta
            # 获取所有True的索引
            true_indices = torch.where(mask == True)[0]
            # 随机选择一些True的位置并设置为False
            indices_to_remove = torch.randperm(len(true_indices))[:num_to_remove]
            mask[true_indices[indices_to_remove]] = False
            
        for c in cls_target[mask]:
            # print(c)
            self.counter[c] += 1
        
        return bbox_feats[mask], cls_target[mask], cls_weight[mask], bbox_target[mask], bbox_weight[mask], rois[mask]
    

        
@MODELS.register_module()
class StandardPrototypeReplayHead(StandardRoIReplayHead):
    def __init__(self,
                 bbox_roi_extractor: OptMultiConfig = None,
                 bbox_head: OptMultiConfig = None,
                 mask_roi_extractor: OptMultiConfig = None,
                 mask_head: OptMultiConfig = None,
                 shared_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 previous_path = None,
                 task_id = 1,
                 task_split = [0,10,20]) -> None:
        super().__init__(bbox_roi_extractor, bbox_head, mask_roi_extractor, mask_head, shared_head, train_cfg, test_cfg, init_cfg)
        self.replay = False
        self.task_split = task_split
        self.task_id = task_id
        device = next(self.parameters()).device
        if previous_path != None and osp.exists(previous_path):
            assert task_id != 1
            self.replay = True
            print("Load previous stuff from ", osp.join(previous_path, "rois_etc.pth"))
            self.bbox_featss, self.cls_targets, self.cls_weights, self.bbox_targets, self.bbox_weights, self.roiss = torch.load(osp.join(previous_path, "rois_etc.pth"), map_location=device)
            previous_cls = range(task_split[0], task_split[task_id-1])
            tmp = []
            for i in previous_cls:
                cls_mask = self.cls_targets == i
                cls_bbox_feats = torch.mean(self.bbox_featss[cls_mask], dim=0, keepdim=True)
                tmp.append(cls_bbox_feats)
            self.bbox_featss = torch.cat(tmp, dim=0)
            
    def loss(self, x: Tuple[Tensor], rpn_results_list, batch_data_samples) -> dict:
        losses = super().loss(x, rpn_results_list, batch_data_samples, replay=False)
        device = next(self.parameters()).device
        if self.replay:
            # do sampling
            # mask = torch.randperm(self.bbox_featss.shape[0])[:64].to(self.bbox_featss.device)
            bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss = self.bbox_featss, self.cls_targets, self.cls_weights, self.bbox_targets, self.bbox_weights, self.roiss
            bbox_featss = self.bbox_featss
            bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss = bbox_featss.to(device), cls_targets.to(device), cls_weights.to(device), bbox_targets.to(device), bbox_weights.to(device), roiss.to(device)
            
            sampling_results = [cls_targets, cls_weights, bbox_targets, bbox_weights]
            replay_results = self.replay_loss(bbox_featss, None, None)
            # print("before update", losses)
            losses.update(replay_results['replay_loss'])
            # print("replay loss", replay_results['replay_loss'])
            # print("after update", losses)
        return losses
    
    
    def replay_loss(self, bbox_feats: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  rois) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        # bbox_feats = torch.rot90(bbox_feats, torch.randint(0, 4, (1,)).item(), dims=(-2,-1))
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        
        # teacher model
        # teacher_cls_score, teacher_bbox_pred = self.teacher_model.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        losses = dict()
        
        pre_idx = self.task_split[self.task_id]
        cls_score = torch.cat([cls_score[:, :pre_idx], cls_score[:,-1:]], dim=-1)

        # print(cls_score.shape)
        
        losses["replay_loss_cls"] = F.cross_entropy(cls_score.softmax(dim=-1), torch.Tensor([i for i in range(cls_score.shape[0])]).long().to(cls_score.device))
    

        bbox_results.update(replay_loss=losses)
        return bbox_results



def adaptive_threshold(svals: torch.Tensor, offset: float = 0):
    points: np.ndarray = svals.cpu().numpy()
    assert points.ndim == 1
    if len(points) >= 128:
        fil_points = scipy.ndimage.gaussian_filter1d(points, sigma=10)
        _delta = 1
        diff_o1 = fil_points[:-_delta] - fil_points[_delta:]
        diff_o2 = diff_o1[:-1] - diff_o1[1:]
        _drop_ratio = 0.03
        drop_num = int(len(points) * _drop_ratio / 2)
        assert len(points) - drop_num >= 10
        valid_o2 = diff_o2[drop_num:-drop_num]
        thres_val = points[np.argmax(valid_o2) + int((len(points) - len(valid_o2)) / 2)]
    else:
        diff_o1 = points[:-1] - points[1:]
        diff_o2 = diff_o1[:-1] - diff_o1[1:]
        thres_val = points[np.argmax(diff_o2) + int((len(points) - len(diff_o2)) / 2)]
    i_thres = np.arange(len(points))[points >= thres_val].max()
    # assert 0 <= offset < 1, offset
    # print(offset)
    if -1 <= offset <= 1:
        i_thres = min(i_thres + int(offset * (i_thres)), len(points) - 1)
        i_thres = max(0, i_thres)
    else:
        i_thres = max(min(i_thres + int(offset), len(points) - 1), 0)

    zero_idx = np.zeros(len(points), dtype=np.int64)
    zero_idx[i_thres:] = 1
    zero_idx = torch.as_tensor(torch.from_numpy(zero_idx), dtype=torch.bool, device=svals.device)
    return zero_idx

def gram_schmidt(A,j):
    # 获取矩阵A的列数
    m, n = A.size()
    # 初始化正交矩阵Q
    Q = torch.zeros(m, n)
    Q = Q.to(A.device)
    Q[:, :j] = A[:, :j]
    # 提取第j列向量
    v = A[:, j]
    
    # 减去v在之前所有正交向量上的投影
    for i in range(j):
        u = Q[:, i]
        # 计算投影
        proj = torch.sum(u * v) / torch.sum(u * u) * u
        # 减去投影
        v = v - proj
    
    # 单位化
    Q[:, j] = v / torch.norm(v)
    
    return Q

def get_norm(a):
    return (a / a.norm(dim=-1, keepdim=True)).unsqueeze(1)

def get_cos_sim(ii, projected_U):
    return (ii/ii.norm(dim=-1, keepdim=True))@(projected_U / projected_U.norm(dim=-1, keepdim=True)).t()



def get_work_dir(previous_path):
    # 用这个根据给出的之前的path去推断当前的work_dir
    if "coco" in previous_path:
        return "./"
    splited_path = previous_path.split("_")
    task_id = int(splited_path[-1])
    splited_path[-1] = str(task_id + 1)
    return "_".join(splited_path)




@MODELS.register_module()
class StandardMultiPrototypeReplayHead(StandardRoIReplayHead):
    def __init__(self,
                 bbox_roi_extractor: OptMultiConfig = None,
                 bbox_head: OptMultiConfig = None,
                 mask_roi_extractor: OptMultiConfig = None,
                 mask_head: OptMultiConfig = None,
                 shared_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 previous_path = None,
                 task_id = 1,
                 task_split = [0,10,20],
                 max_prototype=10,
                 work_dir = None) -> None:
        super().__init__(bbox_roi_extractor, bbox_head, mask_roi_extractor, mask_head, shared_head, train_cfg, test_cfg, init_cfg)
        self.replay = False
        self.task_split = task_split
        self.task_id = task_id
        self.max_proto = max_prototype
        device = next(self.parameters()).device
        with torch.no_grad():
            if previous_path != None and osp.exists(previous_path):
                # DBSCAN baseline
                assert task_id != 1
                self.replay = True
                print("Load previous stuff from ", osp.join(previous_path, "rois_etc.pth"))
                self.bbox_featss, self.cls_targets, self.cls_weights, self.bbox_targets, self.bbox_weights, self.roiss = torch.load(osp.join(previous_path, "rois_etc.pth"), map_location=device)
                previous_cls = range(task_split[0], task_split[task_id-1])
                tmp = []
                tmp_label = []
                if osp.exists(osp.join(previous_path, "mask.pth")):
                    save_idx = torch.load(osp.join(previous_path, "mask.pth"), map_location='cpu')
                else:
                    save_idx = []
                for i in previous_cls:
                    cls_mask = self.cls_targets == i
                    cls_bbox_feats = torch.mean(self.bbox_featss[cls_mask], dim=0, keepdim=True)
                    tmp.append(cls_bbox_feats)
                    tmp_label.append(i)
                    
                    bbox_feats_norm = self.bbox_featss[cls_mask].reshape(-1, 7*7*256) / self.bbox_featss[cls_mask].reshape(-1, 7*7*256).norm(dim=-1, keepdim=True)
                    bbox_feats_sim = bbox_feats_norm @ bbox_feats_norm.t()
                    
                    sim_mask = bbox_feats_sim >= 0.6 # n * n
                    sim_sum, idx = sim_mask.long().sum(dim=-1).sort(dim=-1, descending=True)
                    sim_sum_threash = sim_sum[-sim_sum.shape[0]//3]
                    potential_center = (bbox_feats_sim >= 0.6).long().sum(dim=-1) <= sim_sum_threash
                    
                    if i < len(save_idx):
                        tmp_mask = save_idx[i]
                    else:
                        tmp_mask = []
                        
                    for proto_count in range(self.max_proto-1):
                        for id_ in idx:
                            if proto_count < len(tmp_mask):
                                m = tmp_mask[proto_count].to(device)
  
                            else:
                                if potential_center[id_]:
                                    continue
                                
                                m = sim_mask[id_]
                                tmp_mask.append(m)
                   
                            potential_center = torch.logical_or(potential_center, m)
                            prototype = torch.mean(self.bbox_featss[cls_mask][m], dim=0, keepdim=True)
                            tmp.append(prototype)
                            tmp_label.append(i)
                            break
                    if i >= len(save_idx):
                        save_idx.append(tmp_mask)
                self.bbox_featss = torch.cat(tmp, dim=0)
                self.tmp_label = torch.Tensor(tmp_label, device = self.bbox_featss.device).long()
                work_dir = get_work_dir(previous_path)
                torch.save(save_idx, osp.join(work_dir, "mask.pth"))
        
    def loss(self, x: Tuple[Tensor], rpn_results_list, batch_data_samples) -> dict:
        losses = super().loss(x, rpn_results_list, batch_data_samples, replay=False)
        device = next(self.parameters()).device
        if self.replay:
            bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss = self.bbox_featss, self.cls_targets, self.cls_weights, self.bbox_targets, self.bbox_weights, self.roiss
            bbox_featss = self.bbox_featss
            bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss = bbox_featss.to(device), cls_targets.to(device), cls_weights.to(device), bbox_targets.to(device), bbox_weights.to(device), roiss.to(device)
            
            sampling_results = [cls_targets, cls_weights, bbox_targets, bbox_weights]
            replay_results = self.replay_loss(bbox_featss, None, None)
            losses.update(replay_results['replay_loss'])

        return losses
    
    def replay_loss(self, bbox_feats: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  rois) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """

        bbox_feats = bbox_feats
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        losses = dict()
 
        pre_idx = self.task_split[self.task_id]
        cls_score = torch.cat([cls_score[:, :pre_idx], cls_score[:,-1:]], dim=-1)
        losses["replay_loss_cls"] = F.cross_entropy(cls_score.softmax(dim=-1), self.tmp_label.to(cls_score.device))
        bbox_results.update(replay_loss=losses)
        return bbox_results
    
    