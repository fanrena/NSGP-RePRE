# Copyright (c) OpenMMLab. All rights reserved.

import copy
import logging
import os
import os.path as osp
import pickle
import platform
import time
import warnings
from collections import OrderedDict, defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import re
import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.dataset import worker_init_fn as default_worker_init_fn
from mmengine.device import get_device
from mmengine.dist import (broadcast, get_dist_info, get_rank, get_world_size,
                           init_dist, is_distributed, master_only, all_reduce_dict, all_reduce, all_gather, barrier)
from mmengine.evaluator import Evaluator
from mmengine.fileio import FileClient, join_path
from mmengine.hooks import Hook
from mmengine.logging import MessageHub, MMLogger, print_log
from mmengine.model import (MMDistributedDataParallel, convert_sync_batchnorm,
                            is_model_wrapper, revert_sync_batchnorm)
from mmengine.model.efficient_conv_bn_eval import \
    turn_on_efficient_conv_bn_eval
from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler,
                            build_optim_wrapper)
from mmengine.registry import (DATA_SAMPLERS, DATASETS, EVALUATOR, FUNCTIONS,
                               HOOKS, LOG_PROCESSORS, LOOPS, MODEL_WRAPPERS,
                               MODELS, OPTIM_WRAPPERS, PARAM_SCHEDULERS,
                               RUNNERS, VISUALIZERS, DefaultScope)
from mmengine.utils import apply_to, digit_version, get_git_hash, is_seq_of
from mmengine.utils.dl_utils import (TORCH_VERSION, collect_env,
                                     set_multi_processing)
from mmengine.visualization import Visualizer
from mmengine.runner.activation_checkpointing import turn_on_activation_checkpointing
from mmengine.runner.base_loop import BaseLoop
from mmengine.runner.checkpoint import (_load_checkpoint, _load_checkpoint_to_model,
                         find_latest_checkpoint, save_checkpoint,
                         weights_to_cpu)
from mmengine.runner.log_processor import LogProcessor
from mmengine.runner.loops import EpochBasedTrainLoop, IterBasedTrainLoop, TestLoop, ValLoop
from mmengine.runner.priority import Priority, get_priority
from mmengine.runner.utils import _get_batch_size, set_random_seed

from mmengine.runner.runner import Runner

ConfigType = Union[Dict, Config, ConfigDict]
ParamSchedulerType = Union[List[_ParamScheduler], Dict[str,
                                                       List[_ParamScheduler]]]
OptimWrapperType = Union[OptimWrapper, OptimWrapperDict]

from mmdet.models.utils import unpack_gt_instances
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def all_gather_different_shape(t):
    bs = torch.Tensor([t.shape[0]])
    t_shape = list(t.shape)
    world_size = get_world_size()
    local_rank = get_rank()
    res = []
    for i in range(world_size):
        if i != local_rank:
            tmp_bs = torch.Tensor([0]).to(t.device)
        else:
            tmp_bs = bs
        all_reduce(tmp_bs)
        
        t_shape[0] = int(tmp_bs.item())
        if i != local_rank:
            tmp_t = torch.zeros(t_shape).to(t)
        else:
            tmp_t = t
        all_reduce(tmp_t)
        res.append(tmp_t)
    return res
        



@RUNNERS.register_module()
class BRNullSpaceRunner(Runner):
    def __init__(
        self,
        model: Union[nn.Module, Dict],
        work_dir: str,
        train_dataloader: Optional[Union[DataLoader, Dict]] = None,
        val_dataloader: Optional[Union[DataLoader, Dict]] = None,
        test_dataloader: Optional[Union[DataLoader, Dict]] = None,
        train_cfg: Optional[Dict] = None,
        val_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
        auto_scale_lr: Optional[Dict] = None,
        optim_wrapper: Optional[Union[OptimWrapper, Dict]] = None,
        param_scheduler: Optional[Union[_ParamScheduler, Dict, List]] = None,
        val_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
        test_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
        default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
        custom_hooks: Optional[List[Union[Hook, Dict]]] = None,
        data_preprocessor: Union[nn.Module, Dict, None] = None,
        load_from: Optional[str] = None,
        resume: bool = False,
        launcher: str = 'none',
        env_cfg: Dict = dict(dist_cfg=dict(backend='nccl')),
        log_processor: Optional[Dict] = None,
        log_level: str = 'INFO',
        visualizer: Optional[Union[Visualizer, Dict]] = None,
        default_scope: str = 'mmengine',
        randomness: Dict = dict(seed=None),
        experiment_name: Optional[str] = None,
        task_id: Optional[int] = None,
        previous_dir: Optional[str] = None,
        ckpt_keywords: Optional[str] = None,
        cfg: Optional[ConfigType] = None,
    ):
        self._work_dir = osp.abspath(work_dir)
        mmengine.mkdir_or_exist(self._work_dir)

        # recursively copy the `cfg` because `self.cfg` will be modified
        # everywhere.
        if cfg is not None:
            if isinstance(cfg, Config):
                self.cfg = copy.deepcopy(cfg)
            elif isinstance(cfg, dict):
                self.cfg = Config(cfg)
        else:
            self.cfg = Config(dict())

        # lazy initialization
        training_related = [train_dataloader, train_cfg, optim_wrapper]
        if not (all(item is None for item in training_related)
                or all(item is not None for item in training_related)):
            raise ValueError(
                'train_dataloader, train_cfg, and optim_wrapper should be '
                'either all None or not None, but got '
                f'train_dataloader={train_dataloader}, '
                f'train_cfg={train_cfg}, '
                f'optim_wrapper={optim_wrapper}.')
        self._train_dataloader = train_dataloader
        self._train_loop = train_cfg

        self.optim_wrapper: Optional[Union[OptimWrapper, dict]]
        self.optim_wrapper = optim_wrapper

        self.auto_scale_lr = auto_scale_lr

        # If there is no need to adjust learning rate, momentum or other
        # parameters of optimizer, param_scheduler can be None
        if param_scheduler is not None and self.optim_wrapper is None:
            raise ValueError(
                'param_scheduler should be None when optim_wrapper is None, '
                f'but got {param_scheduler}')

        # Parse `param_scheduler` to a list or a dict. If `optim_wrapper` is a
        # `dict` with single optimizer, parsed param_scheduler will be a
        # list of parameter schedulers. If `optim_wrapper` is
        # a `dict` with multiple optimizers, parsed `param_scheduler` will be
        # dict with multiple list of parameter schedulers.
        self._check_scheduler_cfg(param_scheduler)
        self.param_schedulers = param_scheduler

        val_related = [val_dataloader, val_cfg, val_evaluator]
        if not (all(item is None
                    for item in val_related) or all(item is not None
                                                    for item in val_related)):
            raise ValueError(
                'val_dataloader, val_cfg, and val_evaluator should be either '
                'all None or not None, but got '
                f'val_dataloader={val_dataloader}, val_cfg={val_cfg}, '
                f'val_evaluator={val_evaluator}')
        self._val_dataloader = val_dataloader
        self._val_loop = val_cfg
        self._val_evaluator = val_evaluator

        test_related = [test_dataloader, test_cfg, test_evaluator]
        if not (all(item is None for item in test_related)
                or all(item is not None for item in test_related)):
            raise ValueError(
                'test_dataloader, test_cfg, and test_evaluator should be '
                'either all None or not None, but got '
                f'test_dataloader={test_dataloader}, test_cfg={test_cfg}, '
                f'test_evaluator={test_evaluator}')
        self._test_dataloader = test_dataloader
        self._test_loop = test_cfg
        self._test_evaluator = test_evaluator

        self._launcher = launcher
        if self._launcher == 'none':
            self._distributed = False
        else:
            self._distributed = True

        # self._timestamp will be set in the `setup_env` method. Besides,
        # it also will initialize multi-process and (or) distributed
        # environment.
        self.setup_env(env_cfg)
        # self._deterministic and self._seed will be set in the
        # `set_randomness`` method
        self._randomness_cfg = randomness
        self.set_randomness(**randomness)

        if experiment_name is not None:
            self._experiment_name = f'{experiment_name}_{self._timestamp}'
        elif self.cfg.filename is not None:
            filename_no_ext = osp.splitext(osp.basename(self.cfg.filename))[0]
            self._experiment_name = f'{filename_no_ext}_{self._timestamp}'
        else:
            self._experiment_name = self.timestamp
        self._log_dir = osp.join(self.work_dir, self.timestamp)
        mmengine.mkdir_or_exist(self._log_dir)
        # Used to reset registries location. See :meth:`Registry.build` for
        # more details.
        if default_scope is not None:
            default_scope = DefaultScope.get_instance(  # type: ignore
                self._experiment_name,
                scope_name=default_scope)
        self.default_scope = default_scope

        # Build log processor to format message.
        log_processor = dict() if log_processor is None else log_processor
        self.log_processor = self.build_log_processor(log_processor)
        # Since `get_instance` could return any subclass of ManagerMixin. The
        # corresponding attribute needs a type hint.
        self.logger = self.build_logger(log_level=log_level)

        # Collect and log environment information.
        self._log_env(env_cfg)

        # Build `message_hub` for communication among components.
        # `message_hub` can store log scalars (loss, learning rate) and
        # runtime information (iter and epoch). Those components that do not
        # have access to the runner can get iteration or epoch information
        # from `message_hub`. For example, models can get the latest created
        # `message_hub` by
        # `self.message_hub=MessageHub.get_current_instance()` and then get
        # current epoch by `cur_epoch = self.message_hub.get_info('epoch')`.
        # See `MessageHub` and `ManagerMixin` for more details.
        self.message_hub = self.build_message_hub()
        # visualizer used for writing log or visualizing all kinds of data
        self.visualizer = self.build_visualizer(visualizer)
        if self.cfg:
            self.visualizer.add_config(self.cfg)

        # 记录上一个任务的位置
        # # Null Space计算用的feature in
        self.task_id = task_id if task_id is not None else 1
        self.task_split = cfg.get("train_task_split")
        self.previous_dir = previous_dir if self.task_id != 1 else None
        self.ckpt_keywords = ckpt_keywords
        if self.previous_dir == None or not osp.exists(self.previous_dir):
            assert self.task_id == 1, f"Error, previous task dir should be fed into the runner."
        
        if self.previous_dir != None and load_from == None:
            for i in os.listdir(self.previous_dir):
                if self.ckpt_keywords in i:
                    break
            load_from = osp.join(self.previous_dir, i)
            
        self._load_from = load_from
        self._resume = resume
        # flag to mark whether checkpoint has been loaded or resumed
        self._has_loaded = False

        # build a model
        if isinstance(model, dict) and data_preprocessor is not None:
            # Merge the data_preprocessor to model config.
            model.setdefault('data_preprocessor', data_preprocessor)
        self.model = self.build_model(model)

        
        # wrap model
        self.model = self.wrap_model(
            self.cfg.get('model_wrapper_cfg'), self.model)

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._hooks: List[Hook] = []
        # register hooks to `self._hooks`
        self.register_hooks(default_hooks, custom_hooks)
        # log hooks information
        self.logger.info(f'Hooks will be executed in the following '
                         f'order:\n{self.get_hooks_info()}')

        
        self.logger.info(f"Task id start from 1, Current Task id = {task_id}")
        
        self.fea_in_save_path = osp.join(self.work_dir, "covariance.pth")
        if self.previous_dir != None:
            self.fea_in_load_path = osp.join(self.previous_dir, "covariance.pth")
        else:
            self.fea_in_load_path = None
        self.fea_in_load_path = cfg.get("fea_in_load_path") if cfg.get("fea_in_load_path") else self.fea_in_load_path
        
        self.ignore_keys = cfg.get('ignore_keys') + ["roi_head.bbox_head.fc_cls", "roi_head.bbox_head.fc_reg", "teacher"] if cfg.get('ignore_keys') else ["roi_head.bbox_head.fc_cls", "roi_head.bbox_head.fc_reg", "teacher"]
        self.rr_thresh = cfg.get("rr_thresh") if cfg.get("rr_thresh") else [0.5, 0.5]
        self.offset = cfg.get('offset') if cfg.get('offset') else 0.0
        self.reserve_per_class = cfg.get('reserve_per_class') if cfg.get('reserve_per_class') != None else 0
        self.fea_in_hook = {}
        self.is_trained = cfg.get('is_trained') if cfg.get('is_trained') else False
        
        self.fea_in = defaultdict(dict)
        self.fea_in_count = defaultdict(int)
        # 用在BN上的EWC用参数
        self.ewc_reg_terms = {}
        # dump `cfg` to `work_dir`
        self.dump_config()
        
        
    @classmethod
    def from_cfg(cls, cfg: ConfigType) -> 'Runner':
        """Build a runner from config.

        Args:
            cfg (ConfigType): A config used for building runner. Keys of
                ``cfg`` can see :meth:`__init__`.

        Returns:
            Runner: A runner build from ``cfg``.
        """
        cfg = copy.deepcopy(cfg)
        runner = cls(
            model=cfg['model'],
            work_dir=cfg['work_dir'],
            train_dataloader=cfg.get('train_dataloader'),
            val_dataloader=cfg.get('val_dataloader'),
            test_dataloader=cfg.get('test_dataloader'),
            train_cfg=cfg.get('train_cfg'),
            val_cfg=cfg.get('val_cfg'),
            test_cfg=cfg.get('test_cfg'),
            auto_scale_lr=cfg.get('auto_scale_lr'),
            optim_wrapper=cfg.get('optim_wrapper'),
            param_scheduler=cfg.get('param_scheduler'),
            val_evaluator=cfg.get('val_evaluator'),
            test_evaluator=cfg.get('test_evaluator'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=cfg.get('custom_hooks'),
            data_preprocessor=cfg.get('data_preprocessor'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher', 'none'),
            env_cfg=cfg.get('env_cfg', dict(dist_cfg=dict(backend='nccl'))),
            log_processor=cfg.get('log_processor'),
            log_level=cfg.get('log_level', 'INFO'),
            visualizer=cfg.get('visualizer'),
            default_scope=cfg.get('default_scope', 'mmengine'),
            randomness=cfg.get('randomness', dict(seed=None)),
            experiment_name=cfg.get('experiment_name'),
            task_id=cfg.get('task_id'),
            previous_dir=cfg.get('previous_dir'),
            cfg=cfg,
        )

        return runner
        
    def train(self) -> nn.Module:
        """Launch training.

        Returns:
            nn.Module: The model after training.
        """
        if is_model_wrapper(self.model):
            ori_model = self.model.module
        else:
            ori_model = self.model
                
        ori_model.rpn_thresh = self.rr_thresh[0]
        ori_model.roi_thresh = self.rr_thresh[1]
                
                
        assert hasattr(ori_model, 'train_step'), (
            'If you want to train your model, please make sure your model '
            'has implemented `train_step`.')

        if self._val_loop is not None:
            assert hasattr(ori_model, 'val_step'), (
                'If you want to validate your model, please make sure your '
                'model has implemented `val_step`.')

        if self._train_loop is None:
            raise RuntimeError(
                '`self._train_loop` should not be None when calling train '
                'method. Please provide `train_dataloader`, `train_cfg`, '
                '`optimizer` and `param_scheduler` arguments when '
                'initializing runner.')

        self._train_loop = self.build_train_loop(
            self._train_loop)  # type: ignore

        # `build_optimizer` should be called before `build_param_scheduler`
        #  because the latter depends on the former
        self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)
        # assign names to the params
        recorder = {}
        for i, group in enumerate(self.optim_wrapper.optimizer.param_groups):
            for p in group["params"]:
                recorder[id(p)] = i
            self.optim_wrapper.optimizer.param_groups[i]["params"] = []
            self.optim_wrapper.optimizer.param_groups[i]["names"] = []
            
        for name, param in ori_model.named_parameters():
            if param.requires_grad:
                i = recorder[id(param)]
                self.optim_wrapper.optimizer.param_groups[i]["params"] += [param]
                self.optim_wrapper.optimizer.param_groups[i]["names"] += [name]
        # Automatically scaling lr by linear scaling rule
        self.scale_lr(self.optim_wrapper, self.auto_scale_lr)

        if self.param_schedulers is not None:
            self.param_schedulers = self.build_param_scheduler(  # type: ignore
                self.param_schedulers)  # type: ignore

        if self._val_loop is not None:
            self._val_loop = self.build_val_loop(
                self._val_loop)  # type: ignore
        # TODO: add a contextmanager to avoid calling `before_run` many times
        self.call_hook('before_run')
        
        # initialize the model weights
        self._init_model_weights()
        
        bg_weight = ori_model.roi_head.bbox_head.fc_cls[-1].weight.data.detach().clone()
        bg_bias = ori_model.roi_head.bbox_head.fc_cls[-1].bias.data.detach().clone()

        # try to enable activation_checkpointing feature
        modules = self.cfg.get('activation_checkpointing', None)
        if modules is not None:
            self.logger.info(f'Enabling the "activation_checkpointing" feature'
                             f' for sub-modules: {modules}')
            turn_on_activation_checkpointing(ori_model, modules)

        # try to enable efficient_conv_bn_eval feature
        modules = self.cfg.get('efficient_conv_bn_eval', None)
        if modules is not None:
            self.logger.info(f'Enabling the "efficient_conv_bn_eval" feature'
                             f' for sub-modules: {modules}')
            turn_on_efficient_conv_bn_eval(ori_model, modules)

        # make sure checkpoint-related hooks are triggered after `before_run`
        if self.task_id != 1 and "joint" not in self.work_dir:
            ori_model.teacher_model = copy.deepcopy(ori_model)
            ori_model.roi_head.teacher_model = ori_model.teacher_model.roi_head
        
        self.load_or_resume()
        
        if self.task_id != 1 and not self.is_trained:
            if "joint" not in self.work_dir:
                if hasattr(ori_model, 'teacher_model'):
                    del ori_model.teacher_model
                ori_model.teacher_model = copy.deepcopy(ori_model)
                ori_model.roi_head.teacher_model = ori_model.teacher_model.roi_head
            for name, param in ori_model.named_parameters():
                if "teacher" in name:
                    param.requires_grad_(False)
            
            assert self._resume == False, "Resume from trained model are not allowed! Because teacher model is initialized with ckpt in self.load_from. Resuming from ckpt will degrade your teacher performance in old Tasks."
            
            self.update_optim_transforms(self.train_dataloader)
            
            self.load_importance()
            if is_model_wrapper(self.model):
                model = self.model.module
            else:
                model = self.model
            if "joint" not in self.work_dir:
                model.loss = EWCHook(module=model, reg_params=self.reg_params, ewc_reg_terms=self.ewc_reg_terms)

        # Initiate inner count of `optim_wrapper`.
        self.optim_wrapper.initialize_count_status(
            self.model,
            self._train_loop.iter,  # type: ignore
            self._train_loop.max_iters)  # type: ignore

        # Maybe compile the model according to options in self.cfg.compile
        # This must be called **AFTER** model has been wrapped.
        self._maybe_compile('train_step')
        # self.val_loop.run()
        torch.cuda.empty_cache()      
        if not self.is_trained:  
            model = self.train_loop.run()  # type: ignore
        self.call_hook('after_run')
        self._has_loaded = False
        self.calculate_save_importance(self.train_dataloader)
        self.cal_fea_in(self.train_dataloader)
        self.cal_rois(self.train_dataloader)
        return model    
    
    
    def test(self) -> dict:
        """Launch test.

        Returns:
            dict: A dict of metrics on testing set.
        """
        if self._test_loop is None:
            raise RuntimeError(
                '`self._test_loop` should not be None when calling test '
                'method. Please provide `test_dataloader`, `test_cfg` and '
                '`test_evaluator` arguments when initializing runner.')

        self._test_loop = self.build_test_loop(self._test_loop)  # type: ignore

        self.call_hook('before_run')

        # make sure checkpoint-related hooks are triggered after `before_run`
        self.load_or_resume()
        # self.calculate_save_importance(self.train_dataloader)
        metrics = self.test_loop.run()  # type: ignore
        self.call_hook('after_run')
        self.cal_fea_in(self.train_dataloader)
        self.calculate_save_importance(self.train_dataloader)
        return metrics
   
    @torch.no_grad()
    def update_optim_transforms(self, train_loader):
        
        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model
        
        def check_if_ignore(n):
            ignore = False
            for ignore_key in self.ignore_keys:
                ignore = ignore or bool(re.match(ignore_key, n))

            if not ignore:
                self.logger.info(f"** {n}")
            return ignore
            
        self.logger.info(f"Load Covariance from {self.fea_in_load_path}")
        self.fea_in = torch.load(self.fea_in_load_path, map_location=next(model.parameters()).device)
        self.fea_in = {k: v for k, v in self.fea_in.items() if not check_if_ignore(k)}
        
        self.optim_wrapper.optimizer.get_eigens(self.fea_in)

        self.optim_wrapper.optimizer.get_transforms(offset=self.offset)
        
        del self.fea_in
        
        print("Done getting eigens and transforms.")
        torch.cuda.empty_cache()        

    
    @torch.no_grad()
    def cal_fea_in(self, train_loader):
        self.logger.info("Doing cal_fea_in......")
        # self._has_loaded = None
        self.fea_in = defaultdict(dict)
        
        for i in os.listdir(self.work_dir):
            if self.ckpt_keywords in i:
                break
        load_from = osp.join(self.work_dir, i)
        self._load_from = load_from
        self.load_or_resume()
        
        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model
        
        def check_if_ignore(n):
            ignore = False
            for ignore_key in self.ignore_keys:
                ignore = ignore or bool(re.match(ignore_key, n))

            if not ignore:
                self.logger.info(f"** {n}")
            return ignore
        
        modules = [m for n, m in model.named_modules() if hasattr(
            m, 'weight') and not check_if_ignore(n)]
        
        handles = []
        for m in modules:
            handles.append(m.register_forward_hook(hook=self.compute_cov))
            
        model.eval()
        for i, data_batch in tqdm(enumerate(train_loader), desc=str(get_rank()), disable=False):
            # with self.optim_wrapper.optim_context(self):
            data = model.data_preprocessor(data_batch, True)
            inputs = data["inputs"]
            model(inputs, data["data_samples"], mode="nullspace") # get labels for rpn and gt
            torch.cuda.empty_cache()
        # TODO: sync fea_in, 现在就用单卡存好的内容。
        barrier()
        all_reduce_dict(self.fea_in)
            
        barrier()
        if self.task_id != 1:
            self.logger.info(f"During cal_fea_in, trying to load Covariance from {self.fea_in_load_path}")
            old_fea_in = torch.load(self.fea_in_load_path, map_location=next(model.parameters()).device)
            self.fea_in = {k: v + old_fea_in[k].to(v.device) for k, v in self.fea_in.items() if not check_if_ignore(k)}
        
        
        self.logger.info(f"Trying to save Covariance to {self.fea_in_save_path}")
        torch.save(self.fea_in, self.fea_in_save_path)
        self.logger.info(f"Covariance saved to {self.fea_in_save_path}")
        # get_distinguisher: two parts: optimizer + part
        del self.fea_in
        for h in handles:
            h.remove()
        torch.cuda.empty_cache()        
        
        
    @torch.no_grad()
    def cal_rois(self, train_loader):
        self.logger.info("Doing cal_rois......")
        # self._has_loaded = None
        # self.fea_in = defaultdict(dict)
        
        for i in os.listdir(self.work_dir):
            if self.ckpt_keywords in i:
                break
            load_from = osp.join(self.work_dir, i)
        self._load_from = load_from
        self.load_or_resume()
        # print(get_rank(), 'load done')
        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model
        # print(get_rank(), "eval")
        model.eval()
        # print(get_rank(), len(train_loader))
        bbox_featss = []; roiss = []; cls_targets = []; cls_weights = []; bbox_targets = []; bbox_weights = []
        for i, data_batch in tqdm(enumerate(train_loader), desc=f"{get_rank()}", disable=False):
            # if i > 50:
            #         break
            # with self.optim_wrapper.optim_context(self):
            # print(get_rank(), 1, i)
            data = model.data_preprocessor(data_batch, True)
            # print(get_rank(), 11, i)
            inputs = data["inputs"]
            # print(get_rank(), 2, i)
            bbox_feats, cls_target, cls_weight, bbox_target, bbox_weight, rois = model(inputs, data["data_samples"], mode="roi_replay") # get labels for rpn and gt
            # print(get_rank(), 3, i)
            bbox_featss.append(bbox_feats)
            cls_targets.append(cls_target)
            cls_weights.append(cls_weight)
            bbox_targets.append(bbox_target)
            bbox_weights.append(bbox_weight)
            roiss.append(rois)
            
        gathered_bbox_featss = torch.cat(all_gather_different_shape(torch.cat(bbox_featss, dim=0)))
        gathered_cls_targets = torch.cat(all_gather_different_shape(torch.cat(cls_targets, dim=0)))
        gathered_cls_weights = torch.cat(all_gather_different_shape(torch.cat(cls_weights, dim=0)))
        gathered_bbox_targets = torch.cat(all_gather_different_shape(torch.cat(bbox_targets, dim=0)))
        gathered_bbox_weights = torch.cat(all_gather_different_shape(torch.cat(bbox_weights, dim=0)))
        gathered_roiss = torch.cat(all_gather_different_shape(torch.cat(roiss, dim=0)))
        
        a = [gathered_bbox_featss, gathered_cls_targets, gathered_cls_weights, gathered_bbox_targets, gathered_bbox_weights, gathered_roiss]
        
        
        if self.reserve_per_class != 0:
            self.logger.info(f"Saving {self.reserve_per_class} / class")
            cls_targets = a[1]
            res = []
            masks = {}
            for tns in a:
                tmp = []
                for cls_idx in range(20):
                    cls_mask = cls_targets == cls_idx
                    if cls_idx not in masks.keys():
                        masks[cls_idx] = torch.randperm(cls_mask.sum())[:self.reserve_per_class]
                    mask = masks[cls_idx]
                    cls_bbox_feats = tns[cls_mask][mask]
                    tmp.append(cls_bbox_feats)
                tmp = torch.cat(tmp, dim=0)
                res.append(tmp)
            
            gathered_bbox_featss, gathered_cls_targets, gathered_cls_weights, gathered_bbox_targets, gathered_bbox_weights, gathered_roiss = res
        
        if self.task_id != 1:
            roi_load_file = osp.join(self.previous_dir, "rois_etc.pth")
            self.logger.info(f"During cal_rois, trying to load RoI Features from {roi_load_file}")
            tmp_device = next(model.parameters()).device
            bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss = torch.load(roi_load_file, map_location=tmp_device)
        
            bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss = [bbox_featss, gathered_bbox_featss], [cls_targets, gathered_cls_targets], [cls_weights, gathered_cls_weights], [bbox_targets, gathered_bbox_targets], [bbox_weights, gathered_bbox_weights], [roiss, gathered_roiss]
            bbox_featss = torch.cat(bbox_featss, dim=0)
            cls_targets = torch.cat(cls_targets, dim=0)
            cls_weights = torch.cat(cls_weights, dim=0)
            bbox_targets = torch.cat(bbox_targets, dim=0)
            bbox_weights = torch.cat(bbox_weights, dim=0)
            roiss = torch.cat(roiss, dim=0)
        else:
            bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss = gathered_bbox_featss, gathered_cls_targets, gathered_cls_weights, gathered_bbox_targets, gathered_bbox_weights, gathered_roiss
        
        results = [bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss]
        
        roi_save_file = osp.join(self.work_dir, "rois_etc.pth")
            
        self.logger.info(f"Trying to save rois etc to {roi_save_file}")
        torch.save(results, roi_save_file)
        self.logger.info(f"RoIs etc saved to {roi_save_file}")
        # get_distinguisher: two parts: optimizer + part
        torch.cuda.empty_cache()        

    def compute_cov(self, module, fea_in, fea_out):
        '''
        Hook, 每经过一层，就计算一次covariance。
        '''
        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model
        
        for n, mod in model.named_modules():
            if module == mod:
                name = n + ".weight"
                break
        
        fea_in_0 = fea_in[0]
        
        if isinstance(module, nn.Linear):
            self.update_cov(torch.mean(fea_in_0, 0, True), name)

        elif isinstance(module, nn.Conv2d):
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding

            fea_in_ = F.unfold(torch.mean(fea_in_0, 0, True), kernel_size=kernel_size, padding=padding, stride=stride)
            # 提取所有3x3的features, 并concat到一起。mean消除了batch的影响。
            # padding和stride保证在完成卷积后大小依旧是（32x32），共提取1024个region，concat后的feature大小是（9x64）=576.
            fea_in_ = fea_in_.permute(0, 2, 1)
            fea_in_ = fea_in_.reshape(-1, fea_in_.shape[-1])
            self.update_cov(fea_in_, name)

        torch.cuda.empty_cache()
        return None


    def update_cov(self, fea_in, k):
        cov = torch.mm(fea_in.transpose(0, 1), fea_in)
        if len(self.fea_in[k]) == 0:
            self.fea_in[k] = cov
        else:
            self.fea_in[k] = self.fea_in[k] + cov
         
    def calculate_save_importance(self, dataloader):
        self.logger.info("cal importance")

        self.register_params()
        if len(self.ewc_reg_terms) == 0:
                    self.ewc_reg_terms = {'importance': defaultdict(
                        list), 'task_param': defaultdict(list)}
        importance = {}
        
        for n, p in self.reg_params.items():
            importance[n] = p.clone().detach().fill_(0)
        
        self.model.eval()
        print(len(self.train_dataloader))
        for idx, data_batch in tqdm(enumerate(self.train_dataloader), disable=False):
            # with self.optim_wrapper.optim_context(self):
            data = self.model.module.data_preprocessor(data_batch, True)
            # print([i.gt_instances.labels.unique() for i in data["data_samples"]])
            losses = self.model._run_forward(data, mode='loss')  # type: ignore
            parsed_losses, log_vars = self.model.module.parse_losses(losses)
            loss = self.optim_wrapper.scale_loss(parsed_losses)
            self.optim_wrapper.backward(loss)
            
            for n, p in importance.items():
                if self.reg_params[n].grad is not None:
                    p += ((self.reg_params[n].grad ** 2)
                          * len(data_batch) / len(dataloader))

            self.optim_wrapper.zero_grad()
        
        for n, p in self.reg_params.items():
            self.ewc_reg_terms['importance'][n].append(importance[n].unsqueeze(0))
            self.ewc_reg_terms['task_param'][n].append(p.unsqueeze(0).clone().detach())
            
        torch.save(self.ewc_reg_terms, osp.join(self.work_dir, "ewc_reg_terms_ewc.pth"))
        self.logger.info("cal importance done")
        
    def load_importance(self):
        self.register_params()
        self.ewc_reg_terms = torch.load(osp.join(self.previous_dir, "ewc_reg_terms_ewc.pth"), map_location=list(self.reg_params.values())[0].device)
    
    def register_params(self):
        self.reg_params = {}
        ignore_names = ["teacher_model"]
        must_names = ["bn"]
        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model
        
        for n, p in model.named_parameters():
            ignore = True
            must = False if len(must_names) != 0 else True
            for ignore_name in ignore_names:
                if ignore_name in n:
                    ignore = False

            for must_name in must_names:
                if must_name in n:
                    must = True
            if ignore and must:
                self.reg_params[n] = p
        
class EWCHook:
    def __init__(self, module, reg_params, ewc_reg_terms):
        self.module = module
        self.reg_params = reg_params
        self.ewc_reg_terms = ewc_reg_terms
        self.ori_loss = module.loss
    
    def __call__(self, *args, **kwargs):
        result = self.ori_loss(*args, **kwargs)
        reg_loss = {}
        reg_loss["ewc_loss"] = 0
        for n, p in self.reg_params.items():
            if not p.requires_grad:
                continue
            importance = torch.cat(
                self.ewc_reg_terms['importance'][n], dim=0)
            old_params = torch.cat(
                self.ewc_reg_terms['task_param'][n], dim=0)
            new_params = p.unsqueeze(0).expand(old_params.shape)
            tmp = (importance * (new_params - old_params) ** 2).sum()
            reg_loss["ewc_loss"] = reg_loss['ewc_loss'] + 1000 * tmp
            # print(n, (new_params == old_params).all(), tmp)
        if reg_loss["ewc_loss"] != 0:
            result.update(reg_loss)
        # print(result) 
        return result
    