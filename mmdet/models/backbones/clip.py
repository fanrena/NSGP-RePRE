import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging

from mmengine.logging import MMLogger, print_log
from mmengine.logging import MMLogger
from mmengine.model import (BaseModule, ModuleList, Sequential, constant_init,
                            normal_init, trunc_normal_init)

from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from mmdet.registry import MODELS

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


def load_clip_to_cpu(cfg):
    backbone_name = cfg.clip_model
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class VPTDeepPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        # hyper param
        self.cfg = cfg
        self.n_ctx = cfg.n_prompt
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.visual.conv1.out_channels # 768
        self.clip_imsize = clip_model.visual.input_resolution
        self.layers = clip_model.visual.transformer.layers
        
        ctx_vectors = torch.empty(self.layers, self.n_ctx, self.ctx_dim, dtype=self.dtype)
        for i in range(self.layers):
            nn.init.normal_(ctx_vectors[i], std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
    def forward(self):
        return self.ctx

class Transformer_VPTD(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        # hyper param
        self.cfg = cfg
        self.n_ctx = cfg.n_prompt
        self.prompt = cfg.prompt
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.visual.conv1.out_channels # 768
        self.clip_imsize = clip_model.visual.input_resolution
        self.layers = clip_model.visual.transformer.layers

        # model
        transformer = clip_model.visual.transformer
        self.resblocks: nn.Sequential = transformer.resblocks
        self.layers = transformer.layers

        self.ctx_learner = VPTDeepPromptLearner(cfg, clip_model)


    def forward(self, x):
        ctx = self.ctx_learner()
        ctx = ctx.unsqueeze(2).expand(-1, -1, x.shape[1], -1)
        
        if self.prompt:
            for i in range(self.layers):
                if self.cfg.trim_last and i == self.layers-1:        
                    continue    
                x = torch.cat([x, ctx[i]], dim=0)
                x = self.resblocks[i](x)
                x = x[:-self.n_ctx, :, :]

        else:
            for i in range(self.layers):    
                if self.cfg.trim_last and i == self.layers-1:        
                    continue            
                x = self.resblocks[i](x)

        return x


class ImageEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.cfg = cfg
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = Transformer_VPTD(cfg, clip_model)
        self.ln_post = clip_model.visual.ln_post
        self.proj = clip_model.visual.proj
        
    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        img_size = x.shape[-2:]; bs = x.shape[0]
        
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # class_embedding is class token.
        
        # interpolate positional_embedding into the same size as the x.
        positional_embedding = self.positional_embedding.to(x.dtype)
        cls_token_pos_embed, img_pos_embed = positional_embedding[:1,:], positional_embedding[1:,:]
        spitial_size, ch = img_pos_embed.shape
        img_pos_embed = img_pos_embed.t().reshape(1, -1, int(math.sqrt(spitial_size)), int(math.sqrt(spitial_size)))
        img_pos_embed = F.interpolate(img_pos_embed, img_size).reshape(ch, -1).t()
        
        positional_embedding = torch.cat([cls_token_pos_embed, img_pos_embed], dim=0)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        
        x = x + positional_embedding

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:, :]
        
        if not self.cfg.trim_last:
            x = self.ln_post(x) # only prune class token which is awsome.
            if self.proj is not None:
                x = x @ self.proj

        x = x.permute(0, 2, 1).reshape(bs, -1, img_size[0], img_size[1])
        
        return x

class TmpConfig:
    pass

@MODELS.register_module()
class PromptedCLIPViT(BaseModule):
    def __init__(self, 
                 clip_model, 
                 prompt = True, 
                 n_prompt=4,
                 trim_last = False,
                 init_cfg=None):
        assert "ViT" in clip_model, "Only CLIP ViT models are supported."
        super().__init__(init_cfg=init_cfg)
        
        cfg = TmpConfig()
        cfg.clip_model = clip_model
        cfg.prompt = prompt
        cfg.n_prompt = n_prompt
        cfg.trim_last = trim_last
        
        self.trim_last = trim_last
        clip_model = load_clip_to_cpu(cfg).float()
        
        # visual
        self.image_encoder = ImageEncoder(cfg, clip_model)
        # visual end
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        
        for name, param in self.named_parameters():
            if "ctx_learner" not in name:
                param.requires_grad_(False)
            elif cfg.prompt:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    def init_weights(self):
        """Initialize the weights."""
        print_log(
                f'init_weights of {self.__class__.__name__} has '
                f'been called, yet nothing was done.',
                logger='current',
                level=logging.WARNING)

    def forward(self, image):
        image = image.to(next(self.image_encoder.parameters()).device)
        image_features = self.image_encoder(image.type(self.dtype))

        if not self.trim_last:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # print(image_features.shape)
        return [image_features]

