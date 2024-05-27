# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only MiniCPM-V-2 model compatible with HuggingFace weights."""
import math
from functools import partial
from typing import Iterable, List, Optional, Tuple

import numpy as np

try:
    import timm
except ImportError:
    raise ImportError('Please install timm==0.9.10') from ImportError
import torch
import torch.nn.functional as F
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torch import nn
from torch.nn.init import trunc_normal_
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.minicpm import MiniCPMForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput

_KEYS_TO_MODIFY_MAPPING = {
    "language_model.lm_head": "lm_head",
    "language_model.model": "language_model",
}


def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: (H, W)
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    # tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    return F.interpolate(
        abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
        size=(tgt_size[0], tgt_size[1]),
        mode="bicubic",
        align_corners=False,
    ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)


# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or 
                [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_h_size, grid_w_size = grid_size, grid_size
    else:
        grid_h_size, grid_w_size = grid_size[0], grid_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_h_size, grid_w_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed],
                                   axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    default_norm_layer = partial(nn.LayerNorm, eps=1e-6)

    def __init__(self,
                 grid_size,
                 embed_dim,
                 num_heads,
                 kv_dim=None,
                 norm_layer=default_norm_layer,
                 adaptive=False):
        super().__init__()
        self.num_queries = grid_size**2
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.adaptive = adaptive

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(
                embed_dim, grid_size)).float()).requires_grad_(False)

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=.02)

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Parameter(
            (embed_dim**-0.5) * torch.randn(embed_dim, embed_dim))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, tgt_size=None, attn_mask=None):
        if self.adaptive:
            pos_embed = torch.Tensor(
                get_2d_sincos_pos_embed(self.embed_dim,
                                        tgt_size)).float().to(device=x.device,
                                                              dtype=x.dtype)
        else:
            pos_embed = get_abs_pos(self.pos_embed, tgt_size)

        x = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2)

        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(self._repeat(q, N) + self.pos_embed.unsqueeze(1),
                        x + pos_embed.unsqueeze(1),
                        x,
                        attn_mask=attn_mask)[0]
        x = out.permute(1, 0, 2)

        x = self.ln_post(x)
        x = x @ self.proj
        return x

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


class MiniCPMV(nn.Module):

    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.llm = MiniCPMForCausalLM(config,
                                      cache_config=cache_config,
                                      quant_config=quant_config,
                                      lora_config=lora_config)
        self.vpm = self.init_vision_module()
        self.vpm.to(dtype=torch.bfloat16)
        self.vision_dim = self.vpm.embed_dim
        self.embed_dim = self.llm.config.hidden_size
        self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)
        self.resampler.to(device="cuda", dtype=torch.bfloat16)
        self.sampler = Sampler()

    def init_vision_module(self):
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)
        model = timm.create_model('vit_so400m_patch14_siglip_384.webli',
                                  pretrained=False,
                                  num_classes=0,
                                  dynamic_img_size=True,
                                  dynamic_img_pad=True)
        torch.set_default_dtype(default_dtype)
        if isinstance(
                model,
                timm.models.VisionTransformer) and model.attn_pool is not None:
            model.attn_pool = torch.nn.Identity()

        if self.config.drop_vision_last_layer:
            model.blocks = model.blocks[:-1]

        return model

    def init_resampler(self, embed_dim, vision_dim):
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)
        resampler = Resampler(grid_size=int(math.sqrt(self.config.query_num)),
                              embed_dim=embed_dim,
                              num_heads=embed_dim // 128,
                              kv_dim=vision_dim,
                              adaptive=True)
        torch.set_default_dtype(default_dtype)
        return resampler

    def get_vision_embedding(self, pixel_values):
        res = []
        dtype = self.vpm.pos_embed.data.dtype
        for pixel_value in pixel_values:
            # V2.0 start
            H, W = pixel_value.shape[-2:]
            tgt_size = (math.ceil(H / self.vpm.patch_embed.patch_size[0]),
                        math.ceil(W / self.vpm.patch_embed.patch_size[0]))
            # V2.0 end
            vision_embedding = self.vpm.forward_features(
                pixel_value.unsqueeze(0).type(dtype))
            if hasattr(self.vpm,
                       'num_prefix_tokens') and self.vpm.num_prefix_tokens > 0:
                vision_embedding = vision_embedding[:, self.vpm.
                                                    num_prefix_tokens:]
            res.append(self.resampler(vision_embedding, tgt_size))
        return torch.vstack(res)

    def get_image_bound(self, input_ids, im_start_token_id, im_end_token_id,
                        unk_token_id):
        length = len(input_ids)
        bound = []
        im_start_idx = -1
        flag = False
        for x in range(length):
            if input_ids[x] == im_start_token_id:
                if flag is False:
                    flag = True
                im_start_idx = x + 1
            elif input_ids[x] == im_end_token_id:
                if flag is True:
                    flag = False
                    bound.append(im_start_idx)
                    bound.append(x - 1)
            elif input_ids[x] != unk_token_id:
                if flag is True:
                    flag = False
        if len(bound) > 0:
            bound = torch.tensor(bound).reshape(-1, 2)
        return bound

    def ensure_divide(self, length, patch_size):
        return max(round(length / patch_size) * patch_size, patch_size)

    def find_best_resize(self,
                         original_size,
                         scale_resolution,
                         patch_size,
                         allow_upscale=False):
        width, height = original_size
        if (width * height >
                scale_resolution * scale_resolution) or allow_upscale:
            r = width / height
            height = int(scale_resolution / math.sqrt(r))
            width = int(height * r)
        best_width = self.ensure_divide(width, patch_size)
        best_height = self.ensure_divide(height, patch_size)
        return (best_width, best_height)

    def get_refine_size(self,
                        original_size,
                        grid,
                        scale_resolution,
                        patch_size,
                        allow_upscale=False):
        width, height = original_size
        grid_x, grid_y = grid

        refine_width = self.ensure_divide(width, grid_x)
        refine_height = self.ensure_divide(height, grid_y)

        grid_width = refine_width / grid_x
        grid_height = refine_height / grid_y

        best_grid_size = self.find_best_resize((grid_width, grid_height),
                                               scale_resolution,
                                               patch_size,
                                               allow_upscale=allow_upscale)
        refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)
        return refine_size

    def split_to_patches(self, image, grid):
        patches = []
        width, height = (image.shape[-1], image.shape[-2])
        grid_x = int(width / grid[0])
        grid_y = int(height / grid[1])
        for i in range(0, height, grid_y):
            images = []
            for j in range(0, width, grid_x):
                patch = image[:, i:i + grid_y, j:j + grid_x]
                images.append(patch)
            patches.append(images)

        return patches

    def slice_image(self,
                    image: torch.Tensor,
                    max_slice_nums=9,
                    scale_resolution=448,
                    patch_size=14,
                    never_split=False):
        original_size = (image.shape[-1], image.shape[-2])
        original_width, original_height = original_size
        log_ratio = math.log(original_width / original_height)
        ratio = original_width * original_height / (scale_resolution *
                                                    scale_resolution)
        multiple = min(math.ceil(ratio), max_slice_nums)

        source_image = None
        best_grid = None
        patches = []

        if multiple <= 1 or never_split:
            best_size = self.find_best_resize(original_size, scale_resolution,
                                              patch_size)
            resize_transform = transforms.Compose([
                transforms.Resize((best_size[::-1]),
                                  InterpolationMode.BICUBIC,
                                  antialias=True),
                transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN,
                                     std=IMAGENET_INCEPTION_STD)
            ])
            source_image = resize_transform(image)
        else:
            candidate_split_grids_nums = []
            for i in [multiple - 1, multiple, multiple + 1]:
                if i == 1 or i > max_slice_nums:
                    continue
                candidate_split_grids_nums.append(i)

            best_resize = self.find_best_resize(original_size,
                                                scale_resolution, patch_size)
            resize_transform = transforms.Compose([
                transforms.Resize(best_resize[::-1],
                                  InterpolationMode.BICUBIC,
                                  antialias=True),
                transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN,
                                     std=IMAGENET_INCEPTION_STD)
            ])
            source_image = resize_transform(image.clone())
            candidate_grids = []

            # find best grid
            for split_grids_nums in candidate_split_grids_nums:
                m = 1
                while m <= split_grids_nums:
                    if split_grids_nums % m == 0:
                        candidate_grids.append([m, split_grids_nums // m])
                    m += 1

            best_grid = [1, 1]
            min_error = float("inf")
            for grid in candidate_grids:
                error = abs(log_ratio - math.log(grid[0] / grid[1]))
                if error < min_error:
                    best_grid = grid
                    min_error = error

            refine_size = self.get_refine_size(original_size,
                                               best_grid,
                                               scale_resolution,
                                               patch_size,
                                               allow_upscale=True)

            resize_transform = transforms.Compose([
                transforms.Resize(refine_size[::-1],
                                  InterpolationMode.BICUBIC,
                                  antialias=True),
                transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN,
                                     std=IMAGENET_INCEPTION_STD)
            ])
            refine_image = resize_transform(image.clone())
            patches = self.split_to_patches(refine_image, best_grid)

        return source_image, patches, best_grid

    def get_grid_placeholder(self, grid, query_num):
        image_placeholder = [self.config.im_start_token_id] + \
            [self.config.unk_token_id] * query_num + \
            [self.config.im_end_token_id]

        cols = grid[0]
        rows = grid[1]
        slices = []
        for i in range(rows):
            lines = []
            for j in range(cols):
                lines += image_placeholder
            slices = slices + lines
            if i < rows - 1:
                slices += [5]  # \n
        slice_placeholder = [self.config.slice_start_token_id
                             ] + slices + [self.config.slice_end_token_id]
        return slice_placeholder

    def get_slice_image_placeholder(self, image):
        image_placeholder = [self.config.im_start_token_id] + \
            [self.config.unk_token_id] * self.config.query_num + \
            [self.config.im_end_token_id]
        slice_images = []

        source_image, patches, best_grid = self.slice_image(
            image,
            self.config.max_slice_nums,  # default: 9
            self.config.scale_resolution,  # default: 448
            self.config.patch_size  # default: 14
        )

        slice_images.append(source_image)
        final_placeholder = image_placeholder

        if len(patches) > 0:
            for i in range(len(patches)):
                for j in range(len(patches[0])):
                    slice_images.append(patches[i][j])

            final_placeholder += self.get_grid_placeholder(
                best_grid, self.config.query_num)
        return slice_images, final_placeholder

    def modify_input_ids(self, input_ids, place_holder, im_start_token_id,
                         im_end_token_id):
        if len(torch.where(input_ids == im_end_token_id)[0]) == 0:
            return [], input_ids
        place_holder = torch.tensor(place_holder + [5]).to(
            device=input_ids.device, dtype=input_ids.dtype)
        start_idx = 0
        end_idx = 0
        for x in range(input_ids.shape[0]):
            if input_ids[x] == im_start_token_id:
                start_idx = x
            elif input_ids[x] == im_end_token_id:
                end_idx = x
        input_ids = torch.cat([
            input_ids[:start_idx], place_holder,
            input_ids[end_idx + 1:-place_holder.shape[0] + 2]
        ],
                              dim=0)
        image_start_tokens = torch.where(input_ids == im_start_token_id)[0]
        # 跳过 im_start
        image_start_tokens += 1
        image_end_tokens = torch.where(input_ids == im_end_token_id)[0]
        valid_image_nums = max(len(image_start_tokens), len(image_end_tokens))
        if image_start_tokens[:valid_image_nums].unsqueeze(
                -1).shape[0] == image_end_tokens[:valid_image_nums].unsqueeze(
                    -1).shape[0]:
            image_bound = torch.cat([
                image_start_tokens[:valid_image_nums].unsqueeze(-1),
                image_end_tokens[:valid_image_nums].unsqueeze(-1),
            ],
                                    dim=1)
        else:
            image_bound = torch.tensor([]).to(device=input_ids.device,
                                              dtype=input_ids.dtype)
        return image_bound, input_ids

    def get_embedding(self, data, im_start_token_id, im_end_token_id,
                      unk_token_id):
        if 'vision_hidden_states' not in data:
            pixel_values = data['pixel_values']
            if pixel_values is not None and len(pixel_values) > 0:
                images, places_holder = self.get_slice_image_placeholder(
                    pixel_values[0])
                vision_hidden_states = self.get_vision_embedding(images)
            else:
                vision_hidden_states = torch.tensor([]).to(
                    data['input_ids'].device)
        else:
            vision_hidden_states = data['vision_hidden_states']

        if data['pixel_values'] is not None:
            image_bound, input_ids = self.modify_input_ids(
                data['input_ids'], places_holder, im_start_token_id,
                im_end_token_id)
        else:
            input_ids = data['input_ids']
            image_bound = []

        vllm_embedding = self.llm.model.embed_tokens(
            input_ids) * self.llm.config.scale_emb
        vision_hidden_states = vision_hidden_states.type(vllm_embedding.dtype)
        if len(vision_hidden_states) > 0 and len(image_bound) > 0:
            image_indices = torch.stack([
                torch.arange(r[0], r[1], dtype=torch.long) for r in image_bound
            ]).to(vllm_embedding.device)
            vllm_embedding.scatter_(
                0,
                image_indices.view(-1, 1).repeat(1, vllm_embedding.shape[-1]),
                vision_hidden_states.view(-1, vision_hidden_states.shape[-1]))
        return vllm_embedding, vision_hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        image_input: Optional[torch.Tensor] = None,
    ):
        vllm_embeddings, vision_hidden_states = self.get_embedding(
            {
                'pixel_values': image_input,
                'input_ids': input_ids
            }, self.config.im_start_token_id, self.config.im_end_token_id,
            self.config.unk_token_id)
        output = self.llm(input_ids=None,
                          positions=positions,
                          kv_caches=kv_caches,
                          attn_metadata=attn_metadata,
                          input_embeds=vllm_embeddings)
        return output

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        return self.llm.compute_logits(hidden_states, sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            #     for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
            #         if key_to_modify in name:
            #             name = name.replace(key_to_modify, new_key)
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            use_default_weight_loading = False
            if "vpm" in name or 'resampler' in name:
                # We only do sharding for language model and
                # not vision model for now.
                use_default_weight_loading = True
            else:
                for (param_name, weight_name,
                     shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    param = params_dict[name.replace(weight_name, param_name)]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    use_default_weight_loading = True
            if use_default_weight_loading:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
