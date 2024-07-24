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
import re
from functools import partial
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.nn.init import trunc_normal_
from transformers.configuration_utils import PretrainedConfig
from transformers.models.idefics2.modeling_idefics2 import (
    Idefics2VisionTransformer)

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsVision
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.minicpm import MiniCPMForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import (cached_get_image_processor,
                                   cached_get_tokenizer)
from vllm.sequence import IntermediateTensors, SamplerOutput, SequenceData

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
def get_2d_sincos_pos_embed(embed_dim,
                            grid_size,
                            cls_token=False,
                            version=2.0):
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

    if version == 2.0:
        grid = grid.reshape([2, 1, grid_h_size, grid_w_size])
        pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, version)
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed],
                                       axis=0)
    else:
        pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, version)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid, version=2.0):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0], version)  # (H*W, D/2) or (H, W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1], version)  # (H*W, D/2) or (H, W, D/2)

    if version == 2.0:
        emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    else:
        emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, version=2.0):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,) / (H, W)
    out: (M, D) / (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    if version == 2.0:
        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)
        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    else:
        out = np.einsum('hw,d->hwd', pos, omega)  # (H, W, D/2), outer product
        emb_sin = np.sin(out)  # (H, W, D/2)
        emb_cos = np.cos(out)  # (H, W, D/2)
        emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
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
                 num_queries,
                 grid_size,
                 embed_dim,
                 num_heads,
                 kv_dim=None,
                 norm_layer=default_norm_layer,
                 adaptive=False,
                 max_size=(70, 70),
                 version=2.0):
        super().__init__()

        self.version = version
        if self.version == 2.0:
            self.num_queries = grid_size**2
        else:
            self.num_queries = num_queries
            self.max_size = max_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.adaptive = adaptive

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

        if self.version == 2.0:
            self.pos_embed = nn.Parameter(
                torch.from_numpy(
                    get_2d_sincos_pos_embed(
                        embed_dim, grid_size,
                        version=self.version)).float()).requires_grad_(False)
        else:
            self._set_2d_pos_cache(self.max_size)

        self.apply(self._init_weights)

    def _set_2d_pos_cache(self, max_size, device='cpu'):
        pos_embed = torch.from_numpy(
            get_2d_sincos_pos_embed(self.embed_dim,
                                    max_size,
                                    version=self.version)).float().to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _adjust_pos_cache(self, tgt_sizes, device):
        max_h = torch.max(tgt_sizes[:, 0])
        max_w = torch.max(tgt_sizes[:, 1])
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = [
                max(max_h, self.max_size[0]),
                max(max_w, self.max_size[1])
            ]
            self._set_2d_pos_cache(self.max_size, device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_2_5(self, x, tgt_sizes=None):
        assert x.shape[0] == tgt_sizes.shape[0]
        bs = x.shape[0]

        device = x.device
        dtype = x.dtype

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes, device=device)

        max_patch_len = torch.max(patch_len)
        key_padding_mask = torch.zeros((bs, max_patch_len),
                                       dtype=torch.bool,
                                       device=device)

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            pos_embed.append(self.pos_embed[:tgt_h, :tgt_w, :].reshape(
                (tgt_h * tgt_w, -1)).to(dtype))  # patches * D
            key_padding_mask[i, patch_len[i]:] = True

        pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed,
                                                    batch_first=True,
                                                    padding_value=0.0).permute(
                                                        1, 0,
                                                        2)  # BLD => L * B * D

        x = self.kv_proj(x)  # B * L * D
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D

        q = self.ln_q(self.query)  # Q * D

        out = self.attn(
            self._repeat(q, bs),  # Q * B * D
            x + pos_embed,  # L * B * D +  L * B * D
            x,
            key_padding_mask=key_padding_mask)[0]
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D

        x = self.ln_post(x)
        x = x @ self.proj
        return x

    def forward_2(self, x, tgt_sizes=None, attn_mask=None):
        if self.adaptive:
            pos_embed = torch.Tensor(
                get_2d_sincos_pos_embed(self.embed_dim,
                                        tgt_sizes)).float().to(device=x.device,
                                                               dtype=x.dtype)
        else:
            pos_embed = get_abs_pos(self.pos_embed, tgt_sizes)

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

    def forward(self, x, tgt_sizes=None, attn_mask=None):
        if self.version == 2.0:
            return self.forward_2(x, tgt_sizes=tgt_sizes, attn_mask=attn_mask)
        else:
            return self.forward_2_5(x, tgt_sizes=tgt_sizes)

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


def get_max_minicpmv_image_tokens(ctx: InputContext):
    hf_config = ctx.get_hf_config(PretrainedConfig)
    return getattr(hf_config, "query_num", 64)


def dummy_seq_data_for_minicpmv(seq_len: int):
    token_ids = [0] * seq_len
    return SequenceData(token_ids)


def dummy_image_for_minicpmv(hf_config):
    width = height = hf_config.image_size
    image = Image.new("RGB", (width, height), color=0)
    return {"image": image}


def dummy_data_for_minicpmv(ctx: InputContext, seq_len: int):
    hf_config = ctx.get_hf_config(PretrainedConfig)

    # image_feature_size = get_max_minicpmv_image_tokens(ctx)

    seq_data = dummy_seq_data_for_minicpmv(seq_len)

    mm_data = dummy_image_for_minicpmv(hf_config)

    return seq_data, mm_data


def input_processor_for_minicpmv(ctx: InputContext, llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs

    model_config = ctx.model_config

    prompt = llm_inputs.get("prompt")
    tokenizer = cached_get_tokenizer(model_config.tokenizer,
                                     trust_remote_code=True)
    image_processor = cached_get_image_processor(model_config.tokenizer)

    pattern = "(<image>./</image>)"
    image = multi_modal_data["image"]
    image_tags = re.findall(pattern, prompt)
    assert len(image_tags) <= 1
    text_chunks = prompt.split(pattern)
    new_prompt = text_chunks[0] \
        + image_processor.get_slice_image_placeholder(image.size) \
        + text_chunks[1]

    new_token_ids = tokenizer.encode(new_prompt)

    llm_inputs = LLMInputs(prompt_token_ids=new_token_ids,
                           prompt=new_prompt,
                           multi_modal_data=multi_modal_data)
    return llm_inputs


@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_minicpmv_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_minicpmv)
@INPUT_REGISTRY.register_input_processor(input_processor_for_minicpmv)
class MiniCPMV(nn.Module, SupportsVision):

    def __init__(
        self,
        config,
        multimodal_config: MultiModalConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.multimodal_config = multimodal_config

        self.version = float(self.config.version)
        self.llm = self.init_llm(config, cache_config, quant_config)
        self.vpm = self.init_vision_module()
        param_dtype = torch.get_default_dtype()
        self.vpm.to(dtype=param_dtype)
        self.vision_dim = self.vpm.embed_dim if self.version == 2.0 \
            else self.vpm.embeddings.embed_dim
        self.embed_dim = self.llm.config.hidden_size
        self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)
        self.resampler.to(device="cuda", dtype=param_dtype)
        self.sampler = Sampler()

    def init_llm(self, config, cache_config, quant_config):
        if self.version == 2.0:
            return MiniCPMForCausalLM(config,
                                      cache_config=cache_config,
                                      quant_config=quant_config)
        else:
            return LlamaForCausalLM(config,
                                    cache_config=cache_config,
                                    quant_config=quant_config)

    def init_vision_module(self):
        if self.version == 2.0:
            try:
                import timm
            except ImportError:
                raise ImportError(
                    'Please install timm==0.9.10') from ImportError
            default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float16)
            model = timm.create_model('vit_so400m_patch14_siglip_384.webli',
                                      pretrained=False,
                                      num_classes=0,
                                      dynamic_img_size=True,
                                      dynamic_img_pad=True)
            torch.set_default_dtype(default_dtype)
            if isinstance(model, timm.models.VisionTransformer
                          ) and model.attn_pool is not None:
                model.attn_pool = torch.nn.Identity()

            if self.config.drop_vision_last_layer:
                model.blocks = model.blocks[:-1]
        else:
            model = Idefics2VisionTransformer(self.config.vision_config)
            if self.config.drop_vision_last_layer:
                model.encoder.layers = model.encoder.layers[:-1]
        return model

    def init_resampler(self, embed_dim, vision_dim):
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)
        if self.version == 2.0:
            resampler = Resampler(grid_size=int(
                math.sqrt(self.config.query_num)),
                                  num_queries=None,
                                  embed_dim=embed_dim,
                                  num_heads=embed_dim // 128,
                                  kv_dim=vision_dim,
                                  adaptive=True,
                                  version=self.version)
        else:
            resampler = Resampler(num_queries=self.config.query_num,
                                  grid_size=None,
                                  embed_dim=embed_dim,
                                  num_heads=embed_dim // 128,
                                  kv_dim=vision_dim,
                                  adaptive=True,
                                  version=self.version)
        torch.set_default_dtype(default_dtype)
        return resampler

    def get_vision_embedding(self,
                             pixel_values,
                             patch_attn_mask=None,
                             tgt_sizes=None,
                             version=2.0):
        if version == 2.0:
            res = []
            dtype = self.vpm.pos_embed.data.dtype
            for pixel_value in pixel_values:
                # V2.0 start
                H, W = pixel_value[0].shape[-2:]
                tgt_size = (math.ceil(H / self.vpm.patch_embed.patch_size[0]),
                            math.ceil(W / self.vpm.patch_embed.patch_size[0]))
                # V2.0 end
                vision_embedding = self.vpm.forward_features(
                    pixel_value.unsqueeze(0).type(dtype))
                if hasattr(self.vpm, 'num_prefix_tokens'
                           ) and self.vpm.num_prefix_tokens > 0:
                    vision_embedding = vision_embedding[:, self.vpm.
                                                        num_prefix_tokens:]
                res.append(self.resampler(vision_embedding, tgt_size))
            return torch.vstack(res)
        else:
            vision_embedding = self.vpm(
                pixel_values.type(dtype),
                patch_attention_mask=patch_attn_mask).last_hidden_state
            vision_embedding = self.resampler(vision_embedding, tgt_sizes)

    def get_image_bounds(self, input_ids):
        tokenizer = cached_get_tokenizer(self.config._name_or_path,
                                         trust_remote_code=True)
        im_start_token_id = tokenizer.im_start_id
        im_end_token_id = tokenizer.im_end_id
        image_start_tokens = torch.where(input_ids == im_start_token_id)[0]
        image_start_tokens += 1
        image_end_tokens = torch.where(input_ids == im_end_token_id)[0]
        valid_image_nums = min(len(image_start_tokens), len(image_end_tokens))
        if valid_image_nums == 0:
            return []
        image_bound = torch.hstack([
            image_start_tokens[:valid_image_nums].unsqueeze(-1),
            image_end_tokens[:valid_image_nums].unsqueeze(-1),
        ])

        return image_bound

    def get_vision_hidden_states(self, data):
        if "vision_hidden_states" not in data:
            pixel_values = data["pixel_values"]
            tgt_sizes = data["tgt_sizes"]
            vision_hidden_states = []
            if self.version == 2.0:
                if pixel_values is not None and len(pixel_values) > 0:
                    vision_hidden_states = self.get_vision_embedding(
                        pixel_values)
                else:
                    vision_hidden_states = torch.tensor([]).to(
                        data["input_ids"].device)
            else:
                device = self.vpm.embeddings.position_embedding.weight.device
                dtype = self.vpm.embeddings.position_embedding.weight.dtype
                all_pixel_values = [
                    i.flatten(end_dim=1).permute(1, 0) for i in pixel_values
                ]
                if all_pixel_values:
                    tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)
                    max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])
                    all_pixel_values = torch.nn.utils.rnn.pad_sequence(
                        all_pixel_values, batch_first=True, padding_value=0.0)
                    B, L, _ = all_pixel_values.shape
                    all_pixel_values = all_pixel_values.permute(
                        0, 2, 1).reshape(B, 3, -1, L)

                    patch_attn_mask = torch.zeros((B, 1, max_patches),
                                                  dtype=torch.bool,
                                                  device=device)
                    for i in range(B):
                        patch_attn_mask[i, :tgt_sizes[i][0] *
                                        tgt_sizes[i][1]] = True

                    vision_embedding = self.vpm(
                        all_pixel_values.type(dtype),
                        patch_attention_mask=patch_attn_mask).last_hidden_state
                    vision_hidden_states = self.resampler(
                        vision_embedding, tgt_sizes)

                else:  # no image
                    dummy_feature = []
                    vision_hidden_states = dummy_feature
        else:
            vision_hidden_states = data["vision_hidden_states"]

        return vision_hidden_states

    def get_embedding(self, data):
        input_ids = data["input_ids"]

        vision_hidden_states = self.get_vision_hidden_states(data)
        if vision_hidden_states is not None and len(vision_hidden_states) > 0:
            image_bounds = self.get_image_bounds(input_ids)
        else:
            image_bounds = []

        if hasattr(self.llm.config, 'scale_emb'):
            vlm_embedding = self.llm.model.embed_tokens(
                input_ids) * self.llm.config.scale_emb
        else:
            vlm_embedding = self.llm.model.embed_tokens(input_ids)
        vision_hidden_states = [
            i.type(vlm_embedding.dtype) if isinstance(i, torch.Tensor) else i
            for i in vision_hidden_states
        ]

        if len(vision_hidden_states) > 0 and len(image_bounds) > 0:
            vision_hidden_states = torch.cat(vision_hidden_states, dim=0)
            image_indices = torch.stack([
                torch.arange(r[0], r[1], dtype=torch.long)
                for r in image_bounds
            ]).to(vlm_embedding.device)
            vlm_embedding.scatter_(
                0,
                image_indices.view(-1, 1).repeat(1, vlm_embedding.shape[-1]),
                vision_hidden_states.view(-1, vision_hidden_states.shape[-1]))
        return vlm_embedding, vision_hidden_states

    def process_multimodal_inputs(self, inputs):
        pixel_values = []
        tgt_sizes = []
        for b in range(len(inputs["pixel_values"])):
            pixel_values += inputs["pixel_values"][b]
            tgt_sizes += inputs["tgt_sizes"][b]
        return {
            "pixel_values": pixel_values,
            "input_ids": inputs["input_ids"],
            "tgt_sizes": tgt_sizes
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ):
        inputs = {
            "pixel_values": kwargs.pop("pixel_values", []),
            "input_ids": input_ids,
            "tgt_sizes": kwargs.pop("tgt_sizes", None),
        }

        inputs = self.process_multimodal_inputs(inputs)

        vlm_embeddings, vision_hidden_states = self.get_embedding(inputs)

        output = self.llm(input_ids=None,
                          positions=positions,
                          kv_caches=kv_caches,
                          attn_metadata=attn_metadata,
                          intermediate_tensors=intermediate_tensors,
                          input_embeds=vlm_embeddings)
        return output

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        return self.llm.compute_logits(hidden_states, sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.llm.sample(logits, sampling_metadata)
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
