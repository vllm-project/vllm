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
"""Inference-only MiniCPM-V model compatible with HuggingFace weights."""
import math
import re
from functools import partial
from typing import (Any, Callable, Iterable, List, Optional, Tuple, TypedDict,
                    Union)

import numpy as np
import torch
import torch.nn.functional as F
import torch.types
from PIL import Image
from torch import nn
from torch.nn.init import trunc_normal_
from transformers.configuration_utils import PretrainedConfig

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsVision
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.minicpm import MiniCPMModel
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import (cached_get_image_processor,
                                   cached_get_tokenizer)
from vllm.sequence import IntermediateTensors, SamplerOutput, SequenceData

from .idefics2_vision_model import Idefics2VisionTransformer

logger = init_logger(__name__)

_KEYS_TO_MODIFY_MAPPING = {
    "llm.lm_head": "lm_head",
    "llm.model": "llm",
}


class MiniCPMVImagePixelInputs(TypedDict):
    pixel_values: List[torch.Tensor]
    """
    Shape: `(batch_size * num_images, num_channels, height, width)`

    Note that the image size may vary, so we pass it as a list
    instead of a batched tensor.
    """

    image_bounds: torch.Tensor
    """
    Shape: `(batch_size * num_images, 2)`

    This should be in `(start, stop)` format.
    """

    tgt_sizes: torch.Tensor
    """
    Shape: `(batch_size * num_images, 2)`

    This should be in `(height, width)` format.
    """


MiniCPMVImageInputs = MiniCPMVImagePixelInputs

DEFAULT_LN = partial(nn.LayerNorm, eps=1e-6)


def get_abs_pos(abs_pos: torch.Tensor, tgt_size: torch.Tensor):
    # abs_pos: L, C
    # tgt_size: (H, W)
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    # tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    return (F.interpolate(
        abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
        size=(tgt_size[0], tgt_size[1]),
        mode="bicubic",
        align_corners=False,
    ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype))


# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(
        embed_dim: int,
        grid_size: Union[int, Tuple[int, int]],
        cls_token: bool = False,
        version: Tuple[int, int] = (2, 0),
):
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

    if version == (2, 0):
        grid = grid.reshape([2, 1, grid_h_size, grid_w_size])
        pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, version)
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed],
                                       axis=0)
    else:
        pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, version)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int,
                                      grid: np.ndarray,
                                      version: Tuple[int, int] = (2, 0)):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0], version)  # (H*W, D/2) or (H, W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1], version)  # (H*W, D/2) or (H, W, D/2)

    if version == (2, 0):
        emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    else:
        emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int,
                                      pos: np.ndarray,
                                      version: Tuple[int, int] = (2, 0)):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,) / (H, W)
    out: (M, D) / (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    if version == (2, 0):
        pos = pos.reshape(-1)  # (M,)
        out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)
        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    else:
        out = np.einsum("hw,d->hwd", pos, omega)  # (H, W, D/2), outer product
        emb_sin = np.sin(out)  # (H, W, D/2)
        emb_cos = np.cos(out)  # (H, W, D/2)
        emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
    return emb


class BaseResampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
        self,
        num_queries: int,
        embed_dim: int,
        num_heads: int,
        kv_dim: Optional[int] = None,
        norm_layer: Callable[[int], nn.LayerNorm] = DEFAULT_LN,
    ) -> None:
        super().__init__()

        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=0.02)
        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = ReplicatedLinear(kv_dim, embed_dim, bias=False)
        else:
            # Maintain the same return value with ReplicatedLinear.forward
            self.kv_proj = lambda *args, **kwargs: (
                nn.Identity()(*args, **kwargs),
                None,
            )
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)
        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Parameter(
            (embed_dim**-0.5) * torch.randn(embed_dim, embed_dim))

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


class Resampler2(BaseResampler):

    def __init__(
        self,
        grid_size: int,
        embed_dim: int,
        num_heads: int,
        kv_dim: Optional[int] = None,
        norm_layer: Callable[[int], nn.LayerNorm] = DEFAULT_LN,
        adaptive: bool = False,
    ) -> None:
        super().__init__(grid_size**2, embed_dim, num_heads, kv_dim,
                         norm_layer)

        self.adaptive = adaptive
        pos_embed_arr = get_2d_sincos_pos_embed(embed_dim,
                                                grid_size,
                                                version=(2, 0))
        self.pos_embed = nn.Parameter(
            torch.from_numpy(pos_embed_arr).float()).requires_grad_(False)

        self.apply(self._init_weights)

    def forward(
        self,
        x: torch.Tensor,
        tgt_sizes: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        if self.adaptive:
            pos_embed_arr = get_2d_sincos_pos_embed(self.embed_dim,
                                                    tgt_sizes,
                                                    version=(2, 0))
            pos_embed = torch.from_numpy(pos_embed_arr).to(device=x.device,
                                                           dtype=x.dtype)
        else:
            pos_embed = get_abs_pos(self.pos_embed, tgt_sizes)

        x, _ = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2)

        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(
            self._repeat(q, N) + self.pos_embed.unsqueeze(1),
            x + pos_embed.unsqueeze(1),
            x,
            attn_mask=attn_mask,
        )[0]
        x = out.permute(1, 0, 2)

        x = self.ln_post(x)
        x = x @ self.proj
        return x


class Resampler2_5(BaseResampler):

    def __init__(
            self,
            num_queries: int,
            embed_dim: int,
            num_heads: int,
            kv_dim: Optional[int] = None,
            norm_layer: Callable[[int], nn.LayerNorm] = DEFAULT_LN,
            max_size: Tuple[int, int] = (70, 70),
    ) -> None:
        super().__init__(num_queries, embed_dim, num_heads, kv_dim, norm_layer)

        self.max_size = max_size
        self._set_2d_pos_cache(self.max_size)

        self.apply(self._init_weights)

    def _set_2d_pos_cache(self,
                          max_size: Tuple[int, int],
                          device: torch.types.Device = "cpu") -> None:
        pos_embed_arr = get_2d_sincos_pos_embed(self.embed_dim,
                                                max_size,
                                                version=(2, 5))
        pos_embed = torch.from_numpy(pos_embed_arr).float().to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _adjust_pos_cache(self, tgt_sizes: torch.Tensor,
                          device: torch.types.Device) -> None:
        max_h = tgt_sizes[:, 0].max().item()
        max_w = tgt_sizes[:, 1].max().item()
        assert isinstance(max_h, int) and isinstance(max_w, int)

        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = (
                max(max_h, self.max_size[0]),
                max(max_w, self.max_size[1]),
            )
            self._set_2d_pos_cache(self.max_size, device)

    def forward(self, x: torch.Tensor,
                tgt_sizes: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == tgt_sizes.shape[0]
        bs = x.shape[0]

        device = x.device
        dtype = x.dtype

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes, device=device)

        max_patch_len = patch_len.max().item()
        assert isinstance(max_patch_len, int)

        key_padding_mask = torch.zeros((bs, max_patch_len),
                                       dtype=torch.bool,
                                       device=device)

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i].tolist()
            pos_embed.append(self.pos_embed[:tgt_h, :tgt_w, :].reshape(
                (tgt_h * tgt_w, -1)).to(dtype))  # patches * D
            key_padding_mask[i, patch_len[i]:] = True
        pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed,
                                                    batch_first=True,
                                                    padding_value=0.0).permute(
                                                        1, 0,
                                                        2)  # BLD => L * B * D
        x, _ = self.kv_proj(x)  # B * L * D
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D

        q = self.ln_q(self.query)  # Q * D

        out = self.attn(
            self._repeat(q, bs),  # Q * B * D
            x + pos_embed,  # L * B * D +  L * B * D
            x,
            key_padding_mask=key_padding_mask,
        )[0]
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D

        x = self.ln_post(x)
        x = x @ self.proj
        return x


def get_version_by_config(config: PretrainedConfig) -> Tuple[int, ...]:
    version_float = getattr(config, "version", None)

    # The old configs do not include version number
    # TODO: Remove this after the HF repos are updated
    if version_float is None:
        if config.hidden_size == 2304 and config.query_num == 64:
            return (2, 0)
        return (2, 5)

    version_str = str(version_float)
    return tuple(int(x) for x in version_str.split("."))


def get_max_minicpmv_image_tokens(ctx: InputContext):
    hf_config = ctx.get_hf_config(PretrainedConfig)
    return getattr(hf_config, "query_num", 64)


def dummy_seq_data_for_minicpmv(seq_len: int):
    token_ids = [0] * seq_len
    return SequenceData(token_ids)


def dummy_image_for_minicpmv(hf_config: PretrainedConfig):
    width = height = hf_config.image_size
    image = Image.new("RGB", (width, height), color=0)
    return {"image": image}


def dummy_data_for_minicpmv(ctx: InputContext, seq_len: int):
    hf_config = ctx.get_hf_config(PretrainedConfig)

    seq_data = dummy_seq_data_for_minicpmv(seq_len)
    mm_data = dummy_image_for_minicpmv(hf_config)

    return seq_data, mm_data


def input_processor_for_minicpmv(ctx: InputContext, llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs
    model_config = ctx.model_config
    version = get_version_by_config(model_config.hf_config)
    tokenizer = cached_get_tokenizer(model_config.tokenizer,
                                     trust_remote_code=True)
    image_processor = cached_get_image_processor(model_config.tokenizer)

    def get_placeholder(image_size: Tuple[int, int], num_image: int):
        if version == (2, 0) or version == (2, 5):
            return image_processor. \
                get_slice_image_placeholder(image_size)
        return image_processor. \
            get_slice_image_placeholder(image_size, num_image)

    prompt = llm_inputs.get("prompt")
    if prompt is None:
        token_ids = llm_inputs.get("prompt_token_ids")
        prompt = tokenizer.decode(token_ids)

    pattern = "(<image>./</image>)"
    images = multi_modal_data["image"]
    if isinstance(images, Image.Image):
        images = [images]
    image_tags = re.findall(pattern, prompt)

    if len(image_tags) == 0:
        new_token_ids = token_ids
        new_prompt = prompt
    else:
        text_chunks = prompt.split(pattern)
        new_prompt_chunks: List[str] = []
        for i in range(len(images)):
            new_prompt_chunks += [
                text_chunks[i],
                get_placeholder(images[i].size, i)
            ]
        new_prompt_chunks.append(text_chunks[-1])
        new_prompt = "".join(new_prompt_chunks)
        new_token_ids = tokenizer.encode(new_prompt)

    llm_inputs = LLMInputs(
        prompt_token_ids=new_token_ids,
        prompt=new_prompt,
        multi_modal_data=multi_modal_data,
    )
    return llm_inputs


class MiniCPMVBaseModel(nn.Module, SupportsVision):
    """
    The abstract class of MiniCPMV can only be inherited, but cannot be
    instantiated.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        multimodal_config: MultiModalConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.multimodal_config = multimodal_config

        self.version = get_version_by_config(self.config)
        self.llm = self.init_llm(config, cache_config, quant_config)
        self.vpm = self.init_vision_module()
        param_dtype = torch.get_default_dtype()
        self.vpm.to(dtype=param_dtype)
        self.vision_dim = (self.vpm.embed_dim if self.version == (2, 0) else
                           self.vpm.embeddings.embed_dim)
        self.embed_dim = self.config.hidden_size
        self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)
        self.resampler.to(device="cuda", dtype=param_dtype)
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        image_inputs: Optional[MiniCPMVImageInputs],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vlm_embedding: torch.Tensor = self.llm.embed_tokens(input_ids)
        if hasattr(self.config, "scale_emb"):
            vlm_embedding *= self.config.scale_emb

        if image_inputs is None:  # No image
            vision_hidden_states = torch.tensor([], device=input_ids.device)
        else:
            vision_hidden_states = self.get_vision_hidden_states(image_inputs)

            # See NOTE in _parse_and_validate_inputs
            image_bounds = image_inputs["image_bounds"]
            if len(image_bounds) > 0:
                image_indices = torch.stack([
                    torch.arange(start, end, dtype=torch.long)
                    for start, end in image_bounds.tolist()
                ]).to(vlm_embedding.device)
                vlm_embedding.scatter_(
                    0,
                    image_indices.view(-1, 1).repeat(1,
                                                     vlm_embedding.shape[-1]),
                    vision_hidden_states.view(-1,
                                              vision_hidden_states.shape[-1]),
                )

        return vlm_embedding, vision_hidden_states

    def _get_image_bounds(self, input_ids: torch.Tensor) -> torch.Tensor:
        tokenizer = cached_get_tokenizer(self.config._name_or_path,
                                         trust_remote_code=True)
        start_cond = input_ids == tokenizer.im_start_id
        end_cond = input_ids == tokenizer.im_end_id
        if hasattr(tokenizer, "slice_start_id"):
            start_cond |= (input_ids == tokenizer.slice_start_id)
            end_cond |= (input_ids == tokenizer.slice_end_id)

        image_start_tokens, = torch.where(start_cond)
        image_start_tokens += 1
        image_end_tokens, = torch.where(end_cond)
        valid_image_nums = max(len(image_start_tokens), len(image_end_tokens))

        if valid_image_nums == 0:
            return torch.zeros((0, 2), device=input_ids.device)

        return torch.hstack([
            image_start_tokens[:valid_image_nums].unsqueeze(-1),
            image_end_tokens[:valid_image_nums].unsqueeze(-1),
        ])

    def _parse_and_validate_inputs(
        self,
        input_ids: torch.Tensor,
        **kwargs: object,
    ) -> Optional[MiniCPMVImageInputs]:
        pixel_values = kwargs.pop("pixel_values", [])
        tgt_sizes = kwargs.pop("tgt_sizes", [])

        if not isinstance(pixel_values, (torch.Tensor, list)):
            raise ValueError("Incorrect type of pixel values. "
                             f"Got type: {type(pixel_values)}")

        if not isinstance(tgt_sizes, (torch.Tensor, list)):
            raise ValueError("Incorrect type of target sizes. "
                             f"Got type: {type(tgt_sizes)}")

        if len(pixel_values) != len(tgt_sizes):
            raise ValueError("Inconsistent batch lengths, found: "
                             f"{len(pixel_values)} vs. {len(tgt_sizes)}")

        pixel_values_flat: List[torch.Tensor] = []
        tgt_sizes_flat: List[torch.Tensor] = []
        for b in range(len(pixel_values)):
            pixel_values_flat += pixel_values[b]
            tgt_sizes_flat += tgt_sizes[b]

        # NOTE: Input IDs does not contain image tokens during memory profiling,
        # so we allow it to be empty
        if len(pixel_values_flat) != len(tgt_sizes_flat):
            raise ValueError("Inconsistent flattened lengths, found: "
                             f"{len(pixel_values_flat)} vs. "
                             f"{len(tgt_sizes_flat)}")

        if len(pixel_values_flat) == 0:
            return None

        return MiniCPMVImageInputs(
            image_bounds=self._get_image_bounds(input_ids),
            pixel_values=pixel_values_flat,
            tgt_sizes=torch.stack(tgt_sizes_flat),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        image_inputs = self._parse_and_validate_inputs(input_ids, **kwargs)

        vlm_embeddings, _ = self.get_embedding(input_ids, image_inputs)

        output = self.llm(
            input_ids=None,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=vlm_embeddings,
        )
        return output

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

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
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            use_default_weight_loading = False
            if self.is_default_weight_loading(name):
                use_default_weight_loading = True
            else:
                for param_name, weight_name, shard_id in stacked_params_mapping:
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

    def init_llm(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> nn.Module:
        raise NotImplementedError

    def init_vision_module(self) -> nn.Module:
        raise NotImplementedError

    def init_resampler(self, embed_dim: int, vision_dim: int) -> nn.Module:
        raise NotImplementedError

    def get_vision_embedding(
        self,
        pixel_values: List[torch.Tensor],
        patch_attn_mask: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_vision_hidden_states(self,
                                 data: MiniCPMVImageInputs) -> torch.Tensor:
        raise NotImplementedError

    def is_default_weight_loading(self, name: str) -> bool:
        raise NotImplementedError


class MiniCPMV2_0(MiniCPMVBaseModel):

    def __init__(
        self,
        config: PretrainedConfig,
        multimodal_config: MultiModalConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__(config, multimodal_config, cache_config, quant_config)
        assert self.version == (2, 0)

    def init_llm(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> nn.Module:
        return MiniCPMModel(config,
                            cache_config=cache_config,
                            quant_config=quant_config)

    def init_vision_module(self) -> nn.Module:
        # TODO :refactor this vision model
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm==0.9.10") from ImportError
        with set_default_torch_dtype(torch.float16):
            model = timm.create_model(
                "vit_so400m_patch14_siglip_384.webli",
                pretrained=False,
                num_classes=0,
                dynamic_img_size=True,
                dynamic_img_pad=True,
            )

        if (isinstance(model, timm.models.VisionTransformer)
                and model.attn_pool is not None):
            model.attn_pool = torch.nn.Identity()

        if self.config.drop_vision_last_layer:
            model.blocks = model.blocks[:-1]

        return model

    def init_resampler(self, embed_dim: int, vision_dim: int) -> nn.Module:
        with set_default_torch_dtype(torch.float16):
            resampler = Resampler2(
                embed_dim=embed_dim,
                num_heads=embed_dim // 128,
                grid_size=int(math.sqrt(self.config.query_num)),
                kv_dim=vision_dim,
                adaptive=True,
            )

        return resampler

    def get_vision_embedding(
        self,
        pixel_values: List[torch.Tensor],
        patch_attn_mask: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        res = []
        dtype = self.vpm.pos_embed.data.dtype
        for pixel_value in pixel_values:
            H, W = pixel_value[0].shape[-2:]
            tgt_size = (
                math.ceil(H / self.vpm.patch_embed.patch_size[0]),
                math.ceil(W / self.vpm.patch_embed.patch_size[0]),
            )
            vision_embedding = self.vpm.forward_features(
                pixel_value.unsqueeze(0).type(dtype))
            if (hasattr(self.vpm, "num_prefix_tokens")
                    and self.vpm.num_prefix_tokens > 0):
                vision_embedding = vision_embedding[:, self.vpm.
                                                    num_prefix_tokens:]
            res.append(self.resampler(vision_embedding, tgt_size))
        return torch.vstack(res)

    def get_vision_hidden_states(self,
                                 data: MiniCPMVImageInputs) -> torch.Tensor:
        pixel_values = data["pixel_values"]

        return self.get_vision_embedding(pixel_values)

    def is_default_weight_loading(self, name: str) -> bool:
        return "resampler" in name or "vpm" in name


class MiniCPMV2_5(MiniCPMVBaseModel):

    def __init__(
        self,
        config: PretrainedConfig,
        multimodal_config: MultiModalConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__(config, multimodal_config, cache_config, quant_config)
        assert self.version == (2, 5)

    def init_llm(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> nn.Module:
        return LlamaModel(config,
                          cache_config=cache_config,
                          quant_config=quant_config)

    def init_vision_module(self) -> nn.Module:
        model = Idefics2VisionTransformer(self.config.vision_config)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]
        return model

    def init_resampler(self, embed_dim: int, vision_dim: int) -> nn.Module:
        with set_default_torch_dtype(torch.float16):
            resampler = Resampler2_5(
                num_queries=self.config.query_num,
                embed_dim=embed_dim,
                num_heads=embed_dim // 128,
                kv_dim=vision_dim,
            )
        return resampler

    def get_vision_embedding(
        self,
        pixel_values: List[torch.Tensor],
        patch_attn_mask: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        vision_embedding = self.vpm(pixel_values,
                                    patch_attention_mask=patch_attn_mask)
        vision_embedding = self.resampler(vision_embedding, tgt_sizes)
        return vision_embedding

    def get_vision_hidden_states(self,
                                 data: MiniCPMVImageInputs) -> torch.Tensor:
        pixel_values = data["pixel_values"]
        tgt_sizes = data["tgt_sizes"]

        device = self.vpm.embeddings.position_embedding.weight.device
        dtype = self.vpm.embeddings.position_embedding.weight.dtype
        all_pixel_values_lst = [
            i.flatten(end_dim=1).permute(1, 0) for i in pixel_values
        ]

        max_patches = (tgt_sizes[:, 0] * tgt_sizes[:, 1]).max().item()
        assert isinstance(max_patches, int)

        all_pixel_values = torch.nn.utils.rnn.pad_sequence(
            all_pixel_values_lst, batch_first=True, padding_value=0.0)
        B, L, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2,
                                                    1).reshape(B, 3, -1, L)

        patch_attn_mask = torch.zeros((B, 1, max_patches),
                                      dtype=torch.bool,
                                      device=device)
        for i in range(B):
            patch_attn_mask[i, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

        return self.get_vision_embedding(all_pixel_values.type(dtype),
                                         patch_attn_mask, tgt_sizes)

    def is_default_weight_loading(self, name: str) -> bool:
        return "resampler" in name


class MiniCPMV2_6(MiniCPMVBaseModel):

    def __init__(
        self,
        config: PretrainedConfig,
        multimodal_config: MultiModalConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__(config, multimodal_config, cache_config, quant_config)
        assert self.version == (2, 6)

    def init_llm(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> nn.Module:
        return Qwen2Model(config,
                          cache_config=cache_config,
                          quant_config=quant_config)

    def init_vision_module(self) -> nn.Module:
        # A custom version of SiglipVisionTransformer, won't work with TP
        from vllm.model_executor.models.na_vit import SiglipVisionTransformer

        if self.config._attn_implementation == "flash_attention_2":
            self.config.vision_config._attn_implementation = "flash_attention_2"
        else:
            # not support sdpa
            self.config.vision_config._attn_implementation = "eager"
        model = SiglipVisionTransformer(self.config.vision_config)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]
        return model

    def init_resampler(self, embed_dim: int, vision_dim: int) -> nn.Module:
        with set_default_torch_dtype(torch.float16):
            # The resampler in 2.6 remains consistent with the one in 2.5.
            resampler = Resampler2_5(
                num_queries=self.config.query_num,
                embed_dim=embed_dim,
                num_heads=embed_dim // 128,
                kv_dim=vision_dim,
            )

        return resampler

    def get_vision_embedding(
        self,
        pixel_values: List[torch.Tensor],
        patch_attn_mask: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        vision_embedding = self.vpm(
            pixel_values,
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
        ).last_hidden_state
        return vision_embedding

    def get_vision_hidden_states(self,
                                 data: MiniCPMVImageInputs) -> torch.Tensor:
        pixel_values = data["pixel_values"]
        tgt_sizes = data["tgt_sizes"]

        device = self.vpm.embeddings.position_embedding.weight.device
        dtype = self.vpm.embeddings.position_embedding.weight.dtype
        all_pixel_values_lst = [
            i.flatten(end_dim=1).permute(1, 0) for i in pixel_values
        ]

        max_patches = (tgt_sizes[:, 0] * tgt_sizes[:, 1]).max().item()
        assert isinstance(max_patches, int)

        all_pixel_values = torch.nn.utils.rnn.pad_sequence(
            all_pixel_values_lst, batch_first=True, padding_value=0.0)
        B, L, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2,
                                                    1).reshape(B, 3, -1, L)

        patch_attn_mask = torch.zeros((B, 1, max_patches),
                                      dtype=torch.bool,
                                      device=device)
        for i in range(B):
            patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True
        vision_embedding = self.vpm(
            all_pixel_values.type(dtype),
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
        ).last_hidden_state

        return self.resampler(vision_embedding, tgt_sizes)

    def is_default_weight_loading(self, name: str) -> bool:
        return "resampler" in name or "vpm" in name


_SUPPORT_VERSION = {
    (2, 0): MiniCPMV2_0,
    (2, 5): MiniCPMV2_5,
    (2, 6): MiniCPMV2_6
}


@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_minicpmv_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_minicpmv)
@INPUT_REGISTRY.register_input_processor(input_processor_for_minicpmv)
class MiniCPMV(MiniCPMVBaseModel):
    """
    Different versions of MiniCPMV use different visual encoders and LLMs,
    which is not conducive to the current integration logic of LoRA and
    bitsandbytes in vLLM. Therefore, it is necessary to separate them.
    """

    def __new__(
        cls,
        config: PretrainedConfig,
        multimodal_config: MultiModalConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        if not hasattr(config, "version"):
            if config.hidden_size == 2304 and config.query_num == 64:
                version = (2, 0)
            else:
                version = (2, 5)
        else:
            version = str(config.version).split(".")
            version = tuple([int(x) for x in version])
        # Dispatch class based on version
        instance_class = _SUPPORT_VERSION.get(version, None)
        if instance_class is None:
            raise ValueError(
                "Currently, MiniCPMV only supports versions 2.0, 2.5, and 2.6")
        return instance_class(config, multimodal_config, cache_config,
                              quant_config)
