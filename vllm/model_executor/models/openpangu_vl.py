# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Adapted from vllm/model_executor/models/qwen2_5_vl.py
# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
#
# This file is a part of the vllm-ascend project.
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

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import lru_cache, partial
from typing import Annotated, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.transforms import v2
from transformers.utils import logging

from vllm.config import MultiModalConfig, VllmConfig
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .vision import get_vit_attn_backend

logger = logging.get_logger(__name__)


class OpenPanguVisionAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads
        )
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, self.tp_size
        )

        self.qkv = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
        )
        self.proj = RowParallelLinear(
            input_size=projection_size,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )
        self.attn = MMEncoderAttention(
            num_heads=self.num_attention_heads_per_partition,
            head_size=self.hidden_size_per_attention_head,
            scale=self.hidden_size_per_attention_head**-0.5,
            multimodal_config=multimodal_config,
        )
        self.apply_rotary_emb = ApplyRotaryEmb(enforce_enable=True)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        seq_length, _ = x.size()
        x, bias = self.qkv(x)
        if bias is not None:
            x = x + bias
        q, k, v = x.chunk(3, dim=1)

        q, k, v = (
            rearrange(
                x, "s (b n d) -> b s n d", d=self.hidden_size_per_attention_head, b=1
            ).contiguous()
            for x in (q, k, v)
        )
        qk_concat = torch.cat([q, k], dim=0)
        qk_rotated = self.apply_rotary_emb(qk_concat, cos, sin)
        q, k = torch.chunk(qk_rotated, 2, dim=0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        context_layer = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        context_layer = rearrange(
            context_layer, "b s h d -> s (b h d)", b=1
        ).contiguous()
        output, bias = self.proj(context_layer)
        if bias is not None:
            output = output + bias
        return output


class OpenPanguVisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = False,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        vision_config=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_act = vision_config.hidden_act
        if self.hidden_act == "silu":
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            if hidden_features % tp_size != 0:
                hidden_features = (hidden_features + tp_size - 1) // tp_size * tp_size
            self.gate_up_proj = MergedColumnParallelLinear(
                input_size=in_features,
                output_sizes=[hidden_features] * 2,
                bias=bias,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_up_proj",
            )
        else:
            self.up_proj = ColumnParallelLinear(
                in_features,
                hidden_features,
                bias=bias,
                quant_config=quant_config,
                prefix=f"{prefix}.up_proj",
            )

        self.down_proj = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor):
        if self.hidden_act == "silu":
            x, _ = self.gate_up_proj(x)
        else:
            x, _ = self.up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class OpenPanguVisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        norm_layer: Callable[[int], nn.Module] | None = None,
        vision_config=None,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = OpenPanguVisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = OpenPanguVisionMLP(
            dim,
            mlp_hidden_dim,
            act_fn=act_fn,
            bias=True,
            vision_config=vision_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, cos=cos, sin=sin
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class OpenPanguVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )
        self._seq_len_cached = 0
        self._freqs_cached = None

    def update_freqs_cache(self, seqlen: int) -> None:
        if seqlen > self._seq_len_cached:
            seqlen *= 2
            self._seq_len_cached = seqlen
            seq = torch.arange(
                seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
            )
            freqs = torch.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def forward(self, seqlen: int) -> torch.Tensor:
        self.update_freqs_cache(seqlen)
        return (
            self._freqs_cached[:seqlen]
            if self._freqs_cached is not None
            else self._freqs_cached
        )


class OpenPanguVisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size
        self.input_size = (
            self.patch_size * self.patch_size * in_channels * self.temporal_patch_size
        )

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_size:
            x = torch.cat(
                [
                    x.reshape(-1, self.patch_size * self.patch_size),
                    x.reshape(-1, self.patch_size * self.patch_size),
                ],
                dim=-1,
            ).reshape(-1, self.input_size)
        x = x.matmul(self.proj.weight.data.view(self.hidden_size, -1).transpose(0, 1))
        return x


class OpenPanguVisionPatchMerger(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable[[int], nn.Module] | None = None,
        spatial_merge_size: int = 2,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = norm_layer(context_dim)
        self.mlp = nn.Sequential(
            ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.0",
                return_bias=False,
            ),
            nn.GELU(),
            RowParallelLinear(
                self.hidden_size,
                d_model,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.2",
                return_bias=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln_q(x).view(-1, self.hidden_size))


class OpenPanguVisionTransformer(nn.Module):
    def __init__(
        self,
        vision_config,
        out_hidden_size,
        hidden_size,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
        interleaved=False,
    ) -> None:
        super().__init__()
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.spatial_merge_unit = self.spatial_merge_size**2

        norm_layer = partial(RMSNorm, eps=norm_eps)
        self.interleaved = interleaved
        self.out_hidden_size = vision_config.out_hidden_size
        self.hidden_act = vision_config.hidden_act

        head_dim = self.hidden_size // self.num_heads
        attn_backend_override = (
            multimodal_config.mm_encoder_attn_backend if multimodal_config else None
        )
        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim,
            dtype=torch.get_default_dtype(),
            attn_backend_override=attn_backend_override,
        )

        if self.attn_backend not in {
            AttentionBackendEnum.FLASH_ATTN,
        }:
            raise RuntimeError(
                f"Pangu-VL does not support {self.attn_backend} backend now."
            )
        self.rotary_pos_emb = OpenPanguVisionRotaryEmbedding(head_dim // 2)
        self.patch_embed = OpenPanguVisionPatchEmbed(
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
        )
        self.blocks = nn.ModuleList(
            [
                OpenPanguVisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    act_fn=_ACTIVATION_REGISTRY[vision_config.hidden_act],
                    vision_config=vision_config,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    multimodal_config=multimodal_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
                for layer_idx in range(vision_config.depth)
            ]
        )
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.hidden_size_per_attention_head = dist_utils.divide(
            self.hidden_size, self.num_heads
        )

        self.select_layer = getattr(
            vision_config, "mm_unit_vision_select_layer", [-1, -3]
        )
        self.select_index = [vision_config.depth + i for i in self.select_layer]
        self.select_index = self.select_index[::-1]
        self.select_layer = [-1 * (i + 1) for i in range(len(self.select_index))]

        self.take_indices = self.select_index

        self.final_layernorm = RMSNorm(self.hidden_size, eps=norm_eps)
        self.merger = nn.ModuleList(
            [
                OpenPanguVisionPatchMerger(
                    d_model=vision_config.out_hidden_size,
                    context_dim=self.hidden_size,
                    norm_layer=norm_layer,
                    spatial_merge_size=self.spatial_merge_size,
                    quant_config=quant_config,
                    prefix=f"{prefix}.merger.{i}",
                )
                for i in range(len(self.select_layer))
            ]
        )
        self.vision_projection = ProjectionSingle(out_hidden_size, hidden_size)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def cal_cos_sin(self, rotary_pos_emb):
        cos = rotary_pos_emb.cos()
        sin = rotary_pos_emb.sin()
        return cos, sin

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        # see https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py for details. #L209 # noqa: E501
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = (
                hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos_ids = (
                wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        # see https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py for details. #L238 # noqa: E501
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # compute cu_seqlens
        cu_seqlens = (
            torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
            .to(torch.int32)
            .to(x.device)
        )
        cu_seqlens = torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        x = self.patch_embed(x)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=x.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        seq_len, _ = x.size()
        x = x.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        x = x[window_index, :, :]
        x = x.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        cos, sin = self.cal_cos_sin(rotary_pos_emb.to(x.dtype))

        intermediates = []
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            x = blk(x, cu_seqlens=cu_seqlens_now, cos=cos, sin=sin)
            if layer_num in self.take_indices:
                ln_hs = self.final_layernorm(x)
                intermediates.append(ln_hs)

        image_embeddings_list = []
        for idx, sl in enumerate(self.select_layer):
            image_embeddings_list.append(self.merger[idx](intermediates[sl]))
        x = sum(image_embeddings_list)

        reverse_indices = torch.argsort(window_index)
        x = x[reverse_indices, :]
        x = self.vision_projection(x)
        return x

    def load_weights(self, weights) -> set[str]:
        def _padding_weight(name: str, w: torch.Tensor) -> torch.Tensor:
            if "gate_proj" in name or "up_proj" in name:
                dim, size = 0, w.size(0)
            elif "down_proj" in name:
                dim, size = 1, w.size(-1)
            else:
                return w
            pad_len = -size % self.tp_size
            if pad_len == 0:
                return w
            pad = [0] * (w.ndim * 2)
            pad[-(dim + 1) * 2 + 1] = pad_len
            return F.pad(w, pad, mode="constant", value=0)

        stacked_params_mapping = [
            ("attn.qkv.", "attn.q.", "q"),
            ("attn.qkv.", "attn.k.", "k"),
            ("attn.qkv.", "attn.v.", "v"),
        ]
        if self.hidden_act == "silu":
            stacked_params_mapping.extend(
                [
                    ("gate_up_proj", "gate_proj", 0),
                    ("gate_up_proj", "up_proj", 1),
                ]
            )
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if self.hidden_act == "silu":
                loaded_weight = _padding_weight(name, loaded_weight)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class ProjectionSingle(nn.Module):
    def __init__(self, i_hidden_size: int, t_hidden_size: int):
        super().__init__()
        self.act = F.silu
        self.fc1 = nn.Linear(i_hidden_size, t_hidden_size, bias=True)

    def forward(self, hidden_states):
        x = self.act(hidden_states)
        return self.fc1(x)


class OpenPanguVLProcessingInfo(Qwen2_5_VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.model_config.hf_config

    def get_hf_processor(
        self,
        *,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        size: dict[str, int] | None = None,
        fps: float | list[float] | None = None,
        **kwargs: object,
    ):
        if fps is not None:
            kwargs["fps"] = fps

        return self.ctx.get_hf_processor(
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )


class OpenPanguVLImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]

    pixel_values: Annotated[
        torch.Tensor,
        TensorShape("np", "cps"),
    ]
    image_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("ni", 3),
    ]


class OpenPanguVLImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]

    image_embeds: Annotated[
        torch.Tensor,
        TensorShape("nf", "hs"),
    ]
    image_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("ni", 3),
    ]


class OpenPanguVLVideoPixelInputs(TensorSchema):
    type: Literal["pixel_values_videos"]

    pixel_values_videos: Annotated[
        torch.Tensor,
        TensorShape("np", "ctps"),
    ]
    video_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("nv", 3),
    ]


class OpenPanguVLVideoEmbeddingInputs(TensorSchema):
    type: Literal["video_embeds"]

    video_embeds: Annotated[
        torch.Tensor,
        TensorShape("nf", "hs"),
    ]
    video_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("nv", 3),
    ]


class OpenPanguVLMultiModalProcessor(Qwen2_5_VLMultiModalProcessor):
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        image_token = hf_processor.image_token
        video_token = hf_processor.video_token
        vision_start_token = hf_processor.vision_start_token
        vision_end_token = hf_processor.vision_end_token
        image_token_id = vocab[image_token]
        video_token_id = vocab[video_token]
        vision_start_token_id = vocab[vision_start_token]
        vision_end_token_id = vocab[vision_end_token]
        placeholder = {
            "image": image_token_id,
            "video": video_token_id,
        }

        merge_length = image_processor.merge_size**2

        def get_replacement_openpangu_vision(item_idx: int, modality: str):
            out_item = out_mm_kwargs[modality][item_idx]
            grid_thw = out_item[f"{modality}_grid_thw"].data
            if not isinstance(grid_thw, torch.Tensor):
                raise TypeError("Expected 'grid_thw' to be a Tensor")
            if modality == "image":
                image_token_id_total = [image_token_id] * (
                    int(grid_thw.prod()) // merge_length
                )
                return image_token_id_total
            else:
                # When modality is video
                grid_t, grid_h, grid_w = grid_thw
                video_seq_length_per_time = (grid_h * grid_w).item() // merge_length
                video_token_id_per_time = (
                    [vision_start_token_id]
                    + [video_token_id] * video_seq_length_per_time
                    + [vision_end_token_id]
                )
                video_token_id_total = video_token_id_per_time * grid_t.item()
                video_token_id_middle = video_token_id_total[1:-1]
                return PromptUpdateDetails.select_token_id(
                    video_token_id_middle,
                    embed_token_id=video_token_id,
                )

        return [
            PromptReplacement(
                modality=modality,
                target=[placeholder[modality]],
                replacement=partial(
                    get_replacement_openpangu_vision, modality=modality
                ),
            )
            for modality in ("image", "video")
        ]


class OpenPanguVLDummyInputsBuilder(Qwen2_5_VLDummyInputsBuilder):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    OpenPanguVLMultiModalProcessor,
    info=OpenPanguVLProcessingInfo,
    dummy_inputs=OpenPanguVLDummyInputsBuilder,
)
class OpenPanguVLForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsLoRA, SupportsPP, SupportsMRoPE
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.vllm_config = vllm_config
        self.multimodal_config = multimodal_config
        quant_config = vllm_config.quant_config
        self.visual = OpenPanguVisionTransformer(
            vision_config=config.vision_config,
            out_hidden_size=config.vision_config.out_hidden_size,
            hidden_size=config.hidden_size,
            norm_eps=getattr(config.vision_config, "rms_norm_eps", 1e-6),
            quant_config=self._maybe_ignore_quant_config(quant_config),
            multimodal_config=multimodal_config,
            prefix=maybe_prefix(prefix, "visual"),
        )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix("openpangu", "language_model"),
            architectures=["PanguEmbeddedForCausalLM"],
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self._parse_preprocess_params(config.vision_config)

    def _parse_preprocess_params(self, vision_config):
        self.channel = vision_config.in_channels
        self.patch_size = vision_config.patch_size
        from vllm.multimodal import MULTIMODAL_REGISTRY

        image_processor = (
            MULTIMODAL_REGISTRY.create_processor(self.vllm_config.model_config)
            .info.get_hf_processor()
            .image_processor
        )
        self.do_rescale = image_processor.do_rescale
        self.rescale_factor = image_processor.rescale_factor
        self.do_normalize = image_processor.do_normalize
        self.image_mean = tuple(image_processor.image_mean)
        self.image_std = tuple(image_processor.image_std)

    def _maybe_ignore_quant_config(self, quant_config: QuantizationConfig):
        if isinstance(quant_config, (GPTQConfig, GPTQMarlinConfig)):
            return None
        return quant_config

    def _validate_and_reshape_mm_tensor(
        self, mm_input: object, name: str
    ) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(
                    f"{name} should be 2D or batched 3D tensor. "
                    f"Got ndim: {mm_input.ndim} "
                    f"(shape={mm_input.shape})"
                )
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_image_input(self, **kwargs: object):
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values"
            )
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw"
            )

            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of image pixel values. "
                    f"Got type: {type(pixel_values)}"
                )

            return OpenPanguVLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            image_embeds = self._validate_and_reshape_mm_tensor(
                image_embeds, "image embeds"
            )
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw"
            )

            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError(
                    "Incorrect type of image embeddings. "
                    f"Got type: {type(image_embeds)}"
                )
            return OpenPanguVLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

    def _parse_and_validate_video_input(self, **kwargs: object):
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            pixel_values_videos = self._validate_and_reshape_mm_tensor(
                pixel_values_videos, "video pixel values"
            )
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw"
            )

            return OpenPanguVLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )

        if video_embeds is not None:
            video_embeds = self._validate_and_reshape_mm_tensor(
                video_embeds, "video embeds"
            )
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw"
            )

            if not isinstance(video_embeds, torch.Tensor):
                raise ValueError(
                    "Incorrect type of video embeddings. "
                    f"Got type: {type(video_embeds)}"
                )
            return OpenPanguVLVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw,
            )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}
        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    **kwargs
                )
            if (
                input_key in ("pixel_values_videos", "video_embeds")
                and "video" not in mm_input_by_modality
            ):
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(
                    **kwargs
                )
        return mm_input_by_modality

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return None

        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                vision_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings = (
                    multimodal_embeddings
                    if not vision_embeddings
                    else (multimodal_embeddings + vision_embeddings)
                )
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings = (
                    multimodal_embeddings
                    if not video_embeddings
                    else (multimodal_embeddings + video_embeddings)
                )
        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.embed_input_ids(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = self.embed_input_ids(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                [self.config.image_token_id, self.config.video_token_id],
            )
        return inputs_embeds

    def _process_image_input(self, image_input) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        if grid_thw.ndim != 2:
            raise ValueError(f"grid_thw.ndim must be 2, but it is {grid_thw.ndim}")

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            # rescale and normalize
            pixel_values = pixel_values.reshape(
                -1, self.channel, self.patch_size, self.patch_size
            )
            pixel_values = rescale_and_normalize(
                pixel_values,
                self.do_rescale,
                self.rescale_factor,
                self.do_normalize,
                self.image_mean,
                self.image_std,
            )
            pixel_values = pixel_values.reshape(
                -1, self.channel * self.patch_size * self.patch_size
            )
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size
        return image_embeds.split(sizes.tolist())

    def _process_video_input(self, video_input) -> torch.Tensor:
        grid_thw = video_input["video_grid_thw"]
        if grid_thw.ndim != 2:
            raise ValueError(f"grid_thw.ndim must be 2, but it is {grid_thw.ndim}")

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(
                self.visual.dtype
            )
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)

        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return video_embeds.split(sizes.tolist())

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.merger.",
            tower_model="visual.",
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "[unused18][unused19][unused20]"
        if modality.startswith("video"):
            return "[unused18][unused32][unused20]"

        raise ValueError("Only image or video modality is supported")

    def iter_mm_grid_thw(
        self, mm_features: list[MultiModalFeatureSpec]
    ) -> Iterator[tuple[str, int, int, int, int]]:
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            offset = mm_feature.mm_position.offset
            modality = mm_feature.modality
            if modality == "image":
                t, h, w = mm_feature.data["image_grid_thw"].data.tolist()
                assert t == 1, f"Image must have 1 frame, got {t}"
                yield (
                    modality,
                    offset,
                    1,
                    h // spatial_merge_size,
                    w // spatial_merge_size,
                )
            elif modality == "video":
                t, h, w = mm_feature.data["video_grid_thw"].data.tolist()
                yield (
                    modality,
                    offset,
                    t,
                    h // spatial_merge_size,
                    w // spatial_merge_size,
                )
            else:
                raise ValueError(f"Unsupported modality: {modality}")

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        llm_pos_ids_list: list = []
        st = 0

        for (
            modality,
            offset,
            llm_grid_t,
            llm_grid_h,
            llm_grid_w,
        ) in self.iter_mm_grid_thw(mm_features):
            text_len = offset - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )
            if modality == "video":
                eot_bot_pos = torch.full((3, 1), 0, dtype=torch.long)
                offset_pos = max(llm_grid_h, llm_grid_w)
                current_pos = text_len + st_idx
                grid_h = (
                    torch.arange(llm_grid_h)
                    .view(-1, 1)
                    .expand(-1, llm_grid_w)
                    .flatten()
                )
                grid_w = (
                    torch.arange(llm_grid_w)
                    .view(1, -1)
                    .expand(llm_grid_h, -1)
                    .flatten()
                )
                frame_pos = torch.stack(
                    [
                        torch.full_like(grid_h, 0, dtype=torch.long),  # t
                        grid_h,  # h
                        grid_w,  # w
                    ]
                )
                llm_pos_ids_list.append(frame_pos + current_pos)
                for _ in range(llm_grid_t - 1):
                    current_pos = current_pos + offset_pos
                    llm_pos_ids_list.append(eot_bot_pos + current_pos)
                    llm_pos_ids_list.append(eot_bot_pos + current_pos + 1)
                    llm_pos_ids_list.append(frame_pos + current_pos + 2)
                    current_pos += 2
                st = (
                    offset + llm_grid_t * llm_grid_h * llm_grid_w + (llm_grid_t - 1) * 2
                )
            else:
                t_index = (
                    (
                        torch.arange(llm_grid_t)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                    )
                    .long()
                    .flatten()
                )
                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = offset + llm_grid_t * llm_grid_h * llm_grid_w
        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
        return llm_positions, mrope_position_delta


def rescale(image, scale):
    return image * scale


def normalize(image, mean, std):
    return v2.functional.normalize(image, mean, std)


@lru_cache(maxsize=10)
def _fuse_mean_std_and_rescale_factor(
    do_normalize: bool | None = None,
    image_mean: float | list[float] | None = None,
    image_std: float | list[float] | None = None,
    do_rescale: bool | None = None,
    rescale_factor: float | None = None,
    device: Optional["torch.device"] = None,
) -> tuple:
    if do_rescale and do_normalize:
        # Fused rescale and normalize
        image_mean = torch.tensor(image_mean, device=device) * (1.0 / rescale_factor)
        image_std = torch.tensor(image_std, device=device) * (1.0 / rescale_factor)
        do_rescale = False
    return image_mean, image_std, do_rescale


def rescale_and_normalize(
    images: "torch.Tensor",
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean: float | list[float],
    image_std: float | list[float],
    dtype: torch.dtype = torch.bfloat16,
) -> "torch.Tensor":
    """
    Rescale and normalize images.
    """
    image_mean, image_std, do_rescale = _fuse_mean_std_and_rescale_factor(
        do_normalize=do_normalize,
        image_mean=image_mean,
        image_std=image_std,
        do_rescale=do_rescale,
        rescale_factor=rescale_factor,
        device=images.device,
    )
    # if/elif as we use fused rescale and normalize if both are set to True
    if do_normalize:
        images = normalize(images.to(dtype=torch.float32), image_mean, image_std)
    elif do_rescale:
        images = rescale(images, rescale_factor)
    images = images.to(dtype)

    return images
