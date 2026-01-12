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

from functools import partial
from typing import Callable, Literal, Optional, TypedDict, Union
from collections.abc import Iterable, Mapping, Sequence
import math
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from vllm.config import VllmConfig
from vllm.distributed import parallel_state, tensor_model_parallel_all_gather
from vllm.distributed import utils as dist_utils
from vllm.model_executor.models.interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP, SupportsMRoPE)
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo,
)
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear,
                                               MergedColumnParallelLinear)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader 
from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.multimodal.processing import PromptUpdate, PromptReplacement, PromptUpdateDetails
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from transformers.utils import logging
from transformers import PretrainedConfig

from vllm.model_executor.models.openpangu_processor_vl import OpenPanguVLProcessor
from vllm.model_executor.models.openpangu import PanguEmbeddedForCausalLM
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.activation import SiluAndMul

try:
    import flash_attn
except (ImportError, ModuleNotFoundError):
    flash_attn = None

logger = logging.get_logger(__name__)


class OpenPanguVisionAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads)
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, self.tp_size)

        self.qkv = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv")
        self.proj = RowParallelLinear(input_size=projection_size,
                                      output_size=embed_dim,
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.proj")
        self.scale_value = self.hidden_size_per_attention_head**-0.5


    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2] 
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


    def apply_rotary_pos_emb(self, q, k, cos, sin, offset: int = 0):
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed.to(torch.bfloat16), k_embed.to(torch.bfloat16)


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

        q, k, v = (rearrange(x, "s (b n d) -> b s n d", d=self.hidden_size_per_attention_head, b=1).contiguous()
                   for x in (q, k, v))
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

        if flash_attn is not None:
            from flash_attn import flash_attn_varlen_func
            q, k, v = [
                rearrange(x, "b s h d -> (b s) h d").contiguous()
                for x in (q, k, v)
            ]
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            attn_out = flash_attn_varlen_func(q,
                                            k,
                                            v,
                                            cu_seqlens_q=cu_seqlens,
                                            cu_seqlens_k=cu_seqlens,
                                            max_seqlen_q=max_seqlen,
                                            max_seqlen_k=max_seqlen,
                                            dropout_p=0.0,
                                            causal=False)

            attn_out = rearrange(attn_out, "(b s) h d -> s (b h d)", b=1).contiguous()
        else:
            outputs = []
            for i in range(1, len(cu_seqlens)):
                start_idx = cu_seqlens[i - 1]
                end_idx = cu_seqlens[i]
                q_i = q[:, start_idx:end_idx]
                k_i = k[:, start_idx:end_idx]
                v_i = v[:, start_idx:end_idx]
                q_i, k_i, v_i = (rearrange(x, "b s h d -> b h s d")
                                 for x in [q_i, k_i, v_i])
                output_i = F.scaled_dot_product_attention(q_i,
                                                          k_i,
                                                          v_i,
                                                          dropout_p=0.0)
                output_i = rearrange(output_i, "b h s d -> b s h d ")
                outputs.append(output_i)
            context_layer = torch.cat(outputs, dim=1)
            attn_out = rearrange(context_layer, "b s h d -> s (b h d)", b=1).contiguous()
        output, bias = self.proj(attn_out)
        if bias is not None:
            output = output + bias
        return output


class OpenPanguVisionMLP(nn.Module):

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 bias: bool = False,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
                 vision_config = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.hidden_act = vision_config.hidden_act
        if self.hidden_act == "silu":
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            if hidden_features % tp_size != 0:
                hidden_features = (hidden_features + tp_size - 1) // tp_size * tp_size
            self.gate_up_proj = MergedColumnParallelLinear(input_size=in_features,
                                                           output_sizes=[hidden_features] * 2,
                                                           bias=bias,
                                                           quant_config=quant_config,
                                                           prefix=f"{prefix}.gate_up_proj",)
        else:
            self.up_proj = ColumnParallelLinear(in_features,
                                                hidden_features,
                                                bias=bias,
                                                quant_config=quant_config,
                                                prefix=f"{prefix}.up_proj")
        
        self.down_proj = RowParallelLinear(hidden_features,
                                              in_features,
                                              bias=bias,
                                              quant_config=quant_config,
                                              prefix=f"{prefix}.down_proj")
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor):
        if self.hidden_act == "silu":
            x, _ = self.gate_up_proj(x)
            x = SiluAndMul()(x)
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
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        vision_config = None,
        quant_config: Optional[QuantizationConfig] = None,
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
            prefix=f"{prefix}.attn")
        self.mlp = OpenPanguVisionMLP(dim,
                                     mlp_hidden_dim,
                                     act_fn=act_fn,
                                     bias=True,
                                     vision_config=vision_config,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.mlp")
        
    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens=cu_seqlens, cos=cos, sin=sin)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class OpenPanguVisionRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.inv_freq = 1.0 / (theta**(torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self._seq_len_cached = 0
        self._freqs_cached = None

    def update_freqs_cache(self, seqlen: int) -> None:
        if seqlen > self._seq_len_cached:
            seqlen *= 2
            self._seq_len_cached = seqlen
            seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def forward(self, seqlen: int) -> torch.Tensor:
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]


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
        self.input_size = self.patch_size * self.patch_size * in_channels * self.temporal_patch_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels,
                              hidden_size,
                              kernel_size=kernel_size,
                              stride=kernel_size,
                              bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_size:
            x = torch.cat([x.reshape(-1, self.patch_size * self.patch_size), \
            x.reshape(-1, self.patch_size * self.patch_size)], dim=-1).reshape(-1, self.input_size)
        x = x.matmul(
            self.proj.weight.data.view(self.hidden_size, -1).transpose(0, 1))
        return x


class OpenPanguVisionPatchMerger(nn.Module):

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        spatial_merge_size: int = 2,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = norm_layer(context_dim)
        self.mlp = nn.Sequential(
            ColumnParallelLinear(self.hidden_size,
                                 self.hidden_size,
                                 bias=True,
                                 quant_config=quant_config,
                                 prefix=f"{prefix}.mlp.0",
                                 return_bias=False),
            nn.GELU(),
            RowParallelLinear(self.hidden_size,
                              d_model,
                              bias=True,
                              quant_config=quant_config,
                              prefix=f"{prefix}.mlp.2",
                              return_bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln_q(x).view(-1, self.hidden_size))


class OpenPanguVisionTransformer(nn.Module):
    def __init__(
        self,
        vision_config,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        interleaved=False,
        use_data_parallel: bool = False,
    ) -> None:
        self.use_data_parallel = use_data_parallel
        self._tp_group = self._get_tp_group()
        with parallel_state.patch_tensor_parallel_group(self._tp_group):
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
                        prefix=f"{prefix}.blocks.{layer_idx}",
                    )
                    for layer_idx in range(vision_config.depth)
                ]
            )
            self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
            self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
            self.hidden_size_per_attention_head = dist_utils.divide(self.hidden_size, self.num_heads)

            self.select_layer = getattr(vision_config, "mm_unit_vision_select_layer", [-1, -3])
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
    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.devic

    def cal_cos_sin(self, rotary_pos_emb):
        cos = rotary_pos_emb.cos()
        sin = rotary_pos_emb.sin()

        if not self.interleaved:
            cos_new = torch.cat((cos, cos), dim=-1)
            sin_new = torch.cat((sin, sin), dim=-1)
        else:
            cos_new = rearrange(torch.stack((cos, cos), dim=-1), "... d two -> ...(d two)", two=2)
            sin_new = rearrange(torch.stack((sin, sin), dim=-1), "... d two -> ...(d two)", two=2)
        cos_new = cos_new.reshape(1, -1, 1, self.hidden_size_per_attention_head)
        sin_new = sin_new.reshape(1, -1, 1, self.hidden_size_per_attention_head)
        return cos_new, sin_new

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        see https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py#L209 # noqa: E501
        for details.
        """
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
        """
        see https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py#L238 # noqa: E501
        for details.
        """
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
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
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
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
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).to(torch.int32)
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
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        cos, sin = self.cal_cos_sin(rotary_pos_emb)

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
            return F.pad(w, pad, mode='constant', value=0)

        with parallel_state.patch_tensor_parallel_group(self._tp_group):
            stacked_params_mapping = [
                ("attn.qkv.", "attn.q.", "q"),
                ("attn.qkv.", "attn.k.", "k"),
                ("attn.qkv.", "attn.v.", "v"),
            ]
            if self.hidden_act == "silu":
                stacked_params_mapping.extend([
                    ("gate_up_proj", "gate_proj", 0),
                    ("gate_up_proj", "up_proj", 1),
                ])
            params_dict = dict(self.named_parameters(remove_duplicate=False))
            loaded_params: set[str] = set()

            for name, loaded_weight in weights:
                if self.hidden_act == "silu":
                    loaded_weight = _padding_weight(name, loaded_weight)
                for (param_name, weight_name, shard_id) in stacked_params_mapping:
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

    def _get_tp_group(self) -> None:
        if not self.use_data_parallel:
            return parallel_state.get_tp_group()

        world_size = torch.distributed.get_world_size()
        tensor_model_parallel_size = 1
        group_ranks = torch.arange(world_size).view(-1, tensor_model_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]

        # creates tp process group containing only a subset of gpu ranks
        local_rank = parallel_state.get_world_group().local_rank
        tp_backend = torch.distributed.get_backend(parallel_state.get_tp_group().device_group)
        return parallel_state.init_model_parallel_group(group_ranks, local_rank, tp_backend)


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
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        fps: Optional[Union[float, list[float]]] = None,
        **kwargs: object,
    ):
        if fps is not None:
            kwargs["fps"] = fps
        if False:
            return self.ctx.get_hf_processor(
                OpenPanguVLProcessor,
                image_processor=self.get_image_processor(
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    size=size,
                    use_fast=kwargs.get("use_fast", True),
                ),
                **kwargs,
            )
        else:
            return self.ctx.get_hf_processor(
                OpenPanguVLProcessor,
                use_fast=kwargs.pop("use_fast", True),
                **kwargs,
            )

def get_load_balance_assignment(
    sizes: list[int],
    num_gpus: int = 2,
) -> tuple[list[int], list[int], list[int]]:
    """
    see https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/vision.py#L253 for details.
    """

    n_samples = len(sizes)

    # Handle edge cases
    if n_samples == 0:
        return [], [0] * num_gpus, [0] * num_gpus

    # Use greedy algorithm - balance by total size, not sample count
    gpu_assignments = [list[int]() for _ in range(num_gpus)]
    gpu_loads = [0] * num_gpus  # This tracks total SIZE, not sample count

    # Sort indices by size (largest first for better load balancing)
    large_to_small_indices = sorted(range(n_samples), key=lambda i: sizes[i], reverse=True)

    for idx in large_to_small_indices:
        # Find GPU with minimum current load (by total size)
        min_gpu = min(range(num_gpus), key=lambda i: gpu_loads[i])
        gpu_assignments[min_gpu].append(idx)
        gpu_loads[min_gpu] += sizes[idx]

    # Create shuffle indices and counts
    shuffle_indices = list[int]()
    gpu_sample_counts = list[int]()
    for gpu_id in range(num_gpus):
        shuffle_indices.extend(gpu_assignments[gpu_id])
        gpu_sample_counts.append(len(gpu_assignments[gpu_id]))

    return (shuffle_indices, gpu_sample_counts, gpu_loads)


def run_dp_sharded_mrope_vision_model(
    vision_model: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw_list,
    *,
    rope_type: Literal["rope_3d", "rope_2d"],
) -> tuple[torch.Tensor, ...]:
    """
    https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/vision.py#L322 for details.
    """
    grid_thw_list = grid_thw_list.tolist()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()

    tp_rank_local = parallel_state.get_tensor_model_parallel_rank()

    patches_per_image = [math.prod(grid_thw) for grid_thw in grid_thw_list]
    cum_patches_per_image = [0, *itertools.accumulate(patches_per_image)]

    # Get load balancing assignment with all metadata
    (image_to_tp_rank, gpu_sample_counts, grouped_pixel_values_len) = get_load_balance_assignment(
        patches_per_image, tp_size
    )

    cum_gpu_sample_counts = [0, *itertools.accumulate(gpu_sample_counts)]

    image_idxs_local = image_to_tp_rank[cum_gpu_sample_counts[tp_rank_local] : cum_gpu_sample_counts[tp_rank_local + 1]]

    # Get the pixel values for the local images based on the image_idxs_local
    if len(image_idxs_local) > 0:
        pixel_values_local = torch.cat(
            [pixel_values[cum_patches_per_image[i] : cum_patches_per_image[i + 1]] for i in image_idxs_local]
        )
    else:
        # Handle case where this rank has no images
        pixel_values_local = torch.empty(
            (0, pixel_values.shape[1]),
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
    if rope_type == "rope_2d":
        embed_dim_reduction_factor = vision_model.merge_kernel_size[0] * vision_model.merge_kernel_size[1]
    else:
        embed_dim_reduction_factor = vision_model.spatial_merge_size * vision_model.spatial_merge_size

    # Find the max length across all ranks
    # The output embedding of every DP rank has to be
    # padded to this length for tensor_model_parallel_all_gather
    # to work
    max_len_per_rank = max(grouped_pixel_values_len) // embed_dim_reduction_factor
    local_grid_thw_list = [grid_thw_list[i] for i in image_idxs_local]

    # Run the vision model on the local pixel_values_local
    if rope_type == "rope_2d":
        if pixel_values_local.shape[0] > 0:
            image_embeds_local = vision_model(pixel_values_local, torch.tensor(local_grid_thw_list))
            if isinstance(image_embeds_local, list):
                image_embeds_local = torch.cat(image_embeds_local, dim=0)
        else:
            out_dim = getattr(vision_model.config, "hidden_size", None)
            image_embeds_local = torch.empty(
                (0, embed_dim_reduction_factor, out_dim),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )
    else:
        if pixel_values_local.shape[0] > 0:
            image_embeds_local = vision_model(pixel_values_local, torch.tensor(local_grid_thw_list))
        else:
            # Handle empty case
            image_embeds_local = torch.empty(
                (0, vision_model.out_hidden_size),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )

    # Pad the output based on max_len_per_rank
    # for tensor_model_parallel_all_gather to work
    current_len = image_embeds_local.shape[0]
    if current_len < max_len_per_rank:
        padding_size = max_len_per_rank - current_len
        if rope_type == "rope_2d":
            padding = torch.empty(
                (
                    padding_size,
                    image_embeds_local.shape[1],
                    image_embeds_local.shape[2],
                ),
                dtype=image_embeds_local.dtype,
                device=image_embeds_local.device,
            )
        else:
            padding = torch.empty(
                (padding_size, image_embeds_local.shape[1]),
                dtype=image_embeds_local.dtype,
                device=image_embeds_local.device,
            )
        image_embeds_local_padded = torch.cat([image_embeds_local, padding], dim=0)
    else:
        image_embeds_local_padded = image_embeds_local

    # Do all_gather to collect embeddings from all ranks
    gathered_embeds = tensor_model_parallel_all_gather(image_embeds_local_padded, dim=0)

    # Remove padding and reconstruct per-rank embeddings
    rank_embeddings = list[torch.Tensor]()
    for rank in range(tp_size):
        start_idx = rank * max_len_per_rank
        end_idx = start_idx + (grouped_pixel_values_len[rank] // embed_dim_reduction_factor)
        rank_embeddings.append(gathered_embeds[start_idx:end_idx])

    patches_per_output_image = [(patch_size // embed_dim_reduction_factor) for patch_size in patches_per_image]

    # Reconstruct embeddings in the original order
    original_order_embeddings = [None] * len(grid_thw_list)
    current_idx = 0
    for rank in range(tp_size):
        count = gpu_sample_counts[rank]
        if count > 0:
            # Get images assigned to this rank in shuffled order
            rank_images = image_to_tp_rank[current_idx : current_idx + count]

            rank_embed = rank_embeddings[rank]
            # Split rank embeddings back to individual images
            embed_start = 0
            for img_idx in rank_images:
                img_patches = patches_per_output_image[img_idx]
                original_order_embeddings[img_idx] = rank_embed[embed_start : embed_start + img_patches]
                embed_start += img_patches
            current_idx += count
    out_embeddings = tuple(embed for embed in original_order_embeddings if embed is not None)
    if len(out_embeddings) != len(original_order_embeddings):
        raise ValueError("Found unassigned embeddings")

    return torch.concat(out_embeddings)


class OpenPanguVLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor


class OpenPanguVLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: torch.Tensor
    image_grid_thw: torch.Tensor


class OpenPanguVLVideoPixelInputs(TypedDict):
    type: Literal["pixel_values_videos"]
    pixel_values_videos: torch.Tensor
    video_grid_thw: torch.Tensor
    second_per_grid_ts: torch.Tensor


class OpenPanguVLVideoEmbeddingInputs(TypedDict):
    type: Literal["video_embeds"]
    video_embeds: torch.Tensor
    video_grid_thw: torch.Tensor


class OpenPanguVLMultiModalProcessor(Qwen2_5_VLMultiModalProcessor):

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(
            **hf_processor_mm_kwargs)
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
            try:
                grid_thw = out_mm_kwargs[f"{modality}_grid_thw"][item_idx]
            except Exception as e:
                out_item = out_mm_kwargs[modality][item_idx]
                grid_thw = out_item[f"{modality}_grid_thw"].data
            if not isinstance(grid_thw, torch.Tensor):
                raise TypeError("Expected 'grid_thw' to be a Tensor")
            if modality == "image":
                image_token_id_total = [image_token_id] * (int(grid_thw.prod()) // merge_length)
                return image_token_id_total
            else:
                # When modality is video
                grid_t, grid_h, grid_w = grid_thw
                video_seq_length_per_time = (grid_h * grid_w).item() // merge_length
                video_token_id_per_time = [vision_start_token_id] + [video_token_id] * video_seq_length_per_time + \
                                    [vision_end_token_id]
                video_token_id_total = video_token_id_per_time * grid_t
                video_token_id_middle = video_token_id_total[1:-1]
                return PromptUpdateDetails.select_token_id(
                    video_token_id_middle,
                    embed_token_id=video_token_id,
                )

        return [
            PromptReplacement(
                modality=modality,
                target=[placeholder[modality]],
                replacement=partial(get_replacement_openpangu_vision,
                                    modality=modality),
            ) for modality in ("image", "video")
        ]


class OpenPanguVLDummyInputsBuilder(Qwen2_5_VLDummyInputsBuilder):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    OpenPanguVLMultiModalProcessor,
    info=OpenPanguVLProcessingInfo,
    dummy_inputs=OpenPanguVLDummyInputsBuilder,
)
class OpenPanguVLForConditionalGeneration(nn.Module, SupportsMultiModal,
                                         SupportsLoRA, SupportsPP, SupportsMRoPE):

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        })
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):

        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.use_data_parallel = getattr(vllm_config.parallel_config, 'enable_multimodal_encoder_data_parallel', False)
        self.config = config
        self.multimodal_config = multimodal_config
        quant_config = vllm_config.quant_config
        self.visual = OpenPanguVisionTransformer(
            vision_config=config.vision_config,
            norm_eps=getattr(config.vision_config, "rms_norm_eps", 1e-6),
            quant_config=self._maybe_ignore_quant_config(quant_config),
            prefix=maybe_prefix(prefix, "visual"),
            use_data_parallel=self.use_data_parallel,
        )
        self.visual.vision_projection = ProjectionSingle(config.vision_config.out_hidden_size, config.hidden_size)

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix("openpangu", "language_model"),
            architectures=["PanguEmbeddedForCausalLM"],
        )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def _maybe_ignore_quant_config(self, quant_config: QuantizationConfig):
            if isinstance(quant_config, (GPTQConfig, GPTQMarlinConfig)):
                return None
            return quant_config

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                    name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                                f"Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(f"{name} should be 2D or batched 3D tensor. "
                                    f"Got ndim: {mm_input.ndim} "
                                    f"(shape={mm_input.shape})")
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
                pixel_values, "image pixel values")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return OpenPanguVLImagePixelInputs(type="pixel_values",
                                              pixel_values=pixel_values,
                                              image_grid_thw=image_grid_thw)

        if image_embeds is not None:
            image_embeds = self._validate_and_reshape_mm_tensor(
                image_embeds, "image embeds")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")
            return OpenPanguVLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw)
        
    def _parse_and_validate_video_input(self, **kwargs: object):
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            pixel_values_videos = self._validate_and_reshape_mm_tensor(
                pixel_values_videos, "video pixel values")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            return OpenPanguVLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )

        if video_embeds is not None:
            video_embeds = self._validate_and_reshape_mm_tensor(
                video_embeds, "video embeds")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            if not isinstance(video_embeds, torch.Tensor):
                raise ValueError("Incorrect type of video embeddings. "
                                 f"Got type: {type(video_embeds)}")
            return OpenPanguVLVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw)
        
    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}
        for input_key in kwargs:
            if input_key in ("pixel_values", "image_embeds"
                             ) and "image" not in mm_input_by_modality:
                mm_input_by_modality[
                    "image"] = self._parse_and_validate_image_input(**kwargs)
            if input_key in ("pixel_values_videos", "video_embeds"
                             ) and "video" not in mm_input_by_modality:
                mm_input_by_modality[
                    "video"] = self._parse_and_validate_video_input(**kwargs)
        return mm_input_by_modality
    
    def get_language_model(self) -> torch.nn.Module:
        return self.language_model
    
    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:

        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(
            **kwargs)
        if not mm_input_by_modality:
            return None

        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                vision_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += vision_embeddings
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += video_embeddings
        return multimodal_embeddings
    
    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.embed_input_ids(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [self.config.image_token_id, self.config.video_token_id])
        return inputs_embeds

    def get_input_embeddings_v0(
        self,
        input_ids: torch.Tensor,
        image_input = None,
        video_input = None,
    ) -> torch.Tensor:
        inputs_embeds = self.get_input_embeddings(input_ids)
        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                image_embeds,
                placeholder_token_id=self.config.image_token_id,
            )

        if video_input is not None:
            video_embeds = self._process_video_input(video_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                video_embeds,
                placeholder_token_id=self.config.video_token_id,
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
            if self.use_data_parallel:
                image_embeds = run_dp_sharded_mrope_vision_model(
                    self.visual, pixel_values, grid_thw, rope_type="rope_3d"
                )
            else:
                image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

            image_embeds = self.visual.vision_projection(image_embeds)
        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size
        return image_embeds.split(sizes.tolist())

    def _process_video_input(
        self,
        video_input
    ) -> torch.Tensor:
        grid_thw = video_input["video_grid_thw"]
        if grid_thw.ndim != 2:
            raise ValueError(f"grid_thw.ndim must be 2, but it is {grid_thw.ndim}")

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
            if self.use_data_parallel:
                video_embeds = run_dp_sharded_mrope_vision_model(
                    self.visual, pixel_values_videos, grid_thw, rope_type="rope_3d"
                )
            else:
                video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)

            video_embeds = self.visual.vision_projection(video_embeds)
        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return video_embeds.split(sizes.tolist())

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            video_input = self._parse_and_validate_video_input(**kwargs)

            if image_input is None and video_input is None:
                inputs_embeds = None
            else:
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids, image_input=image_input, video_input=video_input
                )
                input_ids = None
        
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
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
 
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
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "[unused18][unused19][unused20]"
        if modality.startswith("video"):
            return "[unused18][unused32][unused20]"

        raise ValueError("Only image or video modality is supported")

    @classmethod
    def get_mrope_input_positions(
        cls,
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[list[list[int]], torch.Tensor],
        video_grid_thw: Union[list[list[int]], torch.Tensor],
        second_per_grid_ts: Optional[list[float]] = None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
        audio_feature_lengths: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
    ) -> tuple[torch.Tensor, int]:
        """Get mrope input positions and delta value."""
        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        vision_end_token_id = hf_config.vision_end_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        tokens_per_second = getattr(hf_config.vision_config, "tokens_per_second", 1.0)

        if isinstance(image_grid_thw, list):
            image_grid_thw = torch.tensor(image_grid_thw)
        if isinstance(video_grid_thw, list):
            video_grid_thw = torch.tensor(video_grid_thw)

        src_item = input_tokens
        if not second_per_grid_ts:
            second_per_grid_ts = [1] * video_grid_thw.shape[0]
        video_idx = 0
        image_idx = 0
        new_src_item: list[int] = []
        llm_pos_ids_list: list[torch.Tensor] = []

        idx = 0
        while idx < len(src_item):
            new_src_item_len = len(new_src_item)
            start_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            if src_item[idx] not in [
                    video_token_id, image_token_id
            ]:
                new_src_item.append(src_item[idx])
                llm_pos_ids = torch.tensor([start_idx],
                                            dtype=torch.long).expand(3, -1)
                llm_pos_ids_list.append(llm_pos_ids)
            elif src_item[idx] == image_token_id:
                grid_t = image_grid_thw[image_idx][0]
                grid_hs = image_grid_thw[:, 1]
                grid_ws = image_grid_thw[:, 2]
                t_index = (torch.arange(grid_t) * 1 * tokens_per_second).long()
                llm_pos_ids = cls._get_llm_pos_ids_for_vision(
                    start_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                )
                llm_pos_ids_list.append(llm_pos_ids)
                vision_seqlen = image_grid_thw[image_idx].prod() // (
                    spatial_merge_size**2)
                new_src_item.extend([image_token_id] * vision_seqlen)
                image_idx += 1
            else:
                # Processing video token position
                # Get the grid information of the current video
                T = video_grid_thw[video_idx][0].item()
                H = video_grid_thw[video_idx][1].item()
                W = video_grid_thw[video_idx][2].item()
                llm_H = H // spatial_merge_size
                llm_W = W // spatial_merge_size
                tokens_per_frame = llm_H * llm_W
                # Get timestamps (one t value per frame)
                t_index_all = (torch.arange(T)).long()
                # Calculate the current starting position
                start_pos = llm_pos_ids_list[-1].max().item() + 1 if llm_pos_ids_list else 0
                current_pos = start_pos
                # frame by frame processing
                final_frame_time = T - 1 # Record the order of the last frame
                for t in range(T):
                    # 1. Calculate the left placeholder position of the first frame, skip
                    if t != 0:
                        new_src_item.append(vision_start_token_id) # For looping, count
                        bot_pos = torch.full((3, 1), current_pos, dtype=torch.long)
                        llm_pos_ids_list.append(bot_pos)
                        current_pos += 1
                    # 2. Video tokens for frame t
                    # Construct a single frame of (t, h, w)
                    grid_h = torch.arange(llm_H).view(-1, 1).expand(-1, llm_W).flatten()
                    grid_w = torch.arange(llm_W).view(1, -1).expand(llm_H, -1).flatten()
                    # Here we don't add current_pos to h/w, just keep the original (t, h, w)
                    frame_pos = torch.stack([
                        torch.full_like(grid_h, 0, dtype=torch.long),      # t
                        grid_h,                                            # h
                        grid_w                                             # w
                    ])  # shape: (3, tokens_per_frame)
                    frame_pos_with_offset = frame_pos + current_pos # Current frame position offset
                    new_src_item.extend([video_token_id] * tokens_per_frame) # For looping, count
                    llm_pos_ids_list.append(frame_pos_with_offset)
                    current_pos += max(llm_H, llm_W)
                    # 3. Calculate the right placeholder position of the last frame and skip it
                    if t != final_frame_time:
                        new_src_item.append(vision_end_token_id) # For looping, count
                        eot_pos = torch.full((3, 1), current_pos, dtype=torch.long)
                        llm_pos_ids_list.append(eot_pos)
                        current_pos += 1
                video_idx += 1
            # move to the next token
            idx += len(new_src_item) - new_src_item_len

        llm_positions = torch.cat(llm_pos_ids_list, dim=1)
        mrope_position_delta = torch.cat(llm_pos_ids_list, dim=1).max() + 1 - len(src_item)
        llm_positions = llm_positions[:, context_len:seq_len]

        return llm_positions, mrope_position_delta

    @staticmethod
    def _get_llm_pos_ids_for_vision(
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: list[int],
        grid_hs: torch.Tensor,
        grid_ws: torch.Tensor,
    ) -> torch.Tensor:
        llm_pos_ids_list = []
        llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
        llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
        h_index = (torch.arange(llm_grid_h).view(1, -1, 1).expand(
            len(t_index), -1, llm_grid_w).flatten())
        w_index = (torch.arange(llm_grid_w).view(1, 1, -1).expand(
            len(t_index), llm_grid_h, -1).flatten())
        t_index_tensor = torch.Tensor(t_index).to(llm_grid_h.device).view(
            -1, 1).expand(-1, llm_grid_h * llm_grid_w).long().flatten()
        _llm_pos_ids = torch.stack([t_index_tensor, h_index, w_index])
        llm_pos_ids_list.append(_llm_pos_ids + start_idx)
        llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
        return llm_pos_ids
