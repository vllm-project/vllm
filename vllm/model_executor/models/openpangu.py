# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import typing
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ParallelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import (
    Attention,
    StaticSinkAttention,
)
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mhc import (
    HCHeadOp,
    MHCPostOp,
    MHCPreOp,
)
from vllm.model_executor.layers.mla import (
    MLAModules,
    MultiHeadLatentAttentionWrapper,
    StaticSinkMultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import (
    Indexer as _DeepseekIndexer,
)
from vllm.model_executor.models.deepseek_v2 import (
    _try_load_fp8_indexer_wk,
)
from vllm.model_executor.models.interfaces import (
    MixtureOfExperts,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.utils.torch_utils import (
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn_diffkv import FlashAttentionDiffKVBackend
from vllm.v1.kv_cache_interface import SlidingWindowMomeSpec


def check_ffn_act_fn(act_fn: str):
    if act_fn != "silu":
        raise ValueError(
            f"Unsupported activation: {act_fn}. Only silu is supported for now."
        )


class PanguSinkAttentionBase:
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        output_dim = getattr(param, "output_dim", None)
        is_sharded_weight = getattr(param, "is_sharded_weight", False)
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow
        is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit
        # Special case for GGUF
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()
        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, nn.UninitializedParameter):
            final_shape = list(loaded_weight.shape)
            if output_dim is not None:
                tp_size = getattr(self, "tp_size", 1)
                assert final_shape[output_dim] % tp_size == 0
                final_shape[output_dim] = final_shape[output_dim] // tp_size
            param.materialize(final_shape, dtype=loaded_weight.dtype)
        param_data = param.data
        if output_dim is not None and not is_sharded_weight:
            shard_size = param_data.shape[output_dim]
            tp_rank = getattr(self, "tp_rank", 0)
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


@CustomOp.register("MomeAttention")
class MomeAttention(MambaBase, CustomOp):
    """
    MoME attention layer with vLLM KV cache management.
    Handles 3-part convolution state (q, kv, o).
    """

    def __init__(
        self,
        kernel_size: int,
        num_spec_tokens: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        num_heads: int,
        num_local_heads: int,
        v_head_dim: int,
        vllm_config: VllmConfig,
        prefix: str,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_spec_tokens = num_spec_tokens
        self.dtype = vllm_config.model_config.dtype
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.num_local_heads = num_local_heads
        self.v_head_dim = v_head_dim
        self.o_dim = num_local_heads * v_head_dim
        self.cache_head_size = q_lora_rank + kv_lora_rank + self.o_dim
        self.prefix = prefix
        self.kv_cache_dtype = vllm_config.cache_config.cache_dtype

        # 3 Conv1d weights for q, kv, o
        # These names match the original weights in the checkpoint
        self.qa_conv = nn.Conv1d(
            q_lora_rank,
            q_lora_rank,
            kernel_size,
            groups=q_lora_rank,
            bias=False,
            dtype=self.dtype,
        )
        self.compresskv_conv = nn.Conv1d(
            kv_lora_rank,
            kv_lora_rank,
            kernel_size,
            groups=kv_lora_rank,
            bias=False,
            dtype=self.dtype,
        )
        self.o_conv = nn.Conv1d(
            self.o_dim,
            self.o_dim,
            kernel_size,
            groups=self.o_dim,
            bias=False,
            dtype=self.dtype,
        )

        # Set weight loading attributes
        set_weight_attrs(self.qa_conv.weight, {"weight_loader": self.weight_loader})
        set_weight_attrs(
            self.compresskv_conv.weight, {"weight_loader": self.weight_loader}
        )
        set_weight_attrs(
            self.o_conv.weight,
            {"weight_loader": self.weight_loader, "output_parallel": True},
        )

        # In v1, the KV cache tensors are set by the ModelRunner
        self.kv_cache = (torch.tensor([]), torch.tensor([]), torch.tensor([]))

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @property
    def mamba_type(self) -> str:
        return "mome"

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return ((self.q_lora_rank,), (self.kv_lora_rank,), (self.o_dim,))

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return (self.qa_conv.weight.dtype,) * 3

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> SlidingWindowMomeSpec:
        kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        # FIXME(runze): block_size and sliding_window are hardcoded to be 8 now;
        # make it general later
        return SlidingWindowMomeSpec(
            block_size=8,
            num_kv_heads=1,
            head_size=self.cache_head_size,
            dtype=kv_cache_dtype,
            sliding_window=8,
            cache_dtype_str=vllm_config.cache_config.cache_dtype,
            alignment=576,
            component_dims=(self.q_lora_rank, self.kv_lora_rank, self.o_dim),
        )

    def get_attn_backend(self) -> type:
        from vllm.v1.attention.backends.mome_attn import MomeAttentionBackend

        return MomeAttentionBackend

    def forward(self, hidden_states: torch.Tensor, state_indice: int) -> torch.Tensor:
        output = torch.empty_like(hidden_states)
        torch.ops.vllm.mome_attention_fused_op(
            hidden_states,
            self.qa_conv.weight,
            self.compresskv_conv.weight,
            self.o_conv.weight,
            self.q_lora_rank,
            self.kv_lora_rank,
            self.o_dim,
            self.kernel_size,
            state_indice,
            self.prefix,
            output,
        )
        return output

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        tp_rank = get_tensor_model_parallel_rank()
        output_parallel = getattr(param, "output_parallel", False)
        param_data = param.data
        if output_parallel:
            shard_size = param_data.shape[0]
            loaded_weight = loaded_weight.narrow(0, tp_rank * shard_size, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


def _select_mome_conv_params(
    state_indice: int,
    q_conv_weight: torch.Tensor,
    compresskv_conv_weight: torch.Tensor,
    o_conv_weight: torch.Tensor,
    q_lora_rank: int,
    kv_lora_rank: int,
    o_dim: int,
    kernel_size: int,
) -> tuple[torch.Tensor, int]:
    if state_indice == 0:
        conv_weight = q_conv_weight
        hidden_size = q_lora_rank
    elif state_indice == 1:
        conv_weight = compresskv_conv_weight
        hidden_size = kv_lora_rank
    else:
        conv_weight = o_conv_weight
        hidden_size = o_dim
    return conv_weight.view(hidden_size, kernel_size), hidden_size


def mome_attention_fused_op(
    hidden_states: torch.Tensor,
    q_conv_weight: torch.Tensor,
    compresskv_conv_weight: torch.Tensor,
    o_conv_weight: torch.Tensor,
    q_lora_rank: int,
    kv_lora_rank: int,
    o_dim: int,
    kernel_size: int,
    state_indice: int,
    layer_name: str,
    output: torch.Tensor,
) -> None:
    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    if forward_context.attn_metadata is None:
        output.fill_(0)
        return

    mome_metadata = forward_context.attn_metadata[layer_name]
    self_kv_cache = layer.kv_cache

    def _get_request_state_indices(state_indices: torch.Tensor) -> torch.Tensor:
        if state_indices.ndim > 1:
            return state_indices[:, 0]
        return state_indices

    conv_weight, hidden_size = _select_mome_conv_params(
        state_indice,
        q_conv_weight,
        compresskv_conv_weight,
        o_conv_weight,
        q_lora_rank,
        kv_lora_rank,
        o_dim,
        kernel_size,
    )
    conv_weight = conv_weight.to(hidden_states.dtype)
    conv_state = self_kv_cache[state_indice]
    conv_state = conv_state.transpose(-1, -2)

    output_chunks = []

    num_decode_tokens = mome_metadata.num_decode_tokens
    num_decodes = mome_metadata.num_decodes
    if num_decodes > 0:
        decode_hidden_states = hidden_states[:num_decodes].clone()
        decode_state_indices = _get_request_state_indices(
            mome_metadata.state_indices_tensor_d[:num_decodes]
        )
        decode_output = causal_conv1d_update(
            decode_hidden_states,
            conv_state,
            conv_weight,
            bias=None,
            activation=None,
            conv_state_indices=decode_state_indices,
        )
        output_chunks.append(decode_output)

    num_prefills = mome_metadata.num_prefills
    if num_prefills > 0:
        query_start_loc = mome_metadata.query_start_loc_p
        assert query_start_loc is not None
        prefill_hidden_states = hidden_states[
            num_decode_tokens : num_decode_tokens + mome_metadata.num_prefill_tokens
        ]
        prefill_state_indices = _get_request_state_indices(
            mome_metadata.state_indices_tensor_p[:num_prefills]
        )
        prefill_output = causal_conv1d_fn(
            prefill_hidden_states.transpose(0, 1),
            conv_weight,
            bias=None,
            activation=None,
            conv_states=conv_state,
            has_initial_state=mome_metadata.has_initial_states_p,
            cache_indices=prefill_state_indices,
            query_start_loc=query_start_loc,
            metadata=mome_metadata,
            zero_initial_state_output=True,
        ).transpose(0, 1)
        output_chunks.append(prefill_output)

    if len(output_chunks) == 0:
        output.fill_(0)
        return
    if len(output_chunks) == 1:
        result = output_chunks[0]
    else:
        result = torch.cat(output_chunks, dim=0)

    output[: result.shape[0]] = result


def mome_attention_fake(
    hidden_states: torch.Tensor,
    q_conv_weight: torch.Tensor,
    compresskv_conv_weight: torch.Tensor,
    o_conv_weight: torch.Tensor,
    q_lora_rank: int,
    kv_lora_rank: int,
    o_dim: int,
    kernel_size: int,
    state_indice: int,
    layer_name: str,
    output: torch.Tensor,
) -> None:
    return


direct_register_custom_op(
    op_name="mome_attention_fused_op",
    op_func=mome_attention_fused_op,
    mutates_args=["output"],
    fake_impl=mome_attention_fake,
)


class PanguIndexer(_DeepseekIndexer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_norm = RMSNorm(self.head_dim, self.config.rms_norm_eps)
        self.deepgemm_n_head = 32 if self.n_head <= 32 else 64

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions,
        rotary_emb,
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)

        if current_platform.is_rocm():
            kw, _ = self.wk_weights_proj(hidden_states)
            k = kw[:, : self.head_dim]
            weights = kw[:, self.head_dim :]

            k = self.k_norm(k)

            rotary_emb(
                positions, q[..., : self.rope_dim], k[..., : self.rope_dim].unsqueeze(1)
            )
        else:
            q_pe, q_nope = torch.split(
                q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
            )
            kw, _ = self.wk_weights_proj(hidden_states)
            k = kw[:, : self.head_dim]
            weights = kw[:, self.head_dim :]

            k = self.k_norm(k)
            k_pe, k_nope = torch.split(
                k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
            )

            q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
            q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
            k_pe = k_pe.reshape(-1, 1, self.rope_dim)

        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe.reshape(-1, self.rope_dim), k_nope], dim=-1)

        q = q.view(-1, self.head_dim)
        q_fp8, q_scale = per_token_group_quant_fp8(
            q,
            self.quant_block_size,
            column_major_scales=False,
            use_ue8m0=self.scale_fmt is not None,
        )
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1)

        weights = (
            weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head**-0.5
        )
        weights = weights.squeeze(-1)

        if self.deepgemm_n_head != self.n_head:
            pad_heads = self.deepgemm_n_head - self.n_head
            q_fp8 = torch.cat(
                [
                    q_fp8,
                    q_fp8.new_zeros(q_fp8.shape[0], pad_heads, q_fp8.shape[2]),
                ],
                dim=1,
            )
            weights = torch.cat(
                [
                    weights,
                    weights.new_zeros(weights.shape[0], pad_heads),
                ],
                dim=1,
            )

        return self.indexer_op(hidden_states, q_fp8, k, weights)


class OpenPanguMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        reduce_results: bool = True,
        is_sequence_parallel=False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )

        check_ffn_act_fn(hidden_act)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)[0]))[0]


class OpenPanguMoE(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tp_group().rank_in_group

        self.routed_scaling_factor = config.routed_scaling_factor
        self.ep_group = get_ep_group().device_group
        self.ep_rank = self.ep_group.rank()
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe
        check_ffn_act_fn(config.hidden_act)

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        if (
            hasattr(config, "router_enable_expert_bias")
            and config.router_enable_expert_bias
        ):
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32)
            )
        else:
            self.gate.e_score_correction_bias = None

        # Load balancing settings.
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = OpenPanguMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            prefix=f"{prefix}.experts",
            scoring_func="sigmoid",
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=True,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        router_logits, _ = self.gate(hidden_states)

        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.view(num_tokens, hidden_dim)


class OpenPanguMLAAttention(PanguSinkAttentionBase, nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        vllm_config: VllmConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.tp_size = get_tensor_model_parallel_world_size()
        if num_heads % self.tp_size != 0:
            raise ValueError(
                f"num_heads {num_heads} is not divisible by tp_size {self.tp_size}."
            )
        self.num_local_heads = num_heads // self.tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.prefix = prefix

        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True,
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )

        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )

        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # TODO: remove hard coding
        set_default_rope_theta(config, default_theta=10000)
        rope_parameters = {
            "rope_theta": config.rope_parameters["rope_theta"],
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": max_position_embeddings,
            "type": "yarn",
            "rope_type": "deepseek_yarn",
        }
        self.rope_interleaved = getattr(config, "rope_interleaved", True)
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=(not self.rope_interleaved),
        )
        self.param_sink_number = getattr(config, "param_sink_number", 0)
        self.param_sink_with_value = getattr(config, "param_sink_with_value", False)
        # SWA
        layer_idx = extract_layer_index(prefix)
        is_dsa = hasattr(config, "index_topk") and (
            not hasattr(config, "dsa_layers") or layer_idx in config.dsa_layers
        )
        is_sliding = (
            hasattr(config, "sliding_window") or hasattr(config, "sliding_window_list")
        ) and (not hasattr(config, "swa_layers") or layer_idx in config.swa_layers)
        if is_sliding:
            if hasattr(config, "sliding_window_list"):
                sliding_window = config.sliding_window_list[
                    config.swa_layers.index(layer_idx)
                ]
            else:
                sliding_window = config.sliding_window
        else:
            sliding_window = None
        if is_dsa:
            self.indexer_rope_emb = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=rope_parameters,
                is_neox_style=(not self.rope_interleaved),
            )
            self.indexer = PanguIndexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
            )
        else:
            self.indexer_rope_emb = None
            self.indexer = None
        self.sliding_window = sliding_window
        # MOME
        if getattr(config, "use_mome", False):
            spec_token_num = 0
            if vllm_config.speculative_config:
                spec_token_num = vllm_config.speculative_config.num_speculative_tokens
            self.mome_attn = MomeAttention(
                kernel_size=config.router_sliding_window,
                num_spec_tokens=spec_token_num,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=self.kv_lora_rank,
                num_heads=self.num_heads,
                num_local_heads=self.num_local_heads,
                v_head_dim=self.v_head_dim,
                vllm_config=vllm_config,
                prefix=f"{prefix}.mome_attn",
            )
        else:
            self.mome_attn = None
        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj
            if self.q_lora_rank is not None
            else None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa
            if self.q_lora_rank is None
            else None,
            q_a_layernorm=self.q_a_layernorm if self.q_lora_rank is not None else None,
            q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else None,
            indexer=self.indexer,
            indexer_rotary_emb=self.indexer_rope_emb,
            is_sparse=is_dsa,
            topk_indices_buffer=topk_indices_buffer,
        )
        if self.param_sink_number == 0:
            self.mla_attn = MultiHeadLatentAttentionWrapper(
                self.hidden_size,
                self.num_local_heads,
                self.scaling,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                self.v_head_dim,
                self.q_lora_rank,
                self.kv_lora_rank,
                mla_modules,
                cache_config,
                quant_config,
                prefix,
            )
        else:
            self.mla_attn = StaticSinkMultiHeadLatentAttentionWrapper(
                self.hidden_size,
                self.num_local_heads,
                self.scaling,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                self.v_head_dim,
                self.q_lora_rank,
                self.kv_lora_rank,
                mla_modules,
                cache_config,
                quant_config,
                prefix,
                sink_len=self.param_sink_number,
                sliding_window=sliding_window,
                mome_attn=self.mome_attn,
            )
            self.param_sink_k_pe = torch.nn.Parameter(
                torch.empty(
                    (
                        self.param_sink_number,
                        self.qk_rope_head_dim,
                    ),
                    device=current_platform.current_device(),
                    dtype=config.torch_dtype,
                )
            )
            set_weight_attrs(
                self.param_sink_k_pe,
                {
                    "output_dim": 1,
                    "weight_loader": self.weight_loader,
                },
            )
            if self.param_sink_with_value:
                self.param_sink_compressed_kv = torch.nn.Parameter(
                    torch.empty(
                        (
                            self.param_sink_number,
                            self.kv_lora_rank,
                        ),
                        device=current_platform.current_device(),
                        dtype=config.torch_dtype,
                    )
                )
                set_weight_attrs(
                    self.param_sink_compressed_kv,
                    {
                        "output_dim": 1,
                        "weight_loader": self.weight_loader,
                    },
                )
            else:
                self.param_sink_compressed_kv = torch.zeros(
                    (
                        self.param_sink_number,
                        self.kv_lora_rank,
                    ),
                    device=current_platform.current_device(),
                    dtype=config.torch_dtype,
                )
        # To enable dummy run with out weight
        self.post_weight_load()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states)

    def post_weight_load(self) -> None:
        if getattr(self, "param_sink_number", 0) > 0:
            if getattr(self, "kv_a_layernorm", None) is not None:
                param_sink_compressed_kv = self.kv_a_layernorm(
                    self.param_sink_compressed_kv
                )
            else:
                param_sink_compressed_kv = self.param_sink_compressed_kv
            self.mla_attn.mla_attn.update_sink_kv(
                self.param_sink_k_pe, param_sink_compressed_kv
            )


class OpenPanguEmbeddedAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        if self.total_num_heads % tp_size != 0:
            raise ValueError(
                f"total_num_heads {self.total_num_heads} "
                f"is not divisible by tp_size {tp_size}."
            )
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads > tp_size and self.total_num_kv_heads % tp_size != 0:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel ranks.
            raise ValueError(
                "Number of KV heads is greater than TP size, "
                f"but total_num_kv_heads {self.total_num_kv_heads} "
                f"is not divisible by tp_size {tp_size}."
            )
        elif (
            self.total_num_kv_heads < tp_size and tp_size % self.total_num_kv_heads != 0
        ):
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel ranks.
            raise ValueError(
                f"Number of KV heads is less than TP size, but tp_size {tp_size} "
                f"is not divisible by total_num_kv_heads {self.total_num_kv_heads}."
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_size // self.total_num_heads
        self.head_dim = head_dim
        self.qk_nope_dim = getattr(config, "qk_nope_dim", None)
        self.qk_rope_dim = getattr(config, "qk_rope_dim", self.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self._init_rotary_emb(config, quant_config=quant_config)

        if hasattr(config, "interleaved_sliding_window"):
            interleaved_sliding_window = config.interleaved_sliding_window
            if isinstance(interleaved_sliding_window, int):
                sliding_window = interleaved_sliding_window
            elif isinstance(interleaved_sliding_window, list):
                sw_idx = layer_idx % len(interleaved_sliding_window)
                sliding_window = interleaved_sliding_window[sw_idx]
            else:
                raise ValueError(
                    f"{type(interleaved_sliding_window)} "
                    "for interleaved_sliding_window is not supported."
                )
        else:
            sliding_window = None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def _init_rotary_emb(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type == "PanguEmbedded":
            is_neox_style = False

        rope_parameters = config.rope_parameters or {}
        if rope_parameters is not None and rope_parameters.get(
            "mrope_interleaved", False
        ):
            rope_parameters["rope_type"] = "openpangu"

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=is_neox_style,
        )


class OpenPanguSinkAttention(PanguSinkAttentionBase, nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.total_num_heads = num_heads
        if self.total_num_heads % self.tp_size != 0:
            raise ValueError(
                f"total_num_heads {self.total_num_heads} "
                f"is not divisible by tp_size {self.tp_size}."
            )
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = num_kv_heads
        if (
            self.total_num_kv_heads > self.tp_size
            and self.total_num_kv_heads % self.tp_size != 0
        ):
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel ranks.
            raise ValueError(
                "Number of KV heads is greater than TP size, "
                f"but total_num_kv_heads {self.total_num_kv_heads} "
                f"is not divisible by tp_size {self.tp_size}."
            )
        elif self.total_num_kv_heads < self.tp_size:
            # TODO: Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel ranks.
            raise ValueError(
                f"Number of KV heads {self.total_num_kv_heads} is less than "
                f"TP size {self.tp_size}, KV heads replication is not support yet."
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.qk_nope_dim = getattr(config, "qk_nope_dim", None)
        self.qk_rope_dim = getattr(config, "qk_rope_dim", None)
        self.v_channels = getattr(config, "v_channels", None)
        self.head_dim = self.qk_rope_dim + self.qk_nope_dim
        self.q_size = self.num_heads * self.head_dim
        self.k_size = self.num_kv_heads * self.head_dim
        self.v_size = self.num_kv_heads * self.v_channels
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.param_sink_number = getattr(config, "param_sink_number", 0)
        self.param_sink_with_value = getattr(config, "param_sink_with_value", False)
        self.param_sink_scalar = getattr(config, "param_sink_scalar", None)
        self.param_sink_of_head_num = getattr(config, "param_sink_of_head_dim", False)

        self.qkv_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[
                self.q_size * self.tp_size,
                self.k_size * self.tp_size,
                self.v_size * self.tp_size,
            ],
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.v_channels,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.k_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self._init_rotary_emb(
            config, rope_parameters=rope_parameters, quant_config=quant_config
        )

        if hasattr(config, "interleaved_sliding_window"):
            interleaved_sliding_window = config.interleaved_sliding_window
            if isinstance(interleaved_sliding_window, int):
                sliding_window = interleaved_sliding_window
            elif isinstance(interleaved_sliding_window, list):
                sw_idx = layer_idx % len(interleaved_sliding_window)
                sliding_window = interleaved_sliding_window[sw_idx]
            else:
                raise ValueError(
                    f"{type(interleaved_sliding_window)} "
                    "for interleaved_sliding_window is not supported."
                )
        else:
            sliding_window = None

        FlashAttentionDiffKVBackend.set_head_size_v(self.v_channels)
        self.attn = StaticSinkAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            sink_len=self.param_sink_number,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
            attn_backend=FlashAttentionDiffKVBackend,
            head_size_v=self.v_channels,
        )

        if self.param_sink_number > 0:
            self.param_sink_key = torch.nn.Parameter(
                torch.empty(
                    (
                        self.param_sink_number,
                        self.num_kv_heads,
                        self.head_dim,
                    ),
                    device=current_platform.current_device(),
                    dtype=config.torch_dtype,
                )
            )
            set_weight_attrs(
                self.param_sink_key,
                {
                    "output_dim": 1,
                    "weight_loader": self.weight_loader,
                },
            )

            if self.param_sink_with_value:
                self.param_sink_value = torch.nn.Parameter(
                    torch.empty(
                        (
                            self.param_sink_number,
                            self.num_kv_heads,
                            self.v_channels,
                        ),
                        device=current_platform.current_device(),
                        dtype=config.torch_dtype,
                    )
                )
                set_weight_attrs(
                    self.param_sink_value,
                    {
                        "output_dim": 1,
                        "weight_loader": self.weight_loader,
                    },
                )
            else:
                self.param_sink_value = torch.zeros(
                    (
                        self.param_sink_number,
                        self.num_kv_heads,
                        self.v_channels,
                    ),
                    device=current_platform.current_device(),
                    dtype=config.torch_dtype,
                )
        # To enable dummy run with out weight
        self.post_weight_load()

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        output_dim = getattr(param, "output_dim", None)

        is_sharded_weight = getattr(param, "is_sharded_weight", False)
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow
        is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

        # Special case for GGUF
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, nn.UninitializedParameter):
            final_shape = list(loaded_weight.shape)
            if output_dim is not None:
                assert final_shape[output_dim] % self.tp_size == 0
                final_shape[output_dim] = final_shape[output_dim] // self.tp_size
            param.materialize(final_shape, dtype=loaded_weight.dtype)

        param_data = param.data
        if output_dim is not None and not is_sharded_weight:
            shard_size = param_data.shape[output_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        k = self.k_layernorm(k.view(-1, self.num_kv_heads, self.head_dim))
        q, k = self.rotary_emb(positions, q, k)

        q = q.view(-1, self.q_size)
        k = k.view(-1, self.k_size)

        attn_output = self.attn(
            q,
            k,
            v,
            output_shape=torch.Size(
                [q.shape[0], q.shape[1] // self.head_dim * self.v_channels]
            ),
        )
        output, _ = self.o_proj(attn_output)
        return output

    def _init_rotary_emb(
        self,
        config: PretrainedConfig,
        rope_parameters: dict[str, Any],
        quant_config: QuantizationConfig | None,
    ) -> None:
        rope_parameters["partial_rotary_factor"] = self.qk_rope_dim / self.head_dim
        is_neox_style = True
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=is_neox_style,
        )

    def post_weight_load(self) -> None:
        if hasattr(self, "k_layernorm") and self.k_layernorm is not None:
            param_sink_key = self.k_layernorm(self.param_sink_key)
        else:
            param_sink_key = self.param_sink_key

        self.attn.update_sink_kv(param_sink_key, self.param_sink_value)


@CustomOp.register("mHCModule")
class mHCModule(CustomOp):
    def __init__(
        self,
        config: PretrainedConfig,
        merge_layer_only_pre=False,
        prefix: str = "",
    ):
        super().__init__()
        self.num_stream = config.mhc_num_stream
        self.hidden_size = config.hidden_size
        self.merge_layer_only_pre = merge_layer_only_pre

        if not self.merge_layer_only_pre:
            phi_output_hidden_size = (self.num_stream + 2) * self.num_stream
            self.branch_alpha = nn.Parameter(torch.empty(3, dtype=torch.float32))
            self.branch_beta = nn.Parameter(
                torch.empty(
                    self.num_stream * (self.num_stream + 2), dtype=torch.float32
                )
            )
        else:
            phi_output_hidden_size = self.num_stream
            self.branch_alpha_pre = nn.Parameter(torch.empty(1, dtype=torch.float32))
            self.branch_beta_pre = nn.Parameter(
                torch.empty(self.num_stream, dtype=torch.float32)
            )
        self.phi = ReplicatedLinear(
            self.hidden_size * self.num_stream,
            phi_output_hidden_size,
            bias=False,
            prefix=f"{prefix}.phi",
        )
        self.mhc_use_gamma = config.mhc_use_gamma
        self.hc_eps = 1e-6
        self.norm_eps = config.rms_norm_eps
        self.mhc_recur_norm = config.mhc_recur_norm
        self.hc_post_alpha = 2.0
        if self.mhc_use_gamma:
            self.norm_gamma = nn.Parameter(
                torch.empty(self.hidden_size * self.num_stream, dtype=torch.bfloat16)
            )
        self.mhc_pre = MHCPreOp()
        self.mhc_post = MHCPostOp()

    def post_weight_load(self) -> None:
        if self.mhc_use_gamma:
            self.phi_weight = (self.phi.weight * self.norm_gamma).contiguous().float()

    def hc_pre(
        self,
        x: torch.Tensor,
    ):
        x = x.view(-1, self.num_stream, self.hidden_size)
        fn = (
            self.phi_weight if hasattr(self, "phi_weight") else self.phi.weight
        ).float()
        post_mix, res_mix, layer_input = self.mhc_pre(
            residual=x,
            fn=fn,
            hc_scale=self.branch_alpha,
            hc_base=self.branch_beta,
            rms_eps=self.norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.hc_post_alpha,
            sinkhorn_repeat=self.mhc_recur_norm,
        )
        return layer_input, post_mix, res_mix

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
    ):
        residual = residual.view(-1, self.num_stream, self.hidden_size)
        res = self.mhc_post(x, residual, h_post, h_res)
        res = res.view(-1, self.num_stream * self.hidden_size)
        return res


class OpenPanguDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        vllm_config: VllmConfig,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        self.use_mla = (
            hasattr(config, "qk_nope_head_dim")
            and hasattr(config, "qk_rope_head_dim")
            and hasattr(config, "v_head_dim")
            and hasattr(config, "kv_lora_rank")
        )
        self.use_sink_attention = (
            hasattr(config, "param_sink_number") and config.param_sink_number > 0
        )
        self.router_sliding_window = getattr(config, "router_sliding_window", 0)
        if self.use_mla:
            self.self_attn = OpenPanguMLAAttention(
                config=config,
                vllm_config=vllm_config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=(
                    config.q_lora_rank if hasattr(config, "q_lora_rank") else None
                ),
                kv_lora_rank=config.kv_lora_rank,
                max_position_embeddings=max_position_embeddings,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
            )
        elif self.use_sink_attention:
            attention_bias = getattr(config, "attention_bias", False) or getattr(
                config, "bias", False
            )
            bias_o_proj = attention_bias
            if hasattr(config, "qkv_bias"):
                attention_bias = config.qkv_bias
            if getattr(config, "is_causal", True):
                attn_type = AttentionType.DECODER
            else:
                raise ValueError(
                    f"is_causal={config.is_causal} is not support "
                    "for attention with sink"
                )
            rope_parameters = getattr(config, "rope_scaling", None)
            if rope_parameters is None:
                rope_parameters = {
                    "rope_type": "default",
                    "rope_theta": config.rope_theta,
                }
            self.self_attn = OpenPanguSinkAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=getattr(
                    config, "num_key_value_heads", config.num_attention_heads
                ),
                rope_parameters=rope_parameters,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                bias=attention_bias,
                bias_o_proj=bias_o_proj,
                cache_config=cache_config,
                prefix=f"{prefix}.self_attn",
                attn_type=attn_type,
            )
        else:
            attention_bias = getattr(config, "attention_bias", False) or getattr(
                config, "bias", False
            )
            bias_o_proj = attention_bias
            if hasattr(config, "qkv_bias"):
                attention_bias = config.qkv_bias
            # By default, PanguEmbedded uses causal attention
            # as it is a decoder-only model.
            # You can override the HF config with `is_causal=False` to enable
            # bidirectional attention, which is used in some embedding models
            if getattr(config, "is_causal", True):
                attn_type = AttentionType.DECODER
            else:
                attn_type = AttentionType.ENCODER_ONLY
            self.self_attn = OpenPanguEmbeddedAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=getattr(
                    config, "num_key_value_heads", config.num_attention_heads
                ),
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                bias=attention_bias,
                bias_o_proj=bias_o_proj,
                cache_config=cache_config,
                prefix=f"{prefix}.self_attn",
                attn_type=attn_type,
            )

        if (
            getattr(config, "n_routed_experts", None) is not None
            and layer_idx >= config.first_k_dense_replace
        ):
            self.mlp = OpenPanguMoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = OpenPanguMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                bias=getattr(config, "mlp_bias", False),
                prefix=f"{prefix}.mlp",
            )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.num_hidden_layers = config.num_hidden_layers
        self.first_k_dense_replace = getattr(
            config, "first_k_dense_replace", self.num_hidden_layers
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_mhc = getattr(config, "use_mhc", False)
        self.sandwich_norm = getattr(config, "sandwich_norm", False)
        if self.use_mhc or (not self.use_mhc and self.sandwich_norm):
            self.pre_mlp_layernorm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        if not self.use_mhc or (self.use_mhc and self.sandwich_norm):
            self.post_attention_layernorm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        if self.sandwich_norm:
            self.post_mlp_layernorm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        block_post_layernorm_hidden_size = config.hidden_size
        mtp_idx = self.layer_idx - self.num_hidden_layers
        mtp_layer_num = getattr(config, "num_nextn_predict_layers", -1)
        self.is_mtp_layer = mtp_idx >= 0 and mtp_idx < mtp_layer_num
        self.use_mhc = getattr(config, "use_mhc", False)
        if self.use_mhc and not self.is_mtp_layer:
            self.attn_mhc_module = mHCModule(
                config=config,
                prefix=f"{prefix}.attn_mhc_module",
            )
            self.mlp_mhc_module = mHCModule(
                config=config,
                prefix=f"{prefix}.mlp_mhc_module",
            )
            block_post_layernorm_hidden_size *= getattr(config, "mhc_num_stream", 4)
        self.has_block_post_layernorm = layer_idx in getattr(
            config, "block_post_layernorm_idx", []
        )
        if self.has_block_post_layernorm:
            self.block_post_layernorm = RMSNorm(
                block_post_layernorm_hidden_size, eps=config.rms_norm_eps
            )
        self.layer_name = prefix

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.use_mhc and not self.is_mtp_layer:
            return self.forward_mhc(positions, hidden_states, residual)
        else:
            return self.forward_normal(positions, hidden_states, residual)

    def forward_normal(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        if (
            self.routed_scaling_factor is not None
            and hidden_states.dtype == torch.float16
        ):
            # Fix FP16 overflow
            # We scale both hidden_states and residual before
            # rmsnorm, and rmsnorm result would not affect by scale.
            hidden_states *= 1.0 / self.routed_scaling_factor
            if self.layer_idx == 0:
                # The residual is shared by all layers, we only scale it on
                # first layer.
                residual *= 1.0 / self.routed_scaling_factor

        if self.sandwich_norm:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states, residual = self.pre_mlp_layernorm(hidden_states, residual)
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )

        # Fully Connected
        hidden_states = self.mlp(hidden_states)

        if (
            self.routed_scaling_factor is not None
            and isinstance(self.mlp, OpenPanguMLP)
            and hidden_states.dtype == torch.float16
        ):
            hidden_states *= 1.0 / self.routed_scaling_factor

        if self.sandwich_norm:
            hidden_states = self.post_mlp_layernorm(hidden_states)

        if self.has_block_post_layernorm:
            hidden_states, _ = self.block_post_layernorm(hidden_states, residual)
            residual = None

        return hidden_states, residual

    def forward_mhc(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states, h_post, h_res = self.attn_mhc_module.hc_pre(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        if self.sandwich_norm:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.attn_mhc_module.hc_post(
            hidden_states, residual, h_post, h_res
        )
        residual = hidden_states
        hidden_states, h_post, h_res = self.mlp_mhc_module.hc_pre(hidden_states)
        hidden_states = self.pre_mlp_layernorm(hidden_states)

        # Fully Connected
        hidden_states = self.mlp(hidden_states)

        if self.sandwich_norm:
            hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = self.mlp_mhc_module.hc_post(
            hidden_states, residual, h_post, h_res
        )
        if self.has_block_post_layernorm:
            hidden_states = self.block_post_layernorm(hidden_states)

        return hidden_states, None


@support_torch_compile
class OpenPanguModel(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        eplb_config = vllm_config.parallel_config.eplb_config
        self.config = config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        if hasattr(config, "index_topk"):
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=current_platform.device_type,
            )
        else:
            topk_indices_buffer = None

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: OpenPanguDecoderLayer(
                config, prefix, vllm_config, topk_indices_buffer
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        self.use_mhc = getattr(config, "use_mhc", False)
        if self.use_mhc:
            self.num_stream = getattr(config, "mhc_num_stream", 4)
            self.merge_mhc_module = mHCModule(
                config=config,
                merge_layer_only_pre=True,
                prefix=f"{prefix}.attn_mhc_module",
            )
            self.hc_head_op = HCHeadOp()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def hc_head(self, x) -> torch.Tensor:
        x = x.view(-1, self.num_stream, self.merge_mhc_module.hidden_size)
        res = self.hc_head_op(
            x,
            self.merge_mhc_module.phi_weight
            if hasattr(self.merge_mhc_module, "phi_weight")
            else self.merge_mhc_module.phi.weight.float(),
            self.merge_mhc_module.branch_alpha_pre,
            self.merge_mhc_module.branch_beta_pre,
            self.merge_mhc_module.norm_eps,
            self.merge_mhc_module.hc_eps,
        )
        return res

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
                if self.use_mhc:
                    hidden_states = hidden_states.repeat(1, self.num_stream)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if self.use_mhc:
            hidden_states = self.hc_head(hidden_states)
        else:
            hidden_states = (
                hidden_states + residual if residual is not None else hidden_states
            )

        return hidden_states

    def load_attn_mlp_weight(
        self,
        attn_mlp_replace_mapping: list[tuple[str, str, int]],
        params_dict: dict[str, Any],
        weight_name: str,
        loaded_weight: torch.Tensor,
        loaded_params: set[str],
    ) -> bool:
        for param_name, origin_name, shard_id in attn_mlp_replace_mapping:
            if origin_name not in weight_name or (
                ("mlp.experts." in weight_name) and weight_name not in params_dict
            ):
                continue
            weight_name_mapped = weight_name.replace(origin_name, param_name)
            if (
                param_name == "fused_qkv_a_proj"
                and weight_name_mapped not in params_dict
            ):
                continue
            else:
                weight_name = weight_name_mapped
            if weight_name.endswith(".bias") and weight_name not in params_dict:
                continue
            if is_pp_missing_parameter(weight_name, self):
                continue

            param = params_dict[weight_name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            loaded_params.add(weight_name)
            return True
        return False

    def load_expert_weight(
        self,
        expert_merge_mapping: list[tuple[str, str, int, str]],
        params_dict: dict[str, Any],
        weight_name: str,
        loaded_weight: torch.Tensor,
        loaded_params: set[str],
        flag_dict: dict[str, bool],
    ) -> bool:
        for mapping in expert_merge_mapping:
            param_name, origin_name, expert_id, shard_id = mapping
            if origin_name not in weight_name:
                continue
            flag_dict["is_expert_weight"] = True
            weight_name_mapped = weight_name.replace(origin_name, param_name)
            if is_pp_missing_parameter(weight_name_mapped, self):
                continue
            param = params_dict[weight_name_mapped]
            weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
            success = weight_loader(
                param,
                loaded_weight,
                weight_name_mapped,
                shard_id=shard_id,
                expert_id=expert_id,
                return_success=True,
            )
            if success:
                weight_name = weight_name_mapped
                loaded_params.add(weight_name_mapped)
                return True
        return False

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        attn_mlp_replace_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            (".indexer.wk_weights_proj", ".indexer.wk", 0),
            (".indexer.wk_weights_proj", ".indexer.weights_proj", 1),
        ]
        has_experts = hasattr(self.config, "n_routed_experts")
        if has_experts:
            expert_merge_mapping = FusedMoE.make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.n_routed_experts,
                num_redundant_experts=self.num_redundant_experts,
            )

        _pending_wk_fp8: dict = {}
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "layers" in name:
                layer_idx = int(name.split("layers.")[-1].split(".")[0])
                if layer_idx >= self.config.num_hidden_layers:
                    continue
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            if (
                "layers" in name
                and hasattr(self.config, "num_nextn_predict_layers")
                and (self.config.num_nextn_predict_layers > 0)
            ):
                layer_idx = int(name.split("layers.")[-1].split(".")[0])
                mtp_idx = layer_idx - self.config.num_hidden_layers
                if mtp_idx >= 0 and mtp_idx < self.config.num_nextn_predict_layers:
                    continue  # skip spec decode layers for main model

            if _try_load_fp8_indexer_wk(
                name, loaded_weight, _pending_wk_fp8, params_dict, loaded_params
            ):
                continue

            flag_dict = {"is_expert_weight": False}
            if (
                self.load_attn_mlp_weight(
                    attn_mlp_replace_mapping,
                    params_dict,
                    name,
                    loaded_weight,
                    loaded_params,
                )
                or has_experts
                and self.load_expert_weight(
                    expert_merge_mapping,
                    params_dict,
                    name,
                    loaded_weight,
                    loaded_params,
                    flag_dict,
                )
            ):
                continue
            else:
                if flag_dict["is_expert_weight"]:
                    continue
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name.endswith("e_score_correction_bias"):
                    name = name.replace(
                        "e_score_correction_bias", "gate.e_score_correction_bias"
                    )
                if ".self_attn.qa_conv.weight" in name:
                    name = name.replace(
                        ".self_attn.qa_conv.weight",
                        ".self_attn.mome_attn.qa_conv.weight",
                    )
                if ".self_attn.compresskv_conv.weight" in name:
                    name = name.replace(
                        ".self_attn.compresskv_conv.weight",
                        ".self_attn.mome_attn.compresskv_conv.weight",
                    )
                if ".self_attn.o_conv.weight" in name:
                    name = name.replace(
                        ".self_attn.o_conv.weight",
                        ".self_attn.mome_attn.o_conv.weight",
                    )
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        self.post_weight_load()
        return loaded_params

    def post_weight_load(self) -> None:
        for name, module in self.named_modules():
            if module is self:
                continue
            if hasattr(module, "post_weight_load"):
                module.post_weight_load()


class OpenPanguModelBase(nn.Module, SupportsPP, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.fuse_qkv_a_proj = (
            hasattr(config, "q_lora_rank") and config.q_lora_rank is not None
        )
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping["fused_qkv_a_proj"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]

        self.model = OpenPanguModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, self.model.norm(hidden_states))
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


class OpenPanguMoEModel(OpenPanguModelBase, MixtureOfExperts):
    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        config = vllm_config.model_config.hf_config
        return (
            (config.q_lora_rank,),
            (config.kv_lora_rank,),
            (config.num_attention_heads * config.v_head_dim,),
        )

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return (vllm_config.model_config.dtype,) * 3

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config

        # Set MoE hyperparameters
        self.expert_weights = []
        self.num_moe_layers = config.num_hidden_layers - config.first_k_dense_replace
        self.num_expert_groups = 1

        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, OpenPanguDecoderLayer)
            if isinstance(layer.mlp, OpenPanguMoE):
                # Pick last one layer since the first ones may be dense layers.
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError("No MOE layer found in model.layers.")

        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.n_routed_experts = example_moe.n_routed_experts
        self.n_shared_experts = example_moe.n_shared_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, OpenPanguMoE):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()


class OpenPanguEmbeddedModel(OpenPanguModelBase):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)


class PanguEmbeddedForCausalLM(OpenPanguEmbeddedModel):
    pass


class PanguUltraMoEForCausalLM(OpenPanguMoEModel):
    pass


class PanguProMoEV2ForCausalLM(OpenPanguMoEModel):
    pass
