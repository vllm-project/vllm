# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Apache License, Version 2.0:
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
#
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Inference-only Flash model compatible with HuggingFace weights."""

import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import ignore_torch_compile, support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE, ZeroExpertFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.int8_utils import block_dequant
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


class FlashConfig(PretrainedConfig):
    """Flash model configuration."""

    model_type = "longcat_flash"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=4096,
        intermediate_size=8192,
        num_layers=28,
        num_hidden_layers=None,
        num_attention_heads=96,
        num_key_value_heads=128,
        ep_size=1,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        num_experts_per_tok=None,
        norm_topk_prob=False,
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=100000,
        eos_token_id=100001,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        mla_scale_q_lora=False,
        mla_scale_kv_lora=False,
        dtype="bfloat16",
        params_dtype="bfloat16",
        router_dtype="float32",
        router_bias=False,
        topk_method=None,
        routed_scaling_factor=1.0,
        zero_expert_num=0,
        zero_expert_type=None,
        nextn_use_scmoe=False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            dtype=dtype,
            params_dtype=params_dtype,
            router_dtype=router_dtype,
            topk_method=topk_method,
            router_bias=router_bias,
            nextn_use_scmoe=nextn_use_scmoe,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        # Longcat Flash HF configs use num_layers as the actual layer count.
        # num_hidden_layers is often derived (e.g., 2 * num_layers) and should
        # not drive vLLM layer construction.
        if num_layers is not None:
            self.num_hidden_layers = num_layers
        else:
            self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.ep_size = ep_size
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        # Try to set `rope_scaling` if available, otherwise use `rope_parameters`
        rope_scaling = kwargs.pop("rope_scaling", None)
        rope_parameters = rope_scaling or rope_parameters or {"rope_type": "default"}
        rope_theta = kwargs.pop("rope_theta", 1000000.0)
        if "rope_theta" not in rope_parameters:
            rope_parameters["rope_theta"] = rope_theta
        self.rope_parameters = rope_parameters
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mla_scale_q_lora = mla_scale_q_lora
        self.mla_scale_kv_lora = mla_scale_kv_lora
        self.zero_expert_num = zero_expert_num
        self.zero_expert_type = zero_expert_type
        self.routed_scaling_factor = routed_scaling_factor
        self.hidden_act = "silu"
        self.intermediate_size = (
            self.ffn_hidden_size
            if hasattr(self, "ffn_hidden_size")
            else intermediate_size
        )
        if hasattr(self, "moe_intermediate_size"):
            self.moe_intermediate_size = self.moe_intermediate_size
        elif hasattr(self, "expert_ffn_hidden_size"):
            self.moe_intermediate_size = self.expert_ffn_hidden_size
        else:
            self.moe_intermediate_size = self.intermediate_size


class NgramEmbedding(nn.Module):
    """N-gram enhanced embeddings."""

    def __init__(
        self,
        config: FlashConfig,
        base_embeddings: VocabParallelEmbedding,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.word_embeddings = base_embeddings

        if config.emb_neighbor_num is None or config.emb_split_num is None:
            raise ValueError("N-gram embedding config is missing.")
        if config.ngram_vocab_size_ratio is None:
            raise ValueError("ngram_vocab_size_ratio is missing in config.")

        self.n = int(config.emb_neighbor_num)
        self.k = int(config.emb_split_num)
        self.context_len = self.n - 1
        if self.context_len <= 0:
            raise ValueError("emb_neighbor_num must be >= 2 for N-gram embedding.")

        self.m = config.ngram_vocab_size_ratio * config.vocab_size
        self._vocab_mods_cache: dict[tuple[int, int], list[int]] | None = None

        self._init_ngram_embeddings(prefix)

    def _init_ngram_embeddings(self, prefix: str) -> None:
        num_embedders = self.k * (self.n - 1)
        emb_dim = self.config.hidden_size // num_embedders
        if emb_dim * num_embedders != self.config.hidden_size:
            raise ValueError("hidden_size must be divisible by k*(n-1).")

        embedders = []
        post_projs = []
        for i in range(num_embedders):
            vocab_size = int(self.m + i * 2 + 1)
            embedders.append(
                VocabParallelEmbedding(
                    vocab_size,
                    emb_dim,
                    prefix=maybe_prefix(prefix, f"embedders.{i}"),
                )
            )
            post_projs.append(
                ReplicatedLinear(
                    emb_dim,
                    self.config.hidden_size,
                    bias=False,
                    prefix=maybe_prefix(prefix, f"post_projs.{i}"),
                )
            )

        self.embedders = nn.ModuleList(embedders)
        self.post_projs = nn.ModuleList(post_projs)

    def _shift_right_ignore_eos(
        self, tensor: torch.Tensor, n: int, eos_token_id: int
    ) -> torch.Tensor:
        """Shift tensor right by n positions, resetting at EOS tokens."""
        batch_size, seq_len = tensor.shape
        idx = torch.arange(seq_len, device=tensor.device, dtype=torch.int64)
        eos_mask = tensor == eos_token_id
        eos_pos = torch.where(eos_mask, idx, -1)
        prev_eos_inclusive = torch.cummax(eos_pos, dim=1).values
        prev_eos = torch.cat(
            [eos_pos.new_full((batch_size, 1), -1), prev_eos_inclusive[:, :-1]],
            dim=1,
        )
        segment_start = prev_eos + 1
        dist = idx.unsqueeze(0) - segment_start
        shift_mask = dist >= n

        src_idx = idx - n
        src_idx_clamped = torch.clamp(src_idx, min=0)
        gather_idx = src_idx_clamped.unsqueeze(0).expand(batch_size, -1)
        shifted = tensor.gather(dim=1, index=gather_idx)
        shifted = shifted.masked_fill(src_idx.unsqueeze(0) < 0, 0)
        return shifted.masked_fill(~shift_mask, 0)

    def _precompute_vocab_mods(self) -> dict[tuple[int, int], list[int]]:
        if self._vocab_mods_cache is not None:
            return self._vocab_mods_cache

        vocab_mods: dict[tuple[int, int], list[int]] = {}
        vocab_size = int(self.config.vocab_size)

        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)

                mods = [pow(vocab_size, p, emb_vocab_dim) for p in range(1, i)]

                vocab_mods[(i, j)] = mods

        self._vocab_mods_cache = vocab_mods
        return vocab_mods

    def _get_ngram_ids(
        self,
        input_ids: torch.Tensor,
        shifted_ids: dict[int, torch.Tensor],
        vocab_mods: list[int],
        ngram: int,
    ) -> torch.Tensor:
        ngram_ids = input_ids.clone()
        for k in range(2, ngram + 1):
            ngram_ids = ngram_ids + shifted_ids[k] * vocab_mods[k - 2]
        return ngram_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        ngram_context: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() == 1:
            if query_start_loc is None:
                raise ValueError("query_start_loc is required for flat N-gram input.")
            query_start_loc = query_start_loc.to(
                device=input_ids.device, dtype=torch.int64
            )
            num_reqs = query_start_loc.numel() - 1
            eos_token_id = int(self.config.eos_token_id)

            if ngram_context is None:
                ngram_context = input_ids.new_full(
                    (num_reqs, self.context_len), eos_token_id
                )
            elif ngram_context.shape != (num_reqs, self.context_len):
                raise ValueError("ngram_context shape mismatch.")

            x = self.word_embeddings(input_ids)
            vocab_mods = self._precompute_vocab_mods()

            token_positions = torch.arange(
                input_ids.shape[0], device=input_ids.device, dtype=torch.int64
            )
            req_indices = torch.searchsorted(
                query_start_loc[1:], token_positions, right=True
            )
            seq_start = query_start_loc[req_indices]
            pos_in_seq = token_positions - seq_start

            ctx_len = self.context_len
            virtual_pos = pos_in_seq + ctx_len

            eos_mask_input = input_ids == eos_token_id
            eos_pos_rel = torch.where(
                eos_mask_input,
                pos_in_seq,
                pos_in_seq.new_full(pos_in_seq.shape, -1),
            )
            offset = req_indices * (input_ids.shape[0] + ctx_len + 1)
            eos_pos_rel_offset = eos_pos_rel + offset
            last_eos_rel_offset = torch.cummax(eos_pos_rel_offset, dim=0).values
            last_eos_input_rel = last_eos_rel_offset - offset
            is_req_start = pos_in_seq == 0
            prev_eos_input_rel = torch.where(
                is_req_start,
                pos_in_seq.new_full(pos_in_seq.shape, -1),
                last_eos_input_rel.roll(1),
            )
            prev_eos_input_virtual = torch.where(
                prev_eos_input_rel >= 0,
                prev_eos_input_rel + ctx_len,
                prev_eos_input_rel,
            )

            ctx_positions = torch.arange(
                ctx_len, device=input_ids.device, dtype=torch.int64
            )
            ctx_eos = ngram_context == eos_token_id
            ctx_eos_pos = torch.where(ctx_eos, ctx_positions, -1)
            last_eos_context = ctx_eos_pos.max(dim=1).values
            last_eos_context_for_token = last_eos_context[req_indices]

            last_eos_virtual = torch.maximum(
                last_eos_context_for_token, prev_eos_input_virtual
            )
            segment_start_virtual = last_eos_virtual + 1

            ngram_range = range(2, self.n + 1)
            shifted_ids = {}
            for i in ngram_range:
                shift = i - 1
                src_virtual = virtual_pos - shift
                valid = src_virtual >= segment_start_virtual
                valid = valid & (src_virtual >= 0)

                src_in_context = src_virtual < ctx_len
                context_idx = src_virtual.clamp(min=0, max=ctx_len - 1)
                context_vals = ngram_context[req_indices, context_idx]

                src_abs = token_positions - shift
                src_abs_clamped = src_abs.clamp(min=0)
                input_vals = input_ids.gather(0, src_abs_clamped)

                shifted = torch.where(src_in_context, context_vals, input_vals)
                shifted = shifted.masked_fill(~valid, 0)
                shifted_ids[i] = shifted

            for i in ngram_range:
                base = (i - 2) * self.k
                for j in range(self.k):
                    index = base + j
                    emb_vocab_dim = int(self.m + index * 2 + 1)

                    ngram_ids = self._get_ngram_ids(
                        input_ids, shifted_ids, vocab_mods[(i, j)], ngram=i
                    )
                    new_ids = (ngram_ids % emb_vocab_dim).to(input_ids.dtype)

                    x_ngram = self.embedders[index](new_ids)
                    x_proj, _ = self.post_projs[index](x_ngram)
                    x = x + x_proj

            x = x / (1 + self.k * (self.n - 1))
            return x

        if input_ids.dim() != 2:
            raise ValueError("input_ids must be a 1D or 2D tensor.")

        batch_size, seq_len = input_ids.shape
        eos_token_id = int(self.config.eos_token_id)

        x = self.word_embeddings(input_ids)

        if ngram_context is None:
            ngram_context = input_ids.new_full(
                (batch_size, self.context_len), eos_token_id
            )
        elif ngram_context.shape != (batch_size, self.context_len):
            raise ValueError("ngram_context shape mismatch.")

        context = torch.cat([ngram_context, input_ids], dim=-1).to(torch.int64)
        vocab_mods = self._precompute_vocab_mods()

        ngram_range = range(2, self.n + 1)
        shifted_ids = {
            i: self._shift_right_ignore_eos(context, i - 1, eos_token_id)
            for i in ngram_range
        }

        for i in ngram_range:
            base = (i - 2) * self.k
            for j in range(self.k):
                index = base + j
                emb_vocab_dim = int(self.m + index * 2 + 1)

                ngram_ids = self._get_ngram_ids(
                    context, shifted_ids, vocab_mods[(i, j)], ngram=i
                )
                new_ids = (ngram_ids % emb_vocab_dim)[..., -seq_len:].to(
                    input_ids.dtype
                )

                x_ngram = self.embedders[index](new_ids)
                x_proj, _ = self.post_projs[index](x_ngram)
                x = x + x_proj

        x = x / (1 + self.k * (self.n - 1))
        return x


class FlashMLP(nn.Module):
    """Flash MLP layer."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class LongcatRouter(nn.Module):
    def __init__(
        self,
        config,
        zero_expert_num=0,
        rounter_params_dtype=torch.bfloat16,
        prefix: str = "",
    ):
        super().__init__()
        self.n_routed_experts = (
            config.n_routed_experts
            if hasattr(config, "n_routed_experts")
            else config.num_experts[0]
        )
        self.n_routed_experts = self.n_routed_experts + zero_expert_num
        self.classifier = ReplicatedLinear(
            config.hidden_size,
            self.n_routed_experts,
            bias=config.router_bias,
            params_dtype=rounter_params_dtype,
            quant_config=None,
            prefix=f"{prefix}.classifier",
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros((self.n_routed_experts), dtype=rounter_params_dtype)
        )

    def forward(self, hidden_states):
        logits, _ = self.classifier(hidden_states)
        return logits


class LongcatMoe(nn.Module):
    def __init__(
        self,
        config: FlashConfig,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        # Gate always runs at half / full precision for now.
        self.rounter_params_dtype = params_dtype
        if config.router_dtype == "float32":
            self.rounter_params_dtype = torch.float32

        self.router = LongcatRouter(
            config=config,
            zero_expert_num=config.zero_expert_num,
            rounter_params_dtype=self.rounter_params_dtype,
            prefix=f"{prefix}.gate",
        )

        assert config.zero_expert_num is not None
        assert config.zero_expert_type is not None
        self.experts = ZeroExpertFusedMoE(
            zero_expert_num=config.zero_expert_num,
            zero_expert_type=config.zero_expert_type,
            router=self.router,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            reduce_results=True,
            params_dtype=params_dtype,
            renormalize=False,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=enable_eplb,
            routed_scaling_factor=config.routed_scaling_factor,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Align to FusedMoE padded hidden size to avoid dim mismatch
        padded_hidden = self.experts.hidden_size
        if hidden_dim < padded_hidden:
            hidden_states_padded = torch.nn.functional.pad(
                hidden_states,
                (0, padded_hidden - hidden_dim),
                mode="constant",
                value=0.0,
            )
        else:
            hidden_states_padded = hidden_states

        router_logits_full = self.router(
            hidden_states_padded.to(self.rounter_params_dtype)
        )

        # ZeroExpertFusedMoE handles routing memoization and zero expert computation
        # internally. Pass full router_logits (including zero experts) so that
        # zero experts can be properly identified in routing.
        final_hidden_states = self.experts(
            hidden_states=hidden_states_padded,
            router_logits=router_logits_full,  # Full logits (includes zero experts)
        )

        # Crop back to original hidden dimension if padded earlier
        if padded_hidden != hidden_dim:
            final_hidden_states = final_hidden_states[..., :hidden_dim]

        return final_hidden_states.view(num_tokens, hidden_dim)


class FlashDecoderLayer(nn.Module):
    """Flash decoder layer with dual attention and MLP structure."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: FlashConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ) -> None:
        super().__init__()
        self.layer_idx = int(prefix.split(sep=".")[-1])
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        # Dual attention structure
        self.self_attn = nn.ModuleList(
            [
                DeepseekV2MLAAttention(
                    vllm_config=vllm_config,
                    config=config,
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
                    quant_config=None
                    if "self_attn" in getattr(config, "disable_quant_module", [])
                    else quant_config,
                    prefix=f"{prefix}.self_attn.{i}",
                )
                for i in range(2)
            ]
        )
        self.input_layernorm = nn.ModuleList(
            [RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for i in range(2)]
        )
        self.post_attention_layernorm = nn.ModuleList(
            [RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for i in range(2)]
        )

        # Dual MLP structure
        self.mlps = nn.ModuleList(
            [
                FlashMLP(
                    hidden_size=self.hidden_size,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=None
                    if "mlps" in getattr(config, "disable_quant_module", [])
                    else quant_config,
                    prefix=f"{prefix}.mlps.{i}",
                )
                for i in range(2)
            ]
        )

        self.mlp = LongcatMoe(
            config=config,
            num_experts=config.n_routed_experts
            if hasattr(config, "n_routed_experts")
            else config.num_experts[self.layer_idx],
            top_k=config.moe_topk
            if hasattr(config, "moe_topk")
            else config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=(f"{prefix}.mlp"),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm[0](hidden_states)
        else:
            hidden_states, residual = self.input_layernorm[0](hidden_states, residual)

        hidden_states = self.self_attn[0](
            positions=positions,
            hidden_states=hidden_states,
            llama_4_scaling=None,
        )

        hidden_states, residual = self.post_attention_layernorm[0](
            hidden_states, residual
        )

        # moe
        hidden_states_copy = hidden_states.clone()
        moe_hidden_states = self.mlp(hidden_states_copy)

        # first mlp
        hidden_states = self.mlps[0](hidden_states)

        hidden_states, residual = self.input_layernorm[1](hidden_states, residual)

        # second_attn
        hidden_states = self.self_attn[1](
            positions=positions,
            hidden_states=hidden_states,
            llama_4_scaling=None,
        )
        hidden_states, residual = self.post_attention_layernorm[1](
            hidden_states, residual
        )

        # second_mlp
        hidden_states = self.mlps[1](hidden_states)

        hidden_states = hidden_states + moe_hidden_states

        return hidden_states, residual


@support_torch_compile
class FlashModel(nn.Module):
    """Flash model."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = FlashConfig(**vllm_config.model_config.hf_config.__dict__)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "embed_tokens"),
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: FlashDecoderLayer(
                vllm_config,
                config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
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

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@support_torch_compile
class FlashNgramModel(FlashModel):
    """Flash model with N-gram enhanced embeddings."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        if get_pp_group().is_first_rank:
            self.ngram_embeddings = NgramEmbedding(
                self.config,
                self.embed_tokens,
                prefix=maybe_prefix(prefix, "ngram_embeddings"),
            )
        else:
            self.ngram_embeddings = PPMissingLayer()

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.numel() == 0:
            return input_ids.new_empty((0, self.config.hidden_size))

        return self.ngram_embeddings(
            input_ids,
            ngram_context=ngram_context,
            query_start_loc=query_start_loc,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if query_start_loc is None:
                    raise ValueError("query_start_loc is required for N-gram input.")
                hidden_states = self.embed_input_ids(
                    input_ids,
                    query_start_loc=query_start_loc,
                    ngram_context=ngram_context,
                )
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LongcatFlashForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    """Flash model for causal language modeling."""

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = FlashConfig(**vllm_config.model_config.hf_config.__dict__)
        quant_config = vllm_config.quant_config

        self.config = config
        config.intermediate_size = (
            config.ffn_hidden_size
            if hasattr(config, "ffn_hidden_size")
            else config.intermediate_size
        )

        self.quant_config = quant_config

        self.model = FlashModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
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
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            if hasattr(self.config, "n_routed_experts")
            else self.config.num_experts[0],
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        expert_params_mapping = self.get_expert_mapping()
        loaded_params: set[str] = set()

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp" in name and "mlps" not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue
                # Skip mtp
                if ".mtp." in name:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    # Skip mtp
                    if ".mtp." in name_mapped:
                        continue
                    if (
                        name_mapped.endswith(".bias") or name_mapped.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = param.weight_loader
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    # Skip loading kv_scale from ckpts towards new design.
                    if name.endswith(".kv_scale") and name not in params_dict:
                        continue
                    # Skip mtp
                    if ".mtp." in name:
                        continue
                    if name is None:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        for layer_id in range(self.config.num_hidden_layers):
            for i in range(2):
                if isinstance(self.model.layers[layer_id], PPMissingLayer):
                    continue
                self_attn = self.model.layers[layer_id].self_attn[i]
                if hasattr(
                    self.quant_config, "weight_block_size"
                ) and self_attn.kv_b_proj.weight.dtype in (
                    torch.float8_e4m3fn,
                    torch.float8_e4m3fnuz,
                ):
                    weight_block_size = self.quant_config.weight_block_size
                    if weight_block_size is not None:
                        assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                        dtype = torch.get_default_dtype()
                        w = block_dequant(
                            self_attn.kv_b_proj.weight,
                            self_attn.kv_b_proj.weight_scale_inv,
                            weight_block_size,
                        ).to(dtype)
                else:
                    w = self_attn.kv_b_proj.weight

                w_kc, w_vc = w.unflatten(
                    0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
                ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
                self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
                self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
                if self.config.mla_scale_q_lora:
                    self_attn.q_a_layernorm.weight.data *= (
                        self.config.hidden_size / self.config.q_lora_rank
                    ) ** 0.5
                if self.config.mla_scale_kv_lora:
                    self_attn.kv_a_layernorm.weight.data *= (
                        self.config.hidden_size / self.config.kv_lora_rank
                    ) ** 0.5
        return loaded_params


@ignore_torch_compile
class LongcatFlashNgramForCausalLM(
    LongcatFlashForCausalLM,
):
    """Flash model for causal LM with N-gram embeddings."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = FlashConfig(**vllm_config.model_config.hf_config.__dict__)
        quant_config = vllm_config.quant_config

        self.config = config
        config.intermediate_size = (
            config.ffn_hidden_size
            if hasattr(config, "ffn_hidden_size")
            else config.intermediate_size
        )
        self.quant_config = quant_config

        self.model = FlashNgramModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        *,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(
            input_ids,
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        expert_params_mapping = self.get_expert_mapping()
        loaded_params: set[str] = set()
        params_dict = dict(self.named_parameters())
        alias_to_param: dict[str, torch.nn.Parameter] = {}
        param_aliases: dict[int, list[str]] = {}
        for param_name, param in self.named_parameters(remove_duplicate=False):
            alias_to_param[param_name] = param
            param_aliases.setdefault(id(param), []).append(param_name)
        canonical_name_by_id = {id(param): name for name, param in params_dict.items()}

        def _mark_loaded(param: torch.nn.Parameter) -> None:
            for alias in param_aliases.get(id(param), ()):
                loaded_params.add(alias)

        def _canonical_name(name: str) -> str | None:
            if name in params_dict:
                return name
            param = alias_to_param.get(name)
            if param is None:
                return None
            return canonical_name_by_id.get(id(param), name)

        def _insert_mla_attn(name: str) -> str | None:
            token = ".self_attn."
            if token not in name or ".mla_attn." in name:
                return None
            prefix, rest = name.split(token, 1)
            parts = rest.split(".", 1)
            if len(parts) != 2:
                return None
            idx, tail = parts
            return f"{prefix}{token}{idx}.mla_attn.{tail}"

        def _resolve_param_name(name: str) -> str | None:
            resolved = _canonical_name(name)
            if resolved is not None:
                return resolved
            if ".self_attn." in name and ".mla_attn." not in name:
                alt = _insert_mla_attn(name)
                if alt is not None:
                    resolved = _canonical_name(alt)
                    if resolved is not None:
                        return resolved
            if ".mla_attn." in name:
                alt = name.replace(".mla_attn.", ".")
                resolved = _canonical_name(alt)
                if resolved is not None:
                    return resolved
            if ".mlp.gate." in name:
                alt = name.replace(".mlp.gate.", ".mlp.router.")
                resolved = _canonical_name(alt)
                if resolved is not None:
                    return resolved
            return None

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            loaded_name = None
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ".mlp.experts." in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                name_mapped = _resolve_param_name(name_mapped)
                if name_mapped is None:
                    continue
                # Skip loading extra bias for GPTQ models.
                if (
                    name_mapped.endswith(".bias") or name_mapped.endswith("_bias")
                ) and name_mapped not in params_dict:
                    continue
                # Skip mtp
                if ".mtp." in name_mapped:
                    continue
                if is_pp_missing_parameter(name_mapped, self):
                    continue
                param = params_dict[name_mapped]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                _mark_loaded(param)
                loaded_name = name_mapped
                break

            if loaded_name is not None:
                continue

            is_expert_weight = False
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                is_expert_weight = True
                name_mapped = name.replace(weight_name, param_name)
                name_mapped = _resolve_param_name(name_mapped)
                if name_mapped is None:
                    continue
                # Skip mtp
                if ".mtp." in name_mapped:
                    continue
                if (
                    name_mapped.endswith(".bias") or name_mapped.endswith("_bias")
                ) and name_mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(name_mapped, self):
                    continue
                param = params_dict[name_mapped]
                weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                success = weight_loader(
                    param,
                    loaded_weight,
                    name_mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    loaded_name = name_mapped
                    _mark_loaded(param)
                    break

            if loaded_name is not None:
                continue
            if is_expert_weight:
                continue

            name_mapped = _resolve_param_name(name)
            if name_mapped is None:
                continue
            # Skip loading extra bias for GPTQ models.
            if (
                name_mapped.endswith(".bias") or name_mapped.endswith("_bias")
            ) and name_mapped not in params_dict:
                continue
            # Skip loading kv_scale from ckpts towards new design.
            if name_mapped.endswith(".kv_scale") and name_mapped not in params_dict:
                continue
            # Skip mtp
            if ".mtp." in name_mapped:
                continue
            if is_pp_missing_parameter(name_mapped, self):
                continue
            param = params_dict[name_mapped]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            _mark_loaded(param)

        for layer_id in range(self.config.num_hidden_layers):
            for i in range(2):
                if isinstance(self.model.layers[layer_id], PPMissingLayer):
                    continue
                self_attn = self.model.layers[layer_id].self_attn[i]
                if hasattr(
                    self.quant_config, "weight_block_size"
                ) and self_attn.kv_b_proj.weight.dtype in (
                    torch.float8_e4m3fn,
                    torch.float8_e4m3fnuz,
                ):
                    weight_block_size = self.quant_config.weight_block_size
                    if weight_block_size is not None:
                        assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                        dtype = torch.get_default_dtype()
                        w = block_dequant(
                            self_attn.kv_b_proj.weight,
                            self_attn.kv_b_proj.weight_scale_inv,
                            weight_block_size,
                        ).to(dtype)
                else:
                    w = self_attn.kv_b_proj.weight

                w_kc, w_vc = w.unflatten(
                    0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
                ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
                self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
                self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
                if self.config.mla_scale_q_lora:
                    self_attn.q_a_layernorm.weight.data *= (
                        self.config.hidden_size / self.config.q_lora_rank
                    ) ** 0.5
                if self.config.mla_scale_kv_lora:
                    self_attn.kv_a_layernorm.weight.data *= (
                        self.config.hidden_size / self.config.kv_lora_rank
                    ) ** 0.5
        return loaded_params
