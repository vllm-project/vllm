# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions adapted from LongSpark at commit 9ab6526e481bbea67489f3a73ebc782218d56f71:
# https://github.com/Hao-Yuan-He/LongSpark/tree/9ab6526e481bbea67489f3a73ebc782218d56f71/vendor/specforge/modeling/draft
#
# MIT License
# Copyright (c) 2026 The LongSpark Authors
#
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
"""LongSpark DraftFreeKV model.

The module implements the inference-only portion of the MIT-licensed
LongSpark/SpecForge DraftFreeKV checkpoint format. Target K/V ownership and
request state live in the MRV2 speculator; this file only owns draft weights.
"""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.worker.gpu.spec_decode.longspark.kernels import (
    longspark_attention,
    longspark_paged_attention,
    prepare_longspark_shifted_input,
)


class _FeedForward(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, out_features, bias=False),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class _LongSparkRMSNorm(nn.Module):
    """Eager-friendly, shape-parameterized RMSNorm for the draft graph.

    The target model normally lets ``torch.compile`` lower the IR RMSNorm.
    LongSpark is captured directly by a CUDA graph instead, so the IR-native
    implementation would otherwise expand every norm into several ATen
    kernels.  Use vLLM's unquantized CUDA kernel explicitly while retaining a
    portable native fallback for tests and unsupported layouts.
    """

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        set_weight_attrs(self.weight, {"dummy_weight_value": 1.0})
        self.variance_epsilon = eps

    def _can_use_cuda_kernel(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ) -> bool:
        return (
            hidden_states.is_cuda
            and hidden_states.dtype in (torch.bfloat16, torch.float16)
            and self.weight.dtype == hidden_states.dtype
            and hidden_states.is_contiguous()
            and (
                residual is None
                or (
                    residual.dtype == hidden_states.dtype
                    and residual.device == hidden_states.device
                    and residual.shape == hidden_states.shape
                    and residual.is_contiguous()
                )
            )
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self._can_use_cuda_kernel(hidden_states, residual):
            if residual is None:
                output = torch.empty_like(hidden_states)
                ops.rms_norm(
                    output,
                    hidden_states,
                    self.weight.data,
                    self.variance_epsilon,
                )
                return output
            ops.fused_add_rms_norm(
                hidden_states,
                residual,
                self.weight.data,
                self.variance_epsilon,
            )
            return hidden_states, residual

        original_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        if residual is not None:
            normalized = normalized + residual.float()
            residual = normalized.to(original_dtype)
        variance = normalized.pow(2).mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.variance_epsilon)
        normalized = (normalized.to(self.weight.dtype) * self.weight).to(original_dtype)
        if residual is None:
            return normalized
        return normalized, residual


class _DraftStateEncoder(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, eps: float):
        super().__init__()
        self.proj = _FeedForward(num_layers * hidden_size, hidden_size, hidden_size)
        self.norm = _LongSparkRMSNorm(hidden_size, eps=eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(hidden_states.flatten(start_dim=1)))


class _ParallelShiftedInput(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        block_size: int,
        mask_token_id: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.position_embedding = nn.Embedding(block_size, hidden_size)
        self.proj = _FeedForward(2 * hidden_size, hidden_size, hidden_size)
        self.norm = _LongSparkRMSNorm(hidden_size, eps=eps)

    def forward(
        self,
        token_ids: torch.Tensor,
        latest_target_state: torch.Tensor,
        target_embed_tokens: nn.Module,
    ) -> torch.Tensor:
        batch_size = token_ids.shape[0]
        embedding_weight = getattr(target_embed_tokens, "weight", None)
        if (
            isinstance(embedding_weight, torch.Tensor)
            and token_ids.is_cuda
            and latest_target_state.is_cuda
            and latest_target_state.dtype == torch.bfloat16
            and embedding_weight.dtype == torch.bfloat16
            and self.position_embedding.weight.dtype == torch.bfloat16
            and embedding_weight.device == latest_target_state.device
            and self.position_embedding.weight.device == latest_target_state.device
            and embedding_weight.ndim == 2
            and embedding_weight.shape[1] == latest_target_state.shape[-1]
            and embedding_weight.stride(1) == 1
            and self.position_embedding.weight.stride(1) == 1
        ):
            shifted_input = prepare_longspark_shifted_input(
                token_ids,
                latest_target_state,
                embedding_weight,
                self.position_embedding.weight,
                self.mask_token_id,
            )
            return self.norm(self.proj(shifted_input))
        block_token_ids = torch.full(
            (batch_size, self.block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=token_ids.device,
        )
        block_token_ids[:, 0] = token_ids
        token_states = target_embed_tokens(block_token_ids)
        positions = torch.arange(self.block_size, device=token_ids.device)
        token_states = token_states + self.position_embedding(positions)[None]
        latest = latest_target_state[:, None].expand(-1, self.block_size, -1)
        return self.norm(self.proj(torch.cat((token_states, latest), dim=-1)))


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class _LongSparkAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        raw_rows: int,
        rope_theta: float,
        eps: float,
    ) -> None:
        super().__init__()
        if num_attention_heads % num_key_value_heads:
            raise ValueError(
                "LongSpark query head count must be divisible by target KV heads"
            )
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.head_dim = head_dim
        self.raw_rows = raw_rows
        self.scaling = head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.qkv_proj: nn.Linear | None = None
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def fuse_weights(self) -> None:
        if self.qkv_proj is not None:
            return
        qkv_weight = torch.cat(
            (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), dim=0
        )
        fused = nn.Linear(
            qkv_weight.shape[1],
            qkv_weight.shape[0],
            bias=False,
            device=qkv_weight.device,
            dtype=qkv_weight.dtype,
        )
        fused.weight = nn.Parameter(qkv_weight, requires_grad=False)
        self.qkv_proj = fused
        self.q_proj = None  # type: ignore[assignment]
        self.k_proj = None  # type: ignore[assignment]
        self.v_proj = None  # type: ignore[assignment]

    def _apply_rope(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        frequencies = position_ids.float()[..., None] * self.inv_freq[None, None]
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        cos = embeddings.cos().to(hidden_states.dtype)[:, None]
        sin = embeddings.sin().to(hidden_states.dtype)[:, None]
        return hidden_states * cos + _rotate_half(hidden_states) * sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        global_keys: torch.Tensor,
        global_values: torch.Tensor,
        raw_keys: torch.Tensor,
        raw_values: torch.Tensor,
        raw_mask: torch.Tensor,
        raw_cache: torch.Tensor | None = None,
        raw_block_table: torch.Tensor | None = None,
        raw_seq_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, block_size, _ = hidden_states.shape
        if self.qkv_proj is None:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        else:
            qkv_states = self.qkv_proj(hidden_states)
            # LongSpark uses Qwen3's QK RMSNorm followed by Neox-style RoPE.
            # Keep the packed QKV output intact so vLLM can normalize and
            # rotate Q/K in one in-place CUDA kernel.  The previous path ran
            # two RMSNorm kernels plus frequency, cos, sin, rotate, and add
            # kernels for every one of the five draft layers.
            if hidden_states.is_cuda and hidden_states.dtype in (
                torch.bfloat16,
                torch.float16,
            ):
                ops.fused_qk_norm_rope(
                    qkv_states.view(-1, qkv_states.shape[-1]),
                    self.num_attention_heads,
                    self.num_key_value_heads,
                    self.num_key_value_heads,
                    self.head_dim,
                    self.q_norm.variance_epsilon,
                    self.q_norm.weight,
                    self.k_norm.weight,
                    cos_sin_cache,
                    True,
                    position_ids.reshape(-1),
                )
                qk_is_ready = True
            else:
                qk_is_ready = False
            query_states, key_states, value_states = qkv_states.split(
                (
                    self.num_attention_heads * self.head_dim,
                    self.num_key_value_heads * self.head_dim,
                    self.num_key_value_heads * self.head_dim,
                ),
                dim=-1,
            )
        query = query_states.view(
            batch_size,
            block_size,
            self.num_attention_heads,
            self.head_dim,
        )
        local_keys = key_states.view(
            batch_size,
            block_size,
            self.num_key_value_heads,
            self.head_dim,
        )
        local_values = value_states.view_as(local_keys)
        if self.qkv_proj is None or not qk_is_ready:
            query = self.q_norm(query)
            local_keys = self.k_norm(local_keys)
            query = self._apply_rope(query.transpose(1, 2), position_ids)
            local_keys = self._apply_rope(local_keys.transpose(1, 2), position_ids)
        else:
            query = query.transpose(1, 2)
            local_keys = local_keys.transpose(1, 2)
        local_values = local_values.transpose(1, 2)

        if (
            raw_cache is not None
            and raw_block_table is not None
            and raw_seq_lens is not None
            and query.is_cuda
            and query.dtype == torch.bfloat16
        ):
            attended = longspark_paged_attention(
                query=query,
                global_keys=global_keys,
                global_values=global_values,
                raw_cache=raw_cache,
                raw_block_table=raw_block_table,
                raw_seq_lens=raw_seq_lens,
                raw_rows=self.raw_rows,
                local_keys=local_keys,
                local_values=local_values,
                scale=self.scaling,
            )
            attended = (
                attended.transpose(1, 2)
                .contiguous()
                .view(
                    batch_size,
                    block_size,
                    self.num_attention_heads * self.head_dim,
                )
            )
            return self.o_proj(attended)

        if (
            query.is_cuda
            and query.dtype == torch.bfloat16
            and raw_keys.ndim == 4
            and raw_keys.shape[2] == self.raw_rows
        ):
            attended = longspark_attention(
                query=query,
                global_keys=global_keys,
                global_values=global_values,
                raw_keys=raw_keys,
                raw_values=raw_values,
                raw_mask=raw_mask,
                local_keys=local_keys,
                local_values=local_values,
                scale=self.scaling,
            )
            attended = (
                attended.transpose(1, 2)
                .contiguous()
                .view(
                    batch_size,
                    block_size,
                    self.num_attention_heads * self.head_dim,
                )
            )
            return self.o_proj(attended)

        raw_keys = raw_keys.repeat_interleave(self.num_key_value_groups, dim=1)
        raw_values = raw_values.repeat_interleave(self.num_key_value_groups, dim=1)
        local_keys = local_keys.repeat_interleave(self.num_key_value_groups, dim=1)
        local_values = local_values.repeat_interleave(self.num_key_value_groups, dim=1)
        key = torch.cat((global_keys, raw_keys, local_keys), dim=2)
        value = torch.cat(
            (
                global_values.to(dtype=query.dtype),
                raw_values.to(dtype=query.dtype),
                local_values,
            ),
            dim=2,
        )
        global_mask = torch.ones(
            (batch_size, global_keys.shape[2]),
            dtype=torch.bool,
            device=query.device,
        )
        local_mask = torch.ones(
            (batch_size, block_size), dtype=torch.bool, device=query.device
        )
        attention_mask = torch.cat((global_mask, raw_mask, local_mask), dim=1)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask[:, None, None],
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
        )
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                block_size,
                self.num_attention_heads * self.head_dim,
            )
        )
        return self.o_proj(attended)


class _LongSparkMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_up_proj: nn.Linear | None = None
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def fuse_weights(self) -> None:
        if self.gate_up_proj is not None:
            return
        gate_up_weight = torch.cat((self.gate_proj.weight, self.up_proj.weight), dim=0)
        fused = nn.Linear(
            gate_up_weight.shape[1],
            gate_up_weight.shape[0],
            bias=False,
            device=gate_up_weight.device,
            dtype=gate_up_weight.dtype,
        )
        fused.weight = nn.Parameter(gate_up_weight, requires_grad=False)
        self.gate_up_proj = fused
        self.gate_proj = None  # type: ignore[assignment]
        self.up_proj = None  # type: ignore[assignment]

    def forward_with_residual_norm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        norm: RMSNorm,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = norm(hidden_states, residual)
        return self(hidden_states), residual

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.gate_up_proj is None:
            activated = F.silu(self.gate_proj(hidden_states)) * self.up_proj(
                hidden_states
            )
        else:
            activated = self.act_fn(self.gate_up_proj(hidden_states))
        return self.down_proj(activated)


class _LongSparkDecoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.self_attn = _LongSparkAttention(
            config.hidden_size,
            config.qwen_num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.raw_target_rows,
            config.qwen_rope_theta,
            config.qwen_rms_norm_eps,
        )
        self.mlp = _LongSparkMLP(config.hidden_size, config.qwen_intermediate_size)
        self.input_layernorm = _LongSparkRMSNorm(
            config.hidden_size, eps=config.qwen_rms_norm_eps
        )
        self.post_attention_layernorm = _LongSparkRMSNorm(
            config.hidden_size, eps=config.qwen_rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        position_ids: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        global_keys: torch.Tensor,
        global_values: torch.Tensor,
        raw_keys: torch.Tensor,
        raw_values: torch.Tensor,
        raw_mask: torch.Tensor,
        raw_cache: torch.Tensor | None = None,
        raw_block_table: torch.Tensor | None = None,
        raw_seq_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            hidden_states,
            position_ids,
            cos_sin_cache,
            global_keys,
            global_values,
            raw_keys,
            raw_values,
            raw_mask,
            raw_cache,
            raw_block_table,
            raw_seq_lens,
        )
        return self.mlp.forward_with_residual_norm(
            hidden_states, residual, self.post_attention_layernorm
        )


class _GlobalQueryGenerator(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.learned_query = nn.Parameter(
            torch.empty(
                len(config.target_layer_ids),
                config.qwen_num_attention_heads,
                config.global_memory_slots,
                config.head_dim,
            )
        )


class _Global16Raw256Reference(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.global_query_generator = _GlobalQueryGenerator(config)


class LongSparkDraftModel(nn.Module):
    """Inference-only DraftFreeKV model used by ``LongSparkSpeculator``."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        del prefix
        super().__init__()
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        self.config = speculative_config.draft_model_config.hf_config
        config = self.config
        self._validate_config()

        num_layers = len(config.target_layer_ids)
        self.draft_state_encoder = _DraftStateEncoder(
            num_layers, config.hidden_size, config.qwen_rms_norm_eps
        )
        self.parallel_shifted_input = _ParallelShiftedInput(
            config.hidden_size,
            config.block_size,
            config.mask_token_id,
            config.qwen_rms_norm_eps,
        )
        self.parallel_qwen_layers = nn.ModuleList(
            [_LongSparkDecoderLayer(config) for _ in range(num_layers)]
        )
        self.parallel_markov_head = DSparkMarkovHead(
            config.vocab_size,
            config.vocab_size,
            config.parallel_markov_rank,
            prefix="parallel_markov_head",
        )
        self.global16_raw256_reference = _Global16Raw256Reference(config)
        # One shared cache is sufficient for all five layers.  It also lets
        # the fused QK-Norm+RoPE kernel replace per-layer trigonometry.
        self.rotary_emb = RotaryEmbedding(
            head_size=config.head_dim,
            rotary_dim=config.head_dim,
            max_position_embeddings=(
                vllm_config.model_config.max_model_len + config.block_size
            ),
            base=config.qwen_rope_theta,
            is_neox_style=True,
            dtype=vllm_config.model_config.dtype,
        )

        self.target_embed_tokens: nn.Module | None = None
        self.target_final_norm: nn.Module | None = None
        self.target_lm_head: nn.Module | None = None
        self.target_logits_processor = None

    def _validate_config(self) -> None:
        config = self.config
        if getattr(config, "draft_mode", None) != "parallel_shifted_qwen":
            raise ValueError("LongSpark requires draft_mode='parallel_shifted_qwen'")
        if getattr(config, "parallel_markov_head_type", None) != "vanilla":
            raise ValueError("LongSpark requires parallel_markov_head_type='vanilla'")
        if not config.global16_raw256_enabled:
            raise ValueError("LongSpark requires global16_raw256_enabled=true")
        expected = {
            "global_memory_source_window": "full",
            "target_kv_memory_layout": "direct",
            "target_kv_window_size": None,
            "global_checkpoint_compatibility": "retrain_required",
        }
        for name, value in expected.items():
            if getattr(config, name) != value:
                raise ValueError(f"LongSpark requires {name}={value}")
        positive_sizes = {
            "block_size": config.block_size,
            "global_memory_slots": config.global_memory_slots,
            "raw_target_rows": config.raw_target_rows,
            "parallel_markov_rank": config.parallel_markov_rank,
            "qwen_num_attention_heads": config.qwen_num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
        }
        invalid_sizes = [name for name, value in positive_sizes.items() if value <= 0]
        if invalid_sizes:
            raise ValueError(
                "LongSpark dimensions must be positive: " + ", ".join(invalid_sizes)
            )
        if config.qwen_num_attention_heads % config.num_key_value_heads:
            raise ValueError(
                "LongSpark query head count must be divisible by target KV heads"
            )
        if config.local_draft_rows != config.block_size:
            raise ValueError("local_draft_rows must match block_size")
        if config.max_accepted_tokens != config.block_size + 1:
            raise ValueError("max_accepted_tokens must equal block_size + 1")
        if config.parallel_mixer_layers != len(config.target_layer_ids):
            raise ValueError("parallel_mixer_layers must match target_layer_ids")
        if config.mask_token_id is None:
            raise ValueError("LongSpark requires mask_token_id")

    @property
    def global_queries(self) -> torch.Tensor:
        return self.global16_raw256_reference.global_query_generator.learned_query

    def set_target_modules(
        self,
        embed_tokens: nn.Module,
        final_norm: nn.Module,
        lm_head: nn.Module,
        logits_processor,
    ) -> None:
        self.target_embed_tokens = embed_tokens
        self.target_final_norm = final_norm
        self.target_lm_head = lm_head
        self.target_logits_processor = logits_processor

    def optimize_for_inference(self) -> None:
        """Fuse checkpoint projection matrices used together during serving."""
        for layer in self.parallel_qwen_layers:
            layer.self_attn.fuse_weights()
            layer.mlp.fuse_weights()

    def forward(
        self,
        selected_hidden: torch.Tensor,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        global_output: torch.Tensor,
        raw_keys: torch.Tensor,
        raw_values: torch.Tensor,
        raw_mask: torch.Tensor,
        raw_caches: tuple[torch.Tensor, ...] | None = None,
        raw_block_tables: tuple[torch.Tensor, ...] | None = None,
        raw_seq_lens: torch.Tensor | None = None,
        return_residual: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.target_embed_tokens is None:
            raise RuntimeError("LongSpark target modules are not attached")
        hidden_states = self.prepare_hidden(selected_hidden, token_ids)
        return self.forward_layers(
            hidden_states,
            token_ids,
            position_ids,
            global_output,
            raw_keys,
            raw_values,
            raw_mask,
            raw_caches,
            raw_block_tables,
            raw_seq_lens,
            return_residual,
        )

    def prepare_hidden(
        self,
        selected_hidden: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.target_embed_tokens is None:
            raise RuntimeError("LongSpark target modules are not attached")
        latest_target_state = self.draft_state_encoder(selected_hidden)
        return self.parallel_shifted_input(
            token_ids,
            latest_target_state,
            self.target_embed_tokens,
        )

    def forward_layers(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        global_output: torch.Tensor,
        raw_keys: torch.Tensor,
        raw_values: torch.Tensor,
        raw_mask: torch.Tensor,
        raw_caches: tuple[torch.Tensor, ...] | None = None,
        raw_block_tables: tuple[torch.Tensor, ...] | None = None,
        raw_seq_lens: torch.Tensor | None = None,
        return_residual: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        global_keys = self.global_queries[None].expand(
            token_ids.shape[0], -1, -1, -1, -1
        )
        residual = None
        cos_sin_cache = self.rotary_emb._match_cos_sin_cache_dtype(hidden_states)
        for layer_index, layer in enumerate(self.parallel_qwen_layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                position_ids,
                cos_sin_cache,
                global_keys[:, layer_index],
                global_output[:, layer_index],
                raw_keys[:, layer_index],
                raw_values[:, layer_index],
                raw_mask,
                None if raw_caches is None else raw_caches[layer_index],
                None if raw_block_tables is None else raw_block_tables[layer_index],
                raw_seq_lens,
            )
        if return_residual:
            return hidden_states, residual
        return hidden_states + residual

    def compute_draft_logits(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            self.target_final_norm is None
            or self.target_lm_head is None
            or self.target_logits_processor is None
        ):
            raise RuntimeError("LongSpark target modules are not attached")
        target_norm = self.target_final_norm
        target_weight = getattr(target_norm, "weight", None)
        target_eps = getattr(target_norm, "variance_epsilon", None)
        if (
            hidden_states.is_cuda
            and hidden_states.dtype in (torch.bfloat16, torch.float16)
            and isinstance(target_weight, torch.Tensor)
            and target_weight.dtype == hidden_states.dtype
            and target_weight.shape == (hidden_states.shape[-1],)
            and isinstance(target_eps, float)
            and hidden_states.is_contiguous()
        ):
            if residual is None:
                normalized = torch.empty_like(hidden_states)
                ops.rms_norm(normalized, hidden_states, target_weight.data, target_eps)
                hidden_states = normalized
            else:
                ops.fused_add_rms_norm(
                    hidden_states,
                    residual,
                    target_weight.data,
                    target_eps,
                )
        else:
            if residual is not None:
                hidden_states = hidden_states + residual
            hidden_states = target_norm(hidden_states)
        return self.target_logits_processor(self.target_lm_head, hidden_states)

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.parallel_markov_head.embed(token_ids)

    def markov_bias(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.target_logits_processor is None:
            raise RuntimeError("LongSpark target modules are not attached")
        return self.parallel_markov_head.bias(
            hidden_states, self.target_logits_processor
        )

    def fused_markov_weights(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ] | None:
        """Return exact weights and optional L2-sized retrieval factors."""
        processor = self.target_logits_processor
        head = self.parallel_markov_head.markov_w2
        embedding_weight = self.parallel_markov_head.markov_w1.weight
        projection_weight = head.weight
        if (
            processor is None
            or processor.scale != 1.0
            or processor.soft_cap is not None
            or processor.head_dtype not in (None, torch.bfloat16)
            or head.tp_size != 1
            or embedding_weight.dtype != torch.bfloat16
            or embedding_weight.shape
            != (self.config.vocab_size, self.config.parallel_markov_rank)
            or projection_weight.dtype != torch.bfloat16
            or projection_weight.shape
            != (self.config.vocab_size, self.config.parallel_markov_rank)
        ):
            return None
        candidate_embedding = candidate_projection = None
        if projection_weight.is_cuda:
            properties = torch.cuda.get_device_properties(projection_weight.device)
            l2_bytes = int(getattr(properties, "L2_cache_size", 0))
            bytes_per_rank = (
                projection_weight.shape[0] * projection_weight.element_size()
            )
            # Leave room for base logits and candidate re-scoring so repeated
            # projections remain resident instead of thrashing the full L2.
            max_cached_rank = int(0.4 * l2_bytes / max(1, bytes_per_rank))
            bounded_rank = min(
                self.config.parallel_markov_rank - 1,
                max_cached_rank,
            )
            candidate_rank = 32 * (bounded_rank // 32)
            if candidate_rank >= 64:
                with torch.no_grad():
                    left, singular_values, right = torch.linalg.svd(
                        projection_weight.float(), full_matrices=False
                    )
                    retained_energy = (
                        singular_values[:candidate_rank].square().sum()
                        / singular_values.square().sum()
                    )
                    if retained_energy.item() >= 0.60:
                        candidate_projection = (
                            left[:, :candidate_rank]
                            * singular_values[:candidate_rank]
                        ).to(torch.bfloat16)
                        candidate_embedding = (
                            embedding_weight
                            @ right[:candidate_rank].t().to(torch.bfloat16)
                        ).to(torch.bfloat16)
                        candidate_projection = candidate_projection.contiguous()
                        candidate_embedding = candidate_embedding.contiguous()
                    del left, singular_values, right
                    torch.cuda.empty_cache()
        return (
            embedding_weight,
            projection_weight,
            candidate_embedding,
            candidate_projection,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)
