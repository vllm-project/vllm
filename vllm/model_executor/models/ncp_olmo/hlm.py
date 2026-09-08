# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Incremental request-scoped HLM for the first Stage3 ConceptLM backend."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm.envs as envs
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.determinism.batch_invariant import linear_batch_invariant
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform
from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

from .contract import ConceptLMBackendConfig
from .state import ConceptRequestState, HLMKVState, append_tensor_buffer
from .token_tower import (
    ConceptLMOlmo3MLP,
    _config_value,
    _rope_parameters,
    _sliding_window,
)
from .weights import required_checkpoint_shards, resolve_checkpoint_weight


def apply_ncp_layer_norm(
    layer_norm: nn.LayerNorm,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Apply mean-subtracting LayerNorm under vLLM batch invariance.

    Batch-invariant mode overrides ``aten::mean`` so reductions accumulate in
    FP32.  ``aten::native_layer_norm`` can then observe FP32 normalization
    intermediates together with BF16 affine parameters and fail with a mixed
    dtype error.  Keep the normalization in FP32 and restore the activation
    dtype at the boundary, matching the mixed-precision LayerNorm contract.
    """

    if not envs.VLLM_BATCH_INVARIANT:
        return layer_norm(hidden_states)

    normalized_shape = tuple(layer_norm.normalized_shape)
    if normalized_shape != tuple(hidden_states.shape[-len(normalized_shape) :]):
        raise ValueError(
            "NCP LayerNorm input shape does not match normalized_shape: "
            f"{tuple(hidden_states.shape)} vs {normalized_shape}"
        )
    reduction_dims = tuple(
        range(hidden_states.ndim - len(normalized_shape), hidden_states.ndim)
    )
    input_dtype = hidden_states.dtype
    normalized = hidden_states.float()
    mean = normalized.mean(dim=reduction_dims, keepdim=True)
    centered = normalized - mean
    variance = centered.square().mean(dim=reduction_dims, keepdim=True)
    normalized = centered * torch.rsqrt(variance + layer_norm.eps)
    if layer_norm.elementwise_affine:
        if layer_norm.weight is not None:
            normalized = normalized * layer_norm.weight.float()
        if layer_norm.bias is not None:
            normalized = normalized + layer_norm.bias.float()
    return normalized.to(input_dtype)


def apply_ncp_linear(
    linear: nn.Linear,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Apply native NCP route linears with vLLM's invariant GEMM.

    vLLM's own parallel linear layers select ``linear_batch_invariant``
    directly. NCP also has small checkpoint-native ``nn.Linear`` modules for
    DD routes and VQ heads; on Hopper those modules otherwise keep the regular
    cuBLASLt path because the global aten linear override is intentionally not
    installed. Route them through the same invariant implementation explicitly.
    """

    if envs.VLLM_BATCH_INVARIANT and current_platform.is_cuda_alike():
        return linear_batch_invariant(
            hidden_states,
            linear.weight,
            linear.bias,
        )
    return linear(hidden_states)


def _causal_prefill_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
    prefix_length: int,
    sliding_window: int | None,
    flash_attn_version: int | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run causal HLM attention with one explicitly selected FA implementation.

    The tensors use ``[heads, sequence, head_dim]`` layout. With an explicit
    FA2/FA3 version, fresh prompt prefills, continuation prefills, and one-token
    HLM decode use vLLM's bundled FlashAttention on CUDA. FA2 and FA3 share
    bottom-right causal alignment, so a shorter query attends to the existing
    K/V prefix. The legacy comparison path and host-only unit tests retain
    PyTorch SDPA.
    """

    query_length = int(query.shape[1])
    key_length = int(key.shape[1])
    if key_length != prefix_length + query_length:
        raise ValueError(
            "HLM prefill key length must equal prefix plus query length: "
            f"{key_length} != {prefix_length} + {query_length}"
        )
    if (
        query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
        and flash_attn_version in (2, 3)
    ):
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        if cu_seqlens_q is None:
            cu_seqlens_q = torch.tensor(
                [0, query_length],
                dtype=torch.int32,
                device=query.device,
            )
        else:
            cu_seqlens_q[1] = query_length
        if cu_seqlens_k is None:
            cu_seqlens_k = torch.tensor(
                [0, key_length],
                dtype=torch.int32,
                device=query.device,
            )
        else:
            cu_seqlens_k[1] = key_length
        window_size = (-1, -1) if sliding_window is None else (int(sliding_window), 0)
        output = flash_attn_varlen_func(
            q=query.transpose(0, 1).contiguous(),
            k=key.transpose(0, 1).contiguous(),
            v=value.transpose(0, 1).contiguous(),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=query_length,
            max_seqlen_k=key_length,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=True,
            window_size=window_size,
            return_softmax_lse=False,
            fa_version=flash_attn_version,
        )
        return output.transpose(0, 1)

    full_window = sliding_window is None or key_length <= sliding_window + 1
    if prefix_length == 0 and full_window:
        if query.is_cuda and query.dtype in (torch.float16, torch.bfloat16):
            with torch.nn.attention.sdpa_kernel(
                torch.nn.attention.SDPBackend.FLASH_ATTENTION
            ):
                return F.scaled_dot_product_attention(
                    query.unsqueeze(0),
                    key.unsqueeze(0),
                    value.unsqueeze(0),
                    is_causal=True,
                    scale=scale,
                ).squeeze(0)
        return F.scaled_dot_product_attention(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            is_causal=True,
            scale=scale,
        ).squeeze(0)

    query_positions = torch.arange(
        prefix_length,
        prefix_length + query_length,
        device=query.device,
    )
    key_positions = torch.arange(key_length, device=query.device)
    attention_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    if sliding_window is not None:
        attention_mask &= key_positions.unsqueeze(0) >= (
            query_positions.unsqueeze(1) - sliding_window
        )
    return F.scaled_dot_product_attention(
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        attn_mask=attention_mask,
        scale=scale,
    ).squeeze(0)


def _legacy_decode_attention(
    query: torch.Tensor,
    cached_key: torch.Tensor,
    cached_value: torch.Tensor,
    *,
    scale: float,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Run the checkpoint-compatible FP32 HLM decode attention math."""

    scores = torch.matmul(
        query.float(),
        cached_key.float().transpose(-1, -2),
    )
    weights = torch.softmax(scores * scale, dim=-1)
    return torch.matmul(weights, cached_value.float()).to(output_dtype)


def _append_hlm_kv(
    kv_state: HLMKVState,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append HLM K/V into geometrically growing request-owned storage."""

    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("HLM key and value must have the same 3D shape")
    num_tokens = int(key.shape[1])
    if num_tokens <= 0:
        raise ValueError("cannot append an empty HLM K/V segment")
    old_length = int(kv_state.length)
    required_length = old_length + num_tokens
    if (kv_state.key is None) != (kv_state.value is None):
        raise ValueError("HLM K/V storage must contain both key and value")

    if kv_state.key is None:
        if old_length != 0:
            raise ValueError("empty HLM K/V storage has non-zero length")
        target = max(16, required_length)
        capacity = 1 << (target - 1).bit_length()
        kv_state.key = key.new_empty(
            key.shape[0],
            capacity,
            key.shape[2],
        )
        kv_state.value = value.new_empty(
            value.shape[0],
            capacity,
            value.shape[2],
        )
    else:
        if (
            kv_state.key.shape[0] != key.shape[0]
            or kv_state.key.shape[2] != key.shape[2]
            or kv_state.value.shape != kv_state.key.shape
        ):
            raise ValueError("HLM K/V append shape does not match storage")
        capacity = int(kv_state.key.shape[1])
        if old_length < 0 or old_length > capacity:
            raise ValueError("HLM K/V length exceeds storage capacity")
        if required_length > capacity:
            target = max(required_length, capacity * 2)
            new_capacity = 1 << (target - 1).bit_length()
            new_key = key.new_empty(
                key.shape[0],
                new_capacity,
                key.shape[2],
            )
            new_value = value.new_empty(
                value.shape[0],
                new_capacity,
                value.shape[2],
            )
            new_key[:, :old_length].copy_(kv_state.key[:, :old_length])
            new_value[:, :old_length].copy_(kv_state.value[:, :old_length])
            kv_state.key = new_key
            kv_state.value = new_value

    kv_state.key[:, old_length:required_length].copy_(key)
    kv_state.value[:, old_length:required_length].copy_(value)
    kv_state.length = required_length
    return (
        kv_state.key[:, :required_length],
        kv_state.value[:, :required_length],
    )


class ConceptLMDepthDD(nn.Module):
    """One DD route over the current module's input and raw layer history."""

    def __init__(
        self,
        *,
        hidden_size: int,
        layer_index: int,
        use_softmax: bool = False,
    ) -> None:
        super().__init__()
        num_previous = layer_index + 2
        self.use_softmax = bool(use_softmax)
        self.static_a = nn.Parameter(torch.empty(num_previous))
        self.w1 = nn.Linear(hidden_size, num_previous, bias=False)
        self.w2 = nn.Linear(num_previous, num_previous, bias=False)

    def forward(
        self,
        current_hidden: torch.Tensor,
        history_states: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the checkpoint's DD weighted sum over a stacked history."""

        if history_states.ndim < 2 or history_states.shape[-2] != self.static_a.numel():
            raise ValueError(
                f"DD expected {self.static_a.numel()} history states, "
                f"got shape {tuple(history_states.shape)}"
            )
        route_weights = apply_ncp_linear(
            self.w2,
            F.gelu(apply_ncp_linear(self.w1, current_hidden)),
        )
        route_weights = route_weights + self.static_a
        if self.use_softmax:
            route_weights = route_weights.softmax(dim=-1)
        return (route_weights.unsqueeze(-1) * history_states).sum(dim=-2)


class ConceptLMSelfDD(nn.Module):
    """Per-layer depth-dynamic routing modules."""

    def __init__(self, *, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.depth_dds = nn.ModuleList(
            [
                ConceptLMDepthDD(
                    hidden_size=hidden_size,
                    layer_index=layer_index,
                )
                for layer_index in range(num_layers)
            ]
        )


class ConceptLMSelfCumsumDD(nn.Module):
    """Single-state depth recurrence used by cumsum checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(()))

    def forward(
        self,
        current_hidden: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the routed hidden state and the next depth state."""

        alpha = torch.tanh(self.alpha).to(
            dtype=current_hidden.dtype,
            device=current_hidden.device,
        )
        next_state = current_hidden + alpha * previous_state
        return next_state, next_state


class ConceptLMDiagResidualRoute(nn.Module):
    """Softmax source mixer followed by the Stage3 diagonal residual add."""

    def __init__(self, *, hidden_size: int, num_sources: int) -> None:
        super().__init__()
        self.residual_diag = nn.Parameter(torch.empty(hidden_size))
        self.w1 = nn.Linear(hidden_size, num_sources, bias=False)
        self.w2 = nn.Linear(num_sources, num_sources, bias=False)

    def forward(
        self,
        target_hidden: torch.Tensor,
        normalized_sources: torch.Tensor,
        residual_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Read a stacked normalized source tensor and add its projection."""

        if (
            normalized_sources.ndim < 2
            or normalized_sources.shape[-2] != self.w2.out_features
        ):
            raise ValueError(
                f"residual route expected {self.w2.out_features} sources, "
                f"got shape {tuple(normalized_sources.shape)}"
            )
        weights = apply_ncp_linear(
            self.w2,
            F.gelu(apply_ncp_linear(self.w1, target_hidden)),
        ).softmax(dim=-1)
        source_mix = (weights.unsqueeze(-1) * normalized_sources).sum(dim=-2)
        residual_update = source_mix * self.residual_diag.to(source_mix.dtype)
        if residual_scale is not None:
            residual_update = residual_update * residual_scale.to(residual_update.dtype)
        return target_hidden + residual_update


class ConceptLMCrossCumsumRoute(nn.Module):
    """Read only the final normalized source with a trained scalar beta."""

    def __init__(self) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.empty(()))

    def forward(
        self,
        target_hidden: torch.Tensor,
        normalized_sources: torch.Tensor,
        residual_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the cumsum checkpoint's final-source residual route."""

        if normalized_sources.ndim < 2 or normalized_sources.shape[-2] == 0:
            return target_hidden
        update = normalized_sources[..., -1, :] * self.beta.to(
            dtype=target_hidden.dtype,
            device=target_hidden.device,
        )
        if residual_scale is not None:
            update = update * residual_scale.to(update.dtype)
        return target_hidden + update.to(target_hidden.dtype)


class ConceptLMHLMIncrementalAttention(nn.Module):
    """One-token causal attention over a request-owned dense HLM K/V cache."""

    def __init__(
        self,
        *,
        vllm_config: Any,
        backend_config: ConceptLMBackendConfig,
        layer_number: int,
        prefix: str,
    ) -> None:
        super().__init__()
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError("the first request-scoped HLM requires TP=1")
        raw_config = vllm_config.model_config.hf_config
        self.num_heads = backend_config.num_attention_heads
        self.num_kv_heads = backend_config.num_key_value_heads
        self.head_dim = backend_config.hidden_size // self.num_heads
        self.query_size = self.num_heads * self.head_dim
        self.key_value_size = self.num_kv_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        configured_fa_version = vllm_config.attention_config.flash_attn_version
        self.flash_attn_version = (
            int(configured_fa_version)
            if configured_fa_version is not None
            else get_flash_attn_version()
        )
        if self.flash_attn_version not in (2, 3):
            raise ValueError(
                "ConceptLM HLM requires vLLM FlashAttention version 2 or 3"
            )
        self.attention_impl = os.environ.get(
            "CONCEPTLM_HLM_ATTENTION_IMPL",
            "legacy_mixed",
        )
        if self.attention_impl not in ("legacy_mixed", "uniform_flash"):
            raise ValueError(
                "CONCEPTLM_HLM_ATTENTION_IMPL must be legacy_mixed or "
                f"uniform_flash, got {self.attention_impl!r}"
            )
        self.register_buffer(
            "_cu_seqlens_q",
            torch.zeros(2, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "_cu_seqlens_k",
            torch.zeros(2, dtype=torch.int32),
            persistent=False,
        )
        self.linear_qkv = QKVParallelLinear(
            backend_config.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.linear_qkv",
        )
        epsilon = float(_config_value(raw_config, "layernorm_epsilon"))
        self.qk_norm_mode = backend_config.qk_norm_mode
        self.q_layernorm = RMSNorm(
            backend_config.qk_norm_weight_size,
            eps=epsilon,
        )
        self.k_layernorm = RMSNorm(
            backend_config.qk_norm_weight_size,
            eps=epsilon,
        )
        self.sliding_window = _sliding_window(raw_config, layer_number)
        use_yarn = str(
            _config_value(raw_config, "position_embedding_type")
        ) == "yarn" and (
            self.sliding_window is None
            or not bool(_config_value(raw_config, "yarn_full_attn_layers_only"))
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=backend_config.max_model_len,
            is_neox_style=not bool(_config_value(raw_config, "rotary_interleaved")),
            rope_parameters=_rope_parameters(raw_config, apply_yarn=use_yarn),
        )
        self.linear_proj = RowParallelLinear(
            backend_config.hidden_size,
            backend_config.hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.linear_proj",
        )

    def _apply_qk_norm(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.qk_norm_mode == "per_head":
            query_shape = query.shape
            key_shape = key.shape
            query = self.q_layernorm(
                query.reshape(*query_shape[:-1], self.num_heads, self.head_dim)
            ).reshape(query_shape)
            key = self.k_layernorm(
                key.reshape(*key_shape[:-1], self.num_kv_heads, self.head_dim)
            ).reshape(key_shape)
            return query, key
        return self.q_layernorm(query), self.k_layernorm(key)

    def forward(
        self,
        position: int,
        hidden_states: torch.Tensor,
        kv_state: HLMKVState,
    ) -> torch.Tensor:
        """Append one concept token and attend to this request's HLM history."""

        mixed_qkv, _ = self.linear_qkv(hidden_states)
        query, key, value = mixed_qkv.split(
            [self.query_size, self.key_value_size, self.key_value_size],
            dim=-1,
        )
        query, key = self._apply_qk_norm(query, key)
        positions = torch.tensor(
            [position],
            dtype=torch.long,
            device=hidden_states.device,
        )
        query, key = self.rotary_emb(positions, query, key)
        query = query.view(1, self.num_heads, self.head_dim).transpose(0, 1)
        key = key.view(1, self.num_kv_heads, self.head_dim).transpose(0, 1)
        value = value.view(1, self.num_kv_heads, self.head_dim).transpose(0, 1)
        cached_key, cached_value = _append_hlm_kv(
            kv_state,
            key,
            value,
        )
        if self.sliding_window is not None:
            # Megatron's (left, 0) window includes the current token.
            keep = int(self.sliding_window) + 1
            cached_key = cached_key[:, -keep:]
            cached_value = cached_value[:, -keep:]
        if self.attention_impl == "legacy_mixed":
            attention_output = _legacy_decode_attention(
                query,
                cached_key,
                cached_value,
                scale=self.scale,
                output_dtype=hidden_states.dtype,
            )
        else:
            attention_output = _causal_prefill_attention(
                query,
                cached_key,
                cached_value,
                scale=self.scale,
                prefix_length=int(cached_key.shape[1]) - 1,
                sliding_window=self.sliding_window,
                flash_attn_version=self.flash_attn_version,
                cu_seqlens_q=self._cu_seqlens_q,
                cu_seqlens_k=self._cu_seqlens_k,
            )
        attention_output = attention_output.transpose(0, 1).reshape(
            1,
            self.num_heads * self.head_dim,
        )
        output, _ = self.linear_proj(attention_output)
        return output

    def forward_batch(
        self,
        position: int,
        hidden_states: torch.Tensor,
        kv_states: Sequence[HLMKVState],
    ) -> torch.Tensor:
        """Advance equal-length request caches with one batched HLM decode."""

        batch_size = int(hidden_states.shape[0])
        if batch_size <= 1 or len(kv_states) != batch_size:
            raise ValueError(
                "batched HLM attention requires matching batch and state sizes"
            )
        if self.attention_impl != "legacy_mixed":
            return torch.cat(
                [
                    self.forward(
                        position,
                        hidden_states[index : index + 1],
                        kv_state,
                    )
                    for index, kv_state in enumerate(kv_states)
                ],
                dim=0,
            )

        mixed_qkv, _ = self.linear_qkv(hidden_states)
        query, key, value = mixed_qkv.split(
            [self.query_size, self.key_value_size, self.key_value_size],
            dim=-1,
        )
        query, key = self._apply_qk_norm(query, key)
        positions = torch.full(
            (batch_size,),
            position,
            dtype=torch.long,
            device=hidden_states.device,
        )
        query, key = self.rotary_emb(positions, query, key)
        query = query.view(
            batch_size,
            self.num_heads,
            self.head_dim,
        ).unsqueeze(2)
        key = key.view(
            batch_size,
            self.num_kv_heads,
            self.head_dim,
        ).unsqueeze(2)
        value = value.view(
            batch_size,
            self.num_kv_heads,
            self.head_dim,
        ).unsqueeze(2)

        cached_keys = []
        cached_values = []
        for request_index, kv_state in enumerate(kv_states):
            request_key = key[request_index]
            request_value = value[request_index]
            prefix_length = int(kv_state.length)
            if prefix_length != position:
                raise ValueError(
                    "batched HLM cache length does not match position: "
                    f"{prefix_length} != {position}"
                )
            request_cached_key, request_cached_value = _append_hlm_kv(
                kv_state,
                request_key,
                request_value,
            )
            cached_keys.append(request_cached_key)
            cached_values.append(request_cached_value)

        cached_key = torch.stack(cached_keys, dim=0)
        cached_value = torch.stack(cached_values, dim=0)
        if self.sliding_window is not None:
            keep = int(self.sliding_window) + 1
            cached_key = cached_key[:, :, -keep:]
            cached_value = cached_value[:, :, -keep:]
        attention_output = _legacy_decode_attention(
            query,
            cached_key,
            cached_value,
            scale=self.scale,
            output_dtype=hidden_states.dtype,
        )
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size,
            self.num_heads * self.head_dim,
        )
        output, _ = self.linear_proj(attention_output)
        return output

    def prefill(
        self,
        position_start: int,
        hidden_states: torch.Tensor,
        kv_state: HLMKVState,
    ) -> torch.Tensor:
        """Append and process multiple concept tokens in one causal attention call."""

        num_tokens = int(hidden_states.shape[0])
        if num_tokens <= 1:
            raise ValueError("HLM batched prefill requires at least two tokens")
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        query, key, value = mixed_qkv.split(
            [self.query_size, self.key_value_size, self.key_value_size],
            dim=-1,
        )
        query, key = self._apply_qk_norm(query, key)
        positions = torch.arange(
            position_start,
            position_start + num_tokens,
            dtype=torch.long,
            device=hidden_states.device,
        )
        query, key = self.rotary_emb(positions, query, key)
        query = query.view(
            num_tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(0, 1)
        key = key.view(
            num_tokens,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(0, 1)
        value = value.view(
            num_tokens,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(0, 1)
        prefix_length = int(kv_state.length)
        if prefix_length != position_start:
            raise ValueError(
                "HLM cache length does not match prefill position: "
                f"{prefix_length} != {position_start}"
            )
        cached_key, cached_value = _append_hlm_kv(
            kv_state,
            key,
            value,
        )
        attention_output = _causal_prefill_attention(
            query,
            cached_key,
            cached_value,
            scale=self.scale,
            prefix_length=prefix_length,
            sliding_window=self.sliding_window,
            flash_attn_version=(
                None
                if self.attention_impl == "legacy_mixed"
                else self.flash_attn_version
            ),
            cu_seqlens_q=self._cu_seqlens_q,
            cu_seqlens_k=self._cu_seqlens_k,
        )
        attention_output = attention_output.transpose(0, 1).reshape(
            num_tokens,
            self.num_heads * self.head_dim,
        )
        output, _ = self.linear_proj(attention_output)
        return output


class ConceptLMHLMIncrementalLayer(nn.Module):
    """OLMo3 HLM layer with request-owned dense attention state."""

    def __init__(
        self,
        *,
        vllm_config: Any,
        backend_config: ConceptLMBackendConfig,
        layer_number: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_index = layer_number - 1
        self._trace_enabled = bool(os.environ.get("CONCEPTLM_VLLM_LOGITS_TRACE_DIR"))
        self._latest_stage_trace: dict[str, torch.Tensor] = {}
        raw_config = vllm_config.model_config.hf_config
        epsilon = float(_config_value(raw_config, "layernorm_epsilon"))
        self.self_attention = ConceptLMHLMIncrementalAttention(
            vllm_config=vllm_config,
            backend_config=backend_config,
            layer_number=layer_number,
            prefix=f"{prefix}.self_attention",
        )
        self.mlp = ConceptLMOlmo3MLP(
            vllm_config=vllm_config,
            backend_config=backend_config,
            prefix=f"{prefix}.mlp",
        )
        self.post_attention_layernorm = RMSNorm(
            backend_config.hidden_size,
            eps=epsilon,
        )
        self.post_feedforward_layernorm = RMSNorm(
            backend_config.hidden_size,
            eps=epsilon,
        )

    def _record_stage(self, name: str, value: torch.Tensor) -> None:
        if self._trace_enabled:
            self._latest_stage_trace[f"hlm.{name}.{self.layer_index}"] = (
                value[-1].detach().float().cpu()
            )

    def stage_trace(self) -> dict[str, torch.Tensor]:
        """Return the latest sublayer vectors used by the parity harness."""

        trace = dict(self._latest_stage_trace)
        trace.update(self.mlp.stage_trace())
        return trace

    def forward(
        self,
        *,
        position: int,
        hidden_states: torch.Tensor,
        kv_state: HLMKVState,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attention(position, hidden_states, kv_state)
        self._record_stage("attention.raw", hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        self._record_stage("attention.norm", hidden_states)
        hidden_states = residual + hidden_states
        self._record_stage("attention.residual", hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        self._record_stage("mlp.raw", hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        self._record_stage("mlp.norm", hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def prefill(
        self,
        *,
        position_start: int,
        hidden_states: torch.Tensor,
        kv_state: HLMKVState,
    ) -> torch.Tensor:
        """Run one HLM layer over a multi-concept causal prefill."""

        residual = hidden_states
        hidden_states = self.self_attention.prefill(
            position_start,
            hidden_states,
            kv_state,
        )
        self._record_stage("attention.raw", hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        self._record_stage("attention.norm", hidden_states)
        hidden_states = residual + hidden_states
        self._record_stage("attention.residual", hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        self._record_stage("mlp.raw", hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        self._record_stage("mlp.norm", hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def forward_batch(
        self,
        *,
        position: int,
        hidden_states: torch.Tensor,
        kv_states: Sequence[HLMKVState],
    ) -> torch.Tensor:
        """Run one HLM layer for multiple equal-position requests."""

        residual = hidden_states
        hidden_states = self.self_attention.forward_batch(
            position,
            hidden_states,
            kv_states,
        )
        self._record_stage("attention.raw", hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        self._record_stage("attention.norm", hidden_states)
        hidden_states = residual + hidden_states
        self._record_stage("attention.residual", hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        self._record_stage("mlp.raw", hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        self._record_stage("mlp.norm", hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class ConceptLMHLMBlock(nn.Module):
    """Locally numbered HLM transformer block."""

    def __init__(
        self,
        *,
        vllm_config: Any,
        backend_config: ConceptLMBackendConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConceptLMHLMIncrementalLayer(
                    vllm_config=vllm_config,
                    backend_config=backend_config,
                    layer_number=layer_index + 1,
                    prefix=f"{prefix}.layers.{layer_index}",
                )
                for layer_index in range(backend_config.hlm_layers)
            ]
        )
        self.final_layernorm = RMSNorm(
            backend_config.hidden_size,
            eps=float(
                _config_value(
                    vllm_config.model_config.hf_config,
                    "layernorm_epsilon",
                )
            ),
        )


class ConceptLMConceptPredictor(nn.Module):
    """Stage3 HLM, DD, encoder-read routes, and VQ prediction heads."""

    def __init__(
        self,
        *,
        vllm_config: Any,
        backend_config: ConceptLMBackendConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        hidden_size = backend_config.hidden_size
        epsilon = float(
            _config_value(
                vllm_config.model_config.hf_config,
                "layernorm_epsilon",
            )
        )
        self.hlm_block = ConceptLMHLMBlock(
            vllm_config=vllm_config,
            backend_config=backend_config,
            prefix=f"{prefix}.hlm_block",
        )
        self.prediction_heads = nn.ModuleList(
            [
                nn.Linear(hidden_size, backend_config.codebook_size)
                for _ in range(backend_config.num_codebooks)
            ]
        )
        if backend_config.dd_self_mode == "cumsum":
            self.concept_self_dd = ConceptLMSelfCumsumDD()
            self.concept_read_encoder_routes = nn.ModuleList(
                [ConceptLMCrossCumsumRoute() for _ in range(backend_config.hlm_layers)]
            )
        else:
            self.concept_self_dd = ConceptLMSelfDD(
                hidden_size=hidden_size,
                num_layers=backend_config.hlm_layers,
            )
            self.concept_read_encoder_routes = nn.ModuleList(
                [
                    ConceptLMDiagResidualRoute(
                        hidden_size=hidden_size,
                        num_sources=backend_config.encoder_layers - 1,
                    )
                    for _ in range(backend_config.hlm_layers)
                ]
            )
        self.concept_read_encoder_shared_source_norm = nn.LayerNorm(
            hidden_size,
            eps=epsilon,
        )


class ConceptLMProductCodebook(nn.Module):
    """Native ParameterList layout for the product codebooks."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_codebooks: int,
        codebook_size: int,
    ) -> None:
        super().__init__()
        head_dim = hidden_size // num_codebooks
        self.codebook = nn.ParameterList(
            [
                nn.Parameter(torch.empty(codebook_size, head_dim))
                for _ in range(num_codebooks)
            ]
        )

    def stacked(self) -> torch.Tensor:
        """Return [num_codebooks, codebook_size, head_dim]."""

        return torch.stack(tuple(self.codebook), dim=0)


class ConceptLMHighLevelBranch(nn.Module):
    """Request-scoped chunk/HLM path with exact checkpoint parameter names."""

    def __init__(
        self,
        *,
        vllm_config: Any,
        backend_config: ConceptLMBackendConfig,
    ) -> None:
        super().__init__()
        if vllm_config.quant_config is not None:
            raise NotImplementedError("the first ConceptLM HLM is unquantized")
        epsilon = float(
            _config_value(
                vllm_config.model_config.hf_config,
                "layernorm_epsilon",
            )
        )
        self.backend_config = backend_config
        self.cumsum_routes = backend_config.dd_self_mode == "cumsum"
        self.concept_vq_input_norm = nn.LayerNorm(
            backend_config.hidden_size,
            eps=epsilon,
        )
        self.concept_quantizer = ConceptLMProductCodebook(
            hidden_size=backend_config.hidden_size,
            num_codebooks=backend_config.num_codebooks,
            codebook_size=backend_config.codebook_size,
        )
        self.concept_predictor = ConceptLMConceptPredictor(
            vllm_config=vllm_config,
            backend_config=backend_config,
            prefix="concept_predictor",
        )
        self._trace_enabled = bool(os.environ.get("CONCEPTLM_VLLM_LOGITS_TRACE_DIR"))
        self._latest_stage_trace: dict[str, torch.Tensor] = {}

    def _record_stage(self, name: str, value: torch.Tensor) -> None:
        if self._trace_enabled:
            self._latest_stage_trace[name] = value[-1].detach().float().cpu()

    def stage_trace(self) -> dict[str, torch.Tensor]:
        """Return the latest HLM vectors used by the parity harness."""

        trace = dict(self._latest_stage_trace)
        for layer in self.concept_predictor.hlm_block.layers:
            trace.update(layer.stage_trace())
        return trace

    def _route_concept_layer(
        self,
        *,
        layer_index: int,
        raw_output: torch.Tensor,
        history_states: torch.Tensor,
        cumsum_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply either the legacy DD mixer or the cumsum depth recurrence."""

        if self.cumsum_routes:
            if cumsum_state is None:
                raise RuntimeError("HLM cumsum route state is missing")
            route = self.concept_predictor.concept_self_dd
            return route(raw_output, cumsum_state)
        route = self.concept_predictor.concept_self_dd
        return (
            route.depth_dds[layer_index](raw_output, history_states),
            cumsum_state,
        )

    def advance(
        self,
        request_state: ConceptRequestState,
        encoder_chunk: torch.Tensor,
        encoder_layer_chunks: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Advance the HLM by one completed token chunk."""

        if len(encoder_layer_chunks) != self.backend_config.encoder_layers:
            raise ValueError(
                f"expected {self.backend_config.encoder_layers} encoder layers, "
                f"got {len(encoder_layer_chunks)}"
            )
        hidden_states = apply_ncp_layer_norm(
            self.concept_vq_input_norm,
            encoder_chunk.reshape(1, self.backend_config.hidden_size),
        )
        self._record_stage("hlm.input.0", hidden_states)
        encoder_sources = torch.stack(
            tuple(
                source.reshape(1, self.backend_config.hidden_size)
                for source in encoder_layer_chunks[:-1]
            ),
            dim=-2,
        )
        encoder_sources = apply_ncp_layer_norm(
            self.concept_predictor.concept_read_encoder_shared_source_norm,
            encoder_sources,
        )
        dd_history = hidden_states.new_empty(
            (
                *hidden_states.shape[:-1],
                self.backend_config.hlm_layers + 1,
                hidden_states.shape[-1],
            )
        )
        dd_history[..., 0, :].copy_(hidden_states)
        cumsum_state = hidden_states if self.cumsum_routes else None
        raw_layer_states = []
        concept_position = request_state.predicted_concepts.length
        for layer_index, layer in enumerate(self.concept_predictor.hlm_block.layers):
            self._record_stage(f"hlm.input.{layer_index}", hidden_states)
            raw_output = layer(
                position=concept_position,
                hidden_states=hidden_states,
                kv_state=request_state.hlm_kv[layer_index],
            )
            self._record_stage(f"hlm.raw.{layer_index}", raw_output)
            raw_layer_states.append(raw_output)
            append_tensor_buffer(
                request_state.hlm_raw_layer_states[layer_index],
                raw_output,
                minimum_capacity=16,
            )
            dd_history[..., layer_index + 1, :].copy_(raw_output)
            hidden_states, cumsum_state = self._route_concept_layer(
                layer_index=layer_index,
                raw_output=raw_output,
                history_states=dd_history[..., : layer_index + 2, :],
                cumsum_state=cumsum_state,
            )
            hidden_states = self.concept_predictor.concept_read_encoder_routes[
                layer_index
            ](hidden_states, encoder_sources)
            self._record_stage(f"hlm.routed.{layer_index}", hidden_states)
        hidden_states = self.concept_predictor.hlm_block.final_layernorm(hidden_states)
        self._record_stage("hlm.final_norm", hidden_states)
        logits = torch.stack(
            [
                apply_ncp_linear(head, hidden_states)
                for head in self.concept_predictor.prediction_heads
            ],
            dim=1,
        )
        codebook = self.concept_quantizer.stacked().to(logits.dtype)
        predicted = torch.einsum("bhk,hkd->bhd", logits, codebook).reshape(
            1,
            self.backend_config.hidden_size,
        )
        append_tensor_buffer(
            request_state.predicted_concepts,
            predicted,
            minimum_capacity=16,
        )
        return predicted, tuple(raw_layer_states)

    def advance_batch(
        self,
        request_states: Sequence[ConceptRequestState],
        encoder_chunks: torch.Tensor,
        encoder_layer_chunks: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Advance equal-position requests through one batched HLM decode."""

        batch_size = len(request_states)
        if batch_size <= 1 or int(encoder_chunks.shape[0]) != batch_size:
            raise ValueError(
                "batched HLM advance requires matching batch and state sizes"
            )
        if len(encoder_layer_chunks) != self.backend_config.encoder_layers:
            raise ValueError(
                f"expected {self.backend_config.encoder_layers} encoder layers, "
                f"got {len(encoder_layer_chunks)}"
            )
        if any(int(source.shape[0]) != batch_size for source in encoder_layer_chunks):
            raise ValueError("all HLM encoder sources must match the batch size")
        concept_positions = {
            request_state.predicted_concepts.length for request_state in request_states
        }
        if len(concept_positions) != 1:
            raise ValueError("batched HLM requests must have the same concept position")
        concept_position = next(iter(concept_positions))

        hidden_states = apply_ncp_layer_norm(
            self.concept_vq_input_norm,
            encoder_chunks,
        )
        self._record_stage("hlm.input.0", hidden_states)
        encoder_sources = torch.stack(
            tuple(encoder_layer_chunks[:-1]),
            dim=-2,
        )
        encoder_sources = apply_ncp_layer_norm(
            self.concept_predictor.concept_read_encoder_shared_source_norm,
            encoder_sources,
        )
        dd_history = hidden_states.new_empty(
            (
                *hidden_states.shape[:-1],
                self.backend_config.hlm_layers + 1,
                hidden_states.shape[-1],
            )
        )
        dd_history[..., 0, :].copy_(hidden_states)
        cumsum_state = hidden_states if self.cumsum_routes else None
        raw_layer_states = []
        for layer_index, layer in enumerate(self.concept_predictor.hlm_block.layers):
            self._record_stage(f"hlm.input.{layer_index}", hidden_states)
            raw_output = layer.forward_batch(
                position=concept_position,
                hidden_states=hidden_states,
                kv_states=[
                    request_state.hlm_kv[layer_index]
                    for request_state in request_states
                ],
            )
            self._record_stage(f"hlm.raw.{layer_index}", raw_output)
            raw_layer_states.append(raw_output)
            for request_index, request_state in enumerate(request_states):
                append_tensor_buffer(
                    request_state.hlm_raw_layer_states[layer_index],
                    raw_output[request_index : request_index + 1],
                    minimum_capacity=16,
                )
            dd_history[..., layer_index + 1, :].copy_(raw_output)
            hidden_states, cumsum_state = self._route_concept_layer(
                layer_index=layer_index,
                raw_output=raw_output,
                history_states=dd_history[..., : layer_index + 2, :],
                cumsum_state=cumsum_state,
            )
            hidden_states = self.concept_predictor.concept_read_encoder_routes[
                layer_index
            ](hidden_states, encoder_sources)
            self._record_stage(f"hlm.routed.{layer_index}", hidden_states)
        hidden_states = self.concept_predictor.hlm_block.final_layernorm(hidden_states)
        self._record_stage("hlm.final_norm", hidden_states)
        logits = torch.stack(
            [
                apply_ncp_linear(head, hidden_states)
                for head in self.concept_predictor.prediction_heads
            ],
            dim=1,
        )
        codebook = self.concept_quantizer.stacked().to(logits.dtype)
        predicted = torch.einsum("bhk,hkd->bhd", logits, codebook).reshape(
            batch_size,
            self.backend_config.hidden_size,
        )
        for request_index, request_state in enumerate(request_states):
            append_tensor_buffer(
                request_state.predicted_concepts,
                predicted[request_index : request_index + 1],
                minimum_capacity=16,
            )
        return predicted, tuple(raw_layer_states)

    def prefill(
        self,
        request_state: ConceptRequestState,
        encoder_chunks: torch.Tensor,
        encoder_layer_chunks: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Advance a request over multiple completed chunks in parallel."""

        num_chunks = int(encoder_chunks.shape[0])
        if num_chunks <= 1:
            raise ValueError("HLM batched prefill requires at least two chunks")
        if len(encoder_layer_chunks) != self.backend_config.encoder_layers:
            raise ValueError(
                f"expected {self.backend_config.encoder_layers} encoder layers, "
                f"got {len(encoder_layer_chunks)}"
            )
        if any(int(source.shape[0]) != num_chunks for source in encoder_layer_chunks):
            raise ValueError("all HLM encoder sources must have the same batch length")

        hidden_states = apply_ncp_layer_norm(
            self.concept_vq_input_norm,
            encoder_chunks,
        )
        self._record_stage("hlm.input.0", hidden_states)
        encoder_sources = torch.stack(
            tuple(encoder_layer_chunks[:-1]),
            dim=-2,
        )
        encoder_sources = apply_ncp_layer_norm(
            self.concept_predictor.concept_read_encoder_shared_source_norm,
            encoder_sources,
        )
        dd_history = hidden_states.new_empty(
            (
                *hidden_states.shape[:-1],
                self.backend_config.hlm_layers + 1,
                hidden_states.shape[-1],
            )
        )
        dd_history[..., 0, :].copy_(hidden_states)
        cumsum_state = hidden_states if self.cumsum_routes else None
        raw_layer_states = []
        concept_position = request_state.predicted_concepts.length
        for layer_index, layer in enumerate(self.concept_predictor.hlm_block.layers):
            self._record_stage(f"hlm.input.{layer_index}", hidden_states)
            raw_output = layer.prefill(
                position_start=concept_position,
                hidden_states=hidden_states,
                kv_state=request_state.hlm_kv[layer_index],
            )
            self._record_stage(f"hlm.raw.{layer_index}", raw_output)
            raw_layer_states.append(raw_output)
            append_tensor_buffer(
                request_state.hlm_raw_layer_states[layer_index],
                raw_output,
                minimum_capacity=16,
            )
            dd_history[..., layer_index + 1, :].copy_(raw_output)
            hidden_states, cumsum_state = self._route_concept_layer(
                layer_index=layer_index,
                raw_output=raw_output,
                history_states=dd_history[..., : layer_index + 2, :],
                cumsum_state=cumsum_state,
            )
            hidden_states = self.concept_predictor.concept_read_encoder_routes[
                layer_index
            ](hidden_states, encoder_sources)
            self._record_stage(f"hlm.routed.{layer_index}", hidden_states)
        hidden_states = self.concept_predictor.hlm_block.final_layernorm(hidden_states)
        self._record_stage("hlm.final_norm", hidden_states)
        logits = torch.stack(
            [
                apply_ncp_linear(head, hidden_states)
                for head in self.concept_predictor.prediction_heads
            ],
            dim=1,
        )
        codebook = self.concept_quantizer.stacked().to(logits.dtype)
        predicted = torch.einsum("bhk,hkd->bhd", logits, codebook).reshape(
            num_chunks,
            self.backend_config.hidden_size,
        )
        append_tensor_buffer(
            request_state.predicted_concepts,
            predicted,
            minimum_capacity=16,
        )
        return predicted, tuple(raw_layer_states)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load HLM parameters from pure-HF checkpoint keys."""

        params = dict(self.named_parameters(remove_duplicate=False))
        parameter_names = frozenset(params)
        loaded_targets: dict[str, set[str | int | None]] = {}
        for checkpoint_name, loaded_weight in weights:
            if checkpoint_name.endswith("._extra_state"):
                continue
            target = resolve_checkpoint_weight(checkpoint_name, parameter_names)
            if target is None:
                continue
            target_shards = loaded_targets.setdefault(target.parameter_name, set())
            if target.shard_id in target_shards:
                raise ValueError(
                    "duplicate HLM checkpoint target: "
                    f"{checkpoint_name} -> {target.parameter_name}"
                )
            if (target.shard_id is None and target_shards) or (
                target.shard_id is not None and None in target_shards
            ):
                raise ValueError(
                    f"cannot mix full and split HLM tensors for {target.parameter_name}"
                )
            parameter = params[target.parameter_name]
            weight_loader = getattr(
                parameter,
                "weight_loader",
                default_weight_loader,
            )
            if target.shard_id is not None:
                weight_loader(parameter, loaded_weight, target.shard_id)
            else:
                weight_loader(parameter, loaded_weight)
            target_shards.add(target.shard_id)
        complete: set[str] = set()
        for parameter_name, loaded_shards in loaded_targets.items():
            required = required_checkpoint_shards(parameter_name)
            if loaded_shards == {None} or (
                required is not None and loaded_shards == required
            ):
                complete.add(parameter_name)
        missing = sorted(set(params) - complete)
        if missing:
            raise ValueError("missing HLM checkpoint parameters: " + ", ".join(missing))
        return complete
