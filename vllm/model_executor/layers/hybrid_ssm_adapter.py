#! SPDX-License-Identifier: Apache-2.0
#! SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import math
import os

import torch
import torch.nn as nn

from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    model_parallel_is_initialized,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_scan_fn,
    selective_state_update,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.attention.backends.hybrid_attn import HybridAttentionMetadata
from vllm.v1.attention.backends.mamba1_attn import (
    Mamba1AttentionBackend,
    Mamba1AttentionMetadata,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec


class HybridSSMAdapter(nn.Module, AttentionLayerBase):
    """
    History branch based on Mamba-style SSM state.

    This module exposes a minimal interface expected by the v1 KV cache /
    attention stack:

    - It behaves like an ``AttentionLayerBase`` so it can obtain its own
      ``MambaSpec`` KV pool (managed by ``MambaManager``).
    - It provides helper methods that the hybrid attention backend can call to
      obtain an SSM contribution over the same flattened token set as the
      sliding-window attention output.

    The current implementation focuses on wiring and KV-spec integration.
    The actual SSM compute path intentionally reuses the metadata layout of
    Mamba-1 (``Mamba1AttentionMetadata``) but returns a zero contribution for
    now. This keeps the feature opt‑in and avoids touching any CUDA kernels,
    while providing a scaffold to plug in the full Mamba pipeline later.
    """

    def __init__(
        self,
        hidden_size: int,
        ssm_state_size: int,
        conv_kernel_size: int,
        intermediate_size: int,
        *,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ssm_state_size = ssm_state_size
        self.conv_kernel_size = conv_kernel_size
        self.intermediate_size = intermediate_size

        # These are cached so that get_state_shape/get_state_dtype can be
        # computed consistently with existing Mamba layers.
        self.model_config = model_config
        self.cache_config = cache_config

        # Layer name used by vLLM's compilation / forward context.
        self.layer_name = prefix

        # Defaults for Mamba1
        self.time_step_rank = math.ceil(self.hidden_size / 16)
        self.use_conv_bias = True
        self.use_bias = False

        # Detect if we are running in a unit test without distributed env
        is_tp_init = model_parallel_is_initialized()
        disable_tp = not is_tp_init

        # Layers
        self.in_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=self.use_bias,
            prefix=f"{prefix}.in_proj",
            disable_tp=disable_tp,
        )
        self.conv1d = ColumnParallelLinear(
            self.conv_kernel_size,
            self.intermediate_size,
            bias=self.use_conv_bias,
            prefix=f"{prefix}.conv1d",
            disable_tp=disable_tp,
        )
        # Unsqueeze conv1d weight to match Mamba expectations (intermediate_size, 1, kernel_size)
        # But ColumnParallelLinear weight is (output_size, input_size) -> (intermediate_size, kernel_size)
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.x_proj = RowParallelLinear(
            self.intermediate_size,
            self.time_step_rank + self.ssm_state_size * 2,
            bias=False,
            prefix=f"{prefix}.x_proj",
            disable_tp=disable_tp,
        )
        self.dt_proj = ColumnParallelLinear(
            self.time_step_rank,
            self.intermediate_size,
            bias=True,
            skip_bias_add=True,
            prefix=f"{prefix}.dt_proj",
            disable_tp=disable_tp,
        )
        self.out_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=self.use_bias,
            input_is_parallel=True,
            prefix=f"{prefix}.out_proj",
            disable_tp=disable_tp,
        )

        # Parameters A and D
        if is_tp_init:
            tp_size = get_tensor_model_parallel_world_size()
        else:
            tp_size = 1

        self.A = nn.Parameter(
            torch.empty(
                self.intermediate_size // tp_size,
                self.ssm_state_size,
                dtype=torch.float32,
            )
        )
        self.D = nn.Parameter(torch.ones(self.intermediate_size // tp_size))

        # Weight loaders for A and D
        def weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            param.data.copy_(
                loaded_weight.data.split(loaded_weight.shape[0] // tp_size, dim=0)[
                    tp_rank
                ]
            )

        def A_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
            weight_loader(param, -torch.exp(loaded_weight.float()))

        set_weight_attrs(self.D, {"weight_loader": weight_loader})
        set_weight_attrs(self.A, {"weight_loader": A_weight_loader})

        # Simple debug / experimentation knob for the history branch.
        # By default the adapter returns a zero contribution (\"disabled\").
        # Set VLLM_HYBRID_SSM_MODE=prefix_sum to enable a trivial, non-zero
        # SSM rule that accumulates a prefix sum over the flattened token
        # dimension. This keeps the implementation lightweight while
        # allowing end-to-end testing of HybridAttentionImpl fusion without
        # introducing new CUDA kernels.
        self.ssm_mode: str = os.getenv("VLLM_HYBRID_SSM_MODE", "disabled")

        vllm_config = get_current_vllm_config()
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # Placeholder KV cache – this will be replaced by MambaManager via
        # bind_kv_cache() once the v1 engine initializes the cache tensors.
        self.kv_cache: tuple[torch.Tensor, ...] = (
            torch.tensor([]),
            torch.tensor([]),
        )

    # ------------------------------------------------------------------
    # KV cache spec / Mamba state description
    # ------------------------------------------------------------------
    def get_state_shape(self) -> Iterable[tuple[int, ...]]:
        """Return the logical shapes of the Mamba SSM state tensors.

        This mirrors ``MambaMixer.get_state_shape`` by delegating to
        ``MambaStateShapeCalculator`` so that the adapter can share the same
        ``MambaSpec`` / ``MambaManager`` infrastructure.

        In unit tests or single-process runs where model parallel has not been
        initialized yet, we conservatively assume a tensor-parallel world size
        of 1 instead of requiring a full distributed setup.
        """
        if model_parallel_is_initialized():
            tp_world_size = get_tensor_model_parallel_world_size()
        else:
            tp_world_size = 1
        return MambaStateShapeCalculator.mamba1_state_shape(
            tp_world_size=tp_world_size,
            intermediate_size=self.intermediate_size,
            state_size=self.ssm_state_size,
            conv_kernel=self.conv_kernel_size,
        )

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        """Return the dtypes of the Mamba SSM state tensors.

        The adapter mirrors the dtype choices of the Mamba-1 implementation,
        driven by the model and cache configuration.
        """
        # Defer to the runtime vLLM config if explicit configs were not
        # provided at construction time. This keeps the adapter usable from
        # simple unit tests where a full ``ModelConfig`` is not wired yet.
        model_config: ModelConfig
        cache_config: CacheConfig
        if self.model_config is None or self.cache_config is None:
            vllm_config = get_current_vllm_config()
            model_config = vllm_config.model_config
            cache_config = vllm_config.cache_config
        else:
            model_config = self.model_config
            cache_config = self.cache_config

        return MambaStateDtypeCalculator.mamba1_state_dtype(
            model_config.dtype,
            cache_config.mamba_cache_dtype,
            cache_config.mamba_ssm_cache_dtype,
        )

    # ------------------------------------------------------------------
    # AttentionLayerBase integration
    # ------------------------------------------------------------------
    def get_attn_backend(self) -> type[AttentionBackend]:
        """Use the existing Mamba-1 backend for KV grouping / metadata."""
        return Mamba1AttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        """Expose a ``MambaSpec`` so the adapter obtains its own KV pool.

        This allows the v1 KV cache manager to allocate a dedicated Mamba
        state pool (managed by ``MambaManager``) alongside standard
        sliding-window KV pages for attention.
        """
        # Follow the same speculative decoding constraints as MambaBase.
        if (
            vllm_config.speculative_config is not None
            and vllm_config.model_config.hf_config.model_type not in ["qwen3_next"]
        ):
            raise NotImplementedError(
                "Hybrid SSM adapter with speculative decoding is not supported yet."
            )

        mamba_block_size = vllm_config.cache_config.mamba_block_size
        page_size_padded = vllm_config.cache_config.mamba_page_size_padded

        return MambaSpec(
            shapes=tuple(self.get_state_shape()),
            dtypes=self.get_state_dtype(),
            block_size=mamba_block_size,
            page_size_padded=page_size_padded,
            mamba_type="mamba1",
            num_speculative_blocks=(
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config
                else 0
            ),
        )

    # ------------------------------------------------------------------
    # History-branch API used by HybridAttentionImpl
    # ------------------------------------------------------------------
    def _get_mamba_attn_metadata(self) -> Any | None:
        """Fetch the Mamba1AttentionMetadata for this adapter, if present."""
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            return attn_metadata.get(self.layer_name, None)
        return None

    def forward_history_branch_prefill(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: HybridAttentionMetadata | Any | None = None,
    ) -> torch.Tensor:
        """History branch for prefill tokens.

        By default this method returns a zero contribution while ensuring that
        the tensor is correctly shaped and indexed over the same flattened
        token set as the sliding-window attention output.

        When the environment variable ``VLLM_HYBRID_SSM_MODE`` is set to
        ``\"prefix_sum\"``, a simple, fully deterministic SSM rule is enabled:

        - The adapter computes a prefix sum over the first
          ``num_prefill_tokens`` positions along the token dimension and
          returns zeros elsewhere.

        This is intentionally lightweight and does not touch any custom CUDA
        kernels, but it allows the hybrid backend to observe a non‑trivial,
        history‑dependent contribution for experimentation and unit tests.
        """
        if attn_metadata is None:
            attn_metadata = self._get_mamba_attn_metadata()

        # Unwrap composite metadata if needed
        if isinstance(attn_metadata, HybridAttentionMetadata):
            mamba_metadata = attn_metadata.mamba_metadata
        else:
            mamba_metadata = attn_metadata

        if mamba_metadata is None:
            # Profiling / shape-only runs: match the input shape.
            return torch.zeros_like(hidden_states)

        num_actual_tokens: int = getattr(mamba_metadata, "num_prefill_tokens", 0)
        if num_actual_tokens <= 0:
            return torch.zeros_like(hidden_states)

        # Fast path: keep the adapter as a no-op unless explicitly enabled.
        if self.ssm_mode == "prefix_sum":
            # Generic over hidden_states rank: we treat dim 0 as the flattened
            # token dimension and preserve all remaining dimensions.
            prefix = torch.cumsum(hidden_states[:num_actual_tokens], dim=0)
            ssm_out = torch.zeros_like(hidden_states)
            ssm_out[:num_actual_tokens] = prefix
            return ssm_out

        if self.ssm_mode == "disabled":
            return torch.zeros_like(hidden_states)

        # Mamba 1 Forward Pass (Prefill)
        # 1. In Projection: (batch, seq, dim) -> (batch, seq, 2*inner)
        # hidden_states is (total_tokens, dim)
        xz, _ = self.in_proj(hidden_states[:num_actual_tokens])
        x, z = xz.chunk(2, dim=-1)

        # 2. Convolution
        # x needs to be (dim, total_tokens) for causal_conv1d_fn with varlen
        x_t = x.transpose(0, 1).contiguous()
        conv_weight = self.conv1d.weight
        conv_bias = self.conv1d.bias

        # Metadata fields
        # query_start_loc_p: (batch+1,)
        # state_indices_tensor: (batch, n_blocks) or (batch,)
        query_start_loc = mamba_metadata.query_start_loc_p
        cache_indices = mamba_metadata.state_indices_tensor
        has_initial_state = mamba_metadata.has_initial_states_p
        block_idx_first = mamba_metadata.block_idx_first_scheduled_token_p
        block_idx_last = mamba_metadata.block_idx_last_scheduled_token
        num_computed_tokens = mamba_metadata.num_computed_tokens_p

        # kv_cache[0] is conv_state, kv_cache[1] is ssm_state
        conv_state = self.kv_cache[0]
        ssm_state = self.kv_cache[1]

        x_conv = causal_conv1d_fn(
            x_t,
            conv_weight,
            conv_bias,
            conv_states=conv_state,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation="silu",
            block_idx_first_scheduled_token=block_idx_first,
            block_idx_last_scheduled_token=block_idx_last,
            initial_state_idx=block_idx_first,  # Use first token block for init?
            num_computed_tokens=num_computed_tokens,
        )
        # Transpose back to (total_tokens, dim)
        x = x_conv.transpose(0, 1)

        # 3. SSM
        x_dbl = self.x_proj(x)  # (total_tokens, dt_rank + 2*d_state)
        dt, B, C = torch.split(
            x_dbl, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        dt = self.dt_proj(dt)  # (total_tokens, inner_dim)

        # A parameter needs to be passed as -exp(A)
        # But we store it as A. MambaMixer uses A_weight_loader to store -exp(A) in the parameter?
        # In my __init__, I used A_weight_loader which loads it as -exp(float(weight)).
        # So self.A already contains -exp(A).
        # However, selective_scan_fn expects A.
        # Wait, `MambaMixer` in vLLM sets:
        # weight_loader(param, -torch.exp(loaded_weight.float()))
        # So self.A is -exp(A_original).
        # selective_scan_fn uses A directly.
        # So we just pass self.A.

        # x, dt, z are (total_tokens, dim)
        # B, C are (total_tokens, d_state)
        # selective_scan_fn handles varlen with query_start_loc
        y = selective_scan_fn(
            x,
            ssm_state,
            dt,
            self.A,
            B,
            C,
            self.D,
            z,
            self.dt_proj.bias,
            delta_softplus=True,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            block_idx_first_scheduled_token=block_idx_first,
            block_idx_last_scheduled_token=block_idx_last,
            initial_state_idx=block_idx_first,
        )

        # 4. Out Projection
        out = self.out_proj(y)

        # Pad output if needed to match hidden_states size (if we only processed valid tokens)
        # hidden_states is (total_slots, dim) maybe?
        # num_actual_tokens is what we processed.
        if out.shape[0] < hidden_states.shape[0]:
            full_out = torch.zeros_like(hidden_states)
            full_out[:num_actual_tokens] = out
            return full_out
        
        return out

    def forward_history_branch_decode(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: Any | None = None,
    ) -> torch.Tensor:
        """History branch for decode tokens.

        The adapter is expected to produce an SSM contribution aligned with the
        flattened decode token set.

        By default this method returns a zero tensor but wires in the same
        metadata shape as Mamba-1 so that a future implementation can swap in
        the full Mamba pipeline without changing call sites.

        When ``VLLM_HYBRID_SSM_MODE=prefix_sum`` is set, a simple prefix-sum
        history rule is applied over the first ``num_decode_tokens`` (or, if
        unavailable, ``num_actual_tokens``) positions along the token
        dimension, mirroring the prefill behavior.
        """
        if attn_metadata is None:
            attn_metadata = self._get_mamba_attn_metadata()

        # Unwrap composite metadata if needed
        if isinstance(attn_metadata, HybridAttentionMetadata):
            mamba_metadata = attn_metadata.mamba_metadata
        else:
            mamba_metadata = attn_metadata

        if mamba_metadata is None:
            # Profiling / shape-only runs: match the input shape.
            return torch.zeros_like(hidden_states)

        # Prefer decode-specific counts when available (used in unit tests),
        # but fall back to the generic num_actual_tokens field exposed by
        # Triton-style attention metadata.
        num_actual_tokens: int | None = getattr(
            mamba_metadata, "num_decode_tokens", None
        )
        if num_actual_tokens is None:
            # Only if we passed Triton metadata by mistake, but we are unwrapping.
            # Mamba metadata has num_decode_tokens.
            num_actual_tokens = getattr(mamba_metadata, "num_actual_tokens", 0)

        if num_actual_tokens <= 0:
            return torch.zeros_like(hidden_states)

        if self.ssm_mode == "prefix_sum":
            prefix = torch.cumsum(hidden_states[:num_actual_tokens], dim=0)
            ssm_out = torch.zeros_like(hidden_states)
            ssm_out[:num_actual_tokens] = prefix
            return ssm_out

        if self.ssm_mode == "disabled":
            return torch.zeros_like(hidden_states)

        # Mamba 1 Forward Pass (Decode)
        # hidden_states: (num_decodes, dim)
        # Processing one token per sequence.
        
        # 1. In Projection
        xz, _ = self.in_proj(hidden_states[:num_actual_tokens])
        x, z = xz.chunk(2, dim=-1)

        # 2. Conv1d Step
        conv_state = self.kv_cache[0]
        ssm_state = self.kv_cache[1]
        cache_indices = mamba_metadata.state_indices_tensor
        # For decode, block_idx_last_scheduled_token tells where to write/read the step state?
        # Actually cache_indices is (batch, max_blocks) or (batch,).
        # causal_conv1d_update takes conv_state_indices.
        # In Mamba1AttentionMetadata, state_indices_tensor is the block table.
        
        block_idx_last = mamba_metadata.block_idx_last_scheduled_token # (batch,)
        
        x = causal_conv1d_update(
            x,
            conv_state,
            self.conv1d.weight,
            self.conv1d.bias,
            activation="silu",
            conv_state_indices=cache_indices,
            block_idx_last_scheduled_token=block_idx_last,
            # initial_state_idx?
        )

        # 3. SSM Step
        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(
            x_dbl, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        dt = self.dt_proj(dt)
        
        # selective_state_update(state, x, dt, A, B, C, D, z, dt_bias, dt_softplus=True)
        # state: ssm_state (batch, dim, dstate) - wait, here we have KV cache which might be paged?
        # ssm_state in kv_cache[1].
        # selective_state_update supports state_batch_indices (cache_indices).
        
        # Note: state_batch_indices is cache_indices.
        # If we use prefix caching, we have block indices.
        # Does selective_state_update support block indices?
        # It takes `state_batch_indices`.
        # If state is (num_blocks, ...), then state_batch_indices should be (batch,) pointing to the block.
        # But we have multiple blocks per request?
        # No, for Mamba state, we only need the *current* state.
        # But `ssm_state` is allocated as `(num_blocks, ...)` or `(tot_blocks, ...)`?
        # The `MambaSpec` defines the shape.
        # `selective_state_update` updates state in place.
        # We need to pass the index of the block that holds the current state.
        # `block_idx_last_scheduled_token`?
        # Wait, Mamba state is size-fixed per sequence. Why blocks?
        # vLLM uses blocks to manage memory.
        # For Mamba, the state is small (dim*dstate).
        # Does it span multiple blocks?
        # `MambaSpec` defines `block_size`.
        # If `block_size` is 1 (or small), we might need to know which block holds the state.
        # But Mamba state is a *hidden state* (recurrent). It doesn't grow with sequence length like KV cache.
        # So there is only ONE state per sequence (or per head/layer).
        # Why `MambaSpec`?
        # To allocate memory in the block manager.
        # If Mamba state fits in one block, fine.
        # `MambaManager` allocates blocks.
        # For Mamba, we usually just need one block per sequence to store the state?
        # Yes, "SSM State: A fixed-size, recurrent state".
        # So `cache_indices` should point to *that* block.
        # In `Mamba1AttentionMetadata`, `state_indices_tensor` is `block_table_tensor`.
        # If we assume 1 block per seq, `block_table_tensor[:, 0]` gives the index.
        # `Mamba1AttentionMetadataBuilder` handles this:
        # if enable_prefix_caching: returns full table.
        # else: returns `block_table_tensor[:, 0]`.
        
        # `selective_state_update` takes `state_batch_indices`.
        # If `cache_indices` is 1D (batch,), it works.
        # If it is 2D (batch, blocks), we need to select the right one.
        # `block_idx_last_scheduled_token` points to the last block?
        # If Mamba state is always in the *last* block?
        # Or if Mamba state is *distributed*? No, it's fixed size.
        # It should be in *one* block (or a set of blocks representing the state).
        # But vLLM Mamba implementation seems to treat the state as being stored in the "last" scheduled block or using `block_idx_last_scheduled_token` to find it?
        # Actually, looking at `causal_conv1d_update` call above:
        # `conv_state_indices=cache_indices`, `block_idx_last_scheduled_token=block_idx_last`.
        # It seems to handle indirection.
        
        # For `selective_state_update`:
        # It has `state_batch_indices`.
        # Does it accept `block_idx_...`?
        # No, the signature is:
        # def selective_state_update(state, x, dt, A, B, C, D=None, z=None, ..., state_batch_indices=None, ...)
        # It doesn't seem to support the advanced block lookup that `causal_conv1d` does.
        # Wait, `vllm/model_executor/layers/mamba/ops/mamba_ssm.py`:
        # `selective_state_update` documentation says `state_batch_indices: (batch,)`.
        # It doesn't mention `block_idx`.
        
        # So we must pass the *actual* block index for each request.
        # If `cache_indices` is 2D (from metadata with APC), we need to select the correct block index.
        # If APC is enabled, `block_idx_last_scheduled_token` holds the index *into* `cache_indices`?
        # `causal_conv1d_update` docs:
        # "The pointer into conv_state_indices, where the last cache block to be filled is located."
        
        # So `real_index = cache_indices[i, block_idx_last[i]]`.
        # We need to gather these indices if `cache_indices` is 2D.
        
        # Mamba1AttentionMetadata logic:
        # If `enable_prefix_caching`: `state_indices_tensor` is `block_table_tensor` (2D).
        # Else: `block_table_tensor[:, 0]` (1D).
        
        # So if 2D, we need to gather.
        # But `selective_state_update` expects 1D `state_batch_indices`.
        
        real_indices = cache_indices
        if cache_indices.dim() == 2:
             # Gather
             # block_idx_last is (batch,).
             # cache_indices is (batch, max_blocks).
             # We want cache_indices[range(batch), block_idx_last].
             # But wait, block_idx_last is int32 tensor.
             real_indices = cache_indices.gather(1, block_idx_last.unsqueeze(1)).squeeze(1)
        
        out = selective_state_update(
            ssm_state,
            x,
            dt,
            self.A,
            B,
            C,
            self.D,
            z,
            self.dt_proj.bias,
            dt_softplus=True,
            state_batch_indices=real_indices,
        )
        
        return self.out_proj(out)


