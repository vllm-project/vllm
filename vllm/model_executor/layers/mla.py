# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from vllm.config import CacheConfig
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.quantization import QuantizationConfig

# Try to import AITER ops for fused kernels
try:
    from aiter import dtypes
    from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant
    _AITER_AVAILABLE = True
except ImportError:
    _AITER_AVAILABLE = False
    dtypes = None
    fused_rms_fp8_group_quant = None


def _fuse_rmsnorm_quant(
    q_c: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_a_layernorm_variance_epsilon: float,
    kv_c: torch.Tensor,
    kv_a_layernorm_weight: torch.Tensor,
    kv_a_layernorm_variance_epsilon: float,
    dtype_quant=None,  # dtypes.fp8
    group_size: int = 128,
    output_unquantized_inp1: bool = False,
    transpose_scale: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fused dual RMSNorm + FP8 quantization.

    Fuses:
    1. RMSNorm on q_c
    2. FP8 group quantization on q_c
    3. RMSNorm on kv_c (without quantization)

    Based on ATOM's implementation in deepseek_v2.py:283 (_fuse_rmsnorm_quant)

    Returns:
        (q_c_quantized, q_c_scale, kv_c_normed) if successful, else (None, None, None)
    """
    if not _AITER_AVAILABLE:
        return None, None, None

    if dtype_quant is None:
        dtype_quant = dtypes.fp8

    if dtype_quant != dtypes.fp8:
        return None, None, None

    try:
        # Call AITER's fused kernel
        # Returns: (out1_quantized, out1_bs), out1_unquantized, out2, out_res1
        (q_c_quantized, q_c_scale), _, kv_c_normed, _ = fused_rms_fp8_group_quant(
            q_c,                             # x1: first input to normalize + quantize
            q_a_layernorm_weight,            # x1_weight: RMSNorm weight for q_c
            q_a_layernorm_variance_epsilon,  # x1_epsilon: epsilon for q_c
            kv_c,                            # x2: second input to normalize (no quant)
            kv_a_layernorm_weight,           # x2_weight: RMSNorm weight for kv_c
            kv_a_layernorm_variance_epsilon, # x2_epsilon: epsilon for kv_c
            group_size,                      # group_size: 128 elements per group
            dtype_quant,                     # dtype_quant: dtypes.fp8
            None,                            # res1: no residual connection
            output_unquantized_inp1,         # output_unquantized_inp1: False
            transpose_scale,                 # transpose_scale: True
        )

        return q_c_quantized, q_c_scale, kv_c_normed

    except Exception:
        return None, None, None


@dataclass
class MLAModules:
    """Modules used in MLA."""

    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    rotary_emb: torch.nn.Module
    o_proj: torch.nn.Module
    fused_qkv_a_proj: torch.nn.Module | None
    kv_a_proj_with_mqa: torch.nn.Module | None
    q_a_layernorm: torch.nn.Module | None
    q_b_proj: torch.nn.Module | None
    q_proj: torch.nn.Module | None
    indexer: torch.nn.Module | None
    is_sparse: bool
    topk_indices_buffer: torch.Tensor | None
    indexer_rotary_emb: torch.nn.Module | None = None


# --8<-- [start:multi_head_latent_attention]
@PluggableLayer.register("multi_head_latent_attention")
class MultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer which allows OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently oot platforms can still use CustomOp.register_oot to
    replace MLA layer entirly, although we use PluggableLayer to register
    this layer now.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    # --8<-- [end:multi_head_latent_attention]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.indexer = mla_modules.indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse

        if self.indexer is not None:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens
            self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.mla_attn = MLAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
        )

        self.prefix = prefix

        # Determine if RMSNorm+Quant fusion should be enabled (ATOM pattern)
        # Store quant_config and determine fusion dtype at init time
        self.quant_config = quant_config
        self.quant_dtype = None
        self.fuse_qknorm_quant = False

        if _AITER_AVAILABLE and quant_config is not None:
            # Check if quant_config is FP8
            from vllm.model_executor.layers.quantization.fp8 import Fp8Config
            if isinstance(quant_config, Fp8Config):
                self.quant_dtype = dtypes.fp8
                self.fuse_qknorm_quant = True

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None
        q_c_scale = None  # For FP8 quantized path

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None"
            )
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None"
            )

            # Step 1: QKV projection (use existing layer)
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

            # Step 2: Try fused RMSNorm + FP8 quantization
            # Only attempt fusion if enabled in __init__ (based on quant_config)
            if self.fuse_qknorm_quant:
                q_c_fused, q_c_scale, kv_c_normed_fused = _fuse_rmsnorm_quant(
                    q_c,
                    self.q_a_layernorm.weight,
                    self.q_a_layernorm.variance_epsilon,
                    kv_c,
                    self.kv_a_layernorm.weight,
                    self.kv_a_layernorm.variance_epsilon,
                    dtype_quant=self.quant_dtype,  # Use dtype determined in __init__
                    group_size=128,
                    output_unquantized_inp1=False,
                    transpose_scale=True,
                )
            else:
                # Fusion disabled, set to None to trigger unfused path
                q_c_fused = None
                q_c_scale = None
                kv_c_normed_fused = None

            # Try to use fused path
            fused_succeeded = False
            if q_c_fused is not None:
                try:
                    # Attempt to use FP8 quantized q_c
                    if q_c_scale is not None:
                        try:
                            q = self.q_b_proj((q_c_fused, q_c_scale))[0]
                        except (TypeError, IndexError):
                            # q_b_proj doesn't support tuple input, dequantize
                            q_c_dequant = q_c_fused.to(hidden_states.dtype)
                            q = self.q_b_proj(q_c_dequant)[0]
                    else:
                        # No scale (shouldn't happen with FP8, but handle it)
                        q = self.q_b_proj(q_c_fused)[0]

                    # If we got here, use fused kv_c_normed
                    kv_c_normed = kv_c_normed_fused
                    fused_succeeded = True
                except Exception:
                    # Any error in fused path (including dequant), fall back
                    fused_succeeded = False

            if not fused_succeeded:
                # Unfused fallback path
                q_c = self.q_a_layernorm(q_c)
                kv_c_normed = self.kv_a_layernorm(kv_c)
                q = self.q_b_proj(q_c)[0]
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]
            kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        if self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )

        if self.indexer and self.is_sparse:
            _topk_indices = self.indexer(
                hidden_states, q_c, positions, self.indexer_rope_emb
            )

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
        )

        return self.o_proj(attn_out)[0]
