# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.config import CacheConfig
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.quantization import QuantizationConfig

logger = init_logger(__name__)

# Import AITER ops for fused RMSNorm + FP8 quantization
try:
    from aiter import dtypes
    from aiter.jit.utils.torch_guard import torch_compile_guard
    from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant

    _AITER_AVAILABLE = True
except ImportError:
    _AITER_AVAILABLE = False
    dtypes = None
    torch_compile_guard = None
    fused_rms_fp8_group_quant = None


def _fused_rms_fp8_group_quant_fake(
    q_c: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_a_layernorm_variance_epsilon: float,
    kv_c: torch.Tensor,
    kv_a_layernorm_weight: torch.Tensor,
    kv_a_layernorm_variance_epsilon: float,
    dtype_quant: torch.dtype | None = None,
    group_size: int = 128,
    output_unquantized_inp1: bool = False,
    transpose_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake implementation for torch.compile/CUDA graphs."""
    if dtype_quant is None:
        dtype_quant = dtypes.fp8
    m, n1 = q_c.shape
    out1_quantized = torch.empty((m, n1), dtype=dtype_quant, device=q_c.device)
    out1_bs = torch.empty(
        (m, (n1 + group_size - 1) // group_size), dtype=torch.float32, device=q_c.device
    )
    if transpose_scale:
        out1_bs = out1_bs.transpose(0, 1).contiguous().view(*out1_bs.shape)
    out2 = torch.empty_like(kv_c)
    return out1_quantized, out1_bs, out2


def _fuse_rmsnorm_quant_impl(
    q_c: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_a_layernorm_variance_epsilon: float,
    kv_c: torch.Tensor,
    kv_a_layernorm_weight: torch.Tensor,
    kv_a_layernorm_variance_epsilon: float,
    dtype_quant: torch.dtype | None = None,
    group_size: int = 128,
    output_unquantized_inp1: bool = False,
    transpose_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused dual RMSNorm + FP8 quantization using AITER.

    Fuses RMSNorm on q_c with FP8 group quantization, and RMSNorm on kv_c
    without quantization.

    Returns:
        (q_c_quantized, q_c_scale, kv_c_normed)
    """
    (q_c_quantized, q_c_scale), _, kv_c_normed, _ = fused_rms_fp8_group_quant(
        q_c,
        q_a_layernorm_weight,
        q_a_layernorm_variance_epsilon,
        kv_c,
        kv_a_layernorm_weight,
        kv_a_layernorm_variance_epsilon,
        group_size,
        dtype_quant,
        None,
        output_unquantized_inp1,
        transpose_scale,
    )
    return q_c_quantized, q_c_scale, kv_c_normed


# Apply torch_compile_guard decorator when AITER is available
if _AITER_AVAILABLE:
    _fuse_rmsnorm_quant = torch_compile_guard(gen_fake=_fused_rms_fp8_group_quant_fake)(
        _fuse_rmsnorm_quant_impl
    )
else:
    _fuse_rmsnorm_quant = _fuse_rmsnorm_quant_impl


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
    replace MLA layer entirely, although we use PluggableLayer to register
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

        # Extract RoPE caches for AITER fused kernels
        if self.rotary_emb is not None:
            # RoPE stores combined cos_sin_cache, need to split it
            # Format: [seq_len, rotary_dim] where first half is cos, second half is sin
            cos_sin_cache = self.rotary_emb.cos_sin_cache
            rotary_dim = self.rotary_emb.rotary_dim
            half_dim = rotary_dim // 2
            self.cos_cache = cos_sin_cache[:, :half_dim]
            self.sin_cache = cos_sin_cache[:, half_dim:]
            self.is_neox_style = self.rotary_emb.is_neox_style
        else:
            self.cos_cache = None
            self.sin_cache = None
            self.is_neox_style = False

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
            # Pass RoPE caches for AITER fused kernels
            cos_cache=self.cos_cache,
            sin_cache=self.sin_cache,
            is_neox_style=self.is_neox_style,
            # Pass RoPE module (static, doesn't change)
            rotary_emb=self.rotary_emb,
        )

        self.prefix = prefix

        # Enable RMSNorm+Quant fusion when AITER is available with FP8
        self.quant_config = quant_config
        self.quant_dtype = None
        self.fuse_qknorm_quant = False

        if _AITER_AVAILABLE and quant_config is not None:
            from vllm.model_executor.layers.quantization.fp8 import Fp8Config

            if isinstance(quant_config, Fp8Config):
                self.quant_dtype = dtypes.fp8
                self.fuse_qknorm_quant = True
                logger.info(
                    "[MLA_FUSION_INIT] Fusion enabled for %s: "
                    "AITER available and FP8 quantization detected",
                    prefix,
                )

        # VERIFICATION: Confirm all_mla_fused_mixed_batch branch is active
        logger.warning("MLA.PY ALL_MLA_FUSED_MIXED_BATCH BRANCH ACTIVE - 2026-03-20")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None
        q_c_scale = None  # Set when fuse_qknorm_quant is enabled

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
            kv_c, k_pe = kv_lora.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

            # Step 2: Apply RMSNorm and optional FP8 quantization
            if self.fuse_qknorm_quant:
                # Fused RMSNorm + FP8 quantization
                q_c_quantized, q_c_scale, kv_c_normed = _fuse_rmsnorm_quant(
                    q_c,
                    self.q_a_layernorm.weight,
                    self.q_a_layernorm.variance_epsilon,
                    kv_c,
                    self.kv_a_layernorm.weight,
                    self.kv_a_layernorm.variance_epsilon,
                    dtype_quant=self.quant_dtype,
                    group_size=128,
                    output_unquantized_inp1=False,
                    transpose_scale=True,
                )
                q = self.q_b_proj(q_c_quantized, x_scale=q_c_scale)[0]
            else:
                # Unfused path: RMSNorm only
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
            kv_c, k_pe = kv_lora.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)

        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        # VERIFY: Log mla.py outputs before RoPE (EAGER MODE ONLY)
        # COMMENTED OUT: Breaks torch compile / CUDA graph capture
        # from vllm.logger import init_logger
        # logger = init_logger(__name__)
        # logger.warning(
        #     f"[VERIFY MLA] BEFORE RoPE: "
        #     f"q: abs_max={q.float().abs().max().item():.6e}, "
        #     f"first_3={q[0,0,:3].tolist()}, "
        #     f"k_pe: abs_max={k_pe.float().abs().max().item():.6e}, "
        #     f"first_3={k_pe[0,0,:3].tolist()}, "
        #     f"kv_c_normed: abs_max="
        #     f"{kv_c_normed.float().abs().max().item():.6e}, "
        #     f"first_3={kv_c_normed[0,:3].tolist()}"
        # )

        # STEP 3: Determine if fused path can be used (SINGLE CHECK)
        # Check all requirements once and use everywhere
        can_use_fused_path = (
            hasattr(self.mla_attn, "use_aiter_fused")
            and self.mla_attn.use_aiter_fused  # Platform supports fused kernel
            and positions is not None  # Required for RoPE
            and self.rotary_emb is not None  # RoPE module available
        )

        # Apply RoPE based on fused vs unfused path
        if self.rotary_emb is not None:
            if can_use_fused_path:
                # FUSED PATH: Skip RoPE here, custom op will apply it
                # Problem: num_decode_tokens retrieved from forward_context gets
                # frozen as a constant when CUDA graph is captured, causing RoPE
                # to be applied to wrong tokens (e.g., q[512:] instead of q[1:])
                # Solution: Move RoPE to custom op (splitting op, not compiled)
                # where attn_metadata.num_decode_tokens is available dynamically
                pass
            else:
                # UNFUSED PATH: Apply RoPE to ALL tokens
                q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                    positions,
                    q[..., self.qk_nope_head_dim :],  # Q PE part gets RoPE
                    k_pe,  # K PE gets RoPE
                )

                # Log AFTER RoPE
                # logger.warning(
                #     f"[UNFUSED AFTER ROPE] "
                #     f"q_pe_abs_max="
                #     f"{q[..., self.qk_nope_head_dim:].float().abs().max()"
                #     f".item():.6e}, "
                #     f"k_pe_abs_max={k_pe.float().abs().max().item():.6e}, "
                #     f"k_pe_after_first3={k_pe[0, 0, :3].tolist()}"
                # )

        if self.indexer and self.is_sparse:
            _topk_indices = self.indexer(
                hidden_states, q_c, positions, self.indexer_rope_emb
            )

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        # STEP 4: Store rotary_emb in forward_context for custom ops
        # positions is now passed as a parameter to custom ops (no longer
        # stored in context). rotary_emb is still stored in context (not
        # needed in compiled graph)
        from vllm.forward_context import get_forward_context

        forward_context = get_forward_context()
        if self.rotary_emb is not None:
            forward_context._rotary_emb = self.rotary_emb

        # STEP 5: Pass to mla_attention
        attn_out = self.mla_attn(
            q,  # Has RoPE if unfused, NO RoPE if fused
            kv_c_normed,
            k_pe,  # Has RoPE if unfused, NO RoPE if fused
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
            positions=positions,
            slot_mapping=None,  # Retrieved from attn_metadata in mla_attention.py
            use_fused_path=can_use_fused_path,  # Single flag for entire forward pass
            rotary_emb=self.rotary_emb,
        )

        final_out = self.o_proj(attn_out)[0]
        return final_out
