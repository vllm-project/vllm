# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Scaffold for fused QK RMSNorm + RoPE + KV Cache Write (+ implicit FP8/INT8
KV-cache quantization when kv_cache_dtype != "auto").

Architecture
------------
Pattern:      QK RMSNorm → RoPE → unified_kv_cache_update
Replacement:  fused_qk_norm_rope_cache_quant  (single op call)

Stub implementation
-------------------
The registered custom op currently falls back to calling the individual
kernels sequentially so that correctness can be validated before a fused
CUDA kernel is written.

TODO(cuda-fusion): Replace the sequential body of
    _fused_qk_norm_rope_cache_quant_impl
with a single fused CUDA/Triton kernel that avoids intermediate HBM
round-trips.  Performance reference: the AITER (AMD) kernel
    aiter/ops/fused_qk_norm_rope_cache_quant_shuffle.py
achieves meaningful gains for small-batch decode where these ops are
memory-bandwidth bound.  See the vLLM fusion tracker
    vllm-project/vllm#36066  (row: QK Norm + RoPE + Cache + Quant)
for status across hardware platforms.

Relationship to existing passes
--------------------------------
QKNormRoPEFusionPass       – fuses QK norm + RoPE only (no cache write)
RopeKVCacheFusionPass      – fuses RoPE + cache write (ROCm / AITER only)
QKNormRopeCacheQuantFusionPass  – fuses all three stages (CUDA target)

Do NOT enable both QKNormRoPEFusionPass and this pass simultaneously;
this pass is a strict superset.
"""

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops import auto_functionalized
from torch._inductor.fx_passes.post_grad import view_to_reshape
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.utils import Range
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    Attention,
    get_attention_context,
)
from vllm.utils.torch_utils import direct_register_custom_op

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import MatcherRMSNorm, MatcherRotaryEmbedding
from .rms_quant_fusion import empty_bf16, empty_i64

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Fused custom op – stub (sequential fallback)
# ---------------------------------------------------------------------------

def _fused_qk_norm_rope_cache_quant_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: str = "",
) -> torch.Tensor:
    """
    Fused QK RMSNorm + RoPE + KV cache write.

    Tensor shapes
    -------------
    query   : [T, num_heads,    head_dim]  – mutated in-place
    key     : [T, num_kv_heads, head_dim]  – mutated in-place
    value   : [T, num_kv_heads, head_dim]  – read-only
    q_weight: [1, head_dim] or [head_dim]
    k_weight: [1, head_dim] or [head_dim]
    positions: [T]
    cos_sin_cache: [max_seq_len, head_dim]

    Returns a zero-size dummy tensor so that torch.compile sees the
    side-effect dependency on the KV cache, mirroring
    unified_kv_cache_update.

    TODO(cuda-fusion): replace sequential calls with a single fused kernel.
    """
    T = query.shape[0]
    num_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    head_dim = query.shape[2]

    # ---- Step 1: Q RMSNorm (normalise each head independently) ----
    q_norm = torch.empty_like(query)
    torch.ops._C.rms_norm(
        result=q_norm,
        input=query,
        weight=q_weight.view(head_dim),
        epsilon=eps,
    )
    query.copy_(q_norm)

    # ---- Step 2: K RMSNorm ----
    k_norm = torch.empty_like(key)
    torch.ops._C.rms_norm(
        result=k_norm,
        input=key,
        weight=k_weight.view(head_dim),
        epsilon=eps,
    )
    key.copy_(k_norm)

    # ---- Step 3: RoPE (in-place, expects 2-D flat tensors) ----
    q_flat = query.reshape(T, num_heads * head_dim)
    k_flat = key.reshape(T, num_kv_heads * head_dim)
    torch.ops._C.rotary_embedding(
        positions, q_flat, k_flat, head_dim, cos_sin_cache, is_neox
    )
    # query / key are views of q_flat / k_flat, so they see the updated data.

    # ---- Step 4: KV cache write (handles FP8 quant internally) ----
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    if layer_slot_mapping is not None:
        # Import here to avoid a circular import at module load time.
        from vllm.v1.attention.backends.fa_utils import (
            reshape_and_cache_flash,
        )

        key_cache, value_cache = kv_cache.unbind(0)
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            layer_slot_mapping,
            attn_layer.impl.kv_cache_dtype,
            attn_layer._k_scale,
            attn_layer._v_scale,
        )

    return torch.empty(0, device=query.device, dtype=query.dtype)


def _fused_qk_norm_rope_cache_quant_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: str = "",
) -> torch.Tensor:
    return torch.empty(0, device=query.device, dtype=query.dtype)


direct_register_custom_op(
    op_name="fused_qk_norm_rope_cache_quant",
    op_func=_fused_qk_norm_rope_cache_quant_impl,
    mutates_args=["query", "key"],
    fake_impl=_fused_qk_norm_rope_cache_quant_fake,
)

FUSED_QK_NORM_ROPE_CACHE_QUANT_OP = (
    torch.ops.vllm.fused_qk_norm_rope_cache_quant.default
)


# ---------------------------------------------------------------------------
# Pattern (one instance per attention layer × eps × is_neox)
# ---------------------------------------------------------------------------

class QkNormRopeCacheQuantPattern:
    """
    Matches the unfused QK-norm + RoPE + KV-cache-write subgraph and
    replaces it with the single fused op above.

    Unfused graph (conceptually)
    ----------------------------
      q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

      # Q RMSNorm
      q_head  = q.reshape(-1, num_heads, head_dim)
      q_norm  = rms_norm(q_head, q_weight)
      q_flat  = q_norm.reshape(-1, q_size)

      # K RMSNorm
      k_head  = k.reshape(-1, num_kv_heads, head_dim)
      k_norm  = rms_norm(k_head, k_weight)
      k_flat  = k_norm.reshape(-1, kv_size)

      # RoPE
      q_rope, k_rope = rotary_embedding(positions, q_flat, k_flat,
                                        head_dim, cos_sin_cache, is_neox)

      # Reshape for attention / cache
      q_out = q_rope.reshape(-1, num_heads, head_dim)
      k_out = k_rope.reshape(-1, num_kv_heads, head_dim)
      v_out = v.reshape(-1, num_kv_heads, head_dim_v)

      # KV cache write (with implicit quantisation when kv_cache_dtype=fp8)
      dummy = unified_kv_cache_update(k_out, v_out, layer_name)

    Fused replacement
    -----------------
      q, k, v = qkv.split(...)
      q = q.reshape(-1, num_heads, head_dim)
      k = k.reshape(-1, num_kv_heads, head_dim)
      v = v.reshape(-1, num_kv_heads, head_dim_v)
      dummy = fused_qk_norm_rope_cache_quant(
                  q, k, v, q_weight, k_weight, eps,
                  positions, cos_sin_cache, is_neox, layer_name)
    """

    def __init__(
        self,
        layer: Attention,
        eps: float,
        is_neox: bool,
    ) -> None:
        self.layer_name = layer.layer_name
        self.num_heads = layer.num_heads
        self.num_kv_heads = layer.num_kv_heads
        self.head_size = layer.head_size
        self.head_size_v = layer.head_size_v
        self.eps = eps
        self.is_neox = is_neox

        self.q_size = self.num_heads * self.head_size
        self.k_size = self.num_kv_heads * self.head_size
        self.v_size = self.num_kv_heads * self.head_size_v

        self.rmsnorm_matcher = MatcherRMSNorm(eps)
        self.rope_matcher = MatcherRotaryEmbedding(
            is_neox=is_neox,
            head_size=self.head_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )

    def get_inputs(self) -> list[torch.Tensor]:
        T = 5
        L = 4096
        qkv = empty_bf16(T, self.q_size + self.k_size + self.v_size)
        positions = empty_i64(T)
        # Weights: [1, head_dim] – same convention as QkNormRopePattern
        q_weight = empty_bf16(1, self.head_size)
        k_weight = empty_bf16(1, self.head_size)
        cos_sin_cache = empty_bf16(L, self.head_size)
        return [qkv, positions, q_weight, k_weight, cos_sin_cache]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        # Capture loop variables explicitly to avoid closure issues.
        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads
        head_size = self.head_size
        head_size_v = self.head_size_v
        q_size = self.q_size
        k_size = self.k_size
        v_size = self.v_size
        eps = self.eps
        is_neox = self.is_neox
        layer_name = self.layer_name

        rmsnorm_matcher = self.rmsnorm_matcher
        rope_matcher = self.rope_matcher

        def pattern(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)

            # Q RMSNorm: per-head normalisation
            q_by_head = q.reshape(-1, num_heads, head_size)
            q_normed = rmsnorm_matcher(q_by_head, q_weight)
            q_flat = q_normed.reshape(-1, q_size)

            # K RMSNorm: per-head normalisation
            k_by_head = k.reshape(-1, num_kv_heads, head_size)
            k_normed = rmsnorm_matcher(k_by_head, k_weight)
            k_flat = k_normed.reshape(-1, k_size)

            # RoPE
            q_rope, k_rope = rope_matcher(
                positions, q_flat, k_flat, cos_sin_cache
            )

            # Reshape for attention / cache
            q_out = q_rope.reshape(-1, num_heads, head_size)
            k_out = k_rope.reshape(-1, num_kv_heads, head_size)
            v_out = v.reshape(-1, num_kv_heads, head_size_v)

            # KV cache write (includes FP8 quant when kv_cache_dtype=fp8)
            dummy = torch.ops.vllm.unified_kv_cache_update(
                k_out, v_out, layer_name
            )
            return dummy, q_out, k_out, v_out

        def replacement(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)
            q = q.reshape(-1, num_heads, head_size)
            k = k.reshape(-1, num_kv_heads, head_size)
            v = v.reshape(-1, num_kv_heads, head_size_v)

            result = auto_functionalized(
                FUSED_QK_NORM_ROPE_CACHE_QUANT_OP,
                query=q,
                key=k,
                value=v,
                q_weight=q_weight,
                k_weight=k_weight,
                eps=eps,
                positions=positions,
                cos_sin_cache=cos_sin_cache,
                is_neox=is_neox,
                layer_name=layer_name,
            )
            # auto_functionalized returns (dummy, query_mutated, key_mutated, ...)
            # index 0 = dummy, 1 = query (mutated), 2 = key (mutated)
            return result[0], result[1], result[2], v

        def fwd_and_view_to_reshape(*args, **kwargs) -> fx.GraphModule:
            gm = pm.fwd_only(*args, **kwargs)
            view_to_reshape(gm)
            return gm

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            fwd_and_view_to_reshape,
            pm_pass,
        )


# ---------------------------------------------------------------------------
# Pass
# ---------------------------------------------------------------------------

class QKNormRopeCacheQuantFusionPass(VllmPatternMatcherPass):
    """
    Fuse QK RMSNorm + RoPE + KV cache write into a single op for CUDA.

    Enabled via::

        compilation_config:
          pass_config:
            fuse_qk_norm_rope_cache_quant: true

    The pass is restricted to small-batch decode (≤ max_token_num tokens)
    because for large prefill batches these kernels are compute-bound and
    the per-kernel latency is not the bottleneck.

    Models that benefit
    -------------------
    Any model with per-head QK norm, e.g. Qwen3, some Gemma3 variants.
    Combine with FP8 KV cache (--kv-cache-dtype fp8_e4m3) for maximum
    bandwidth savings.

    Current status
    --------------
    The fused op body is a sequential fallback (see module docstring).
    Matching the pattern already reduces kernel-launch overhead; the main
    HBM bandwidth saving requires the TODO fused CUDA kernel.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="qk_norm_rope_cache_quant_fusion_pass"
        )

        cc = config.compilation_config
        self.max_token_num = (
            cc.pass_config.qk_norm_rope_cache_quant_max_token_num
        )

        attn_layers = get_layers_from_vllm_config(config, Attention)
        if not attn_layers:
            logger.warning_once(
                "QKNormRopeCacheQuantFusionPass: no Attention layers found, "
                "pass will have no effect."
            )

        for _, layer in attn_layers.items():
            for eps in [1e-5, 1e-6]:
                for is_neox in [True, False]:
                    QkNormRopeCacheQuantPattern(
                        layer=layer,
                        eps=eps,
                        is_neox=is_neox,
                    ).register(self.patterns)

            # Each (eps, is_neox) combo shares the seen-pattern cache, so
            # clear it between layers to avoid cross-layer collisions.
            torch._inductor.pattern_matcher._seen_patterns.clear()

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug(
            "QKNormRopeCacheQuantFusionPass: replaced %s pattern(s)",
            self.matched_count,
        )

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """Only apply during small-batch decode, not large-batch prefill."""
        return compile_range.end <= self.max_token_num

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self, QkNormRopeCacheQuantPattern)
