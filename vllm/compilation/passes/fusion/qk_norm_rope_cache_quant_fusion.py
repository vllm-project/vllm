# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Scaffold for fused QK RMSNorm + RoPE + KV Cache Write (+ implicit FP8/INT8
KV-cache quantization when kv_cache_dtype != "auto").

Architecture
------------
Pattern:      QK RMSNorm → RoPE → unified_kv_cache_update
Replacement:  fused_qk_norm_rope_cache_quant  (single op call)

Runtime context bridge
----------------------
The fused CUDA kernel still needs runtime-only cache state such as
`kv_cache`, `slot_mapping`, and KV scales.  This module therefore registers a
small custom op wrapper that resolves the attention context from `layer_name`
and forwards the tensors into the native CUDA op.

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

import vllm.ir.ops
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    Attention,
    get_attention_context,
)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.utils.torch_utils import direct_register_custom_op

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import MatcherRotaryEmbedding
from .rms_quant_fusion import empty_bf16, empty_fp32, empty_i64

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
    """Resolve runtime cache context, then dispatch the fused CUDA kernel."""
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)

    if layer_slot_mapping is None:
        return torch.empty(0, device=query.device, dtype=query.dtype)

    key_cache, value_cache = kv_cache.unbind(0)

    head_dim = query.shape[2]

    torch.ops._C.fused_qk_norm_rope_cache_quant(
        query,
        key,
        value,
        key_cache,
        value_cache,
        q_weight.view(head_dim),
        k_weight.view(head_dim),
        cos_sin_cache,
        positions,
        layer_slot_mapping,
        float(attn_layer._k_scale_float),
        float(attn_layer._v_scale_float),
        eps,
        query.shape[1],
        key.shape[1],
        head_dim,
        key_cache.shape[1],
        is_neox,
        attn_layer.kv_cache_dtype.startswith("fp8"),
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
        rope_flashinfer: bool = False,
    ) -> None:
        self.layer_name = layer.layer_name
        self.num_heads = layer.num_heads
        self.num_kv_heads = layer.num_kv_heads
        self.head_size = layer.head_size
        self.head_size_v = layer.head_size_v
        self.eps = eps
        self.is_neox = is_neox
        self.rope_flashinfer = rope_flashinfer

        self.q_size = self.num_heads * self.head_size
        self.k_size = self.num_kv_heads * self.head_size
        self.v_size = self.num_kv_heads * self.head_size_v

        self.rope_matcher = MatcherRotaryEmbedding(
            is_neox=is_neox,
            head_size=self.head_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            use_flashinfer=self.rope_flashinfer,
        )

    def get_inputs(self) -> list[torch.Tensor]:
        T = 5
        L = 4096
        qkv = empty_bf16(T, self.q_size + self.k_size + self.v_size)
        positions = empty_i64(T)
        # Weights: [1, head_dim] – same convention as QkNormRopePattern
        q_weight = empty_bf16(1, self.head_size)
        k_weight = empty_bf16(1, self.head_size)
        if self.rope_flashinfer:
            cos_sin_cache = empty_fp32(L, self.head_size)
        else:
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

        rope_matcher = self.rope_matcher

        def apply_neox_rope_3d(
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
        ) -> torch.Tensor:
            x1, x2 = x.chunk(2, dim=-1)
            cos_expanded = cos.unsqueeze(-2)
            sin_expanded = sin.unsqueeze(-2)
            o1 = x1 * cos_expanded - x2 * sin_expanded
            o2 = x2 * cos_expanded + x1 * sin_expanded
            return torch.cat((o1, o2), dim=-1)

        def pattern_flattened_rope(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)

            # Keep this path structurally aligned with QKNormRoPEFusionPass.
            q_by_head = q.view(*q.shape[:-1], q.shape[-1] // head_size, head_size)
            q_normed_by_head = vllm.ir.ops.rms_norm(q_by_head, q_weight, eps)
            q_flat = q_normed_by_head.view(q.shape)

            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // head_size, head_size)
            k_normed_by_head = vllm.ir.ops.rms_norm(k_by_head, k_weight, eps)
            k_flat = k_normed_by_head.view(k.shape)

            # Flattened RoPE, then reshape to the same 3D layout as RopeKVCacheFusionPass.
            q_rope, k_rope = rope_matcher(positions, q_flat, k_flat, cos_sin_cache)

            q_out = q_rope.view(-1, num_heads, head_size)
            k_out = k_rope.view(-1, num_kv_heads, head_size)
            v_out = v.view(-1, num_kv_heads, head_size_v)

            # KV cache write (includes FP8 quant when kv_cache_dtype=fp8)
            dummy = torch.ops.vllm.unified_kv_cache_update(
                k_out, v_out, layer_name
            )
            return dummy, q_out, k_out, v_out

        def pattern_expanded_3d_rope(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)

            # Q path: view -> RMS
            q_by_head = q.view(*q.shape[:-1], q.shape[-1] // head_size, head_size)
            q_normed_by_head = vllm.ir.ops.rms_norm(q_by_head, q_weight, eps)

            # K path: view -> RMS
            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // head_size, head_size)
            k_normed_by_head = vllm.ir.ops.rms_norm(k_by_head, k_weight, eps)

            cos_sin = cos_sin_cache[positions]
            cos, sin = cos_sin.chunk(2, dim=-1)
            q_out = apply_neox_rope_3d(q_normed_by_head, cos, sin)
            k_out = apply_neox_rope_3d(k_normed_by_head, cos, sin)
            v_out = v.reshape(-1, num_kv_heads, head_size_v)

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

        patterns = [pattern_flattened_rope]
        # The manual 3D expansion below matches the Neox-style branch
        # (`chunk -> mul/add -> cat`). Register it only for that branch to
        # avoid generating duplicate pattern graphs for GPT-J style.
        if is_neox:
            patterns.append(pattern_expanded_3d_rope)

        for pattern in patterns:
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

        attn_layers = get_layers_from_vllm_config(config, Attention)
        if not attn_layers:
            logger.warning_once(
                "QKNormRopeCacheQuantFusionPass: no Attention layers found, "
                "pass will have no effect."
            )

        for _, layer in attn_layers.items():
            for eps in [1e-5, 1e-6]:
                for is_neox in [True, False]:
                    if RotaryEmbedding.enabled():
                        for rope_flashinfer in [False, True]:
                            QkNormRopeCacheQuantPattern(
                                layer=layer,
                                eps=eps,
                                is_neox=is_neox,
                                rope_flashinfer=rope_flashinfer,
                            ).register(self.patterns)
                    else:
                        QkNormRopeCacheQuantPattern(
                            layer=layer,
                            eps=eps,
                            is_neox=is_neox,
                        ).register(self.patterns)

            # Each (eps, is_neox) combo shares the seen-pattern cache, so
            # clear it between layers to avoid cross-layer collisions.
            torch._inductor.pattern_matcher._seen_patterns.clear()

        self.dump_patterns(config, self.patterns)

    @staticmethod
    def _normalize_reshape_neg1(graph: fx.Graph) -> int:
        """Replace literal ``-1`` first-dim in ``aten.reshape.default`` with
        the symbolic batch-size placeholder already present in the graph.

        ``torch.compile`` sometimes preserves the literal ``-1`` from
        ``v.reshape(-1, …)`` while resolving the equivalent expression in
        ``q.view(*q.shape[:-1], …)`` to a symbolic placeholder (e.g.
        ``arg1_1``).  The pattern-matcher's ``check_fn`` re-traces with
        symbolic shapes and maps every batch-dim to the *same* KeywordArg;
        if the real graph mixes symbolic and literal ``-1`` the consistency
        check (``repeated pattern differs``) fails.

        This pre-pass makes all first-dims of ``reshape.default`` use the
        same symbolic node, which is semantically equivalent because every
        tensor in the compiled graph shares the same dynamic batch axis.
        """
        sym_batch: fx.Node | None = None
        for node in graph.nodes:
            if (node.op == "call_function"
                    and node.target is torch.ops.aten.reshape.default):
                shape = node.args[1]
                if isinstance(shape, (list, tuple)) and len(shape) > 0:
                    first = shape[0]
                    if isinstance(first, fx.Node):
                        sym_batch = first
                        break

        if sym_batch is None:
            return 0

        count = 0
        for node in graph.nodes:
            if (node.op == "call_function"
                    and node.target is torch.ops.aten.reshape.default):
                shape = node.args[1]
                if (isinstance(shape, (list, tuple))
                        and len(shape) > 0
                        and shape[0] == -1):
                    new_shape = [sym_batch] + list(shape[1:])
                    node.args = (node.args[0], new_shape)
                    count += 1
        return count

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        norm_count = self._normalize_reshape_neg1(graph)
        if norm_count:
            logger.debug(
                "QKNormRopeCacheQuantFusionPass: normalized %d reshape(-1,…) "
                "nodes to use symbolic batch dim",
                norm_count,
            )
        self.matched_count = self.patterns.apply(graph)
        logger.debug(
            "QKNormRopeCacheQuantFusionPass: replaced %s pattern(s)",
            self.matched_count,
        )

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self, QkNormRopeCacheQuantPattern)
