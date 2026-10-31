# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from collections.abc import Callable, Mapping
from typing import ParamSpec

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

import vllm.ir.ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.utils import Range
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    Attention,
    get_attention_context,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    _USE_LAYERNAME,
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import MROPE_OP, MatcherMRotaryEmbedding, MatcherRotaryEmbedding
from .rms_quant_fusion import empty_bf16, empty_fp32, empty_i64

logger = init_logger(__name__)

P = ParamSpec("P")

# Head sizes the fused kernel fused_qk_norm_rope_cache_pts_quant_shuffle() supports
# Other sizes hard-abort, so skip those layers.
SUPPORTED_FUSED_QK_NORM_ROPE_KVCACHE_HEAD_DIMS: tuple[int, ...] = (64, 128, 256)


# ---------------------------------------------------------------------------
# Custom op: fused QK-norm + RoPE + KV cache update
# ---------------------------------------------------------------------------


def fused_qk_norm_rope_and_unified_kv_cache_update_impl(
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: LayerNameType,
) -> torch.Tensor:
    layer_name = _resolve_layer_name(layer_name)
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    if layer_slot_mapping is not None:
        attn_layer.impl.do_qk_norm_rope_kvcache_update(
            attn_layer,
            qkv,
            q_out,
            k_out,
            positions,
            q_weight,
            k_weight,
            rms_norm_eps,
            cos_sin_cache,
            is_neox,
            kv_cache,
            layer_slot_mapping,
        )
    else:
        # Profiling/dummy run: define q_out/k_out (consumed by attention).
        q_out.zero_()
        k_out.zero_()

    return torch.empty(0, device=qkv.device, dtype=qkv.dtype)


def fused_qk_norm_rope_and_unified_kv_cache_update_fake(
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return torch.empty(0, device=qkv.device, dtype=qkv.dtype)


direct_register_custom_op(
    op_name="fused_qk_norm_rope_and_unified_kv_cache_update",
    op_func=fused_qk_norm_rope_and_unified_kv_cache_update_impl,
    mutates_args=["q_out", "k_out"],
    fake_impl=fused_qk_norm_rope_and_unified_kv_cache_update_fake,
)


def fused_qk_norm_mrope_and_unified_kv_cache_update_impl(
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
    rotary_dim: int,
    gemma_norm: bool,
    gated_q: bool,
    layer_name: LayerNameType,
) -> torch.Tensor:
    layer_name = _resolve_layer_name(layer_name)
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    if layer_slot_mapping is not None:
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1).contiguous()
        attn_layer.impl.do_qk_norm_mrope_kvcache_update(
            attn_layer,
            qkv,
            q_out,
            positions,
            q_weight,
            k_weight,
            rms_norm_eps,
            cos_sin_cache,
            is_neox,
            (mrope_section_t, mrope_section_h, mrope_section_w),
            is_interleaved,
            rotary_dim,
            kv_cache,
            layer_slot_mapping,
            gemma_norm=gemma_norm,
            gated_q=gated_q,
            k_out=k_out,
        )
    else:
        q_out.zero_()
        k_out.zero_()

    return torch.empty(0, device=qkv.device, dtype=qkv.dtype)


def fused_qk_norm_mrope_and_unified_kv_cache_update_fake(
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
    rotary_dim: int,
    gemma_norm: bool,
    gated_q: bool,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return torch.empty(0, device=qkv.device, dtype=qkv.dtype)


direct_register_custom_op(
    op_name="fused_qk_norm_mrope_and_unified_kv_cache_update",
    op_func=fused_qk_norm_mrope_and_unified_kv_cache_update_impl,
    mutates_args=["q_out", "k_out"],
    fake_impl=fused_qk_norm_mrope_and_unified_kv_cache_update_fake,
)


# ---------------------------------------------------------------------------
# Pattern: QK-norm + RoPE + unified_kv_cache_update
# ---------------------------------------------------------------------------


class QkNormRopeKvCachePattern:
    """Match the unfused sequence:
      q, k, v = split(qkv, ...)
      q = rms_norm(q.view(heads), q_weight).view(flat)
      k = rms_norm(k.view(heads), k_weight).view(flat)
      q, k = rotary_embedding(positions, q, k, cos_sin_cache, is_neox)
      q = q.view(num_heads, head_dim)
      k = k.view(num_kv_heads, head_dim)
      v = v.view(num_kv_heads, head_dim)
      dummy = unified_kv_cache_update(k, v, layer_name)

    Replace with:
      q_out = empty(...)
      k_out = empty(...)
      dummy = fused_qk_norm_rope_and_unified_kv_cache_update(
          q_out, k_out, qkv, positions, q_weight, k_weight,
          eps, cos_sin_cache, is_neox, layer_name)
      v = split(qkv, ...)[2].view(num_kv_heads, head_dim)
    """

    FUSED_OP = torch.ops.vllm.fused_qk_norm_rope_and_unified_kv_cache_update.default

    def __init__(
        self,
        layer: Attention,
        eps: float,
        is_neox: bool,
        quant_query: bool,
    ) -> None:
        self.layer_name = layer.layer_name
        self.num_heads = layer.num_heads
        self.num_kv_heads = layer.num_kv_heads
        self.head_size = layer.head_size
        self.head_size_v = layer.head_size_v
        self.eps = eps
        self.is_neox = is_neox
        self.quant_query = quant_query
        self.encoded_layer_name = _encode_layer_name(self.layer_name)
        self.query_quant_group_shape = (
            layer.query_quant.group_shape
            if layer.query_quant is not None
            else GroupShape.PER_TENSOR
        )

        self.q_size = self.num_heads * self.head_size
        self.k_size = self.num_kv_heads * self.head_size
        self.v_size = self.num_kv_heads * self.head_size_v

        self.rope_matcher: MatcherRotaryEmbedding | MatcherMRotaryEmbedding = (
            MatcherRotaryEmbedding(
                is_neox=is_neox,
                head_size=self.head_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
            )
        )

    def get_inputs(self) -> list:
        T = 5
        L = 4096
        qkv = empty_bf16(T, self.q_size + self.k_size + self.v_size)
        positions = empty_i64(T)
        q_weight = empty_bf16(1, self.head_size)
        k_weight = empty_bf16(1, self.head_size)
        cos_sin_cache = empty_bf16(L, self.head_size)
        inputs = [qkv, positions, q_weight, k_weight, cos_sin_cache]
        if self.quant_query:
            q_scale = empty_fp32(1)
            inputs += [q_scale]
        if _USE_LAYERNAME:
            inputs.append(self.encoded_layer_name)
        return inputs

    def _get_layer_name(self, layer_name: LayerNameType | None) -> LayerNameType:
        return self.encoded_layer_name if layer_name is None else layer_name

    def pattern_non_fp8_quant_query(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        layer_name: LayerNameType | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        q_by_head = q.view(-1, self.q_size // self.head_size, self.head_size)
        q_normed = vllm.ir.ops.rms_norm(q_by_head, q_weight, self.eps)
        q_flat = q_normed.view(-1, self.q_size)

        k_by_head = k.view(-1, self.k_size // self.head_size, self.head_size)
        k_normed = vllm.ir.ops.rms_norm(k_by_head, k_weight, self.eps)
        k_flat = k_normed.view(-1, self.k_size)

        q_rope, k_rope = self.rope_matcher(positions, q_flat, k_flat, cos_sin_cache)

        q_rope = q_rope.view(-1, self.num_heads, self.head_size)
        k_rope = k_rope.view(-1, self.num_kv_heads, self.head_size)
        v = v.view(-1, self.num_kv_heads, self.head_size_v)
        dummy = torch.ops.vllm.unified_kv_cache_update(
            k_rope, v, self._get_layer_name(layer_name)
        )
        return dummy, q_rope, k_rope, v

    def replacement_non_fp8_quant_query(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        layer_name: LayerNameType | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_out = torch.empty(
            qkv.shape[0],
            self.num_heads,
            self.head_size,
            device=qkv.device,
            dtype=qkv.dtype,
        )
        k_out = torch.empty(
            qkv.shape[0],
            self.num_kv_heads,
            self.head_size,
            device=qkv.device,
            dtype=qkv.dtype,
        )
        _, _, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        v = v.view(qkv.shape[0], self.num_kv_heads, self.head_size_v)
        results = auto_functionalized(
            self.FUSED_OP,
            q_out=q_out,
            k_out=k_out,
            qkv=qkv,
            positions=positions,
            q_weight=q_weight,
            k_weight=k_weight,
            rms_norm_eps=self.eps,
            cos_sin_cache=cos_sin_cache,
            is_neox=self.is_neox,
            layer_name=self._get_layer_name(layer_name),
        )
        return results[0], results[1], results[2], v

    def _quantize_query(
        self,
        query: torch.Tensor,
        q_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.query_quant_group_shape.is_per_tensor():
            q_out = torch.empty_like(query, dtype=current_platform.fp8_dtype())
            q_quant = auto_functionalized(
                torch.ops.vllm.rocm_aiter_per_tensor_quant.default,
                out=q_out,
                x=query,
                scale=q_scale,
                is_dynamic=False,
            )
            # The AITER op's schema marks scale mutable, so preserve its
            # functionalized write-back as an output of the matched region.
            return q_quant[1], q_quant[2]

        # QuantFP8.forward_hip falls back to the generic static op for the
        # historical per-head shape (-1, block_size). This also covers the
        # degenerate one-local-KV-head case where a scalar scale is represented
        # as (-1, q_size), rather than changing Attention's classification.
        q_out = torch.empty(
            query.shape,
            device=query.device,
            dtype=current_platform.fp8_dtype(),
        )
        q_quant = auto_functionalized(
            torch.ops._C.static_scaled_fp8_quant.default,
            result=q_out,
            input=query,
            scale=q_scale,
            group_shape=[
                self.query_quant_group_shape.row,
                self.query_quant_group_shape.col,
            ],
        )
        return q_quant[1], None

    def pattern_fp8_quant_query(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        q_scale: torch.Tensor,
        layer_name: LayerNameType | None = None,
    ) -> tuple[torch.Tensor, ...]:
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        q_by_head = q.view(-1, self.q_size // self.head_size, self.head_size)
        q_normed = vllm.ir.ops.rms_norm(q_by_head, q_weight, self.eps)
        q_flat = q_normed.view(-1, self.q_size)

        k_by_head = k.view(-1, self.k_size // self.head_size, self.head_size)
        k_normed = vllm.ir.ops.rms_norm(k_by_head, k_weight, self.eps)
        k_flat = k_normed.view(-1, self.k_size)

        q_rope, k_rope = self.rope_matcher(positions, q_flat, k_flat, cos_sin_cache)
        q_rope_fp8, q_scale_out = self._quantize_query(q_rope, q_scale)

        k_rope = k_rope.view(-1, self.num_kv_heads, self.head_size)
        v = v.view(-1, self.num_kv_heads, self.head_size_v)
        dummy = torch.ops.vllm.unified_kv_cache_update(
            k_rope, v, self._get_layer_name(layer_name)
        )
        if q_scale_out is None:
            return dummy, q_rope_fp8, k_rope, v
        return dummy, q_rope_fp8, k_rope, v, q_scale_out

    def replacement_fp8_quant_query(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        q_scale: torch.Tensor,
        layer_name: LayerNameType | None = None,
    ) -> tuple[torch.Tensor, ...]:
        q_out = torch.empty(
            qkv.shape[0],
            self.num_heads,
            self.head_size,
            device=qkv.device,
            dtype=qkv.dtype,
        )
        k_out = torch.empty(
            qkv.shape[0],
            self.num_kv_heads,
            self.head_size,
            device=qkv.device,
            dtype=qkv.dtype,
        )
        _, _, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        v = v.view(qkv.shape[0], self.num_kv_heads, self.head_size_v)
        results = auto_functionalized(
            self.FUSED_OP,
            q_out=q_out,
            k_out=k_out,
            qkv=qkv,
            positions=positions,
            q_weight=q_weight,
            k_weight=k_weight,
            rms_norm_eps=self.eps,
            cos_sin_cache=cos_sin_cache,
            is_neox=self.is_neox,
            layer_name=self._get_layer_name(layer_name),
        )
        # Re-apply the same quant form on the kernel's bf16 q_out; the fused
        # operation does not quantize Q.
        q_fp8_flat = results[1].view(-1, self.q_size)
        q_fp8, q_scale_out = self._quantize_query(q_fp8_flat, q_scale)
        if q_scale_out is None:
            return results[0], q_fp8, results[2], v
        return results[0], q_fp8, results[2], v, q_scale_out

    @staticmethod
    def wrap_trace_fn(
        trace_fn: Callable[P, fx.GraphModule],
        *process_fx_fns: Callable[[fx.GraphModule], None],
    ) -> Callable[P, fx.GraphModule]:
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> fx.GraphModule:
            gm = trace_fn(*args, **kwargs)
            for process_fx in process_fx_fns:
                process_fx(gm)

            return gm

        return wrapped

    @staticmethod
    def fx_view_to_reshape(gm: torch.fx.GraphModule) -> None:
        from torch._inductor.fx_passes.post_grad import view_to_reshape

        view_to_reshape(gm)

    @staticmethod
    def fx_eliminate_noop_reshapes(gm: torch.fx.GraphModule) -> None:
        # Mirror NoOpEliminationPass on the pattern: fold reshape chains, drop no-ops.
        reshape = torch.ops.aten.reshape.default

        def shape_of(node: fx.Node):
            val = node.meta.get("val")
            return tuple(val.shape) if val is not None else None

        for node in list(gm.graph.nodes):
            if node.op != "call_function" or node.target != reshape:
                continue
            base = node.args[0]
            while (
                isinstance(base, fx.Node)
                and base.op == "call_function"
                and base.target == reshape
            ):
                base = base.args[0]
            if base is not node.args[0]:
                node.update_arg(0, base)
            if (
                isinstance(base, fx.Node)
                and shape_of(node) is not None
                and shape_of(node) == shape_of(base)
            ):
                node.replace_all_uses_with(base)
                gm.graph.erase_node(node)
        gm.graph.eliminate_dead_code()
        gm.recompile()

    def _extra_check(self, match: pm.Match) -> bool:
        # A weight of another dtype (e.g. Gemma's fp32 w + 1) must not match plain.
        qkv = match.kwargs.get("qkv")
        qkv_val = qkv.meta.get("val") if isinstance(qkv, fx.Node) else None
        if qkv_val is None:
            return True
        for name in ("q_weight", "k_weight"):
            weight = match.kwargs.get(name)
            val = weight.meta.get("val") if isinstance(weight, fx.Node) else None
            if val is not None and val.dtype != qkv_val.dtype:
                return False
        return True

    def _register(self, pattern, replacement, pm_pass) -> None:
        trace_fn = QkNormRopeKvCachePattern.wrap_trace_fn(
            pm.fwd_only,
            QkNormRopeKvCachePattern.fx_view_to_reshape,
            QkNormRopeKvCachePattern.fx_eliminate_noop_reshapes,
        )

        # Pre-build the search pattern with `ignore_types=(int, torch.SymInt)`
        # and pass it via `search_fn_pattern=` so torch skips both of its
        # internal `fx_to_pattern` calls and treats dynamic-shape SymInts as
        # wildcards.
        inputs = self.get_inputs()
        argnames = [*inspect.signature(pattern).parameters.keys()]
        search_gm = trace_fn(pattern, inputs)
        search_fn_pattern = pm.fx_to_pattern(
            search_gm,
            ignore_types=(int, torch.SymInt),
            argnames=argnames,
        )

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            trace_fn,
            pm_pass,
            extra_check=self._extra_check,
            search_fn_pattern=search_fn_pattern,
        )

    def register(self, pm_pass: PatternMatcherPass) -> None:
        # make_fx counts `self` in bound-method code params; wrap as plain fns.
        if self.quant_query:
            if _USE_LAYERNAME:

                def pattern_q_with_layer(
                    qkv,
                    positions,
                    q_weight,
                    k_weight,
                    cos_sin_cache,
                    q_scale,
                    layer_name,
                ):
                    return self.pattern_fp8_quant_query(
                        qkv,
                        positions,
                        q_weight,
                        k_weight,
                        cos_sin_cache,
                        q_scale,
                        layer_name,
                    )

                def replacement_q_with_layer(
                    qkv,
                    positions,
                    q_weight,
                    k_weight,
                    cos_sin_cache,
                    q_scale,
                    layer_name,
                ):
                    return self.replacement_fp8_quant_query(
                        qkv,
                        positions,
                        q_weight,
                        k_weight,
                        cos_sin_cache,
                        q_scale,
                        layer_name,
                    )

                self._register(pattern_q_with_layer, replacement_q_with_layer, pm_pass)
            else:

                def pattern_q_without_layer(
                    qkv, positions, q_weight, k_weight, cos_sin_cache, q_scale
                ):
                    return self.pattern_fp8_quant_query(
                        qkv, positions, q_weight, k_weight, cos_sin_cache, q_scale
                    )

                def replacement_q_without_layer(
                    qkv, positions, q_weight, k_weight, cos_sin_cache, q_scale
                ):
                    return self.replacement_fp8_quant_query(
                        qkv, positions, q_weight, k_weight, cos_sin_cache, q_scale
                    )

                self._register(
                    pattern_q_without_layer, replacement_q_without_layer, pm_pass
                )
        else:
            if _USE_LAYERNAME:

                def pattern_noq_with_layer(
                    qkv,
                    positions,
                    q_weight,
                    k_weight,
                    cos_sin_cache,
                    layer_name,
                ):
                    return self.pattern_non_fp8_quant_query(
                        qkv,
                        positions,
                        q_weight,
                        k_weight,
                        cos_sin_cache,
                        layer_name,
                    )

                def replacement_noq_with_layer(
                    qkv,
                    positions,
                    q_weight,
                    k_weight,
                    cos_sin_cache,
                    layer_name,
                ):
                    return self.replacement_non_fp8_quant_query(
                        qkv,
                        positions,
                        q_weight,
                        k_weight,
                        cos_sin_cache,
                        layer_name,
                    )

                self._register(
                    pattern_noq_with_layer, replacement_noq_with_layer, pm_pass
                )
            else:

                def pattern_noq_without_layer(
                    qkv, positions, q_weight, k_weight, cos_sin_cache
                ):
                    return self.pattern_non_fp8_quant_query(
                        qkv, positions, q_weight, k_weight, cos_sin_cache
                    )

                def replacement_noq_without_layer(
                    qkv, positions, q_weight, k_weight, cos_sin_cache
                ):
                    return self.replacement_non_fp8_quant_query(
                        qkv, positions, q_weight, k_weight, cos_sin_cache
                    )

                self._register(
                    pattern_noq_without_layer,
                    replacement_noq_without_layer,
                    pm_pass,
                )


class QkNormMRopeKvCachePattern(QkNormRopeKvCachePattern):
    """MRoPE variant; gemma_norm = (1 + w) norms, gated_q = Qwen3.5 [q | gate] heads."""

    FUSED_OP = torch.ops.vllm.fused_qk_norm_mrope_and_unified_kv_cache_update.default

    def __init__(
        self,
        layer: Attention,
        eps: float,
        is_neox: bool,
        quant_query: bool,
        rotary_dim: int,
        mrope_section: tuple[int, int, int],
        is_interleaved: bool,
        gemma_norm: bool = False,
        gated_q: bool = False,
    ) -> None:
        super().__init__(
            layer=layer,
            eps=eps,
            is_neox=is_neox,
            quant_query=quant_query,
        )
        self.rotary_dim = rotary_dim
        self.mrope_section = mrope_section
        self.is_interleaved = is_interleaved
        self.gemma_norm = gemma_norm
        self.gated_q = gated_q
        self.q_cols = self.q_size * (2 if gated_q else 1)
        self.rope_matcher = MatcherMRotaryEmbedding(
            is_neox=is_neox,
            head_size=self.head_size,
            rotary_dim=rotary_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            mrope_section=mrope_section,
            mrope_interleaved=is_interleaved,
        )

    def get_inputs(self) -> list[torch.Tensor]:
        T = 5
        L = 4096
        qkv = empty_bf16(T, self.q_cols + self.k_size + self.v_size)
        positions = empty_i64(3, T)
        q_weight = empty_bf16(1, self.head_size)
        k_weight = empty_bf16(1, self.head_size)
        cos_sin_cache = empty_bf16(L, self.rotary_dim)
        inputs = [qkv, positions, q_weight, k_weight, cos_sin_cache]
        if self.quant_query:
            inputs.append(empty_fp32(1))
        if _USE_LAYERNAME:
            inputs.append(self.encoded_layer_name)
        return inputs

    def _norm_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.float() + 1.0 if self.gemma_norm else weight

    def _split_qkv(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Mirror the model's split: (q_flat, k, v, gate), gate None if ungated."""
        if not self.gated_q:
            q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
            return q, k, v, None
        q_gate, k, v = qkv.split([self.q_cols, self.k_size, self.v_size], dim=-1)
        orig_shape = q_gate.shape[:-1]
        q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        return q.reshape(*orig_shape, -1), k, v, gate.reshape(*orig_shape, -1)

    @staticmethod
    def _with_gate(outputs: tuple, gate: torch.Tensor | None) -> tuple:
        return outputs if gate is None else (*outputs, gate)

    def _unfused_views(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # V is untouched, the gate view is re-derived, K comes back as k_out.
        _, _, v, gate = self._split_qkv(qkv)
        v = v.view(qkv.shape[0], self.num_kv_heads, self.head_size_v)
        return v, gate

    def _norm_rope(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        q, k, v, gate = self._split_qkv(qkv)
        q_by_head = q.view(-1, self.q_size // self.head_size, self.head_size)
        q_normed = vllm.ir.ops.rms_norm(
            q_by_head, self._norm_weight(q_weight), self.eps
        )
        q_flat = q_normed.view(-1, self.q_size)

        k_by_head = k.view(-1, self.k_size // self.head_size, self.head_size)
        k_normed = vllm.ir.ops.rms_norm(
            k_by_head, self._norm_weight(k_weight), self.eps
        )
        k_flat = k_normed.view(-1, self.k_size)

        q_rope, k_rope = self.rope_matcher(positions, q_flat, k_flat, cos_sin_cache)
        return q_rope, k_rope, v, gate

    def pattern_non_fp8_quant_query(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        layer_name: LayerNameType | None = None,
    ) -> tuple[torch.Tensor, ...]:
        q_rope, k_rope, v, gate = self._norm_rope(
            qkv, positions, q_weight, k_weight, cos_sin_cache
        )
        q_rope = q_rope.view(-1, self.num_heads, self.head_size)
        k_rope = k_rope.view(-1, self.num_kv_heads, self.head_size)
        v = v.view(-1, self.num_kv_heads, self.head_size_v)
        dummy = torch.ops.vllm.unified_kv_cache_update(
            k_rope, v, self._get_layer_name(layer_name)
        )
        return self._with_gate((dummy, q_rope, k_rope, v), gate)

    def pattern_fp8_quant_query(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        q_scale: torch.Tensor,
        layer_name: LayerNameType | None = None,
    ) -> tuple[torch.Tensor, ...]:
        q_rope, k_rope, v, gate = self._norm_rope(
            qkv, positions, q_weight, k_weight, cos_sin_cache
        )
        q_rope_fp8, q_scale_out = self._quantize_query(q_rope, q_scale)

        k_rope = k_rope.view(-1, self.num_kv_heads, self.head_size)
        v = v.view(-1, self.num_kv_heads, self.head_size_v)
        dummy = torch.ops.vllm.unified_kv_cache_update(
            k_rope, v, self._get_layer_name(layer_name)
        )
        outputs: tuple = (dummy, q_rope_fp8, k_rope, v)
        if q_scale_out is not None:
            outputs = (*outputs, q_scale_out)
        return self._with_gate(outputs, gate)

    def _fused_results(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        layer_name: LayerNameType | None,
    ) -> tuple[torch.Tensor, ...]:
        q_out = torch.empty(
            qkv.shape[0],
            self.num_heads,
            self.head_size,
            device=qkv.device,
            dtype=qkv.dtype,
        )
        k_out = torch.empty(
            qkv.shape[0],
            self.num_kv_heads,
            self.head_size,
            device=qkv.device,
            dtype=qkv.dtype,
        )
        return auto_functionalized(
            self.FUSED_OP,
            q_out=q_out,
            k_out=k_out,
            qkv=qkv,
            positions=positions,
            q_weight=q_weight,
            k_weight=k_weight,
            rms_norm_eps=self.eps,
            cos_sin_cache=cos_sin_cache,
            is_neox=self.is_neox,
            mrope_section_t=self.mrope_section[0],
            mrope_section_h=self.mrope_section[1],
            mrope_section_w=self.mrope_section[2],
            is_interleaved=self.is_interleaved,
            rotary_dim=self.rotary_dim,
            gemma_norm=self.gemma_norm,
            gated_q=self.gated_q,
            layer_name=self._get_layer_name(layer_name),
        )

    def replacement_non_fp8_quant_query(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        layer_name: LayerNameType | None = None,
    ) -> tuple[torch.Tensor, ...]:
        v, gate = self._unfused_views(qkv)
        results = self._fused_results(
            qkv, positions, q_weight, k_weight, cos_sin_cache, layer_name
        )
        return self._with_gate((results[0], results[1], results[2], v), gate)

    def replacement_fp8_quant_query(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        q_scale: torch.Tensor,
        layer_name: LayerNameType | None = None,
    ) -> tuple[torch.Tensor, ...]:
        v, gate = self._unfused_views(qkv)
        results = self._fused_results(
            qkv, positions, q_weight, k_weight, cos_sin_cache, layer_name
        )
        q_flat = results[1].view(-1, self.q_size)
        q_fp8, q_scale_out = self._quantize_query(q_flat, q_scale)
        outputs: tuple = (results[0], q_fp8, results[2], v)
        if q_scale_out is not None:
            outputs = (*outputs, q_scale_out)
        return self._with_gate(outputs, gate)

    def _extra_check(self, match: pm.Match) -> bool:
        if not super()._extra_check(match):
            return False
        expected = {
            "head_size": self.head_size,
            "rotary_dim": self.rotary_dim,
            "mrope_section_t": self.mrope_section[0],
            "mrope_section_h": self.mrope_section[1],
            "mrope_section_w": self.mrope_section[2],
            "mrope_interleaved": self.is_interleaved,
            "is_neox_style": self.is_neox,
        }
        mrope_node = next(
            (node for node in match.nodes if node.args and node.args[0] == MROPE_OP),
            None,
        )
        # K is handed back through k_out, so any consumer of K is fine.
        return mrope_node is not None and all(
            mrope_node.kwargs.get(name) == value for name, value in expected.items()
        )


def _discover_mrope_configs(
    config: VllmConfig,
) -> tuple[tuple[tuple[int, int, int], bool], ...]:
    model_config = config.model_config
    hf_configs = (
        getattr(model_config, "hf_text_config", None),
        getattr(model_config, "hf_config", None),
    )
    configs: set[tuple[tuple[int, int, int], bool]] = set()
    for hf_config in hf_configs:
        if hf_config is None:
            continue
        rope_parameters = getattr(hf_config, "rope_parameters", None)
        if rope_parameters is None:
            rope_parameters = getattr(hf_config, "rope_scaling", None)
        if not isinstance(rope_parameters, Mapping):
            continue
        section = rope_parameters.get("mrope_section")
        if not isinstance(section, (list, tuple)) or len(section) != 3:
            continue
        try:
            parsed_section = (
                int(section[0]),
                int(section[1]),
                int(section[2]),
            )
        except (TypeError, ValueError):
            continue
        if any(value <= 0 for value in parsed_section):
            continue
        configs.add(
            (
                parsed_section,
                bool(rope_parameters.get("mrope_interleaved", False)),
            )
        )
    return tuple(sorted(configs))


def _uses_attn_output_gate(config: VllmConfig) -> bool:
    """Qwen3.5-family attention packs each Q head as [q | gate] in qkv."""
    hf_config = getattr(config.model_config, "hf_text_config", None)
    return bool(getattr(hf_config, "attn_output_gate", False))


# ---------------------------------------------------------------------------
# Pass class
# ---------------------------------------------------------------------------


class QkNormRopeKvCacheFusionPass(VllmPatternMatcherPass):
    """Fuse QK-norm + RoPE/MRoPE + KV cache update into an AITER HIP kernel.

    Supersedes both QKNormRoPEFusionPass and RopeKVCacheFusionPass for
    attention layers that support the combined operation, eliminating two
    separate kernel launches and the intermediate memory traffic.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="qk_norm_rope_kvcache_fusion_pass"
        )

        cc = config.compilation_config
        self.max_token_num = cc.pass_config.rope_kvcache_fusion_max_token_num
        self.mrope_configs = (
            _discover_mrope_configs(config)
            if cc.is_custom_op_enabled("rotary_embedding")
            else ()
        )
        # Gemma norms leave no config trace: register both; gated is Gemma-only.
        gated = _uses_attn_output_gate(config)
        if gated and not rocm_aiter_ops.is_triton_qk_norm_mrope_cache_supported():
            logger.warning_once(
                "QK Norm+MRoPE+KVCache fusion: the installed AITER Triton kernel lacks "
                "MRoPE/shuffle support; gated attention layers stay unfused."
            )
            gated = False
        self.gated_q_variants = (False, True) if gated else (False,)

        dtype = config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.warning_once(
                "QK Norm+RoPE/MRoPE+KVCache fusion not enabled: unsupported dtype %s",
                dtype,
            )
            return

        attn_layers = get_layers_from_vllm_config(config, Attention)

        patterns_registered = False
        for _, layer in attn_layers.items():
            supports_rope = layer.impl.fused_qk_norm_rope_kvcache_supported()
            supports_mrope = (
                bool(self.mrope_configs)
                and layer.impl.fused_qk_norm_mrope_kvcache_supported()
            )
            if not supports_rope and not supports_mrope:
                continue
            if layer.head_size not in SUPPORTED_FUSED_QK_NORM_ROPE_KVCACHE_HEAD_DIMS:
                logger.warning_once(
                    "QK Norm+RoPE/MRoPE+KVCache fusion not enabled for a layer: "
                    "head_size=%d is not supported by the "
                    "fused_qk_norm_rope_cache_pts_quant_shuffle kernel "
                    "(supported: %s). Falling back to the unfused path.",
                    layer.head_size,
                    SUPPORTED_FUSED_QK_NORM_ROPE_KVCACHE_HEAD_DIMS,
                )
                continue
            if layer.head_size_v != layer.head_size:
                # The fused kernel uses a single head_dim for q/k/v.
                logger.warning_once(
                    "QK Norm+RoPE/MRoPE+KVCache fusion not enabled for a layer: "
                    "head_size_v=%d differs from head_size=%d, which the fused "
                    "kernel does not support. Falling back to the unfused path.",
                    layer.head_size_v,
                    layer.head_size,
                )
                continue
            # Every layer the fused writer may serve must read and write the
            # interleaved V cache, including layers whose patterns are not
            # registered separately below.
            layer.impl.set_interleaved_v_cache()
            # Opaque LayerName is a pattern wildcard on torch >= 2.11, so
            # homogeneous attention layers produce duplicate search patterns.
            # As in the other LayerName-aware fusion passes, one supported
            # layer registers all shape/style variants for the model.
            if _USE_LAYERNAME and patterns_registered:
                continue
            patterns_registered = True
            if supports_rope:
                for epsilon in [1e-5, 1e-6]:
                    for neox in [True, False]:
                        for quant_q in [False, True]:
                            QkNormRopeKvCachePattern(
                                layer=layer,
                                eps=epsilon,
                                is_neox=neox,
                                quant_query=quant_q,
                            ).register(self.patterns)
            if supports_mrope:
                for section, is_interleaved in self.mrope_configs:
                    rotary_dim = 2 * sum(section)
                    if rotary_dim > layer.head_size:
                        logger.warning_once(
                            "QK Norm+MRoPE+KVCache fusion not enabled for a "
                            "layer: rotary_dim=%d exceeds head_size=%d.",
                            rotary_dim,
                            layer.head_size,
                        )
                        continue
                    vector_width = layer.head_size // 32
                    for epsilon in [1e-5, 1e-6]:
                        for neox in [True, False]:
                            # Each warp lane processes `vector_width` adjacent
                            # values. NeoX additionally exchanges vectors
                            # across the half-rotary boundary.
                            rotary_alignment = vector_width * (2 if neox else 1)
                            if rotary_dim % rotary_alignment != 0:
                                logger.warning_once(
                                    "QK Norm+MRoPE+KVCache fusion not enabled "
                                    "for a layer/style: rotary_dim=%d must be "
                                    "divisible by %d for head_size=%d and "
                                    "is_neox=%s.",
                                    rotary_dim,
                                    rotary_alignment,
                                    layer.head_size,
                                    neox,
                                )
                                continue
                            for quant_q in [False, True]:
                                for gemma_norm in [False, True]:
                                    for gated_q in self.gated_q_variants:
                                        if gated_q and not gemma_norm:
                                            continue
                                        QkNormMRopeKvCachePattern(
                                            layer=layer,
                                            eps=epsilon,
                                            is_neox=neox,
                                            quant_query=quant_q,
                                            rotary_dim=rotary_dim,
                                            mrope_section=section,
                                            is_interleaved=is_interleaved,
                                            gemma_norm=gemma_norm,
                                            gated_q=gated_q,
                                        ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.info(
            "QK-Norm+RoPE/MRoPE+KVCache fusion: replaced %s pattern(s) "
            "with an AITER fused attention-prologue kernel",
            self.matched_count,
        )

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return compile_range.end <= self.max_token_num

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(
            self,
            QkNormRopeKvCachePattern,
            QkNormMRopeKvCachePattern,
            _discover_mrope_configs,
            _uses_attn_output_gate,
            repr(self.mrope_configs),
            repr(self.gated_q_variants),
        )
