# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic SplitK + zero-init fusion for blockscale FP8 GEMM.

This pass rewrites every

    Y = blockscale_gemm(A, B, As, Bs, ...)

site whose `A` (and `As`) are produced by a "zero-init capable" producer
into

    Y = torch.empty(M, N, dtype=output_dtype, device=...)
    producer_with_zero_init(..., gemm_out_zero_init=Y)
    blockscale_gemm_out(A, B, As, Bs, output=Y, y_is_zeroed=True)

The producer kernel does a grid-strided uint4 zero-fill of Y as its prologue,
the SplitK GEMM skips its internal hipMemsetAsync, and only one
``torch.empty`` survives DCE. The rewrite is performed entirely at the FX
level, so no model code has to change to thread the preallocated output
buffer through the producer.

The pass is organized around two registries:

* ``ZERO_INIT_PRODUCERS`` -- maps each producer op to a
  :class:`ProducerSpec` that records its mutating zero-init alias plus
  tuple-output indices, residual handling, etc.
* ``BLOCKSCALE_GEMM_OPS`` -- maps each functional blockscale GEMM op to a
  :class:`GemmSpec` that names the mutating out-style GEMM op used by the
  replacement.

Adding a producer or GEMM backend that matches an existing builder is a
registry entry; new FX call shapes can add a small builder and reuse the same
registration loop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch.fx.experimental.symbolic_shapes import is_concrete_int

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import (
    VllmInductorPass,
    VllmPatternMatcherPass,
    VllmPatternReplacement,
)
from .matcher_utils import RMSNORM_EPS_VALUES

# ---------------------------------------------------------------------------
# Registry types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProducerSpec:
    """Describes one "zero-init capable" producer op.

    A producer here is any op that consumes the bf16 RMSNorm/activation
    stream and emits the FP8 + scales tuple that flows into the blockscale
    GEMM. ``with_zero_init_op`` is its mutating alias for the replacement.

    The ``*_index`` fields point into the tuple returned by ``op`` so the
    pass knows which output is the FP8 (i.e. the GEMM's ``A``) and which is
    the scales (``As``). ``residual_output_index`` is used by the fused-add
    family of producers so we preserve the residual edge in the rewrite.
    """

    name: str
    op: torch._ops.OpOverload
    with_zero_init_op: torch._ops.OpOverload
    fp8_output_index: int
    scales_output_index: int
    residual_output_index: int | None = None
    # Extra kwargs always forwarded to both ops (e.g. group_size=128). These
    # are appended to whatever the call site provides.
    static_kwargs: dict[str, object] = field(default_factory=dict)
    eps_values: tuple[float | None, ...] = (None,)


@dataclass(frozen=True)
class GemmSpec:
    """Describes one blockscale GEMM backend.

    ``op`` is the functional GEMM as it appears in the pre-fusion FX graph;
    ``out_op`` is the mutating out-style op that writes into a preallocated
    output buffer and can skip the internal zeroing when that buffer has already
    been zeroed by a producer prologue.
    """

    name: str
    op: torch._ops.OpOverload
    out_op: torch._ops.OpOverload


# ---------------------------------------------------------------------------
# Concrete registries
# ---------------------------------------------------------------------------


def build_default_registries() -> tuple[list[ProducerSpec], list[GemmSpec]]:
    producers = [
        ProducerSpec(
            name="aiter_group_fp8_quant",
            op=rocm_aiter_ops.get_group_quant_op(),
            with_zero_init_op=(
                torch.ops.vllm.rocm_aiter_group_fp8_quant_with_zero_init.default
            ),
            fp8_output_index=0,
            scales_output_index=1,
            static_kwargs={"group_size": 128},
        ),
        ProducerSpec(
            name="aiter_rmsnorm_fp8_group_quant",
            op=rocm_aiter_ops.get_rmsnorm_group_fused_quant_op(),
            with_zero_init_op=(
                torch.ops.vllm.rocm_aiter_rmsnorm_fp8_group_quant_with_zero_init.default
            ),
            fp8_output_index=0,
            scales_output_index=1,
            static_kwargs={"group_size": 128},
            eps_values=RMSNORM_EPS_VALUES,
        ),
        ProducerSpec(
            name="aiter_rmsnorm_with_add_fp8_group_quant",
            op=rocm_aiter_ops.get_rmsnorm_group_add_fused_quant_op(),
            with_zero_init_op=(
                torch.ops.vllm.rocm_aiter_rmsnorm_with_add_fp8_group_quant_with_zero_init.default
            ),
            fp8_output_index=0,
            residual_output_index=1,
            scales_output_index=2,
            static_kwargs={"group_size": 128},
            eps_values=RMSNORM_EPS_VALUES,
        ),
        ProducerSpec(
            name="aiter_act_mul_and_fp8_group_quant",
            op=rocm_aiter_ops.get_act_mul_fused_fp8_group_quant_op(),
            with_zero_init_op=(
                torch.ops.vllm.rocm_aiter_act_mul_and_fp8_group_quant_with_zero_init.default
            ),
            fp8_output_index=0,
            scales_output_index=1,
            static_kwargs={"group_size": 128},
        ),
        ProducerSpec(
            name="aiter_gemma_rmsnorm_fp8_group_quant",
            op=rocm_aiter_ops.get_gemma_rmsnorm_fp8_group_quant_op(),
            with_zero_init_op=(
                torch.ops.vllm.rocm_aiter_gemma_rmsnorm_fp8_group_quant_with_zero_init.default
            ),
            fp8_output_index=0,
            scales_output_index=1,
            static_kwargs={"group_size": 128},
            eps_values=RMSNORM_EPS_VALUES,
        ),
        ProducerSpec(
            name="aiter_gated_rmsnorm_fp8_group_quant",
            op=rocm_aiter_ops.get_gated_rmsnorm_fp8_group_quant_op(),
            with_zero_init_op=(
                torch.ops.vllm.rocm_aiter_gated_rmsnorm_fp8_group_quant_with_zero_init.default
            ),
            fp8_output_index=0,
            scales_output_index=1,
            static_kwargs={"group_size": 128},
            eps_values=RMSNORM_EPS_VALUES,
        ),
    ]
    gemms = [
        GemmSpec(
            name="aiter_gemm_a8w8_blockscale",
            op=torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale.default,
            out_op=torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale_out.default,
        )
    ]
    return producers, gemms


__all__ = [
    "BlockScaleSplitKZeroInitFusionPass",
    "GemmSpec",
    "ProducerSpec",
    "build_default_registries",
]


def _make_extra_check(gemm: GemmSpec, min_k: int) -> Callable[[pm.Match], bool]:
    """Build the per-match shape-driven gate.

    Returns True iff the GEMM's K satisfies the configured SplitK threshold.
    The actual SplitK count is deferred to AITER's runtime CSV, so the gate
    reads only the statically known K from the weight tensor and never touches
    the symbolic batch dim M.
    """

    def extra_check(match: pm.Match) -> bool:
        gemm_node = None
        for node in match.nodes:
            if node.target == gemm.op:
                gemm_node = node
        if gemm_node is None:
            return False
        b_node = gemm_node.args[1]
        if not isinstance(b_node, fx.Node):
            return False
        b_val = b_node.meta.get("val")
        b_shape = getattr(b_val, "shape", None)
        if b_shape is None or len(b_shape) < 2:
            return False
        # K comes from the weight tensor B (shape [N, K]). Using B.shape[1]
        # (instead of x.shape[-1]) keeps the gate correct for producers
        # where x's last dim is not K -- e.g. the silu+mul producer takes x
        # with last dim 2*K, and the gated-RMSNorm producer can wrap x into
        # a head-aware (M, H*D) view where H*D == K. B.shape[1] is always
        # the GEMM's true K and is concrete at FX rewrite time.
        k_dim = b_shape[1]
        K = int(k_dim) if is_concrete_int(k_dim) else None
        if K is None:
            return False
        # Small-K GEMMs usually do not benefit from SplitK, so the configured
        # K bound is the only filter we apply. ``y_is_zeroed=True`` stays safe
        # even when AITER doesn't use SplitK at a given M: the GEMM just
        # overwrites a buffer the producer pre-zeroed (at most one extra
        # M*N*sizeof(out) write).
        return K >= min_k

    return extra_check


def _make_2_input_producer_pattern(
    producer: ProducerSpec,
    gemm: GemmSpec,
    output_dtype: torch.dtype,
    min_k: int,
    eps: float | None,
) -> tuple[object, object, list[torch.Tensor], object]:
    """Build (pattern, replacement, example_inputs, extra_check) for a
    producer that takes exactly two tensor inputs (x, weight) plus
    optional static kwargs (e.g. group_size=128).

    Covers the RMSNorm + group-quant producers (incl. Gemma RMSNorm).
    """

    static_kwargs = dict(producer.static_kwargs)
    assert eps is not None

    fp8_idx = producer.fp8_output_index
    scales_idx = producer.scales_output_index

    def pattern(x, weight, B, Bs):
        # These producers are emitted by the upstream RMSNormQuant fusion
        # pass with ALL kwargs (x=..., weight=..., variance_epsilon=...,
        # group_size=...). The pattern must use the same call shape so the
        # FX node structure matches; mixing positional + kwarg would
        # produce a node with different arg layout and never match.
        prod_out = producer.op(
            x=x,
            weight=weight,
            variance_epsilon=eps,
            **static_kwargs,
        )
        A = prod_out[fp8_idx]
        As = prod_out[scales_idx]
        Y = gemm.op(A, B, As, Bs, output_dtype)
        return Y

    def replacement(x, weight, B, Bs):
        M = x.shape[0]
        N = B.shape[0]
        Y = torch.empty(M, N, dtype=output_dtype, device=B.device)
        prod_results = auto_functionalized(
            producer.with_zero_init_op,
            x=x,
            weight=weight,
            gemm_out_zero_init=Y,
            variance_epsilon=eps,
            **static_kwargs,
        )
        A = prod_results[fp8_idx]
        As = prod_results[scales_idx]
        # The last element of prod_results is the mutated `gemm_out_zero_init`
        # tensor; it aliases `Y` so we can pass either, but using the mutated
        # one keeps the SSA edge explicit.
        Y_zeroed = prod_results[-1]
        gemm_results = auto_functionalized(
            gemm.out_op,
            A=A,
            B=B,
            As=As,
            Bs=Bs,
            output=Y_zeroed,
            output_dtype=output_dtype,
            y_is_zeroed=True,
        )
        # auto_functionalized appends the mutated output buffer after the
        # op's return value. Return the mutated buffer for the SSA edge.
        return gemm_results[-1]

    example_inputs = [
        VllmPatternReplacement.empty_bf16(8, 128),  # x (M, K)
        VllmPatternReplacement.empty_bf16(128),  # weight (K,)
        VllmPatternReplacement.empty_fp8(64, 128),  # B (N, K)
        VllmPatternReplacement.empty_fp32(64, 1),  # Bs (N, K/group)
    ]

    return (
        pattern,
        replacement,
        example_inputs,
        _make_extra_check(gemm, min_k),
    )


def _make_2_input_with_residual_producer_pattern(
    producer: ProducerSpec,
    gemm: GemmSpec,
    output_dtype: torch.dtype,
    min_k: int,
    eps: float | None,
) -> tuple[object, object, list[torch.Tensor], object]:
    """Builder for residual-add + RMSNorm + group-quant producers.

    Signature is `(x, residual, weight, eps, group_size) -> (fp8, residual_out,
    scales)`. The blockscale GEMM downstream consumes ``fp8`` and ``scales``;
    ``residual_out`` is the residual stream flowing forward into the next
    layer (we must preserve it in the rewrite so the SSA edge survives).
    """

    static_kwargs = dict(producer.static_kwargs)
    assert eps is not None
    fp8_idx = producer.fp8_output_index
    scales_idx = producer.scales_output_index
    residual_idx = producer.residual_output_index
    assert residual_idx is not None, (
        f"residual_output_index must be set for residual-add producer {producer.name!r}"
    )

    def pattern(x, residual, weight, B, Bs):
        # Upstream AiterFusedAddRMSFp8GroupQuantPattern emits this op with
        # ALL kwargs; we must mirror that to match the FX node.
        prod_out = producer.op(
            x=x,
            residual=residual,
            weight=weight,
            variance_epsilon=eps,
            **static_kwargs,
        )
        A = prod_out[fp8_idx]
        As = prod_out[scales_idx]
        residual_out = prod_out[residual_idx]
        Y = gemm.op(A, B, As, Bs, output_dtype)
        # Return both Y and residual_out so the pattern matcher sees the
        # residual edge survive (otherwise it would be dead-code-eliminated
        # in the pattern but still required by the surrounding graph).
        return Y, residual_out

    def replacement(x, residual, weight, B, Bs):
        M = x.shape[0]
        N = B.shape[0]
        Y = torch.empty(M, N, dtype=output_dtype, device=B.device)
        prod_results = auto_functionalized(
            producer.with_zero_init_op,
            x=x,
            residual=residual,
            weight=weight,
            gemm_out_zero_init=Y,
            variance_epsilon=eps,
            **static_kwargs,
        )
        A = prod_results[fp8_idx]
        As = prod_results[scales_idx]
        residual_out = prod_results[residual_idx]
        Y_zeroed = prod_results[-1]
        gemm_results = auto_functionalized(
            gemm.out_op,
            A=A,
            B=B,
            As=As,
            Bs=Bs,
            output=Y_zeroed,
            output_dtype=output_dtype,
            y_is_zeroed=True,
        )
        return gemm_results[-1], residual_out

    example_inputs = [
        VllmPatternReplacement.empty_bf16(8, 128),  # x (M, K)
        VllmPatternReplacement.empty_bf16(8, 128),  # residual (M, K)
        VllmPatternReplacement.empty_bf16(128),  # weight (K,)
        VllmPatternReplacement.empty_fp8(64, 128),  # B (N, K)
        VllmPatternReplacement.empty_fp32(64, 1),  # Bs (N, K/group)
    ]

    return (
        pattern,
        replacement,
        example_inputs,
        _make_extra_check(gemm, min_k),
    )


def _make_act_mul_group_quant_producer_pattern(
    producer: ProducerSpec,
    gemm: GemmSpec,
    output_dtype: torch.dtype,
    min_k: int,
    eps: float | None,
) -> tuple[object, object, list[torch.Tensor], object]:
    """Builder for `rocm_aiter_act_mul_and_fp8_group_quant(x, group_size)`.

    Same shape as the group-quant producer, but it halves the last dim of
    ``x`` (silu+mul gating: ``x.shape[-1] == 2 * K``). Example inputs reflect
    that so the pattern matcher synthesizes shape-valid placeholders.

    Upstream `RocmAiterSiluMulFp8GroupQuantFusionPass` emits this op with
    all kwargs (`x=..., group_size=...`), so the pattern uses the same call
    shape to match the FX node structure.
    """
    group_size = int(producer.static_kwargs.get("group_size", 128))
    fp8_idx = producer.fp8_output_index
    scales_idx = producer.scales_output_index

    def pattern(x, B, Bs):
        prod_out = producer.op(
            x=x,
            group_size=group_size,
        )
        A = prod_out[fp8_idx]
        As = prod_out[scales_idx]
        Y = gemm.op(A, B, As, Bs, output_dtype)
        return Y

    def replacement(x, B, Bs):
        M = x.shape[0]
        N = B.shape[0]
        Y = torch.empty(M, N, dtype=output_dtype, device=B.device)
        prod_results = auto_functionalized(
            producer.with_zero_init_op,
            x=x,
            gemm_out_zero_init=Y,
            group_size=group_size,
        )
        A = prod_results[fp8_idx]
        As = prod_results[scales_idx]
        Y_zeroed = prod_results[-1]
        gemm_results = auto_functionalized(
            gemm.out_op,
            A=A,
            B=B,
            As=As,
            Bs=Bs,
            output=Y_zeroed,
            output_dtype=output_dtype,
            y_is_zeroed=True,
        )
        return gemm_results[-1]

    example_inputs = [
        VllmPatternReplacement.empty_bf16(8, 256),  # x (M, 2*K)
        VllmPatternReplacement.empty_fp8(64, 128),  # B (N, K)
        VllmPatternReplacement.empty_fp32(64, 1),  # Bs (N, K/group)
    ]

    return (
        pattern,
        replacement,
        example_inputs,
        _make_extra_check(gemm, min_k),
    )


def _make_group_quant_producer_pattern(
    producer: ProducerSpec,
    gemm: GemmSpec,
    output_dtype: torch.dtype,
    min_k: int,
    eps: float | None,
) -> tuple[object, object, list[torch.Tensor], object]:
    """Builder for `rocm_aiter_group_fp8_quant(x, group_size)` producers.

    The HIP per-1x128 group-quant kernel already supports the zero-init
    prologue via aiter.ops.quant.per_group_quant_hip(gemm_out_zero_init=).
    The functional op signature is `(x, group_size) -> (fp8, scales)`; the
    replacement adds `gemm_out_zero_init` to the existing op.
    """
    group_size = int(producer.static_kwargs.get("group_size", 128))
    fp8_idx = producer.fp8_output_index
    scales_idx = producer.scales_output_index

    def pattern(x, B, Bs):
        prod_out = producer.op(x, group_size)
        A = prod_out[fp8_idx]
        As = prod_out[scales_idx]
        Y = gemm.op(A, B, As, Bs, output_dtype)
        return Y

    def replacement(x, B, Bs):
        M = x.shape[0]
        N = B.shape[0]
        Y = torch.empty(M, N, dtype=output_dtype, device=B.device)
        prod_results = auto_functionalized(
            producer.with_zero_init_op,
            x=x,
            gemm_out_zero_init=Y,
            group_size=group_size,
        )
        A = prod_results[fp8_idx]
        As = prod_results[scales_idx]
        Y_zeroed = prod_results[-1]
        gemm_results = auto_functionalized(
            gemm.out_op,
            A=A,
            B=B,
            As=As,
            Bs=Bs,
            output=Y_zeroed,
            output_dtype=output_dtype,
            y_is_zeroed=True,
        )
        return gemm_results[-1]

    example_inputs = [
        VllmPatternReplacement.empty_bf16(8, 128),  # x (M, K)
        VllmPatternReplacement.empty_fp8(64, 128),  # B (N, K)
        VllmPatternReplacement.empty_fp32(64, 1),  # Bs (N, K/group)
    ]

    return (
        pattern,
        replacement,
        example_inputs,
        _make_extra_check(gemm, min_k),
    )


def _make_gated_producer_pattern(
    producer: ProducerSpec,
    gemm: GemmSpec,
    output_dtype: torch.dtype,
    min_k: int,
    eps: float | None,
) -> tuple[object, object, list[torch.Tensor], object]:
    """Builder for gated producers: `(x, z, weight, eps, group_size)`.

    Mirrors ``_make_2_input_producer_pattern`` but threads the extra ``z``
    gate tensor through both pattern and replacement.
    """

    static_kwargs = dict(producer.static_kwargs)
    assert eps is not None
    fp8_idx = producer.fp8_output_index
    scales_idx = producer.scales_output_index

    def pattern(x, z, weight, B, Bs):
        prod_out = producer.op(
            x,
            z,
            weight,
            eps,
            **static_kwargs,
        )
        A = prod_out[fp8_idx]
        As = prod_out[scales_idx]
        Y = gemm.op(A, B, As, Bs, output_dtype)
        return Y

    def replacement(x, z, weight, B, Bs):
        M = x.shape[0]
        N = B.shape[0]
        Y = torch.empty(M, N, dtype=output_dtype, device=B.device)
        prod_results = auto_functionalized(
            producer.with_zero_init_op,
            x=x,
            z=z,
            weight=weight,
            gemm_out_zero_init=Y,
            variance_epsilon=eps,
            **static_kwargs,
        )
        A = prod_results[fp8_idx]
        As = prod_results[scales_idx]
        Y_zeroed = prod_results[-1]
        gemm_results = auto_functionalized(
            gemm.out_op,
            A=A,
            B=B,
            As=As,
            Bs=Bs,
            output=Y_zeroed,
            output_dtype=output_dtype,
            y_is_zeroed=True,
        )
        return gemm_results[-1]

    example_inputs = [
        VllmPatternReplacement.empty_bf16(8, 128),  # x
        VllmPatternReplacement.empty_bf16(8, 128),  # z
        VllmPatternReplacement.empty_bf16(128),  # weight
        VllmPatternReplacement.empty_fp8(64, 128),  # B
        VllmPatternReplacement.empty_fp32(64, 1),  # Bs
    ]

    return (
        pattern,
        replacement,
        example_inputs,
        _make_extra_check(gemm, min_k),
    )


# Producer name -> pattern builder. The builder isolates argument-shape
# differences between producers (number of inputs, which kwargs the FX node
# carries) so the fusion pass body stays uniform.
_BUILDERS: dict[str, Callable] = {
    "aiter_group_fp8_quant": _make_group_quant_producer_pattern,
    "aiter_rmsnorm_fp8_group_quant": _make_2_input_producer_pattern,
    "aiter_rmsnorm_with_add_fp8_group_quant": (
        _make_2_input_with_residual_producer_pattern
    ),
    "aiter_act_mul_and_fp8_group_quant": _make_act_mul_group_quant_producer_pattern,
    "aiter_gemma_rmsnorm_fp8_group_quant": _make_2_input_producer_pattern,
    "aiter_gated_rmsnorm_fp8_group_quant": _make_gated_producer_pattern,
}


class BlockScaleSplitKZeroInitFusionPass(VllmPatternMatcherPass):
    """FX-level zero-init + SplitK fusion for FP8 blockscale GEMM."""

    @enable_fake_mode
    def __init__(
        self,
        config: VllmConfig,
        output_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(config)
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="blockscale_splitk_zero_init_fusion_pass"
        )

        # Honor the configured model dtype when the caller didn't override.
        if output_dtype is None:
            output_dtype = self.model_dtype or torch.bfloat16
        self.output_dtype = output_dtype

        producers, gemms = build_default_registries()
        min_k = (
            config.compilation_config.pass_config.blockscale_splitk_zero_init_min_k
        )

        for producer in producers:
            builder = _BUILDERS[producer.name]
            for gemm in gemms:
                for eps in producer.eps_values:
                    pattern, replacement, inputs, extra_check = builder(
                        producer, gemm, output_dtype, min_k, eps
                    )
                    pm.register_replacement(
                        pattern,
                        replacement,
                        inputs,
                        pm.fwd_only,
                        self.patterns,
                        extra_check=extra_check,
                    )

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        if self.matched_count:
            VllmPatternMatcherPass.match_table[self.patterns.pass_name] += (
                self.matched_count
            )

    def uuid(self) -> str:
        return self.hash_source(
            self,
            ProducerSpec,
            GemmSpec,
            BlockScaleSplitKZeroInitFusionPass,
            _make_extra_check,
            _make_2_input_producer_pattern,
            _make_2_input_with_residual_producer_pattern,
            _make_act_mul_group_quant_producer_pattern,
            _make_group_quant_producer_pattern,
            _make_gated_producer_pattern,
            build_default_registries,
        )
