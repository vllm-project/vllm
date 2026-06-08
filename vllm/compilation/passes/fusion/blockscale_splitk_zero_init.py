# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic SplitK + zero-init fusion for blockscale FP8 GEMM.

This pass rewrites every

    Y = blockscale_gemm(A, B, As, Bs, ...)

site whose `A` (and `As`) are produced by a "zero-init capable" producer
into

    Y = torch.empty(M, N, dtype=output_dtype, device=...)
    producer_with_zero_init(..., gemm_out_zero_init=Y)
    blockscale_gemm_splitk(A, B, As, Bs, output=Y, split_k=k, y_is_zeroed=True)

The producer kernel does a grid-strided uint4 zero-fill of Y as its prologue,
the SplitK GEMM skips its internal hipMemsetAsync, and only one
``torch.empty`` survives DCE. The rewrite is performed entirely at the FX
level, so no model code has to change to thread the preallocated output
buffer through the producer.

The pass is organized around two registries:

* ``ZERO_INIT_PRODUCERS`` -- maps each functional producer op to a
  :class:`ProducerSpec` that names its mutating ``_with_zero_init`` twin
  plus tuple-output indices, residual handling, etc.
* ``BLOCKSCALE_GEMM_OPS`` -- maps each functional blockscale GEMM op to a
  :class:`GemmSpec` that names its mutating ``_splitk`` twin plus the
  SplitK picker (typically a tuned-CSV lookup).

Adding a producer or GEMM backend that matches an existing builder is a
registry entry; new FX call shapes can add a small builder and reuse the same
registration loop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

import torch

# ---------------------------------------------------------------------------
# Registry types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProducerSpec:
    """Describes one "zero-init capable" producer op.

    A producer here is any op that consumes the bf16 RMSNorm/activation
    stream and emits the FP8 + scales tuple that flows into the blockscale
    GEMM. ``op`` is its functional flavor (as appears in the pre-fusion FX
    graph); ``with_zero_init_op`` is the mutating twin we rewrite to.

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
    # kwarg name on `with_zero_init_op` that takes the GEMM's zero-init buffer
    zero_init_kwarg: str = "gemm_out_zero_init"
    # Extra kwargs always forwarded to both ops (e.g. group_size=128). These
    # are appended to whatever the call site provides.
    static_kwargs: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GemmSpec:
    """Describes one blockscale GEMM backend.

    ``op`` is the functional GEMM as it appears in the pre-fusion FX graph;
    ``splitk_op`` is the mutating twin that takes a preallocated output and a
    SplitK value. The fusion bakes ``split_k=0`` (a "consult CSV" sentinel) and
    AITER's tuned CSV picks the actual SplitK count at kernel launch.
    """

    name: str
    op: torch._ops.OpOverload
    splitk_op: torch._ops.OpOverload
    # Kwarg names on the splitk op for the new (output, split_k, y_is_zeroed)
    # trio. Defaults match the names we registered in vllm/_aiter_ops.py.
    out_kwarg: str = "output"
    split_k_kwarg: str = "split_k"
    y_is_zeroed_kwarg: str = "y_is_zeroed"


# ---------------------------------------------------------------------------
# Concrete registries
# ---------------------------------------------------------------------------


def build_default_registries() -> tuple[list[ProducerSpec], list[GemmSpec]]:
    """Construct the default producer and GEMM registries.

    Imports are kept local so this module is safe to import on platforms
    that don't have the AITER ops registered (the registries will simply be
    empty in that case, and the fusion pass becomes a no-op).
    """

    producers: list[ProducerSpec] = []
    gemms: list[GemmSpec] = []

    try:
        from vllm._aiter_ops import rocm_aiter_ops
    except ImportError:
        return producers, gemms

    # Each accessor below raises AttributeError on non-ROCm builds where the
    # ops were never registered; we guard every producer in its own try so a
    # missing op degrades gracefully instead of poisoning module import.

    # aiter HIP per-1x128 group FP8 quant.
    try:
        group_quant_functional = rocm_aiter_ops.get_group_quant_op()
        group_quant_with_zero_init = rocm_aiter_ops.get_group_quant_with_zero_init_op()
        producers.append(
            ProducerSpec(
                name="aiter_group_fp8_quant",
                op=group_quant_functional,
                with_zero_init_op=group_quant_with_zero_init,
                fp8_output_index=0,
                scales_output_index=1,
                static_kwargs={"group_size": 128},
            )
        )
    except AttributeError:
        pass

    # aiter Triton-fused RMSNorm + per-1x128 group FP8 quant.
    try:
        rmsnorm_quant_functional = rocm_aiter_ops.get_rmsnorm_group_fused_quant_op()
        rmsnorm_quant_with_zero_init = (
            rocm_aiter_ops.get_rmsnorm_fp8_group_quant_with_zero_init_op()
        )
        producers.append(
            ProducerSpec(
                name="aiter_rmsnorm_fp8_group_quant",
                op=rmsnorm_quant_functional,
                with_zero_init_op=rmsnorm_quant_with_zero_init,
                fp8_output_index=0,
                scales_output_index=1,
                static_kwargs={"group_size": 128},
            )
        )
    except AttributeError:
        pass

    # residual-add + RMSNorm + group quant. Returns 3 outputs.
    try:
        rmsnorm_add_functional = rocm_aiter_ops.get_rmsnorm_group_add_fused_quant_op()
        rmsnorm_add_with_zero_init = (
            rocm_aiter_ops.get_rmsnorm_with_add_fp8_group_quant_with_zero_init_op()
        )
        producers.append(
            ProducerSpec(
                name="aiter_rmsnorm_with_add_fp8_group_quant",
                op=rmsnorm_add_functional,
                with_zero_init_op=rmsnorm_add_with_zero_init,
                fp8_output_index=0,
                residual_output_index=1,
                scales_output_index=2,
                static_kwargs={"group_size": 128},
            )
        )
    except AttributeError:
        pass

    # silu+mul + FP8 group quant (MLP down-proj path: gate_up_proj
    # produces (M, 2K), this op applies silu * mul + quant to give (M, K)
    # FP8 feeding down_proj).
    try:
        act_mul_quant_functional = rocm_aiter_ops.get_act_mul_fused_fp8_group_quant_op()
        act_mul_quant_with_zero_init = (
            rocm_aiter_ops.get_act_mul_fused_fp8_group_quant_with_zero_init_op()
        )
        producers.append(
            ProducerSpec(
                name="aiter_act_mul_and_fp8_group_quant",
                op=act_mul_quant_functional,
                with_zero_init_op=act_mul_quant_with_zero_init,
                fp8_output_index=0,
                scales_output_index=1,
                static_kwargs={"group_size": 128},
            )
        )
    except AttributeError:
        pass

    try:
        gemma_functional = rocm_aiter_ops.get_gemma_rmsnorm_fp8_group_quant_op()
        gemma_with_zero_init = (
            rocm_aiter_ops.get_gemma_rmsnorm_fp8_group_quant_with_zero_init_op()
        )
        producers.append(
            ProducerSpec(
                name="aiter_gemma_rmsnorm_fp8_group_quant",
                op=gemma_functional,
                with_zero_init_op=gemma_with_zero_init,
                fp8_output_index=0,
                scales_output_index=1,
                static_kwargs={"group_size": 128},
            )
        )
    except AttributeError:
        pass

    try:
        gated_functional = rocm_aiter_ops.get_gated_rmsnorm_fp8_group_quant_op()
        gated_with_zero_init = (
            rocm_aiter_ops.get_gated_rmsnorm_fp8_group_quant_with_zero_init_op()
        )
        producers.append(
            ProducerSpec(
                name="aiter_gated_rmsnorm_fp8_group_quant",
                op=gated_functional,
                with_zero_init_op=gated_with_zero_init,
                fp8_output_index=0,
                scales_output_index=1,
                static_kwargs={"group_size": 128},
            )
        )
    except AttributeError:
        pass

    # Gemma and gated RMSNorm producers use AITER kernels that already support
    # a zero-init prologue. Registering them lets the pass cover future model
    # graphs that feed these producers directly into blockscale GEMM.

    try:
        splitk_op = rocm_aiter_ops.get_gemm_a8w8_blockscale_splitk_op()
        gemms.append(
            GemmSpec(
                name="aiter_gemm_a8w8_blockscale",
                op=torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale.default,
                splitk_op=splitk_op,
            )
        )
    except AttributeError:
        pass

    return producers, gemms


__all__ = [
    "BlockScaleSplitKZeroInitFusionPass",
    "GemmSpec",
    "ProducerSpec",
    "build_default_registries",
]


# ---------------------------------------------------------------------------
# The fusion pass
# ---------------------------------------------------------------------------

import torch._inductor.pattern_matcher as pm  # noqa: E402  (kept near use)
from torch import fx  # noqa: E402
from torch._higher_order_ops.auto_functionalize import (  # noqa: E402
    auto_functionalized,
)
from torch._inductor.pattern_matcher import PatternMatcherPass  # noqa: E402

from vllm.config import VllmConfig  # noqa: E402
from vllm.logger import init_logger  # noqa: E402
from vllm.platforms import current_platform  # noqa: E402

from ..inductor_pass import enable_fake_mode  # noqa: E402
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass  # noqa: E402

_logger = init_logger(__name__)


def _example_bf16(*shape: int) -> torch.Tensor:
    return torch.empty(*shape, dtype=torch.bfloat16, device="cuda")


def _example_fp8(*shape: int) -> torch.Tensor:
    return torch.empty(*shape, dtype=current_platform.fp8_dtype(), device="cuda")


def _example_fp32(*shape: int) -> torch.Tensor:
    return torch.empty(*shape, dtype=torch.float32, device="cuda")


def _shape_meta(node: fx.Node) -> torch.Size | None:
    val = node.meta.get("val")
    if val is None:
        return None
    return getattr(val, "shape", None)


def _concrete_int(dim: object) -> int | None:
    """Return ``dim`` as a Python int iff it is already concrete.

    Critical: ``int(SymInt)`` is a guard-creating operation in
    ``torch._dynamo``. Calling it on the model's symbolic batch dim
    forces specialization (e.g. ``s0 == 8192``), which then surfaces as
    a ``ConstraintViolationError`` at ``build_guards`` time because the
    user marked that dim dynamic. We therefore *only* return an int
    when ``dim`` is already a plain Python int (or a torch SymInt whose
    underlying value is a static int with no symbolic part).
    """
    if isinstance(dim, bool):
        return None
    if isinstance(dim, int):
        return dim
    # ``torch.SymInt`` exposes ``.node`` whose ``.expr`` is the SymPy
    # expression. A symbolic expression contains free symbols; a fully
    # specialized expression evaluates to a literal int.
    node = getattr(dim, "node", None)
    if node is None:
        return None
    expr = getattr(node, "expr", None)
    if expr is None:
        return None
    try:
        free = expr.free_symbols
    except AttributeError:
        return None
    if free:
        # Symbolic: do NOT call int() on it (would specialize).
        return None
    # No free symbols -> safe to materialize as a concrete int.
    try:
        return int(expr)
    except (TypeError, ValueError):
        return None


def _make_extra_check(gemm: GemmSpec) -> Callable[[pm.Match], bool]:
    """Build the per-match shape-driven gate.

    Returns True iff the GEMM's K is large enough for SplitK to be worth
    attempting. The actual SplitK count is deferred to AITER's runtime CSV
    (the fusion bakes ``split_k=0``), so the gate reads only the statically
    known K from the weight tensor and never touches the symbolic batch dim
    M -- reading M would force a specialization guard on the dynamic shape.
    """

    def extra_check(match: pm.Match) -> bool:
        gemm_node = None
        for node in match.nodes:
            if node.target == gemm.op:
                gemm_node = node
        if gemm_node is None:
            return False
        b_node = gemm_node.args[1]
        b_shape = _shape_meta(b_node) if isinstance(b_node, fx.Node) else None
        if b_shape is None or len(b_shape) < 2:
            return False
        # K comes from the weight tensor B (shape [N, K]). Using B.shape[1]
        # (instead of x.shape[-1]) keeps the gate correct for producers
        # where x's last dim is not K -- e.g. the silu+mul producer takes x
        # with last dim 2*K, and the gated-RMSNorm producer can wrap x into
        # a head-aware (M, H*D) view where H*D == K. B.shape[1] is always
        # the GEMM's true K and is concrete at FX rewrite time.
        K = _concrete_int(b_shape[1])
        if K is None:
            # Weight shapes are static in vLLM, so this would be very
            # surprising. Fall back to "allow" so we don't accidentally
            # disable the fusion on an edge case we didn't anticipate.
            return True
        # Small-K GEMMs never benefit from SplitK, so the static K bound is
        # the only filter we apply. ``y_is_zeroed=True`` stays safe even when
        # AITER doesn't use SplitK at a given M: the GEMM just overwrites a
        # buffer the producer pre-zeroed (at most one extra M*N*sizeof(out)
        # write).
        return K >= 2048

    return extra_check


# vLLM passes split_k=0 to the SplitK GEMM op; the value is ignored by the
# AITER side (the dispatcher always consults its tuned CSV for both splitK
# *and* kernelName, which is keyed on Python-resolved string lookup in the
# current AITER cktile dispatch). The fusion's value is in threading the
# zero-init buffer + y_is_zeroed=True, not in picking SplitK or the kernel
# -- those decisions stay with AITER's per-shape tuning data.
_SPLIT_K_SENTINEL = 0


def _make_2_input_producer_pattern(
    producer: ProducerSpec,
    gemm: GemmSpec,
    output_dtype: torch.dtype,
) -> tuple[object, object, list[torch.Tensor], object]:
    """Build (pattern, replacement, example_inputs, extra_check) for a
    producer that takes exactly two tensor inputs (x, weight) plus
    optional static kwargs (e.g. group_size=128).

    Covers the RMSNorm + group-quant producers (incl. Gemma RMSNorm).
    """

    static_kwargs = dict(producer.static_kwargs)
    eps = 1e-6  # captured as a closure constant; epsilon doesn't affect shape

    fp8_idx = producer.fp8_output_index
    scales_idx = producer.scales_output_index

    def pattern(x, weight, B, Bs):
        # These producers are emitted by the upstream RMSNormQuant fusion
        # pass with ALL kwargs (x=..., weight=..., variance_epsilon=...,
        # group_size=...). The pattern must use the same call shape so the
        # FX node structure matches; mixing positional + kwarg would
        # produce a node with different arg layout and never match.
        prod_out = producer.op(
            x=x, weight=weight, variance_epsilon=eps, **static_kwargs
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
            gemm.splitk_op,
            A=A,
            B=B,
            As=As,
            Bs=Bs,
            output=Y_zeroed,
            output_dtype=output_dtype,
            split_k=_SPLIT_K_SENTINEL,
            y_is_zeroed=True,
        )
        # auto_functionalized returns (orig_return, output_mutated); both are
        # the same storage. Return the mutated one for the SSA edge.
        return gemm_results[-1]

    example_inputs = [
        _example_bf16(8, 128),  # x (M, K)
        _example_bf16(128),  # weight (K,)
        _example_fp8(64, 128),  # B (N, K)  (FP8 weight)
        _example_fp32(64, 1),  # Bs (N, K/group)  (block scales)
    ]

    return (
        pattern,
        replacement,
        example_inputs,
        _make_extra_check(gemm),
    )


def _make_2_input_with_residual_producer_pattern(
    producer: ProducerSpec,
    gemm: GemmSpec,
    output_dtype: torch.dtype,
) -> tuple[object, object, list[torch.Tensor], object]:
    """Builder for residual-add + RMSNorm + group-quant producers.

    Signature is `(x, residual, weight, eps, group_size) -> (fp8, residual_out,
    scales)`. The blockscale GEMM downstream consumes ``fp8`` and ``scales``;
    ``residual_out`` is the residual stream flowing forward into the next
    layer (we must preserve it in the rewrite so the SSA edge survives).
    """

    static_kwargs = dict(producer.static_kwargs)
    eps = 1e-6
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
            gemm.splitk_op,
            A=A,
            B=B,
            As=As,
            Bs=Bs,
            output=Y_zeroed,
            output_dtype=output_dtype,
            split_k=_SPLIT_K_SENTINEL,
            y_is_zeroed=True,
        )
        return gemm_results[-1], residual_out

    example_inputs = [
        _example_bf16(8, 128),  # x (M, K)
        _example_bf16(8, 128),  # residual (M, K)
        _example_bf16(128),  # weight (K,)
        _example_fp8(64, 128),  # B (N, K)
        _example_fp32(64, 1),  # Bs (N, K/group)
    ]

    return (
        pattern,
        replacement,
        example_inputs,
        _make_extra_check(gemm),
    )


def _make_act_mul_group_quant_producer_pattern(
    producer: ProducerSpec,
    gemm: GemmSpec,
    output_dtype: torch.dtype,
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
        prod_out = producer.op(x=x, group_size=group_size)
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
            gemm.splitk_op,
            A=A,
            B=B,
            As=As,
            Bs=Bs,
            output=Y_zeroed,
            output_dtype=output_dtype,
            split_k=_SPLIT_K_SENTINEL,
            y_is_zeroed=True,
        )
        return gemm_results[-1]

    example_inputs = [
        _example_bf16(8, 256),  # x (M, 2*K); silu+mul halves last dim
        _example_fp8(64, 128),  # B (N, K)
        _example_fp32(64, 1),  # Bs (N, K/group)
    ]

    return (
        pattern,
        replacement,
        example_inputs,
        _make_extra_check(gemm),
    )


def _make_group_quant_producer_pattern(
    producer: ProducerSpec,
    gemm: GemmSpec,
    output_dtype: torch.dtype,
) -> tuple[object, object, list[torch.Tensor], object]:
    """Builder for `rocm_aiter_group_fp8_quant(x, group_size)` producers.

    The HIP per-1x128 group-quant kernel already supports the zero-init
    prologue via aiter.ops.quant.per_group_quant_hip(gemm_out_zero_init=).
    The functional op signature is `(x, group_size) -> (fp8, scales)`; the
    mutating twin adds `gemm_out_zero_init` as a positional tensor between
    them.
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
            gemm.splitk_op,
            A=A,
            B=B,
            As=As,
            Bs=Bs,
            output=Y_zeroed,
            output_dtype=output_dtype,
            split_k=_SPLIT_K_SENTINEL,
            y_is_zeroed=True,
        )
        return gemm_results[-1]

    example_inputs = [
        _example_bf16(8, 128),  # x (M, K)
        _example_fp8(64, 128),  # B (N, K)
        _example_fp32(64, 1),  # Bs (N, K/group)
    ]

    return (
        pattern,
        replacement,
        example_inputs,
        _make_extra_check(gemm),
    )


def _make_gated_producer_pattern(
    producer: ProducerSpec,
    gemm: GemmSpec,
    output_dtype: torch.dtype,
) -> tuple[object, object, list[torch.Tensor], object]:
    """Builder for gated producers: `(x, z, weight, eps, group_size)`.

    Mirrors ``_make_2_input_producer_pattern`` but threads the extra ``z``
    gate tensor through both pattern and replacement.
    """

    static_kwargs = dict(producer.static_kwargs)
    eps = 1e-6
    fp8_idx = producer.fp8_output_index
    scales_idx = producer.scales_output_index

    def pattern(x, z, weight, B, Bs):
        prod_out = producer.op(x, z, weight, eps, **static_kwargs)
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
            gemm.splitk_op,
            A=A,
            B=B,
            As=As,
            Bs=Bs,
            output=Y_zeroed,
            output_dtype=output_dtype,
            split_k=_SPLIT_K_SENTINEL,
            y_is_zeroed=True,
        )
        return gemm_results[-1]

    example_inputs = [
        _example_bf16(8, 128),  # x
        _example_bf16(8, 128),  # z
        _example_bf16(128),  # weight
        _example_fp8(64, 128),  # B
        _example_fp32(64, 1),  # Bs
    ]

    return (
        pattern,
        replacement,
        example_inputs,
        _make_extra_check(gemm),
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
    """FX-level zero-init + SplitK fusion for FP8 blockscale GEMM.

    See module docstring for the rewrite shape. The pass is a no-op when
    the producer / GEMM registries come up empty (e.g. on CUDA-only builds
    where the AITER ops aren't registered).
    """

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

        # Cache the resolved debug dump path so ``_dump_graph_to_file`` can
        # cheaply short-circuit in production. ``compile_debug_dump_path``
        # is rank-aware and returns ``None`` when no path is configured.
        try:
            self._debug_dump_path = config.compile_debug_dump_path()
        except Exception:  # noqa: BLE001
            self._debug_dump_path = None

        producers, gemms = build_default_registries()
        self._registered = 0
        # Per-(producer, gemm) attribution. Populated alongside pattern
        # registration; consumed by ``__call__`` to walk the post-apply
        # graph and count how many auto_functionalized HOPs reference each
        # ``producer.with_zero_init_op`` (one inserted per applied match).
        # This is a ground-truth count -- we read it off the rewritten FX
        # graph rather than relying on extra_check call counts (which can
        # diverge from actual applies when the pattern matcher rejects a
        # candidate after the gate has already fired).
        self._pair_specs: dict[str, tuple[ProducerSpec, GemmSpec]] = {}
        if not producers or not gemms:
            _logger.debug(
                "BlockScaleSplitKZeroInitFusionPass: no producers (%d) or "
                "GEMMs (%d) registered; pass is a no-op",
                len(producers),
                len(gemms),
            )
            return

        for producer in producers:
            builder = _BUILDERS.get(producer.name)
            if builder is None:
                _logger.warning(
                    "BlockScaleSplitKZeroInitFusionPass: no pattern builder "
                    "for producer %r; skipping",
                    producer.name,
                )
                continue
            for gemm in gemms:
                pattern, replacement, inputs, extra_check = builder(
                    producer, gemm, output_dtype
                )
                pm.register_replacement(
                    pattern,
                    replacement,
                    inputs,
                    pm.fwd_only,
                    self.patterns,
                    extra_check=extra_check,
                )
                self._registered += 1
                pair_key = f"{producer.name}__x__{gemm.name}"
                self._pair_specs[pair_key] = (producer, gemm)

        # NB: INFO (not DEBUG) on purpose so that the registration count is
        # visible at server-boot time without raising the vLLM log level.
        # The pass currently registers a small handful of patterns, so the
        # noise is one extra line per compile; the alternative (debug-only)
        # made it impossible to tell from the server log whether the
        # producer/gemm registries actually shipped with the build.
        _logger.info(
            "BlockScaleSplitKZeroInitFusionPass: registered %d patterns "
            "(producers=%d, gemms=%d)",
            self._registered,
            len(producers),
            len(gemms),
        )

        self.dump_patterns(config, self.patterns)

    def _count_per_pair(self, graph: fx.Graph) -> dict[str, int]:
        """Ground-truth per-(producer, gemm) attribution.

        Each successful pattern application inserts exactly one
        ``auto_functionalized(producer.with_zero_init_op, ...)`` whose
        third output (``getitem(node, 2)``) is the zero-inited Y buffer,
        which is then consumed by an ``auto_functionalized(gemm.splitk_op,
        output=Y, ...)``. We pair them by walking the SSA edge so we can
        distinguish different ``producer x gemm`` pairs that fuse through
        the same ``producer.with_zero_init_op`` overload but target
        different gemm ops.
        """
        if not self._pair_specs:
            return {}
        # Reverse lookups: producer.with_zero_init_op -> producer.name and
        # gemm.splitk_op -> gemm.name. Keys are OpOverload identities so a
        # stale call site (e.g. an unfused producer/gemm) is ignored.
        producer_by_op: dict[object, str] = {}
        gemm_by_op: dict[object, str] = {}
        # auto_functionalized appends the mutated `gemm_out_zero_init` buffer
        # after the producer's functional returns, so its getitem index equals
        # the number of functional outputs: 2 for the standard (fp8, scales)
        # producers and 3 for the residual (fp8, residual, scales) producer.
        producer_zinit_idx_by_op: dict[object, int] = {}
        for pair_key, (producer, gemm) in self._pair_specs.items():
            producer_by_op[producer.with_zero_init_op] = producer.name
            gemm_by_op[gemm.splitk_op] = gemm.name
            _defined_indices = [
                producer.fp8_output_index,
                producer.scales_output_index,
                producer.residual_output_index,
            ]
            producer_zinit_idx_by_op[producer.with_zero_init_op] = (
                max(i for i in _defined_indices if i is not None) + 1
            )
        counts: dict[str, int] = {pair_key: 0 for pair_key in self._pair_specs}
        # Walk in topological order: for each fused-producer
        # auto_functionalized, find the gemm auto_functionalized that consumes
        # its zero-inited Y buffer (the getitem at `zinit_idx`). The
        # PatternMatcher always wires the replacement this way, so a producer
        # with no such consumer is a sign of a broken rewrite -- we count it as
        # "<producer>__x__<missing-gemm>" so the discrepancy is visible.
        import operator as _operator

        for node in graph.nodes:
            if node.op != "call_function" or node.target is not auto_functionalized:
                continue
            if not node.args:
                continue
            producer_name = producer_by_op.get(node.args[0])
            if producer_name is None:
                continue
            zinit_idx = producer_zinit_idx_by_op[node.args[0]]
            gemm_name: str | None = None
            for user in node.users:
                if (
                    user.op == "call_function"
                    and user.target is _operator.getitem
                    and len(user.args) >= 2
                    and user.args[1] == zinit_idx
                ):
                    for gemm_candidate in user.users:
                        if (
                            gemm_candidate.op == "call_function"
                            and gemm_candidate.target is auto_functionalized
                            and gemm_candidate.args
                        ):
                            maybe = gemm_by_op.get(gemm_candidate.args[0])
                            if maybe is not None:
                                gemm_name = maybe
                                break
                    if gemm_name is not None:
                        break
            if gemm_name is None:
                # Producer fused but no downstream splitk gemm consumer -
                # surface as a synthetic counter row so we can tell this
                # apart from a regular match in the log.
                counts.setdefault(f"{producer_name}__x__<no-gemm-consumer>", 0)
                counts[f"{producer_name}__x__<no-gemm-consumer>"] += 1
                continue
            pair_key = f"{producer_name}__x__{gemm_name}"
            counts.setdefault(pair_key, 0)
            counts[pair_key] += 1
        return counts

    def _dump_graph_to_file(self, graph: fx.Graph, stage: str) -> None:
        """Diagnostic-only pre/post FX graph dump.

        Writes ``<debug_dump_path>/<pass_name>.<stage>.<seq>.fx.txt`` per
        invocation when the pass is configured with a debug dump path (set
        via ``VLLM_DEBUG_DUMP_PATH`` or ``CompilationConfig.debug_dump_path``).
        The dump is otherwise a silent no-op, so leaving the call in
        ``__call__`` is free in production. We don't rely on
        ``VllmInductorPass.dump_graph`` because that method constructs a
        ``LazyString`` and discards it -- it relies on PyTorch's TORCH_LOGS
        handling to capture the graph, which is not always enabled in vLLM's
        standard compilation path.
        """
        debug_dump_path = self._debug_dump_path
        if debug_dump_path is None:
            return
        try:
            debug_dump_path.mkdir(parents=True, exist_ok=True)
            seq = self.__class__._dump_seq
            self.__class__._dump_seq += 1
            from vllm.utils.system_utils import unique_filepath

            file_path = unique_filepath(
                lambda i: debug_dump_path
                / f"{self.patterns.pass_name}.{stage}.{seq}.{i}.fx.txt"
            )
            owning_module = graph.owning_module
            with file_path.open("w") as f:
                f.write(f"# stage={stage} seq={seq} pass={self.patterns.pass_name}\n")
                if owning_module is not None:
                    try:
                        f.write(owning_module.print_readable(print_output=False))
                        f.write("\n\n# ---- raw graph ----\n")
                    except Exception:  # noqa: BLE001
                        pass
                f.write(str(graph))
        except Exception as exc:  # noqa: BLE001
            # Diagnostic-only: never fail the pass because the dump failed.
            _logger.debug(
                "BlockScaleSplitKZeroInitFusionPass: graph dump failed (%s)",
                exc,
            )

    # Class-level counter so ``before`` and ``after`` from the same call
    # land in monotonically increasing files even when multiple compile
    # ranges share a process. Reset on each fresh test (the test backend
    # constructs a new pass instance per ``compile`` cycle, so accumulation
    # across tests is irrelevant).
    _dump_seq: ClassVar[int] = 0

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self._dump_graph_to_file(graph, "before")
        self.matched_count = self.patterns.apply(graph)
        self._dump_graph_to_file(graph, "after")
        # Per-pair attribution: walk the rewritten graph to count which
        # (producer, gemm) pairs actually fired. The total should agree
        # with ``self.matched_count`` modulo overlapping rewrites; a
        # divergence (most importantly an obviously-expected pair sitting
        # at 0) is the signal we use to identify which pattern is being
        # silently missed by the matcher.
        per_pair = self._count_per_pair(graph)
        nonzero = sorted(
            ((count, name) for name, count in per_pair.items() if count > 0),
            reverse=True,
        )
        zero = sorted(name for name, count in per_pair.items() if count == 0)
        attr_str = (
            ", ".join(f"{name}={count}" for count, name in nonzero)
            if nonzero
            else "none"
        )
        # Surface per-pair counts into VllmPatternMatcherPass.match_table so
        # the end-of-run summary line picks them up too.
        for pair_key, count in per_pair.items():
            if count:
                VllmPatternMatcherPass.match_table[
                    f"{self.patterns.pass_name}::{pair_key}"
                ] += count
        # Match count is logged at INFO (and recorded into
        # VllmPatternMatcherPass.match_table below) so a quick `grep` of the
        # server log tells you whether the fusion actually fired. For models
        # whose graph doesn't expose any of the registered producers this
        # will be 0 on every post-grad invocation; that's a feature, not a
        # bug -- it surfaces a missing producer registration immediately.
        _logger.info(
            "%s replaced %d patterns (per-pair: %s; zero-match: %s)",
            self.__class__.__name__,
            self.matched_count,
            attr_str,
            ", ".join(zero) if zero else "none",
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
