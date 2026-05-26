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
``torch.empty`` survives DCE. This is the vLLM analogue of the ATOM
``preallocate_y`` + ``external_y`` plumbing that was wired through
``Qwen3NextDecoderLayer``; in vLLM the rewrite is purely FX-level, so no
model code touches.

Generic-ness comes from two registries:

* ``ZERO_INIT_PRODUCERS`` -- maps each functional producer op to a
  :class:`ProducerSpec` that names its mutating ``_with_zero_init`` twin
  plus tuple-output indices, residual handling, etc.
* ``BLOCKSCALE_GEMM_OPS`` -- maps each functional blockscale GEMM op to a
  :class:`GemmSpec` that names its mutating ``_splitk`` twin plus the
  SplitK picker (typically a tuned-CSV lookup).

Adding a new fusable producer or GEMM backend is then a one-entry registry
change; the fusion-pass body iterates the registries and registers one
pattern per ``(producer, gemm)`` pair via the standard Inductor pattern
matcher.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

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
    ``splitk_op`` is the mutating twin that takes a preallocated output and
    a SplitK value. ``pick_split_k`` is invoked at FX-rewrite time (via the
    pattern-matcher's ``extra_check``) with concrete M/N/K/dtype and must
    return either 0 (skip fusion) or the SplitK value to bake into the
    rewritten graph.
    """

    name: str
    op: torch._ops.OpOverload
    splitk_op: torch._ops.OpOverload
    pick_split_k: Callable[[int, int, int, torch.dtype], int]
    # Kwarg names on the splitk op for the new (output, split_k, y_is_zeroed)
    # trio. Defaults match the names we registered in vllm/_aiter_ops.py.
    out_kwarg: str = "output"
    split_k_kwarg: str = "split_k"
    y_is_zeroed_kwarg: str = "y_is_zeroed"


# ---------------------------------------------------------------------------
# Concrete registries
# ---------------------------------------------------------------------------


def _default_pick_split_k(M: int, N: int, K: int, dtype: torch.dtype) -> int:
    """Default heuristic deciding whether SplitK pays off for a shape.

    The fusion pass uses this only as a yes/no gate: it returns a positive
    value when the GEMM is K-reduction bound (small M, large K) and 0
    otherwise. The actual SplitK count used at kernel launch comes from
    AITER's tuned CSV at runtime (the fusion passes ``split_k=0`` as a
    "consult CSV" sentinel). Production deployments can install a stricter
    CSV-driven picker on ``GemmSpec.pick_split_k``.
    """
    if M <= 0 or K < 4096:
        return 0
    # M-skinny / K-fat region observed to benefit from SplitK on the
    # Qwen3-Next-80B-A3B-Instruct-FP8 tuning sweep (see
    # qwen3_next_80b_a3b_per1x128_cktile_splitk_yz_gfx950.csv in the aiter
    # configs/zero_init_demo/robust directory).
    if M <= 128:
        return 1
    return 0


def build_default_registries() -> (
    tuple[list[ProducerSpec], list[GemmSpec]]
):
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

    # The op-overload accessors raise AttributeError on non-ROCm builds
    # where the ops were never registered. Guard with a single try so we
    # degrade gracefully rather than poisoning import of the pass module.
    try:
        per_token_with_zero_init = (
            rocm_aiter_ops.get_per_token_quant_with_zero_init_op()
        )
        per_token_functional = rocm_aiter_ops.get_per_token_quant_op()
        producers.append(
            ProducerSpec(
                name="aiter_per_token_quant",
                op=per_token_functional,
                with_zero_init_op=per_token_with_zero_init,
                fp8_output_index=0,
                scales_output_index=1,
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

    # Already-fused vLLM producers (Triton-backed). These are wired in for
    # future-proofing: their mutating `_with_zero_init` variants depend on
    # AITER Triton kernel extensions that are tracked as a separate prereq.
    # Once those land, append matching ProducerSpec entries here -- the
    # fusion pass picks them up with no other change.
    #
    # Examples (intentionally left out of the default registry until the
    # AITER prereqs ship):
    #   rocm_aiter_ops.get_rmsnorm_group_fused_quant_op()
    #   rocm_aiter_ops.get_rmsnorm_group_add_fused_quant_op()
    #   rocm_aiter_ops.get_rmsnorm_fused_dynamic_quant_op()
    #   rocm_aiter_ops.get_rmsnorm_fused_add_dynamic_quant_op()
    #   rocm_aiter_ops.get_group_quant_op()
    #   rocm_aiter_ops.get_act_mul_fused_fp8_group_quant_op()

    try:
        gemms.append(
            GemmSpec(
                name="aiter_gemm_a8w8_blockscale",
                op=torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale.default,
                splitk_op=rocm_aiter_ops.get_gemm_a8w8_blockscale_splitk_op(),
                pick_split_k=_default_pick_split_k,
            )
        )
    except AttributeError:
        pass

    try:
        gemms.append(
            GemmSpec(
                name="aiter_triton_gemm_a8w8_blockscale",
                op=torch.ops.vllm.rocm_aiter_triton_gemm_a8w8_blockscale.default,
                splitk_op=(
                    rocm_aiter_ops.get_triton_gemm_a8w8_blockscale_splitk_op()
                ),
                pick_split_k=_default_pick_split_k,
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


def _make_extra_check(
    producer: ProducerSpec,
    gemm: GemmSpec,
    output_dtype: torch.dtype,
) -> Callable[[pm.Match], bool]:
    """Build the per-match shape-driven gate.

    Returns True iff fusing this producer/GEMM pair pays off for the matched
    shape, according to ``gemm.pick_split_k``. Returns False on symbolic
    shapes that haven't specialized to ints (in which case the fusion would
    be applied speculatively without proof it helps).
    """

    def extra_check(match: pm.Match) -> bool:
        prod_node = None
        gemm_node = None
        for node in match.nodes:
            if node.target == producer.op:
                prod_node = node
            elif node.target == gemm.op:
                gemm_node = node
        if prod_node is None or gemm_node is None:
            return False
        x_node = prod_node.args[0]
        b_node = gemm_node.args[1]
        x_shape = _shape_meta(x_node) if isinstance(x_node, fx.Node) else None
        b_shape = _shape_meta(b_node) if isinstance(b_node, fx.Node) else None
        if x_shape is None or b_shape is None:
            return False
        # N and K come from the weight tensor and are always concrete at FX
        # rewrite time; bail out only if the FX node lost that info.
        try:
            K = int(x_shape[-1])
            N = int(b_shape[0])
        except (TypeError, ValueError):
            return False
        # The batch dim M is symbolic in vLLM's compile mode (one FX graph
        # for the whole dynamic shape range). When that's the case we can't
        # call pick_split_k(M, N, K) and have to defer the SplitK decision
        # to AITER's runtime CSV lookup. The y_is_zeroed=True semantics is
        # safe regardless: when AITER ends up not using SplitK at a given
        # M, the GEMM simply overwrites a buffer that the producer pre-
        # zeroed, costing one extra M*N*sizeof(out) write per GEMM (us-
        # scale and well within MoE/attention noise). We still apply a
        # coarse K-bound gate because K < 2048 GEMMs are never SplitK
        # candidates in our tuning sweep -- those wouldn't benefit even
        # if fused.
        try:
            M = int(x_shape[0])
        except (TypeError, ValueError):
            return K >= 2048
        return gemm.pick_split_k(M, N, K, output_dtype) > 0

    return extra_check


# vLLM passes split_k=0 to the SplitK GEMM op so AITER consults its own
# tuned CSV at runtime to pick the actual SplitK count. The fusion's value
# is in plumbing the zero-init buffer + y_is_zeroed=True, not in picking
# SplitK -- that decision stays with AITER's per-shape tuning data.
_SPLIT_K_SENTINEL = 0


def _make_2_input_producer_pattern(
    producer: ProducerSpec,
    gemm: GemmSpec,
    output_dtype: torch.dtype,
) -> tuple[object, object, list[torch.Tensor], object]:
    """Build (pattern, replacement, example_inputs, extra_check) for a
    producer that takes exactly two tensor inputs (x, weight) plus
    optional static kwargs (e.g. group_size=128).

    Covers the P2/P3-style RMSNorm + group-quant producers; P1 (per_token_quant)
    uses a slightly different builder below.
    """

    static_kwargs = dict(producer.static_kwargs)
    eps = 1e-6  # captured as a closure constant; epsilon doesn't affect shape

    fp8_idx = producer.fp8_output_index
    scales_idx = producer.scales_output_index

    def pattern(x, weight, B, Bs):
        # P2-style ops take a static `variance_epsilon` -- the producer's
        # functional Python entry takes it as a positional/kwarg; the FX
        # node carries it as a non-tensor arg, so we pass the same constant
        # in pattern and replacement.
        prod_out = producer.op(x, weight, eps, **static_kwargs)
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
        _make_extra_check(producer, gemm, output_dtype),
    )


def _make_per_token_producer_pattern(
    producer: ProducerSpec,
    gemm: GemmSpec,
    output_dtype: torch.dtype,
) -> tuple[object, object, list[torch.Tensor], object]:
    """Builder for P1-style producers: `per_token_quant(x, quant_dtype)`.

    The functional op signature is `(x, quant_dtype, scale=None) -> (out, scale)`.
    We pass scale=None explicitly so the pattern matches the standalone
    invocation (no upstream scale).
    """

    fp8_dtype = current_platform.fp8_dtype()
    fp8_idx = producer.fp8_output_index
    scales_idx = producer.scales_output_index

    def pattern(x, B, Bs):
        prod_out = producer.op(x, fp8_dtype, None)
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
            quant_dtype=fp8_dtype,
            scale=None,
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
        _example_fp8(64, 128),  # B
        _example_fp32(64, 1),  # Bs
    ]

    return (
        pattern,
        replacement,
        example_inputs,
        _make_extra_check(producer, gemm, output_dtype),
    )


def _make_gated_producer_pattern(
    producer: ProducerSpec,
    gemm: GemmSpec,
    output_dtype: torch.dtype,
) -> tuple[object, object, list[torch.Tensor], object]:
    """Builder for P3-style gated producers: `(x, z, weight, eps, group_size)`.

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
        _make_extra_check(producer, gemm, output_dtype),
    )


# Producer name -> pattern builder. The builder isolates argument-shape
# differences between producers (number of inputs, which kwargs the FX node
# carries) so the fusion pass body stays uniform.
_BUILDERS: dict[str, Callable] = {
    "aiter_per_token_quant": _make_per_token_producer_pattern,
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

        producers, gemms = build_default_registries()
        self._registered = 0
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

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        # Match count is logged at INFO (and recorded into
        # VllmPatternMatcherPass.match_table below) so a quick `grep` of the
        # server log tells you whether the fusion actually fired. For models
        # whose graph doesn't expose any of the registered producers this
        # will be 0 on every post-grad invocation; that's a feature, not a
        # bug -- it surfaces a missing producer registration immediately.
        _logger.info(
            "%s replaced %d patterns", self.__class__.__name__, self.matched_count
        )

    def uuid(self) -> str:
        return self.hash_source(
            self,
            ProducerSpec,
            GemmSpec,
            BlockScaleSplitKZeroInitFusionPass,
            _make_2_input_producer_pattern,
            _make_per_token_producer_pattern,
            build_default_registries,
        )
