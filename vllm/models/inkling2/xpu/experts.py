# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 routed experts on Intel XPU: the `FlatExperts` backends of
`common.moe.flat_experts_class`.

MXFP8 runs the SYCL grouped-GEMM fused MoE of vllm-xpu-kernels
(`XpuFusedMoe`): bf16 activations (W8A16, as Humming on Hopper) where the
built `XpuFusedMoe` runs its xe_2 grouped GEMM, MXFP8-quantized ones where
it runs xe_3 (`mxfp8_quantizes_activations`). It takes `[gate | up]` halves where the
checkpoint interleaves the `w13` rows, and applies the routing weight to
the down-projection rows in fp32 while combining them (`combines`), where
Hopper scales the SiTU output before the down projection and rounds it to
bf16.

NVFP4 runs Hopper's flow with the vendor-neutral NVFP4 emulation MoE Triton
kernel (`fused_moe/experts/nvfp4_emulation_moe.py`) on bf16 activations
(W4A16, as Humming on Hopper; the checkpoint's own recipe quantizes the
expert activations to NVFP4 on SM10x / SM12x) in place of Humming, on the
checkpoint layout (interleaved `w13` rows, consumed by `swiglu`).

Both backends keep the layer's Parameters (`keeps_parameters`): the kernel
layout lives in them, visible as module state.
"""

from collections.abc import Callable
from functools import partial

import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
    invoke_fused_moe_nvfp4_emulation_kernel,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.models.inkling2.common.moe import FlatExperts, QuantFormat
from vllm.models.inkling2.common.ops.moe import swiglu
from vllm.triton_utils import tl, triton

ExpertWeights = dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]


def deinterleave_rows(t: torch.Tensor) -> None:
    """In place: each expert's rows `(g0, u0, g1, u1, ...)` of `t`
    `[E, 2I, ...]` become `[g0 .. g(I-1) | u0 .. u(I-1)]`. One expert is
    copied at a time, so this step needs one expert's rows of extra
    memory."""
    rows = t.view(torch.uint8) if t.element_size() == 1 else t
    for expert in rows:
        expert.copy_(torch.cat((expert[0::2], expert[1::2])))


_SITU_ARGS = ("activation_situ_beta", "activation_situ_linear_beta")


def check_fused_moe_support(fused_moe_interface) -> None:
    """Fail at model init, before any weight is touched, unless the installed
    vllm-xpu-kernels `fused_moe_interface` has what the MXFP8 experts use:
    `XpuFusedMoe._uses_xe2` (from `_uses_xe2_grouped_gemm`, added in 0.1.17),
    which `mxfp8_quantizes_activations` reads, and the SiTU betas of
    `XpuFusedMoe` (since 0.1.13). The private module function stands in for
    the private attribute it sets: a release that renames the function but
    keeps the attribute is rejected here until this check follows it."""
    import inspect

    params = inspect.signature(fused_moe_interface.XpuFusedMoe).parameters
    if hasattr(fused_moe_interface, "_uses_xe2_grouped_gemm") and all(
        name in params for name in _SITU_ARGS
    ):
        return
    from importlib.metadata import PackageNotFoundError, version

    try:
        found = version("vllm-xpu-kernels")
    except PackageNotFoundError:
        found = "unknown"
    raise NotImplementedError(
        "Inkling 2 MXFP8 experts on XPU need vllm-xpu-kernels >= 0.1.17 "
        f"(XpuFusedMoe._uses_xe2); found {found}"
    )


def _situ_activation(gate_cap: float, up_cap: float) -> dict:
    """`XpuFusedMoe` activation arguments of the Inkling2 SiTU."""
    if gate_cap > 0:
        return {
            "activation": "situ",
            "activation_situ_beta": gate_cap,
            "activation_situ_linear_beta": up_cap if up_cap > 0 else None,
        }
    if up_cap == 0:
        return {"activation": "silu"}
    raise NotImplementedError(
        "XPU fused MoE has no SiTU with only the up projection softcapped"
    )


def mxfp8_quantizes_activations(moe) -> bool:
    """Whether the MXFP8 experts feed the built `XpuFusedMoe` `moe`
    MXFP8-quantized activations: exactly when its grouped GEMM is the xe_3
    one, which takes block-scaled activations (`not moe._uses_xe2`, the
    library's own kernel choice: devices that are neither Xe2 nor Xe3, or
    `FORCE_XE3P_KERNEL`); the xe_2 one takes bf16 activations (W8A16).
    Upstream `XPUExperts` decides from the device test alone
    (`fused_moe/experts/xpu_moe.py:64-65`), which disagrees with the library
    under `FORCE_XE3P_KERNEL`."""
    return not moe._uses_xe2


def quantize_mxfp8_activations(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """MXFP8 activations as the modular kernel's prepare step makes them for
    `XPUExperts` (`prepare_finalize/no_dp_ep.py:_quantize_input` with a
    dynamic MXFP8 quant config): `moe_kernel_quantize_input(...,
    quant_dtype="mxfp8")`, on XPU `xpu_mxfp8_quantize`: e4m3 values and e8m0
    scales `[T, H / 32]`."""
    return moe_kernel_quantize_input(
        A=x, A_scale=None, quant_dtype="mxfp8", per_act_token_quant=False
    )


class XpuMxfp8Experts:
    """MXFP8 experts on the SYCL fused MoE (`XpuFusedMoe`).

    `XpuFusedMoe` (vllm-xpu-kernels' fused MoE interface) is called
    directly, because upstream `XPUExperts` has no SiTU yet
    (vllm-project/vllm#49598). The activation decision
    (`mxfp8_quantizes_activations`, from the built `XpuFusedMoe`) and the
    activation quantization (`quantize_mxfp8_activations`, upstream's
    prepare step) are separate functions; `XPUExperts` is a modular-kernel
    expert class with its own config and prepare / finalize, so a later
    switch to it is not shown to be local to this class.

    Load-time memory: the de-interleave copies one expert at a time, but
    `XpuFusedMoe` then builds the relayout (`[E, K, N]` on the xe_2 GEMM)
    of the whole `w13` and `w2` of the local experts before it releases
    the loaded storage. The extra peak per layer is therefore about the
    local `w13` plus `w2` weights (on the xe_2 GEMM the scales are used as
    loaded): 0.95 GiB for Inkling 2 Mini (TP1) and 2.85 GiB for Small
    (TP4), as for upstream `XPUExperts`; a test checks it as an upper
    bound.

    Args:
        weights: Per projection (`w13`, `w2`), the checkpoint weights
            `[E, N, K]` e4m3 and their e8m0 block scales `[E, N, K / 32]`
            (uint8), one per 32 along K: the layer's Parameters.
            `deinterleave_rows` and then `XpuFusedMoe` rewrite them in place
            (`.data`) into the kernel layout, so they stay module state and
            the library's rebuild mark (`xpu_fused_moe`) is set on the `w13`
            Parameter itself.
        shapes: `(N, K)` of each projection.
        nvfp4: Must be False.
        topk: Slots per token.
        gate_cap, up_cap: SiTU softcaps.

    """

    # The fused MoE sums the routed rows itself (`FlatExperts`).
    combines = True
    # Built from the Parameters, which hold the kernel layout afterwards.
    keeps_parameters = True

    def __init__(
        self,
        weights: ExpertWeights,
        shapes: tuple[tuple[int, int], tuple[int, int]],
        nvfp4: bool,
        *,
        topk: int,
        gate_cap: float,
        up_cap: float,
    ) -> None:
        from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

        assert not nvfp4
        (w13, w13_scale, _), (w2, w2_scale, _) = weights["w13"], weights["w2"]
        for (n, k), w, s in zip(shapes, (w13, w2), (w13_scale, w2_scale)):
            assert w.shape[1:] == (n, k) and s.shape[1:] == (n, k // 32)
            assert w.dtype == torch.float8_e4m3fn and s.dtype == torch.uint8
        # Validated before the Parameters are rewritten.
        activation = _situ_activation(gate_cap, up_cap)
        deinterleave_rows(w13.data)
        deinterleave_rows(w13_scale.data)
        self.moe = XpuFusedMoe(
            w13=w13,
            w13_scales=w13_scale,
            w13_bias=None,
            w2=w2,
            w2_scales=w2_scale,
            w2_bias=None,
            n_experts_per_token=topk,
            num_experts=w13.shape[0],
            **activation,
        )
        self.quantize_activations = mxfp8_quantizes_activations(self.moe)

    def __call__(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        src2dst: torch.Tensor,
        num_global_experts: int,
        gate_cap: float,
        up_cap: float,
    ) -> torch.Tensor:
        """bf16 `[T, H]`: each token's local rows (`ids`, -1: not local)
        times their fp32 routing `weights`, summed and rounded once."""
        out = torch.empty_like(x)
        if x.shape[0] > 0:
            a, a_scale = x, None
            if self.quantize_activations:
                a, a_scale = quantize_mxfp8_activations(x)
            self.moe.apply(out, a, weights.view(ids.shape), ids, a1q_scale=a_scale)
        return out


# Batches with fewer rows (token x slot) per expert than `split_dots_bound`
# take the split-dots path of the NVFP4 emulation kernel (`use_split_dots`).
# The bound depends on the shape and on the weight loads (kernel timings,
# one Intel Arc Pro B70, Inkling 2 expert shapes, uniform top-8 routing,
# rows per expert 8 / 16 / 32 / 64 / 128; default / split time, w13 and w2):
# - Mini (hidden 2304, 128 experts), pointer loads: 2.1x at 8 rows,
#   0.82-0.83 at 16, 0.53-0.70 at 32: split below 16 rows;
# - Mini, tensor-descriptor loads: 8.4-9.8x at 8, 1.9-2.3x at 16,
#   0.85-0.93 at 32: split below 32 rows;
# - Small (hidden 4608, 384 experts), either load path: 1.2x / 3.2-3.4x at
#   8, 2.0-2.3x / 11x at 16, 1.6-1.85x / 4.2-4.3x at 32, 0.75-0.86 /
#   0.35-0.36 at 64: split below 64 rows.
# One bound serves both GEMMs, and it uses the mean rows per expert, so a
# hot expert can get more rows than the bound.
SPLIT_DOTS_SMALL_HIDDEN_BOUNDS = {False: 16, True: 32}
SPLIT_DOTS_LARGE_HIDDEN_BOUND = 64
SPLIT_DOTS_LARGE_HIDDEN = 4608


def split_dots_bound(hidden: int, td: bool) -> int:
    """Rows per expert below which the split-dots path is faster at hidden
    size `hidden`, with (`td`) or without tensor-descriptor weight loads."""
    if hidden >= SPLIT_DOTS_LARGE_HIDDEN:
        return SPLIT_DOTS_LARGE_HIDDEN_BOUND
    return SPLIT_DOTS_SMALL_HIDDEN_BOUNDS[td]


# Under VLLM_BATCH_INVARIANT every batch takes the default path with the
# launch config of a batch of this many tokens, so that a row's result does
# not depend on its batch (the path and BLOCK_SIZE_K set the rounding).
BATCH_INVARIANT_CONFIG_TOKENS = 64
# The config entries this backend passes to the emulation kernel as launch
# options: the split-dots config's own, and `get_default_config`'s on the
# default path (no tuned file exists for Inkling 2's expert shapes on XPU;
# its 8 warps above 128 tokens are several times faster there than Triton's
# default 4 on a B70).
_LAUNCH_OPTION_KEYS = ("num_warps", "num_stages", "grf_mode")


def _launch_options(config: dict) -> dict:
    return {key: config[key] for key in _LAUNCH_OPTION_KEYS if key in config}


def _rows_per_expert(num_tokens: int, top_k: int, num_experts: int) -> int:
    return triton.cdiv(num_tokens * top_k, max(num_experts, 1))


def use_split_dots(
    num_tokens: int, top_k: int, num_experts: int, hidden: int, td: bool = False
) -> bool:
    """Whether a batch of `num_tokens` tokens routed to `top_k` of
    `num_experts` experts at hidden size `hidden` takes the split-dots
    path: fewer rows per expert than `split_dots_bound(hidden, td)` (decode,
    and short prefills of models with many experts), and never under
    `VLLM_BATCH_INVARIANT`."""
    if envs.VLLM_BATCH_INVARIANT:
        return False
    return _rows_per_expert(num_tokens, top_k, num_experts) < split_dots_bound(
        hidden, td
    )


def split_dots_config(num_tokens: int, top_k: int, num_experts: int) -> dict:
    """Launch config of the split-dots path: a row block of about one
    expert's rows (16 to 64) and 128 x 128 weight tiles over 8 warps, the
    best single tile of a sweep at Inkling 2 expert shapes on an Intel Arc
    Pro B70 for 1-64 tokens (with `get_default_config` this path was up to
    20x slower there at 64 tokens)."""
    rows_per_expert = _rows_per_expert(num_tokens, top_k, num_experts)
    return {
        "BLOCK_SIZE_M": min(64, max(16, triton.next_power_of_2(rows_per_expert))),
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 8,
        "num_stages": 2,
    }


class XpuNvfp4Experts:
    """NVFP4 experts on the NVFP4 emulation MoE Triton kernel
    (`invoke_fused_moe_nvfp4_emulation_kernel`) with bf16 activations
    (W4A16): `w13` GEMM, `swiglu` (routing weight before the down
    projection, as on Hopper) and `w2` GEMM over the flat rows. Batches of
    fewer rows per expert (over all experts) than `split_dots_bound` take
    the kernel's split-dots path (`use_split_dots`), the
    others its default path with `try_get_optimal_moe_config`; the two
    paths round differently, and `VLLM_BATCH_INVARIANT` pins the default
    path and one config.

    Args:
        weights: Per projection (`w13`, `w2`), the checkpoint weights
            `[E, N, K / 2]` (uint8, two e2m1 each), e4m3 block scales
            `[E, N, K / 16]` (the layer's Parameters, used as they are) and
            fp32 global scales `[E]` (`amax / outer_scale_denominator`).
        shapes: `(N, K)` of each projection.
        nvfp4: Must be True.
        topk, gate_cap, up_cap: Unused (passed per call).

    """

    # Reads the Parameters in the checkpoint layout (no copy).
    keeps_parameters = True

    def __init__(
        self,
        weights: ExpertWeights,
        shapes: tuple[tuple[int, int], tuple[int, int]],
        nvfp4: bool,
        *,
        topk: int,
        gate_cap: float,
        up_cap: float,
    ) -> None:
        assert nvfp4
        self.weights = {}
        for name, (n, k) in zip(("w13", "w2"), shapes):
            weight, scale, global_scale = weights[name]
            assert global_scale is not None and global_scale.dtype == torch.float32
            assert weight.shape[1:] == (n, k // 2) and scale.shape[1:] == (n, k // 16)
            assert weight.dtype == torch.uint8 and scale.dtype == torch.float8_e4m3fn
            self.weights[name] = (weight, scale, global_scale)
        self.num_experts = weights["w13"][0].shape[0]

    def _gemm(self, name, a, top_k, alignment, config, split_dots):
        """bf16 `[rows, N]`: row `r` of the aligned rows is
        `a[r // top_k] @ W[e].T` for its expert `e`; the rows the alignment
        does not list are left unset."""
        weight, scale, global_scale = self.weights[name]
        sorted_ids, expert_ids, num_post_padded = alignment
        out = torch.empty(
            (a.shape[0], top_k, weight.shape[1]), dtype=torch.bfloat16, device=a.device
        )
        invoke_fused_moe_nvfp4_emulation_kernel(
            a,
            weight,
            out,
            scale.view(torch.uint8),
            None,
            global_scale,
            None,
            sorted_ids,
            expert_ids,
            num_post_padded,
            False,
            top_k,
            config,
            compute_type=tl.bfloat16,
            launch_options=_launch_options(config),
            split_dots=split_dots,
        )
        return out.view(-1, weight.shape[1])

    def __call__(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        src2dst: torch.Tensor,
        num_global_experts: int,
        gate_cap: float,
        up_cap: float,
    ) -> torch.Tensor:
        """bf16 `[T * topk, H]` down-projection rows of the local slots of
        `ids` (-1: not local); the other rows are left unset."""
        tokens, topk = ids.shape
        w13, w2 = self.weights["w13"][0], self.weights["w2"][0]
        if tokens == 0:
            return x.new_empty((0, w2.shape[1]))
        # Rows per expert over all experts (the routing's), as the bound
        # was measured; the default path's config as
        # `Nvfp4QuantizationEmulationTritonExperts` gets it.
        # This kernel loads the weights through pointers (no TD path).
        split_dots = use_split_dots(tokens, topk, num_global_experts, x.shape[-1])
        if split_dots:
            config = split_dots_config(tokens, topk, num_global_experts)
        else:
            config_tokens = (
                BATCH_INVARIANT_CONFIG_TOKENS if envs.VLLM_BATCH_INVARIANT else tokens
            )
            config = try_get_optimal_moe_config(
                w13.size(), w2.size(), topk, None, config_tokens
            )
        alignment = moe_align_block_size(
            ids, config["BLOCK_SIZE_M"], self.num_experts, ignore_invalid_experts=True
        )[:3]
        gate_up = self._gemm("w13", x, topk, alignment, config, split_dots)
        activated = swiglu(gate_up, weights, src2dst, gate_cap, up_cap)
        del gate_up
        return self._gemm("w2", activated, 1, alignment, config, split_dots)


def xpu_flat_experts(
    quant_format: QuantFormat, topk: int, gate_cap: float, up_cap: float
) -> Callable[..., FlatExperts]:
    """The XPU `FlatExperts` constructor of `quant_format` experts."""
    cls: type = XpuMxfp8Experts if quant_format == "mxfp8" else XpuNvfp4Experts
    if quant_format == "mxfp8":
        from vllm_xpu_kernels import fused_moe_interface

        check_fused_moe_support(fused_moe_interface)
    return partial(cls, topk=topk, gate_cap=gate_cap, up_cap=up_cap)
