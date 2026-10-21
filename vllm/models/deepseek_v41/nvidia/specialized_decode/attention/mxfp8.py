# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 attention projections.

At up to ``LOW_BATCH_MAX_ROWS`` rows the executor runs ``fused_wqa_wkv``,
``wq_b`` and ``wo_b`` on FlashInfer ``mm_mxfp8`` with its
``cutedsl_low_latency`` backend whenever their input is already MXFP8; the
stock layers keep ``cute-dsl``. Both backends read the same weights, weight
scales and F8_128x4 activation scales.

The backend's tactics are close when tuned in isolation, but split-K tactics
are slower inside the back-to-back PDL chain of a decode step, so the 5-8-row
bucket of the GB300 TP4 shapes is tuned over one pinned tactic.

Up to ``WIDE_LOW_LATENCY_MAX_ROWS`` rows the GB300 TP4 shapes keep the
low-latency kernel, which the backend's runner offers only with an 8-row
activation tile: this module compiles FlashInfer's kernel class itself with a
16-, 32- or 48-row tile and a fixed tactic per tile. The kernel loads the
weights before its programmatic grid wait and lets its dependents launch once
its weight loads are issued, so in the PDL chain of a decode step the weight
read overlaps the previous kernel; ``cute-dsl`` waits before any load.

Above that, the projections stay on ``cute-dsl``, whose tuner picks a tactic
per power-of-two row bucket at every server start. Timed in isolation the
candidates are close, so the pick varies from start to start, and inside a
decode step one start's pick can be much slower than another's. The GB300
TP4 shapes therefore run a fixed tactic per row range, chosen in the chain of
the three projections, and skip the tuner.
"""

import importlib
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from functools import cache
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutedslMxfp8LinearKernel,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.models.deepseek_v41.common.ops.query_quant import mxfp8_scale_bytes
from vllm.platforms import current_platform
from vllm.utils import flashinfer as vllm_flashinfer

from .. import LOW_BATCH_MAX_ROWS, WIDE_LOW_LATENCY_MAX_ROWS, uses_checked_flashinfer

if TYPE_CHECKING:
    from flashinfer.autotuner import TunableRunner

BACKEND = "cutedsl_low_latency"

# (cta_k, ab_stages, sfb_tmem_stages, split_k) per (N, K) on GB300 for 5-8
# rows, chosen jointly for the chain of the three projections.
Tactic = tuple[int, int, int, int]
PINNED_TACTICS: dict[tuple[int, int], Tactic] = {
    (1792, 5120): (128, 10, 1, 4),
    (8192, 1280): (256, 6, 1, 1),
    (5120, 2048): (256, 4, 1, 2),
}
_PINNED_MIN_ROWS = LOW_BATCH_MAX_ROWS // 2 + 1

# The low-latency tactic per (N, K) and activation tile (rows rounded up to
# 16) on GB300 above LOW_BATCH_MAX_ROWS rows, chosen jointly for the chain of
# the three projections.
WIDE_TACTICS: dict[tuple[int, int], dict[int, Tactic]] = {
    (1792, 5120): {16: (128, 9, 1, 4), 32: (128, 8, 1, 4), 48: (128, 7, 1, 4)},
    (8192, 1280): {16: (256, 5, 1, 1), 32: (256, 5, 1, 1), 48: (128, 9, 1, 1)},
    (5120, 2048): {16: (128, 8, 1, 2), 32: (128, 10, 1, 1), 48: (128, 9, 1, 1)},
}

# ((tile_m, tile_n), (cluster_m, cluster_n), swap_ab, prefetch, split_k) of
# the cute-dsl GEMM per (N, K) on GB300 above LOW_BATCH_MAX_ROWS rows, as
# (largest rows, tactic) in increasing rows.
CuteDslTactic = tuple[tuple[int, int], tuple[int, int], bool, bool, int]
CUTE_DSL_TACTICS: dict[tuple[int, int], tuple[tuple[int, CuteDslTactic], ...]] = {
    (1792, 5120): (
        (12, ((128, 16), (1, 1), True, False, 2)),
        (24, ((128, 8), (1, 1), True, False, 1)),
        (48, ((256, 64), (2, 1), True, False, 1)),
        (96, ((256, 64), (4, 1), True, False, 1)),
        (384, ((256, 64), (2, 1), True, False, 1)),
    ),
    (8192, 1280): (
        (12, ((128, 8), (1, 1), True, False, 1)),
        (96, ((128, 64), (1, 1), True, False, 1)),
        (192, ((256, 128), (2, 1), False, False, 1)),
        (384, ((256, 192), (2, 1), True, False, 1)),
    ),
    (5120, 2048): (
        (24, ((128, 16), (1, 1), True, False, 1)),
        (192, ((256, 64), (2, 1), True, False, 1)),
        (384, ((256, 128), (2, 1), True, False, 1)),
    ),
}


def mxfp8_mm(layer: nn.Module, qa: QuantizedActivation) -> torch.Tensor:
    """``layer``'s GEMM on ``cutedsl_low_latency``."""
    out = vllm_flashinfer.mm_mxfp8(
        qa.data,
        layer.weight,
        qa.scale,
        layer.weight_scale,
        out_dtype=qa.orig_dtype,
        backend=BACKEND,
    )
    return out.view(*qa.orig_shape[:-1], out.shape[-1])


@cache
def _backend_supported() -> bool:
    capability = current_platform.get_device_capability()
    try:
        from flashinfer import mm_mxfp8

        return capability is not None and bool(
            mm_mxfp8.is_backend_supported(BACKEND, capability.to_int())
        )
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return False


def _mxfp8_kernel(layer: nn.Module) -> FlashInferCutedslMxfp8LinearKernel | None:
    """``layer``'s kernel when it is exactly the ``cute-dsl`` MXFP8 one."""
    method = getattr(layer, "scheme", getattr(layer, "quant_method", None))
    kernel = getattr(method, "kernel", None)
    return kernel if type(kernel) is FlashInferCutedslMxfp8LinearKernel else None


def mxfp8_eligible(layer: nn.Module) -> bool:
    """Whether ``layer`` is a ``cute-dsl`` MXFP8 linear whose weights the
    low-latency backend also takes."""
    kernel = _mxfp8_kernel(layer)
    return (
        kernel is not None
        and kernel.config.weight_shape[1] % 128 == 0
        and getattr(layer, "bias", None) is None
        and _backend_supported()
    )


def low_latency_mm(
    layer: nn.Module, x: torch.Tensor | QuantizedActivation
) -> torch.Tensor | None:
    """``layer``'s GEMM of ``x`` on ``cutedsl_low_latency`` for an eligible
    layer, or None when ``x`` is not an MXFP8 input of the backend's layout."""
    kernel = _mxfp8_kernel(layer)
    assert kernel is not None
    qa = as_quantized_activation(x, kernel.input_quant_key())
    if qa is None or qa.scale.ndim != 1 or not qa.data.is_contiguous():
        return None
    return mxfp8_mm(layer, qa)


@cache
def _cute_dsl_runner(out_dtype: torch.dtype) -> "TunableRunner | None":
    """``mm_mxfp8``'s ``cute-dsl`` runner (with PDL), when this device and
    FlashInfer are the ones the pinned tactics were chosen on."""
    if not (
        current_platform.is_device_capability(103)
        and uses_checked_flashinfer("pinning the MXFP8 GEMM tactics")
    ):
        return None
    from flashinfer.gemm import gemm_base

    return gemm_base._cute_dsl_gemm_mxfp8_runner(10, 3, True, out_dtype)


# The pinned tactic per (N, K, rows); None where the runner cannot run it.
_pinned: dict[tuple[int, int, int], CuteDslTactic | None] = {}


def pinned_mm(
    layer: nn.Module, x: torch.Tensor | QuantizedActivation
) -> torch.Tensor | None:
    """``layer``'s ``cute-dsl`` GEMM of ``x`` with the pinned tactic of its
    shape and row count, quantizing a BF16 ``x`` as the layer does; None when
    no tactic is pinned for them."""
    kernel = _mxfp8_kernel(layer)
    assert kernel is not None
    k, n = layer.weight.shape  # column-major [K, N]
    ranges = CUTE_DSL_TACTICS.get((n, k))
    qa = as_quantized_activation(x, kernel.input_quant_key())
    if ranges is None or (qa is None and not isinstance(x, torch.Tensor)):
        return None
    if qa is not None:
        a, out_dtype, shape = qa.data, qa.orig_dtype, qa.orig_shape
    else:
        assert isinstance(x, torch.Tensor)
        a, out_dtype, shape = x.view(-1, k), x.dtype, x.shape
    runner = _cute_dsl_runner(out_dtype)
    if runner is None:
        return None

    from flashinfer.gemm import gemm_base

    rows = a.shape[0]
    key = (n, k, rows)
    if key not in _pinned:
        tactic = next((t for max_rows, t in ranges if rows <= max_rows), None)
        # The runner reads only the operands' shapes and the output's layout.
        out = torch.empty(rows, n, dtype=out_dtype, device=a.device)
        operands = [a, layer.weight, None, None, out_dtype, out, None]
        valid = runner.get_valid_tactics(operands, None)  # type: ignore[arg-type]
        runs = rows > LOW_BATCH_MAX_ROWS and tactic in valid
        _pinned[key] = tactic if runs else None
    tactic = _pinned[key]
    if tactic is None:
        return None
    if qa is not None:
        data, scale = qa.data, qa.scale
    else:
        data, scale = mxfp8_e4m3_quantize(a, is_sf_swizzled_layout=True)
    out = torch.empty(rows, n, dtype=out_dtype, device=a.device)
    workspace = gemm_base._get_cache_buf(
        "mm_mxfp8_workspace", gemm_base.DEFAULT_WORKSPACE_SIZE, a.device
    )
    inputs = [data, layer.weight, scale, layer.weight_scale, out_dtype, out, workspace]
    runner(inputs=inputs, tactic=tactic)
    return out.view(*shape[:-1], n)


@cache
def _wide_supported() -> bool:
    """Whether this device and FlashInfer are the ones the wide tactics were
    chosen on and the low-latency kernel class was checked against."""
    return current_platform.is_device_capability(103) and uses_checked_flashinfer(
        "the low-latency MXFP8 GEMM above 8 rows"
    )


def wide_tactic(n: int, k: int, rows: int) -> tuple[int, Tactic] | None:
    """(activation tile, tactic) of the wide low-latency GEMM of an (N, K)
    weight at ``rows`` rows; None where it does not run."""
    if not LOW_BATCH_MAX_ROWS < rows <= WIDE_LOW_LATENCY_MAX_ROWS:
        return None
    tile = (rows + 15) // 16 * 16
    tactic = WIDE_TACTICS.get((n, k), {}).get(tile)
    if tactic is None or not _wide_supported():
        return None
    return tile, tactic


def _gmem_ptr(dtype, address: int, align: int):
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)


def compile_low_latency(gemm, tile: int, cta_k: int, device: torch.device):
    """``gemm``, a FlashInfer ``LowLatencyBlockscaledGemmKernel`` with a
    ``tile``-row activation tile, compiled for MXFP8 operands and a BF16
    output without scale or bias, as FlashInfer's runner compiles it."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream

    f8, sf, bf16 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, cutlass.BFloat16
    with torch.accelerator.device_index(device.index):
        return cute.compile(
            gemm,
            _gmem_ptr(f8, 16, 16),
            _gmem_ptr(sf, 32, 32),
            _gmem_ptr(f8, 16, 16),
            _gmem_ptr(sf, 16, 16),
            _gmem_ptr(bf16, 16, 16),
            None,
            None,
            tuple(cutlass.Int32(v) for v in (128, tile, cta_k, 1)),
            make_fake_stream(),
        )


def launch_low_latency(
    compiled,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    a: torch.Tensor,
    a_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Launch a compiled low-latency GEMM on the current stream: ``out`` =
    ``a`` (rows x K, F8_128x4 scales) times the column-major [K, N]
    ``weight``. The kernel computes ``out.T``: its M is the weight's N."""
    import cutlass

    from ...ops.cute_dsl.primitives import current_cu_stream

    k, n = weight.shape
    f8, sf, bf16 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, cutlass.BFloat16
    compiled(
        _gmem_ptr(f8, weight.data_ptr(), 16),
        _gmem_ptr(sf, weight_scale.data_ptr(), 32),
        _gmem_ptr(f8, a.data_ptr(), 16),
        _gmem_ptr(sf, a_scale.data_ptr(), 16),
        _gmem_ptr(bf16, out.data_ptr(), 16),
        None,
        None,
        tuple(cutlass.Int32(v) for v in (n, a.shape[0], k, 1)),
        current_cu_stream(),
    )


# The compiled wide low-latency GEMM per (tactic, tile, device index).
_wide_kernels: dict[tuple[Tactic, int, int | None], object] = {}


def _wide_kernel(tactic: Tactic, tile: int, device: torch.device):
    """The low-latency GEMM of ``tactic`` with a ``tile``-row activation
    tile, compiled once per device."""
    key = (tactic, tile, device.index)
    compiled = _wide_kernels.get(key)
    if compiled is None:
        import cutlass
        from flashinfer.gemm.kernels.cute_dsl.low_latency_blockscaled_gemm import (
            LowLatencyBlockscaledGemmKernel,
        )

        cta_k, ab_stages, sfb_tmem_stages, split_k = tactic
        gemm = LowLatencyBlockscaledGemmKernel(
            acc_dtype=cutlass.Float32,
            mma_tiler_mnk=(128, tile, cta_k),
            num_ab_stage=ab_stages,
            num_sfb_tmem_stage=sfb_tmem_stages,
            sf_vec_size=32,
            use_pdl=True,
            split_k=split_k,
        )
        compiled = compile_low_latency(gemm, tile, cta_k, device)
        _wide_kernels[key] = compiled
    return compiled


def wide_mm(
    layer: nn.Module, x: torch.Tensor | QuantizedActivation
) -> torch.Tensor | None:
    """``layer``'s low-latency GEMM of ``x`` with a wide activation tile,
    quantizing a BF16 ``x`` as the layer does; None where it does not run."""
    kernel = _mxfp8_kernel(layer)
    assert kernel is not None
    k, n = layer.weight.shape  # column-major [K, N]
    qa = as_quantized_activation(x, kernel.input_quant_key())
    if qa is not None:
        if qa.scale.ndim != 1 or not qa.data.is_contiguous():
            return None
        a, out_dtype, shape = qa.data, qa.orig_dtype, qa.orig_shape
    elif isinstance(x, torch.Tensor):
        a, out_dtype, shape = x.reshape(-1, k), x.dtype, x.shape
    else:
        return None
    choice = wide_tactic(n, k, a.shape[0])
    if choice is None or out_dtype != torch.bfloat16:
        return None
    tile, tactic = choice
    if qa is not None:
        data, scale = qa.data, qa.scale
    else:
        data, scale = mxfp8_e4m3_quantize(a, is_sf_swizzled_layout=True)
    out = torch.empty(a.shape[0], n, dtype=out_dtype, device=a.device)
    compiled = _wide_kernel(tactic, tile, a.device)
    launch_low_latency(compiled, layer.weight, layer.weight_scale, data, scale, out)
    return out.view(*shape[:-1], n)


def wide_or_pinned_mm(
    layer: nn.Module, x: torch.Tensor | QuantizedActivation
) -> torch.Tensor | None:
    """An eligible ``layer``'s GEMM of ``x`` above ``LOW_BATCH_MAX_ROWS``
    rows: on the wide low-latency GEMM, else with the pinned ``cute-dsl``
    tactic; None where neither applies."""
    out = wide_mm(layer, x)
    return out if out is not None else pinned_mm(layer, x)


def projection_mm(
    layer: nn.Module, x: torch.Tensor | QuantizedActivation, low_batch: bool
) -> torch.Tensor | None:
    """An eligible ``layer``'s GEMM of ``x``: on ``cutedsl_low_latency`` in a
    step of up to ``LOW_BATCH_MAX_ROWS`` rows (``low_batch``), else
    ``wide_or_pinned_mm``; None where none applies."""
    return low_latency_mm(layer, x) if low_batch else wide_or_pinned_mm(layer, x)


def shape_representatives(linears: Iterable[nn.Module]) -> list[nn.Module]:
    """One eligible linear per distinct (N, K), in a rank-independent order."""
    by_shape: dict[tuple[int, int], nn.Module] = {}
    for linear in linears:
        kernel = _mxfp8_kernel(linear)
        if kernel is not None and mxfp8_eligible(linear):
            by_shape.setdefault(kernel.config.weight_shape, linear)
    return [by_shape[shape] for shape in sorted(by_shape)]


def _zeros(linear: nn.Module, rows: int) -> QuantizedActivation:
    """An input of ``linear`` laid out like the fused quantize producers'."""
    k = linear.weight.shape[0]  # column-major [K, N]
    device = linear.weight.device
    shape = torch.Size((rows, k))
    return QuantizedActivation(
        torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device),
        torch.zeros(mxfp8_scale_bytes(rows, k), dtype=torch.uint8, device=device),
        torch.bfloat16,
        shape,
        kMxfp8Dynamic,
    )


def run_zeros(linear: nn.Module, rows: int) -> torch.Tensor:
    """Call ``linear`` on ``cutedsl_low_latency`` on a zero input."""
    return mxfp8_mm(linear, _zeros(linear, rows))


def compile_pinned(linear: nn.Module) -> None:
    """Compile ``linear``'s pinned ``cute-dsl`` tactics, each at the largest
    row count of its range, and its wide low-latency GEMMs."""
    k, n = linear.weight.shape
    for rows, _ in CUTE_DSL_TACTICS.get((n, k), ()):
        pinned_mm(linear, _zeros(linear, rows))
    for tile in WIDE_TACTICS.get((n, k), {}):
        choice = wide_tactic(n, k, tile)
        if choice is not None:
            _wide_kernel(choice[1], tile, linear.weight.device)


@contextmanager
def pinned_tactics() -> Iterator[None]:
    """Offer FlashInfer's tuner only the pinned tactic for pinned problems. The
    tuner's tactic list is a FlashInfer internal: off the checked version,
    every tactic is tuned."""
    if not (
        current_platform.is_device_capability(103)
        and uses_checked_flashinfer("pinning the MXFP8 GEMM tactics")
    ):
        yield
        return
    module = importlib.import_module(
        "flashinfer.gemm.kernels.cute_dsl.low_latency_blockscaled_gemm"
    )
    original: Callable[..., list[Tactic]] | None = getattr(
        module, "autotune_tactics", None
    )
    if original is None:
        yield
        return
    enumerate_tactics = original

    def autotune_tactics(
        problem_sizes_mnkl: tuple[int, int, int, int], *args: object, **kwargs: object
    ) -> list[Tactic]:
        tactics = enumerate_tactics(problem_sizes_mnkl, *args, **kwargs)
        # The runner computes out.T: its problem is (N, M, K, L).
        n, rows, k, _ = problem_sizes_mnkl
        pinned = PINNED_TACTICS.get((n, k))
        if _PINNED_MIN_ROWS <= rows <= LOW_BATCH_MAX_ROWS and pinned in tactics:
            return [pinned]
        return tactics

    module.autotune_tactics = autotune_tactics  # type: ignore[attr-defined]
    try:
        yield
    finally:
        module.autotune_tactics = original  # type: ignore[attr-defined]
