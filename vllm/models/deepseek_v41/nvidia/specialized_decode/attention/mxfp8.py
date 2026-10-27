# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-batch MXFP8 attention projections.

At up to ``LOW_BATCH_MAX_ROWS`` rows the executor runs ``fused_wqa_wkv``,
``wq_b`` and ``wo_b`` on FlashInfer ``mm_mxfp8`` with its
``cutedsl_low_latency`` backend whenever their input is already MXFP8; the
stock layers keep ``cute-dsl``. Both backends read the same weights, weight
scales and F8_128x4 activation scales.

The backend's tactics differ by under 1 us when tuned in isolation, but split-K
tactics are slower inside the back-to-back PDL chain of a decode step, so the
5-8-row bucket of the GB300 shapes is tuned over one pinned tactic per tensor
parallel size. Without FlashInfer autotuning the pinned tactic is still the
runner's first, rather than FlashInfer's generic default, whose one CTA per
128 output columns leaves most SMs idle for these narrow GEMMs.
"""

import importlib
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from functools import cache

import torch
from torch import nn

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutedslMxfp8LinearKernel,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.models.deepseek_v41.common.ops.query_quant import mxfp8_scale_bytes
from vllm.platforms import current_platform
from vllm.utils import flashinfer as vllm_flashinfer

from .. import LOW_BATCH_MAX_ROWS, uses_checked_flashinfer

BACKEND = "cutedsl_low_latency"

# (cta_k, ab_stages, sfb_tmem_stages, split_k) per (N, K, TP) on GB300 for 5-8
# rows, chosen jointly for the chain of the three projections. fused_wqa_wkv is
# replicated, so it has the same shape at TP4 and TP8; wq_b and wo_b halve.
Tactic = tuple[int, int, int, int]
PINNED_TACTICS: dict[tuple[int, int, int], Tactic] = {
    (1792, 5120, 4): (128, 10, 1, 4),
    (8192, 1280, 4): (256, 6, 1, 1),
    (5120, 2048, 4): (256, 4, 1, 2),
    (1792, 5120, 8): (128, 10, 1, 2),
    (4096, 1280, 8): (128, 10, 1, 2),
    (5120, 1024, 8): (128, 8, 1, 1),
}
_PINNED_MIN_ROWS = LOW_BATCH_MAX_ROWS // 2 + 1


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


def shape_representatives(linears: Iterable[nn.Module]) -> list[nn.Module]:
    """One eligible linear per distinct (N, K), in a rank-independent order."""
    by_shape: dict[tuple[int, int], nn.Module] = {}
    for linear in linears:
        kernel = _mxfp8_kernel(linear)
        if kernel is not None and mxfp8_eligible(linear):
            by_shape.setdefault(kernel.config.weight_shape, linear)
    return [by_shape[shape] for shape in sorted(by_shape)]


def run_zeros(linear: nn.Module, rows: int) -> torch.Tensor:
    """Call ``linear`` on an input laid out like the fused quantize producers'."""
    k = linear.weight.shape[0]  # column-major [K, N]
    device = linear.weight.device
    shape = torch.Size((rows, k))
    qa = QuantizedActivation(
        torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device),
        torch.zeros(mxfp8_scale_bytes(rows, k), dtype=torch.uint8, device=device),
        torch.bfloat16,
        shape,
        kMxfp8Dynamic,
    )
    return mxfp8_mm(linear, qa)


def _pinned(problem_sizes_mnkl: tuple[int, int, int, int], tp: int) -> Tactic | None:
    """The pinned tactic of a runner problem at tensor parallel size ``tp``, if
    any. The runner computes out.T: its problem is (N, M, K, L)."""
    n, rows, k, _ = problem_sizes_mnkl
    if not _PINNED_MIN_ROWS <= rows <= LOW_BATCH_MAX_ROWS:
        return None
    return PINNED_TACTICS.get((n, k, tp))


def _tactics_module():
    """FlashInfer's low-latency GEMM module, whose ``autotune_tactics`` lists
    a problem's tactics (a FlashInfer internal), or None off GB300 and the
    checked FlashInfer version."""
    if not (
        current_platform.is_device_capability(103)
        and uses_checked_flashinfer("pinning the MXFP8 GEMM tactics")
    ):
        return None
    module = importlib.import_module(
        "flashinfer.gemm.kernels.cute_dsl.low_latency_blockscaled_gemm"
    )
    return module if hasattr(module, "autotune_tactics") else None


def prefer_pinned_tactics() -> None:
    """Put the pinned tactic first in FlashInfer's tactic list for pinned
    problems, for good: the runner falls back to the first tactic when no tuned
    one is cached, which is every call when FlashInfer autotuning is off, and
    the publishing wo_b GEMM resolves its tactic the same way."""
    module = _tactics_module()
    if module is None or getattr(module, "_dsv41_prefers_pinned", False):
        return
    enumerate_tactics: Callable[..., list[Tactic]] = module.autotune_tactics
    tp = get_tensor_model_parallel_world_size()

    def autotune_tactics(
        problem_sizes_mnkl: tuple[int, int, int, int], *args: object, **kwargs: object
    ) -> list[Tactic]:
        tactics = enumerate_tactics(problem_sizes_mnkl, *args, **kwargs)
        pinned = _pinned(problem_sizes_mnkl, tp)
        if pinned not in tactics:
            return tactics
        return [pinned] + [tactic for tactic in tactics if tactic != pinned]

    module.autotune_tactics = autotune_tactics  # type: ignore[attr-defined]
    module._dsv41_prefers_pinned = True  # type: ignore[attr-defined]


@contextmanager
def pinned_tactics() -> Iterator[None]:
    """Offer FlashInfer's tuner only the pinned tactic for pinned problems. The
    tuner's tactic list is a FlashInfer internal: off the checked version,
    every tactic is tuned."""
    module = _tactics_module()
    if module is None:
        yield
        return
    original: Callable[..., list[Tactic]] = module.autotune_tactics
    tp = get_tensor_model_parallel_world_size()

    def autotune_tactics(
        problem_sizes_mnkl: tuple[int, int, int, int], *args: object, **kwargs: object
    ) -> list[Tactic]:
        tactics = original(problem_sizes_mnkl, *args, **kwargs)
        pinned = _pinned(problem_sizes_mnkl, tp)
        return [pinned] if pinned in tactics else tactics

    module.autotune_tactics = autotune_tactics  # type: ignore[attr-defined]
    try:
        yield
    finally:
        module.autotune_tactics = original  # type: ignore[attr-defined]
