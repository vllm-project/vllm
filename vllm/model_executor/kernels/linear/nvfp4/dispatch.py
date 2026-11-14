# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 W4A16 linear that picks Marlin or FlashInfer CuTe-DSL per layer shape and
number of tokens.

Which kernel is faster depends on the layer shape and on the number of tokens in the
step: on sm_12x CuTe-DSL is much faster than Marlin above 64 tokens for some shapes
and slower for others. At load time the first layer of each shape times both kernels
on a few token counts and keeps a table of the winner for that shape, which every
layer of the shape shares. A shape where CuTe-DSL never wins keeps only Marlin, so it
does not pay for a second copy of the weights. The choice is made in a custom op, so
torch.compile does not specialize on the number of tokens and CUDA graphs bake in the
kernel for each captured size.

Selected automatically on sm_12x. VLLM_NVFP4_W4A16_DISPATCH=0 disables it.
"""

import bisect
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig
from .flashinfer import FlashInferCuteDslNvFp4W4A16LinearKernel
from .marlin import MarlinNvFp4LinearKernel

logger = init_logger(__name__)

# Token counts at which the two kernels are timed. A step with m tokens follows the
# result of the smallest of them that is at least m, or of the last one above all.
_CALIBRATION_TOKENS = (8, 16, 32, 64, 96, 128, 256, 512, 1024, 2048)
# CuTe-DSL must take less than this fraction of Marlin's time, and save at least this
# many microseconds per call, to count as faster. The second condition keeps timing
# noise from deciding for very small layers.
_MAX_TIME_RATIO = 0.97
_MIN_SAVING_US = 5.0
# Fewer consecutive wins than this are treated as noise.
_MIN_WIN_RUN = 2


def _longest_win_run(wins: list[bool]) -> tuple[bool, ...]:
    """Keep only the longest run of consecutive wins, if it is long enough."""
    best_start = best_len = start = 0
    for i, win in enumerate(list(wins) + [False]):
        if win:
            if i == 0 or not wins[i - 1]:
                start = i
        elif i > 0 and wins[i - 1] and i - start > best_len:
            best_start, best_len = start, i - start
    if best_len < _MIN_WIN_RUN:
        return (False,) * len(wins)
    return tuple(best_start <= i < best_start + best_len for i in range(len(wins)))


@dataclass(frozen=True)
class _Plan:
    use_cute: tuple[bool, ...]

    @property
    def any_cute(self) -> bool:
        return any(self.use_cute)

    def pick_cute(self, num_tokens: int) -> bool:
        i = bisect.bisect_left(_CALIBRATION_TOKENS, num_tokens)
        return self.use_cute[min(i, len(self.use_cute) - 1)]


# Shared by all layers with the same (output size, input size, device).
_PLANS: dict[tuple, _Plan] = {}
# One entry per layer: (marlin kernel, layer, CuTe-DSL kernel, its layer, plan).
_ENTRIES: list = []


def _dispatch_impl(x: torch.Tensor, idx: int, n: int) -> torch.Tensor:
    marlin_k, marlin_l, cute_k, cute_l, plan = _ENTRIES[idx]
    if cute_k is not None and plan.pick_cute(x.shape[0]):
        return cute_k.apply_weights(cute_l, x)
    return marlin_k.apply_weights(marlin_l, x)


def _dispatch_fake(x: torch.Tensor, idx: int, n: int) -> torch.Tensor:
    return x.new_empty((x.shape[0], n))


direct_register_custom_op(
    op_name="nvfp4_w4a16_dispatch",
    op_func=_dispatch_impl,
    mutates_args=[],
    fake_impl=_dispatch_fake,
)


def _time_us(fn, x: torch.Tensor, reps: int = 10) -> float:
    """Microseconds per call, from a CUDA graph of `reps` calls when it can be
    captured, as in serving, and from eager launches otherwise."""
    for _ in range(3):
        fn(x)
    torch.accelerator.synchronize()

    def eager() -> None:
        for _ in range(reps):
            fn(x)

    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            eager()
        run = graph.replay
    except Exception:
        run = eager
    run()
    torch.accelerator.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(3):
        run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / (3 * reps)


class MarlinCuteDslNvFp4W4A16DispatchKernel(NvFp4LinearKernel):
    """Marlin or CuTe-DSL per call, from a per-shape table measured at load time."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        ok_m, why_m = MarlinNvFp4LinearKernel.is_supported()
        ok_c, why_c = FlashInferCuteDslNvFp4W4A16LinearKernel.is_supported(
            compute_capability
        )
        if ok_m and ok_c:
            return True, None
        return False, why_m or why_c

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def __init__(self, config: NvFp4LinearLayerConfig) -> None:
        super().__init__(config)
        self.marlin = MarlinNvFp4LinearKernel(config)
        self.cute = FlashInferCuteDslNvFp4W4A16LinearKernel(config)
        self._cute_plan: _Plan | None = None

    def uses_cutedsl(self, num_tokens: int) -> bool:
        """Whether a step of `num_tokens` tokens runs this layer on CuTe-DSL."""
        return self._cute_plan is not None and self._cute_plan.pick_cute(num_tokens)

    def _calibrate(self, layer: nn.Module, cute_layer: nn.Module) -> _Plan:
        n = layer.output_size_per_partition
        k = layer.input_size_per_partition
        start = time.monotonic()
        try:
            use = []
            for m in _CALIBRATION_TOKENS:
                x = torch.randn(
                    m, k, dtype=layer.params_dtype, device=layer.weight.device
                )
                t_marlin = _time_us(lambda a: self.marlin.apply_weights(layer, a), x)
                t_cute = _time_us(lambda a: self.cute.apply_weights(cute_layer, a), x)
                use.append(
                    t_cute < _MAX_TIME_RATIO * t_marlin
                    and t_marlin - t_cute >= _MIN_SAVING_US
                )
                logger.debug(
                    "NVFP4 W4A16 %dx%d at %d tokens: Marlin %.1f us, CuTe-DSL %.1f us",
                    n,
                    k,
                    m,
                    t_marlin,
                    t_cute,
                )
        except Exception as e:
            logger.warning(
                "NVFP4 W4A16 calibration failed for %dx%d layers, using Marlin only. "
                "Set VLLM_NVFP4_W4A16_DISPATCH=0 to skip it. Error: %s",
                n,
                k,
                e,
            )
            return _Plan((False,) * len(_CALIBRATION_TOKENS))
        seconds = time.monotonic() - start
        use_cute = _longest_win_run(use)
        wins = [m for m, u in zip(_CALIBRATION_TOKENS, use_cute) if u]
        if wins:
            extra_mib = sum(
                p.numel() * p.element_size() for p in cute_layer.parameters()
            ) / (1 << 20)
            logger.info(
                "NVFP4 W4A16 %dx%d layers: CuTe-DSL is faster at %s tokens and Marlin "
                "at the others (calibrated in %.1f s, %.0f MiB of extra weights per "
                "layer; VLLM_NVFP4_W4A16_DISPATCH=0 disables this)",
                n,
                k,
                wins,
                seconds,
                extra_mib,
            )
        else:
            logger.info(
                "NVFP4 W4A16 %dx%d layers: Marlin at every token count, CuTe-DSL is "
                "not faster (calibrated in %.1f s)",
                n,
                k,
                seconds,
            )
        return _Plan(use_cute)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        n = int(layer.output_size_per_partition)
        k = int(layer.input_size_per_partition)
        key = (n, k, layer.weight.device)
        plan = _PLANS.get(key)
        cute_layer = None
        if plan is None or plan.any_cute:
            # Each backend repacks its weights in place, so CuTe-DSL gets a copy to
            # repack before Marlin does.
            cute_layer = nn.Module()
            for name in ("weight", "weight_scale", "weight_global_scale"):
                param = nn.Parameter(
                    getattr(layer, name).data.clone(), requires_grad=False
                )
                setattr(cute_layer, name, param)
            cute_layer.output_size_per_partition = n
            cute_layer.input_size_per_partition = k
            cute_layer.params_dtype = layer.params_dtype
            self.cute.process_weights_after_loading(cute_layer)
        self.marlin.process_weights_after_loading(layer)
        if plan is None:
            plan = _PLANS[key] = self._calibrate(layer, cute_layer)
        if not plan.any_cute:
            cute_layer = None
        self._cute_plan = plan if cute_layer is not None else None
        self._n = n
        layer._nvfp4_dispatch_idx = len(_ENTRIES)
        _ENTRIES.append(
            (self.marlin, layer, self.cute if cute_layer else None, cute_layer, plan)
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1])
        out = torch.ops.vllm.nvfp4_w4a16_dispatch(
            x_2d, layer._nvfp4_dispatch_idx, self._n
        )
        if bias is not None:
            out = out + bias
        return out.view(*x.shape[:-1], self._n)
