# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-k and MXFP8 before EP dispatch; local routing is built after dispatch."""

from collections.abc import Callable
from functools import cache
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import make_fake_stream

from vllm.models.common.ops.moe_tail_fusion.primitives import current_cu_stream
from vllm.models.deepseek_v41.nvidia.specialized_decode.moe.routed_expert_gemm import (
    HIDDEN,
    TOP_K,
    FrontConfig,
    _fake,
    _pointer,
    width,
)

from ..moe.routed_expert_front_cute import (
    RoutedExpertFront,
)
from . import MAX_ROWS


def _build_rows(rows: int) -> int:
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"DEP front requires 1..{MAX_ROWS} rows")
    return width(rows) if rows <= 192 else MAX_ROWS


class FrontOutput(NamedTuple):
    x_quant: torch.Tensor
    x_scale: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor


@cache
def compile_front(device_index: int, image: bool, built: int, quantize: bool = True):
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            RoutedExpertFront(
                image, built, build_routing_record=False, quantize=quantize
            ),
            _fake(cutlass.Float32, 16),
            _fake(cutlass.Float32, 16),
            _fake(cutlass.Float32, 16),
            _fake(cutlass.Int32, 4),
            Int32(0),
            Int32(0),
            Float32(1.0),
            _fake(cutlass.Uint8, 1),
            Int32(0),
            _fake(cutlass.Uint32, 16),
            _fake(cutlass.Int32, 16),
            _fake(cutlass.Float32, 16),
            _fake(cutlass.Uint8, 16),
            _fake(cutlass.Uint8, 16),
            _fake(cutlass.Int32, 16),
            _fake(cutlass.Int32, 16),
            Int32(1),
            make_fake_stream(),
        )


def _tensor(t, shape, dtype, device):
    return (
        isinstance(t, torch.Tensor)
        and tuple(t.shape) == tuple(shape)
        and t.dtype == dtype
        and t.device == device
        and t.is_contiguous()
        and t.data_ptr() % 16 == 0
    )


class DEPFront:
    def __init__(self, config: FrontConfig, workspace=None, *, quantize=True):
        self.config = config
        self.quantize = quantize
        self.workspace = workspace
        self.device = config.bias.device
        self.device_index = self.device.index
        if self.device.type != "cuda" or not _tensor(
            config.bias, (384,), torch.float32, self.device
        ):
            raise ValueError("front needs contiguous CUDA FP32 bias[384]")
        if config.bias_vl is not None and not _tensor(
            config.bias_vl, (384,), torch.float32, self.device
        ):
            raise ValueError("image bias must match the global router")
        self._compiled: dict[int, Callable] = {}
        # Unused pointer arguments retain the existing kernel interface. The
        # pre-dispatch specialization never reads or writes either record.
        self._unused = torch.empty(4, dtype=torch.int32, device=self.device)

    def warmup(self, rows: int):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("compile the front before capture")
        built = _build_rows(rows)
        self._compiled[built] = compile_front(
            self.device_index, self.config.bias_vl is not None, built, self.quantize
        )

    def accepts(self, x, logits, input_ids=None, is_padding=None):
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            return False
        rows = x.size(0)
        if not 1 <= rows <= MAX_ROWS:
            return False
        if not _tensor(x, (rows, HIDDEN), torch.bfloat16, self.device):
            return False
        if not _tensor(logits, (rows, 384), torch.float32, self.device):
            return False
        if self.config.bias_vl is not None and (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.dtype not in (torch.int32, torch.int64)
            or not _tensor(input_ids, (rows,), input_ids.dtype, self.device)
        ):
            return False
        if is_padding is not None and not _tensor(
            is_padding, (rows,), torch.bool, self.device
        ):
            return False
        return _build_rows(rows) in self._compiled

    def __call__(self, x, logits, input_ids=None, is_padding=None, *, output=None):
        if not self.accepts(x, logits, input_ids, is_padding):
            raise ValueError("unsupported or unwarmed front; retain the original path")
        rows = x.size(0)
        config = self.config
        image = config.bias_vl is not None
        ids = self._unused if not image else input_ids
        specs = (
            ("front_x", (rows, HIDDEN), torch.float8_e4m3fn),
            ("front_scale", (rows, HIDDEN // 32), torch.uint8),
            ("front_ids", (rows, TOP_K), torch.int32),
            ("front_weights", (rows, TOP_K), torch.float32),
        )
        if output is None:
            if not self.quantize:
                raise ValueError("routing-only front requires prequantized output")
            output = FrontOutput(
                *(
                    torch.empty(shape, dtype=dtype, device=self.device)
                    if self.workspace is None
                    else self.workspace.view(name, shape, dtype)
                    for name, shape, dtype in specs
                )
            )
        elif not isinstance(output, FrontOutput) or not all(
            _tensor(tensor, shape, dtype, self.device)
            for tensor, (_, shape, dtype) in zip(output, specs)
        ):
            raise ValueError("front output must match the contiguous local row layout")
        self._compiled[_build_rows(rows)](
            _pointer(cutlass.Float32, logits, 16),
            _pointer(cutlass.Float32, config.bias, 16),
            _pointer(cutlass.Float32, config.bias_vl if image else config.bias, 16),
            _pointer(cutlass.Int32, ids, 4),
            Int32(ids.dtype == torch.int64),
            Int32(config.image_sentinel_lo),
            Float32(config.scale),
            _pointer(
                cutlass.Uint8, self._unused if is_padding is None else is_padding, 1
            ),
            Int32(is_padding is not None),
            _pointer(cutlass.Uint32, x, 16),
            _pointer(cutlass.Int32, output.topk_ids, 16),
            _pointer(cutlass.Float32, output.topk_weights, 16),
            _pointer(cutlass.Uint8, output.x_quant, 16),
            _pointer(cutlass.Uint8, output.x_scale, 16),
            _pointer(cutlass.Int32, self._unused, 16),
            _pointer(cutlass.Int32, self._unused, 16),
            Int32(rows),
            current_cu_stream(),
        )
        return output
