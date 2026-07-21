# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up FA4 CuTeDSL kernels owned by model modules."""

from __future__ import annotations

from typing import Any, Protocol

from torch import nn


class _FA4WarmupWorkerProtocol(Protocol):
    def get_model(self) -> nn.Module: ...


def fa4_cutedsl_warmup(worker: _FA4WarmupWorkerProtocol) -> None:
    seen_compile_keys: set[Any] = set()
    for module in worker.get_model().modules():
        warmup_kernel = getattr(module, "fa4_rel_attention_kernel", None)
        if warmup_kernel is not None:
            warmup_kernel.warmup(seen_compile_keys=seen_compile_keys)
