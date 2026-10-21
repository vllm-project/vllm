# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang dispatch layer for the DSpark confidence head kernel.

Implements :class:`VllmJitKernel` so vLLM's startup warmup can pre-compile
every (n_embed, markov_rank) shape the model will hit, and registers the
custom op via :func:`direct_register_custom_op` so CUDA-Graph capture and
``torch.compile`` pass through transparently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.kernels.dspark.tilelang_kernels import (
    confidence_head_fused_tilelang,
)
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


class DsparkConfidenceTileLangKernel(
    VllmJitKernel["DsparkConfidenceTileLangKernel.CompileKey"]
):
    """VllmJitKernel wrapper for the fused confidence head.

    CompileKey fields are *static* kernel parameters that determine which
    TileLang artifact to use. Dynamic values (batch size, strides, temp)
    are passed at call time.
    """

    @dataclass(frozen=True)
    class CompileKey:
        n_embed: int  # H
        markov_rank: int  # r; 0 means with_markov=False

    @staticmethod
    def kernel(*args: Any, **kwargs: Any) -> Any:
        """Dispatch to the TileLang JIT'd kernel; TileLang caches by shape."""
        confidence_head_fused_tilelang(*args, **kwargs)

    def dispatch(
        self,
        *,
        n_embed: int,
        markov_rank: int,
    ) -> DsparkConfidenceTileLangKernel.CompileKey:
        return self.CompileKey(n_embed=n_embed, markov_rank=markov_rank)

    def get_warmup_keys(
        self, **dispatch_kwargs: Any
    ) -> list[DsparkConfidenceTileLangKernel.CompileKey]:
        return self._trace_dispatch(self.dispatch)(**dispatch_kwargs)

    def compile(self, compile_key: DsparkConfidenceTileLangKernel.CompileKey) -> None:
        """Force TileLang to compile and cache this shape, then record the
        shape as compiled in our own cache so subsequent calls skip the
        TileLang dispatch path.

        TileLang holds its own per-shape JIT cache internally; ours only
        acts as a "shape already seen" memo for the VllmJitKernel protocol.
        """
        if not torch.cuda.is_available():
            self._compiled_cache[compile_key] = None
            return
        n_embed = compile_key.n_embed
        markov_rank = compile_key.markov_rank
        B1 = max(markov_rank, 1)
        device = torch.device("cuda")
        dummy_x0 = torch.zeros(1, n_embed, dtype=torch.bfloat16, device=device)
        dummy_x1 = torch.zeros(1, B1, dtype=torch.bfloat16, device=device)
        dummy_w = torch.zeros(n_embed + markov_rank, dtype=torch.float32, device=device)
        dummy_out = torch.zeros(1, dtype=torch.float32, device=device)
        confidence_head_fused_tilelang(
            dummy_x0,
            dummy_x1,
            dummy_w,
            dummy_out,
            1.0,
            n_embed=n_embed,
            markov_rank=markov_rank,
        )
        # Value is unused at call time; presence in the dict is what the
        # parent class's ``_get_or_compile`` checks.
        self._compiled_cache[compile_key] = True

    def __call__(
        self,
        hidden: torch.Tensor,  # (n, H) bf16
        markov_embed: torch.Tensor,  # (n, r) bf16 (or (n, 1) dummy when r=0)
        weight: torch.Tensor,  # (H + r,) fp32
        out: torch.Tensor,  # (n,) fp32
        temp: float,
    ) -> None:
        n_embed = hidden.shape[-1]
        markov_rank = weight.shape[0] - n_embed

        compile_key = self.dispatch(n_embed=n_embed, markov_rank=markov_rank)
        self._get_or_compile(
            compile_key,
            runtime_context={
                "batch": hidden.shape[0],
                "temp": temp,
            },
        )
        self.kernel(
            hidden,
            markov_embed,
            weight,
            out,
            temp,
            n_embed=n_embed,
            markov_rank=markov_rank,
        )


# Module-level singleton (matches other VllmJitKernel users).
DSPARK_CONFIDENCE_TILELANG_KERNEL = DsparkConfidenceTileLangKernel()


def is_dspark_fused_available() -> bool:
    """True iff the fused TileLang path can run on this platform."""
    if not torch.cuda.is_available():
        return False
    if not current_platform.is_cuda():
        return False
    try:
        if torch.cuda.get_device_capability(0)[0] < 9:
            return False  # PDL assumed
    except Exception:
        return False
    return True


# --- torch custom-op registration ------------------------------------------
def _dspark_confidence_tilelang_fake(
    hidden: torch.Tensor,
    markov_embed: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    temp: float,
) -> None:
    """Fake impl for op-registration (used during torch.compile trace)."""
    return None


def _dspark_confidence_tilelang(
    hidden: torch.Tensor,
    markov_embed: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    temp: float,
) -> None:
    DSPARK_CONFIDENCE_TILELANG_KERNEL(hidden, markov_embed, weight, out, temp)


direct_register_custom_op(
    op_name="dspark_confidence_tilelang",
    op_func=_dspark_confidence_tilelang,
    mutates_args=["out"],
    fake_impl=_dspark_confidence_tilelang_fake,
)
