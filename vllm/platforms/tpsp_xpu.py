# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU TPSP backend."""

from __future__ import annotations

import importlib
import logging
from typing import Any

import torch

from vllm.v1.worker.tpsp_profile import ChunkConfig, TPSPBackend

_LOG = logging.getLogger(__name__)


class XPUTPSPBackend(TPSPBackend):
    requires_projection_context = True

    @classmethod
    def create(
        cls,
        group_name: str,
        device: torch.device,
    ) -> XPUTPSPBackend | None:
        if device.type != "xpu":
            return None
        try:
            ops = importlib.import_module("deep_symm.async_tp")
        except ModuleNotFoundError as exc:
            if exc.name not in ("deep_symm", "deep_symm.async_tp"):
                raise
            _LOG.warning("TPSP unavailable: %s", exc)
            return None
        if device.type not in getattr(ops, "tpsp_supported_devices", ("xpu",)):
            _LOG.warning("TPSP fused projection does not support %s", device.type)
            return None
        if not hasattr(ops._C, "fused_matmul_reduce_scatter_norm_all_gather"):
            _LOG.warning("TPSP native fused projection is unavailable")
            return None
        import vllm_xpu_kernels._C  # noqa: F401

        return cls(ops, group_name, device)

    def open(
        self,
        *,
        dtype: torch.dtype,
        tp_size: int,
        hidden_size: int,
        max_batched_tokens: int,
        group_name: str,
        device: torch.device,
    ) -> str | None:
        if self._closed:
            raise RuntimeError("TPSP backend is closed")
        if device != self.device or group_name != self.group_name or tp_size > 8:
            return None
        if not self._valid_open(
            dtype=dtype,
            tp_size=tp_size,
            hidden_size=hidden_size,
            max_batched_tokens=max_batched_tokens,
            group_name=group_name,
            device=device,
        ):
            return None
        return group_name

    def _profile_context(self, context: Any | None) -> str:
        if context != self.group_name:
            raise ValueError("TPSP context belongs to another backend")
        return context

    def run(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        eps: float,
        config: ChunkConfig,
        *,
        norm_type: str = "rms_norm",
        projection_bias: torch.Tensor | None = None,
        norm_bias: torch.Tensor | None = None,
        context: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if context != self.group_name:
            raise ValueError("TPSP context belongs to another backend")
        if (
            norm_type != "rms_norm"
            or projection_bias is not None
            or norm_bias is not None
        ):
            raise ValueError("This TPSP backend does not support bias or LayerNorm")
        return self.ops.fused_matmul_reduce_scatter_norm_all_gather(
            a,
            b,
            weight,
            None,
            self.group_name,
            eps=eps,
            norm_type=norm_type,
            residual=residual,
            microchunk_tokens=config.microchunk_tokens,
        )

    def close(self, context: Any | None = None) -> None:
        if context is not None:
            self._profile_context(context)
            return
        if not self._closed:
            close = getattr(self.ops, "close_tpsp", None)
            if close is not None:
                close(self.group_name)
            else:
                _LOG.warning(
                    "TPSP backend has no close_tpsp API; native pools may remain "
                    "allocated until process exit"
                )
            super().close(context)
