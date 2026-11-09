# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which MiniMax-M3.1 DSpark draft steps run on the drafter kernels
(`VLLM_ROCM_MINIMAX_M3_MONO_DRAFTER=1`).

A draft step of 1..MAX_REQS requests (7 query rows each) runs its 5 layers and
final norm in launches of up to 8 requests; other steps run the stock layers.
Every condition is decided from TP-replicated state: the kernels exchange
partials in-kernel, so ranks that disagreed would wait on each other.
"""

import importlib.util
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.models.minimax_m3.amd.mono.config import MonoUnsupported
from vllm.models.minimax_m3.amd.mono.drafter import layout
from vllm.platforms.rocm import on_gfx950

if TYPE_CHECKING:
    from vllm.models.minimax_m3.amd.mono.drafter.runner import DrafterRunner

logger = init_logger(__name__)


def _flydsl_available() -> bool:
    return importlib.util.find_spec("flydsl") is not None


def _config_refusal(vllm_config: VllmConfig) -> str | None:
    """Why this deployment cannot use the drafter kernels at all, or None."""
    pc = vllm_config.parallel_config
    spec = vllm_config.speculative_config
    checks = (
        (on_gfx950(), "not gfx950"),
        (_flydsl_available(), "FlyDSL not installed"),
        (pc.tensor_parallel_size == 4, f"TP {pc.tensor_parallel_size} != 4"),
        (pc.data_parallel_size == 1, "DP > 1"),
        (pc.pipeline_parallel_size == 1, "PP > 1"),
        (spec is not None and spec.method == "dspark", "no DSpark draft"),
        (
            spec is not None and spec.num_speculative_tokens == layout.QPR,
            f"{layout.QPR} speculative tokens required",
        ),
    )
    for ok, why in checks:
        if not ok:
            return why
    return None


class DrafterDecode:
    """Routes supported draft steps of one `MiniMaxM3DSparkModel` to the kernels."""

    def __init__(self, model, vllm_config: VllmConfig) -> None:
        self._model = model
        self._runner: DrafterRunner | None = None
        why = _config_refusal(vllm_config)
        self._enabled = why is None
        if why is not None:
            logger.info("MiniMax-M3 DSpark drafter kernels off: %s", why)

    def _disable(self, why: str) -> None:
        logger.warning("MiniMax-M3 DSpark drafter kernels off: %s", why)
        self._enabled = False

    def _ensure_runner(self) -> bool:
        """Build (or rebuild after a KV cache re-bind) the runner outside capture."""
        if self._runner is not None and self._runner.stale():
            if torch.cuda.is_current_stream_capturing():
                return False
            self._runner.close()
            self._runner = None
            logger.info("MiniMax-M3 DSpark drafter kernels: KV cache re-bound")
        if self._runner is not None:
            return True
        if torch.cuda.is_current_stream_capturing():
            return False
        from vllm.models.minimax_m3.amd.mono.drafter.runner import DrafterRunner

        try:
            self._runner = DrafterRunner(self._model)
        except MonoUnsupported as why:
            self._disable(str(why))
            return False
        logger.info("MiniMax-M3 DSpark drafter kernels on")
        return True

    def maybe_forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """The step's final hidden states from the kernels, or None for the stock
        layers."""
        if not self._enabled:
            return None
        n = positions.numel()
        if n % layout.QPR or not 0 < n // layout.QPR <= layout.MAX_REQS:
            return None
        if positions.dtype != torch.int64:
            return None
        if not is_forward_context_available():
            return None
        if get_forward_context().attn_metadata is None:
            return None
        if not self._ensure_runner():
            return None
        assert self._runner is not None
        if inputs_embeds is None:
            inputs_embeds = self._model.embed_input_ids(input_ids)
        return self._runner.forward(positions, inputs_embeds)
