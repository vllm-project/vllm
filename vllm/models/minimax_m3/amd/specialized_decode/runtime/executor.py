# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FFN-launch executor: vLLM runs each sparse layer's input norm and attention,
then one fused launch runs the rest of the layer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm.distributed import get_tp_group
from vllm.logger import init_logger

from ..mono.ffn import ERROR_BYTES, ERROR_WORDS, peer_bytes
from ..mono.runner import FfnRunner
from .weights import MonoLayerWeights

logger = init_logger(__name__)

COVERAGE_LOG_EVERY = 1024


@dataclass(frozen=True)
class MonoHealth:
    failed: bool
    rank: int = -1
    epoch: int = -1
    site: int = 0


class FfnExecutor:
    """Collective construction: every TP rank builds it at the same point."""

    def __init__(
        self,
        layers,
        weights: dict[int, MonoLayerWeights],
        widths: tuple[int, ...],
        tp: int,
    ) -> None:
        from vllm.models.deepseek_v41.amd.mono.common.peer_memory import PeerBuffer

        group = get_tp_group()
        self.rank = group.rank_in_group
        self._cpu_group = group.cpu_group
        device = torch.device("cuda", torch.accelerator.current_device_index())
        self._layers = {layer.layer_id: layer for layer in layers}
        self._weights = weights
        self.peer = PeerBuffer(peer_bytes(tp), self._cpu_group, self.rank, tp, device)
        self.peer.bytes.zero_()
        self.runner = FfnRunner(
            tp, self.rank, self.peer.local, self.peer.addresses, device, widths
        )
        self.runner.warm(next(iter(weights.values())))
        torch.accelerator.synchronize()
        dist.barrier(group=self._cpu_group)
        self.launches = dict.fromkeys(self._layers, 0)
        self.steps = 0
        self._kernels = {
            layer_id: self._layer_kernel(layer_id) for layer_id in self._layers
        }

    def _layer_kernel(self, layer_id: int):
        layer = self._layers[layer_id]
        weights = self._weights[layer_id]
        runner = self.runner
        launches = self.launches

        def run(positions, hidden_states, residual):
            hidden_states, residual = layer.input_layernorm(hidden_states, residual)
            part = layer.self_attn(positions=positions, hidden_states=hidden_states)
            launches[layer_id] += 1
            return runner(part, residual, weights)

        return run

    def layer(self, layer_id: int):
        return self._kernels.get(layer_id)

    def bind(self, layer_id, binding) -> None:
        """The FFN launch reads no cache."""

    def health(self) -> MonoHealth:
        words = self.peer.bytes[: 4 * ERROR_WORDS].view(torch.int32).tolist()
        return MonoHealth(bool(words[0]), *words[1:])

    def abort(self) -> None:
        """Mark this rank failed on every rank, so peers' waits end."""
        words = torch.tensor(
            [1, self.rank, -1, 0], dtype=torch.int32, device=self.peer.bytes.device
        )
        for base in self.peer.addresses.tolist():
            view = _device_words(base, ERROR_BYTES // 4, self.peer.bytes.device)
            view[:ERROR_WORDS].copy_(words)

    def after_step(self) -> None:
        self.steps += 1
        if self.steps % COVERAGE_LOG_EVERY == 0 and self.rank == 0:
            covered = sum(1 for n in self.launches.values() if n)
            logger.info(
                "MiniMax-M3 FFN launch: %d steps, %d launches, %d/%d layers",
                self.steps,
                sum(self.launches.values()),
                covered,
                len(self.launches),
            )

    def close(self) -> None:
        self.peer.close()


def _device_words(ptr: int, n: int, device: torch.device) -> torch.Tensor:
    from vllm.models.deepseek_v41.amd.mono.common.peer_memory import _DeviceBytes

    return torch.as_tensor(_DeviceBytes(ptr, 4 * n), device=device).view(torch.int32)
