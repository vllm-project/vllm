# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FFN-launch executor: vLLM runs each sparse layer's input norm and attention,
then one fused launch runs the rest of the layer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform

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


def tp_agree(ok: bool) -> bool:
    from vllm.models.deepseek_v41.amd.mono.common.peer_memory import tp_agree

    return tp_agree(ok, get_tp_group().cpu_group)


class FfnExecutor:
    """Collective construction: every TP rank builds it at the same point, and
    every phase that can fail on one rank ends in a TP agreement."""

    uses_caches = False

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
        device = torch.device("cuda", torch.accelerator.current_device_index())
        self._layers = {layer.layer_id: layer for layer in layers}
        self._weights = weights
        self.peer = PeerBuffer(peer_bytes(tp), group.cpu_group, self.rank, tp, device)
        failure: Exception | None = None
        try:
            self.peer.bytes.zero_()
            self.runner = FfnRunner(
                tp, self.rank, self.peer.local, self.peer.addresses, device, widths
            )
            cus = current_platform.num_compute_units(device.index)
            self.runner.warm(next(iter(weights.values())), compute_units=cus)
            torch.accelerator.synchronize()
        except Exception as err:
            failure = err
        if not tp_agree(failure is None):
            self.peer.close()
            raise RuntimeError(
                "MiniMax-M3 FFN launch setup failed on "
                f"{'this' if failure else 'another'} TP rank"
            ) from failure
        logger.info_once(
            "MiniMax-M3 FFN launch: rows %s, resident CTAs per CU %s",
            widths,
            tuple(sorted(set(self.runner.residency.values()))),
        )
        self.fault_word = self.peer.bytes[: 4 * ERROR_WORDS].view(torch.int32)
        self.launches = dict.fromkeys(self._layers, 0)
        self.steps = 0
        self.outputs = 0
        self._slots: torch.Tensor | None = None
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
            assert self._slots is not None
            return runner(part, residual, weights, self._slots)

        return run

    def begin_step(self, slots: torch.Tensor) -> None:
        """This step's slot mapping (-1: padding), read by every launch on the
        device; under a FULL graph, the runner's persistent buffer."""
        self._slots = slots

    def layer(self, layer_id: int):
        return self._kernels.get(layer_id)

    def bind(self, layer_id, binding) -> None:
        raise NotImplementedError("the FFN launch reads no KV cache")

    def health(self) -> MonoHealth:
        words = self.fault_word.tolist()
        return MonoHealth(bool(words[0]), *words[1:])

    def abort(self) -> None:
        """Mark this rank failed on every rank, so peers' waits end."""
        try:
            words = torch.tensor(
                [1, self.rank, -1, 0], dtype=torch.int32, device=self.peer.bytes.device
            )
            for base in self.peer.addresses.tolist():
                view = _device_words(base, ERROR_BYTES // 4, self.peer.bytes.device)
                view[:ERROR_WORDS].copy_(words)
            torch.accelerator.synchronize()
        except Exception:
            logger.exception("MiniMax-M3 FFN launch: marking peers failed")

    def after_step(self) -> None:
        self.steps += 1

    def output_step(self) -> None:
        """Every step's output (eager or replayed): launch coverage now and then
        from the kernels' own epoch, which counts launches."""
        self.outputs += 1
        if self.outputs % COVERAGE_LOG_EVERY == 0 and self.rank == 0:
            logger.info(
                "MiniMax-M3 FFN launch: %d output steps, %d launches "
                "(%d sparse layers each)",
                self.outputs,
                int(self.runner.epoch[0].item()),
                len(self._layers),
            )

    def close(self) -> None:
        self.peer.close()


def _device_words(ptr: int, n: int, device: torch.device) -> torch.Tensor:
    from vllm.models.deepseek_v41.amd.mono.common.peer_memory import _DeviceBytes

    return torch.as_tensor(_DeviceBytes(ptr, 4 * n), device=device).view(torch.int32)
