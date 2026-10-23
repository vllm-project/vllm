# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FFN-launch executor: vLLM runs each sparse layer's input norm and attention,
then one fused launch runs the rest of the layer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm import envs
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
)
from vllm.models.minimax_m3.amd.ops.gemma_rmsnorm import (
    gemma_fused_add_rmsnorm_fp8,
)
from vllm.platforms import current_platform

from ..mono.ffn import ERROR_BYTES, ERROR_WORDS, peer_bytes
from ..mono.runner import FfnRunner
from .weights import MonoLayerWeights

logger = init_logger(__name__)

COVERAGE_LOG_EVERY = 1024
# a collective reset well before the kernels' tag limit (EPOCH_LIMIT, the
# attention seam's 16-bit tags): every (2^17 - 2^14) / 57 sparse layers ~ 2000
# sampled steps, counted per sampled step (stock-routed ones too); launches no
# step counts (startup's dummy runs) fit in the 2^14 - 2^12 to EPOCH_LIMIT
RESET_LAUNCHES = (1 << 17) - (1 << 14)


@dataclass(frozen=True)
class MonoHealth:
    failed: bool
    rank: int = -1
    epoch: int = -1
    site: int = 0


def tp_agree(ok: bool) -> bool:
    from vllm.models.deepseek_v41.amd.mono.common.peer_memory import tp_agree

    return tp_agree(ok, get_tp_group().cpu_group)


def mark_peers_failed(peer, rank: int) -> None:
    """Set every rank's sticky error word so no failed step returns a token."""
    try:
        words = torch.tensor(
            [1, rank, -1, 0], dtype=torch.int32, device=peer.bytes.device
        )
        for base in peer.addresses.tolist():
            view = _device_words(base, ERROR_BYTES // 4, peer.bytes.device)
            view[:ERROR_WORDS].copy_(words)
        torch.accelerator.synchronize()
    except Exception:
        logger.exception("MiniMax-M3 launch: marking peers failed")


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
        reset_launches: int = RESET_LAUNCHES,
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
        self.resets = 0
        self._since_reset = 0
        self._reset_launches = reset_launches
        self._epoch_host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self._epoch_event: torch.cuda.Event | None = None
        self._slots: torch.Tensor | None = None
        self._kernels = {
            layer_id: self._layer_kernel(layer_id) for layer_id in self._layers
        }

    def _layer_kernel(self, layer_id: int):
        layer = self._layers[layer_id]
        weights = self._weights[layer_id]
        runner = self.runner
        launches = self.launches

        final = layer_id == max(self._layers)
        fuse_input_quant = (
            not envs.VLLM_BATCH_INVARIANT
            and get_input_quant_key(layer.self_attn.qkv_proj) == kFp8DynamicTokenSym
        )

        def run(positions, hidden_states, residual):
            if fuse_input_quant:
                norm = layer.input_layernorm
                quantized, scales, residual = gemma_fused_add_rmsnorm_fp8(
                    hidden_states, residual, norm.weight, norm.variance_epsilon
                )
                hidden_states = QuantizedActivation(
                    quantized,
                    scales,
                    hidden_states.dtype,
                    hidden_states.shape,
                    kFp8DynamicTokenSym,
                )
            else:
                hidden_states, residual = layer.input_layernorm(hidden_states, residual)
            part = layer.self_attn(positions=positions, hidden_states=hidden_states)
            launches[layer_id] += 1
            assert self._slots is not None
            return runner(part, residual, weights, self._slots, final)

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
        mark_peers_failed(self.peer, self.rank)

    def after_step(self) -> None:
        self.steps += 1

    def output_step(self) -> None:
        """Every rank, every sampled step (eager or replayed): the epoch reset
        when the launches since the last one could near the tag limit, and
        launch coverage now and then, read without a host sync."""
        self.outputs += 1
        self._since_reset += 1
        if self._since_reset * len(self._layers) >= self._reset_launches:
            self.reset_epoch()
        if self.outputs % COVERAGE_LOG_EVERY:
            return
        done = self._epoch_event is not None and self._epoch_event.query()
        if done and self.rank == 0:
            logger.info(
                "MiniMax-M3 FFN launch: %d output steps, %d launches since "
                "reset %d (%d sparse layers each)",
                self.outputs,
                int(self._epoch_host[0]),
                self.resets,
                len(self._layers),
            )
        self._epoch_host.copy_(self.runner.epoch[:1], non_blocking=True)
        self._epoch_event = torch.cuda.Event()
        self._epoch_event.record()

    def reset_epoch(self) -> None:
        """Between steps, on every rank together: no launch in flight, every
        mailbox region and the epoch cleared, so tags restart from one. The
        sticky error word is kept. Every rank reaches each phase's agreement;
        a phase that failed anywhere fails the reset on every rank (the
        failing rank marks all ranks' error words), so no launch resumes on a
        partly cleared state."""
        self._reset_phase("drain", torch.accelerator.synchronize)
        self._reset_phase("clear", self._clear)
        self._since_reset = 0
        self.resets += 1
        logger.info("MiniMax-M3 FFN launch: epoch reset %d", self.resets)

    def _clear(self) -> None:
        self.peer.bytes[4 * ERROR_WORDS :].zero_()
        self.runner.reset()
        torch.accelerator.synchronize()

    def _reset_phase(self, phase: str, run) -> None:
        failure: Exception | None = None
        try:
            run()
        except Exception as err:
            failure = err
        try:
            agreed = tp_agree(failure is None)
        except Exception as err:
            agreed, failure = False, failure or err
        if agreed:
            return
        if failure is not None:
            self.abort()
        raise RuntimeError(
            f"MiniMax-M3 FFN launch epoch reset ({phase}) failed on "
            f"{'this' if failure else 'another'} TP rank"
        ) from failure

    def close(self) -> None:
        self.peer.close()


def _device_words(ptr: int, n: int, device: torch.device) -> torch.Tensor:
    from vllm.models.deepseek_v41.amd.mono.common.peer_memory import _DeviceBytes

    return torch.as_tensor(_DeviceBytes(ptr, 4 * n), device=device).view(torch.int32)
