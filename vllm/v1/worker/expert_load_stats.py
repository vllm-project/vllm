# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.config.expert_load import ExpertLoadStatsConfig
from vllm.distributed import get_ep_group, get_pp_group, get_tp_group
from vllm.distributed.expert_load import (
    ExpertLoadLayer,
    finish_expert_load_iteration,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import triton

logger = init_logger(__name__)


class ExpertLoadReporter:
    """CPU-only JSONL writer, called serially after an export event completes."""

    def __init__(
        self,
        config: ExpertLoadStatsConfig,
        layers: list[int],
        num_experts: int,
        ranks: dict[str, int],
    ):
        self.config = config
        self.layers = layers
        self.labels = {
            "scope": "local",
            "model_role": "target",
            "expert_identity": "logical",
            **ranks,
        }
        self.counts = np.zeros((len(layers), num_experts), dtype=np.int64)
        self.step_begin = 1
        self.dropped = 0
        self.export_errors = 0
        self.path: Path | None = None
        if config.output_dir:
            rank_suffix = "-".join(f"{key}{value}" for key, value in ranks.items())
            self.path = Path(config.output_dir) / (
                f"expert-load-{rank_suffix}-{uuid4().hex}.jsonl"
            )

    def consume(
        self,
        summary: np.ndarray,
        traces: np.ndarray,
        iterations: list[int],
        step_end: int,
        dropped: int,
        report_summary: bool,
    ) -> list[dict[str, Any]]:
        self.counts += summary
        self.dropped += dropped
        records = []
        for iteration, trace in zip(iterations, traces):
            for layer, counts in zip(self.layers, trace):
                records.append(
                    {
                        "event": "vllm.expert_load.iteration",
                        **self.labels,
                        "iteration": iteration,
                        "forward_index": 0,
                        "layer": layer,
                        "counts": counts.tolist(),
                    }
                )
        if report_summary:
            for layer, counts in zip(self.layers, self.counts):
                total = int(counts.sum())
                mean = total / len(counts)
                record = {
                    "event": "vllm.expert_load",
                    **self.labels,
                    "layer": layer,
                    "step_begin": self.step_begin,
                    "step_end": step_end,
                    "assignments": total,
                    "max": int(counts.max()),
                    "mean": mean,
                    "max_mean_ratio": float(counts.max() / mean) if total else 0.0,
                    "unused_experts": int(np.count_nonzero(counts == 0)),
                    "dropped_trace_iterations": self.dropped,
                }
                if self.config.detail == "per_expert":
                    record["counts"] = counts.tolist()
                records.append(record)
            if self.config.reset_after_log:
                self.counts.fill(0)
                self.step_begin = step_end + 1
                self.dropped = 0
        return records

    def write(self, records: list[dict[str, Any]]) -> None:
        if self.path is not None and not self.export_errors:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as output:
                    for record in records:
                        output.write(json.dumps(record, separators=(",", ":")) + "\n")
            except OSError:
                self.export_errors += 1
                logger.exception(
                    "Expert-load JSONL export disabled after a write error; "
                    "serving and summary logging will continue."
                )
        summaries = [r for r in records if r["event"] == "vllm.expert_load"]
        if summaries:
            logger.info(
                "Expert-load steps %d-%d: %d layers, %d assignments, "
                "max layer max/mean %.3f, %d dropped trace iterations, "
                "%d export errors.",
                summaries[0]["step_begin"],
                summaries[0]["step_end"],
                len(summaries),
                sum(r["assignments"] for r in summaries),
                max(r["max_mean_ratio"] for r in summaries),
                summaries[0]["dropped_trace_iterations"],
                self.export_errors,
            )


@dataclass
class _ExportSlot:
    device: torch.Tensor
    host: torch.Tensor
    ready: torch.cuda.Event
    copied: torch.cuda.Event
    future: Future | None = None
    iterations: list[int] = field(default_factory=list)


class ExpertLoadStats:
    """Persistent target counters with two bounded asynchronous export slots.

    Graphs only reference `counts` and `num_valid_tokens`, never export slots.
    The inference stream snapshots into an owned slot; the copy stream waits
    on its ready event. A slot is reusable only after both D2H and writing end.
    Slow writers drop trace samples, not summary counts or inference steps.
    """

    @classmethod
    def create(
        cls, vllm_config: VllmConfig, model: torch.nn.Module, device: torch.device
    ) -> ExpertLoadStats | None:
        config = vllm_config.expert_load_stats_config
        if not config.enabled:
            return None
        cls.validate(vllm_config)
        from vllm.model_executor.layers.fused_moe.layer import MoERunner
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            RoutedExpertsCaptureSource,
        )
        from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

        modules = []
        for module in model.modules():
            if isinstance(module, RoutedExpertsCaptureSource):
                raise ValueError(
                    "Expert-load stats do not yet support custom MoE routers"
                )
            if not isinstance(module, MoERunner):
                continue
            if config.layers is not None and module.layer_id not in config.layers:
                continue
            if module._quant_method.is_monolithic or not isinstance(
                module.router, BaseRouter
            ):
                raise ValueError(
                    "Expert-load stats require a non-monolithic BaseRouter MoE backend"
                )
            if module.moe_config.sp_size not in (
                1,
                vllm_config.parallel_config.tensor_parallel_size,
            ):
                raise ValueError("Unsupported expert-load sequence-parallel topology")
            modules.append(module)
        layers = [module.layer_id for module in modules]
        if len(layers) != len(set(layers)):
            raise ValueError("Expert-load stats require unique target MoE layer IDs")
        if config.layers is not None:
            model_config = vllm_config.model_config
            if max(config.layers) >= model_config.get_total_num_hidden_layers():
                raise ValueError("Requested expert-load layer is outside the model")
            start, end = model_config.get_layers_start_end_indices(
                vllm_config.parallel_config
            )
            local_selection = {i for i in config.layers if start <= i < end}
            if local_selection != set(layers):
                raise ValueError(
                    "Requested expert-load layers are not MoE layers on this stage"
                )
        if not modules:
            return None
        tp_rank = get_tp_group().rank_in_group
        # Replicated routing belongs to TP rank zero. Avoid counters and empty
        # exports on replicas; SP routing before dispatch owns a local shard.
        modules = [
            module
            for module in modules
            if tp_rank == 0
            or (module.moe_config.sp_size > 1 and not module.do_naive_dispatch_combine)
        ]
        if not modules:
            return None
        layers = [module.layer_id for module in modules]
        expert_counts = {module.moe_config.num_logical_experts for module in modules}
        if len(expert_counts) != 1 or min(expert_counts) <= 0:
            raise ValueError(
                "Expert-load stats require the same expert count per layer"
            )
        ranks = {
            "dp_rank": vllm_config.parallel_config.data_parallel_rank,
            "tp_rank": tp_rank,
            "pp_rank": get_pp_group().rank_in_group,
            "ep_rank": get_ep_group().rank_in_group,
        }
        stats = cls(config, layers, expert_counts.pop(), ranks, device)
        for index, module in enumerate(modules):
            assert isinstance(module.router, BaseRouter)
            module.router.expert_load_stats = ExpertLoadLayer(
                stats.counts[index],
                stats.num_valid_tokens,
                ranks["dp_rank"],
                ranks["tp_rank"],
                module.moe_config.sp_size,
                module.do_naive_dispatch_combine,
            )
        return stats

    @staticmethod
    def validate(vllm_config: VllmConfig) -> None:
        parallel = vllm_config.parallel_config
        if not current_platform.is_cuda():
            raise ValueError("Expert-load stats currently require CUDA")
        if (
            parallel.use_ubatching
            or parallel.enable_elastic_ep
            or parallel.prefill_context_parallel_size > 1
            or parallel.decode_context_parallel_size > 1
        ):
            raise ValueError(
                "Expert-load stats do not yet support ubatching, elastic EP, or CP"
            )
        if vllm_config.lora_config is not None:
            raise ValueError("Expert-load stats do not yet support LoRA")
        if (
            vllm_config.ec_transfer_config is not None
            or vllm_config.cache_config.kv_sharing_fast_prefill
        ):
            raise ValueError(
                "Expert-load stats do not yet support encoder disaggregation "
                "or KV-sharing fast prefill"
            )
        if vllm_config.speculative_config is not None and getattr(
            vllm_config.speculative_config, "enable_adaptive_verification", False
        ):
            raise ValueError(
                "Expert-load stats do not yet support adaptive verification"
            )

    def __init__(
        self,
        config: ExpertLoadStatsConfig,
        layers: list[int],
        num_experts: int,
        ranks: dict[str, int],
        device: torch.device,
    ):
        self.config = config
        self.device = device
        self.active = False  # Activated only after compilation and startup warmup.
        self.iteration = 0
        self.summary_begin = 1
        self.dropped = 0
        shape = (len(layers), num_experts)
        self.counts = torch.zeros(shape, dtype=torch.int64, device=device)
        self.summary = torch.zeros_like(self.counts)
        self.num_valid_tokens = torch.zeros((), dtype=torch.int32, device=device)
        capacity = config.flush_interval if config.trace else 0
        self.slots = [
            _ExportSlot(
                torch.empty((capacity + 1, *shape), dtype=torch.int64, device=device),
                torch.empty(
                    (capacity + 1, *shape),
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=True,
                ),
                torch.cuda.Event(),
                torch.cuda.Event(),
            )
            for _ in range(2)
        ]
        self.current: _ExportSlot | None = self.slots[0]
        self.copy_stream = torch.cuda.Stream(device=device)
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="expert-load"
        )
        self.reporter = ExpertLoadReporter(config, layers, num_experts, ranks)
        self.closed = False
        logger.info(
            "Expert-load stats enabled for %d target layers; logical assignments, "
            "local token ownership, draft forwards excluded. "
            "Buffer bytes: %d GPU, %d pinned CPU.",
            len(layers),
            self.counts.nbytes
            + self.summary.nbytes
            + sum(slot.device.nbytes for slot in self.slots),
            sum(slot.host.nbytes for slot in self.slots),
        )

    def _acquire(self) -> None:
        if self.current is not None:
            return
        for slot in self.slots:
            if slot.future is None or slot.future.done():
                if slot.future is not None:
                    slot.future.result()  # I/O errors are handled by the reporter.
                    slot.future = None
                slot.iterations = []
                self.current = slot
                return

    def begin(self, num_tokens: int) -> None:
        self._acquire()
        self.num_valid_tokens.fill_(num_tokens)

    def warmup(self) -> None:
        """Compile export kernels before the serving-time JIT guard is enabled."""
        self.num_valid_tokens.zero_()
        self.counts.zero_()
        self.summary.zero_()
        destinations = [None]
        if self.config.trace:
            destinations.append(self.slots[0].device[1])
        for destination in destinations:
            finish_expert_load_iteration[(triton.cdiv(self.counts.numel(), 256),)](
                self.counts,
                self.summary,
                destination,
                self.counts.numel(),
                destination is not None,
                256,
            )

    def end(self) -> None:
        self.num_valid_tokens.zero_()
        self.iteration += 1
        trace = (
            self.config.trace
            and self.iteration <= self.config.trace_max_iterations
            and (self.iteration - 1) % self.config.trace_interval == 0
        )
        destination = None
        if trace:
            if self.current is None:
                self.dropped += 1
            else:
                self.current.iterations.append(self.iteration)
                destination = self.current.device[len(self.current.iterations)]
        finish_expert_load_iteration[(triton.cdiv(self.counts.numel(), 256),)](
            self.counts,
            self.summary,
            destination,
            self.counts.numel(),
            destination is not None,
            256,
        )
        report = self.iteration - self.summary_begin + 1 >= self.config.log_interval
        trace_full = (
            self.current is not None
            and self.config.trace
            and len(self.current.iterations) == self.config.flush_interval
        )
        trace_done = (
            self.config.trace and self.iteration == self.config.trace_max_iterations
        )
        if self.current is not None and (report or trace_full or trace_done):
            self._flush(report)

    def _flush(self, report_summary: bool) -> None:
        slot = self.current
        assert slot is not None
        slot.device[0].copy_(self.summary)
        self.summary.zero_()
        slot.ready.record()
        size = len(slot.iterations) + 1
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(slot.ready)
            slot.host[:size].copy_(slot.device[:size], non_blocking=True)
            slot.copied.record()
        slot.future = self.executor.submit(
            self._write_slot, slot, self.iteration, self.dropped, report_summary
        )
        self.dropped = 0
        if report_summary:
            self.summary_begin = self.iteration + 1
        self.current = None

    def _write_slot(
        self, slot: _ExportSlot, step_end: int, dropped: int, report_summary: bool
    ) -> None:
        # Only the reporter thread waits. The inference thread never does.
        slot.copied.synchronize()
        records = self.reporter.consume(
            slot.host[0].numpy(),
            slot.host[1 : len(slot.iterations) + 1].numpy(),
            slot.iterations,
            step_end,
            dropped,
            report_summary,
        )
        self.reporter.write(records)

    @contextmanager
    def record(self, num_tokens: int) -> Iterator[None]:
        """Bound collection to a real target forward, outside graph capture."""
        if not self.active or num_tokens == 0:
            yield
            return
        self.begin(num_tokens)
        try:
            yield
        except BaseException:
            self.num_valid_tokens.zero_()
            self.counts.zero_()
            raise
        self.end()

    @torch.inference_mode()
    def close(self) -> None:
        if self.closed:
            return
        self.active = False
        try:
            # Shutdown is allowed to wait; serving never waits on the writer.
            for slot in self.slots:
                if slot.future is not None:
                    slot.future.result()
            self._acquire()
            if self.iteration >= self.summary_begin:
                self._flush(True)
            for slot in self.slots:
                if slot.future is not None:
                    slot.future.result()
        finally:
            self.executor.shutdown(wait=True)
            self.closed = True
