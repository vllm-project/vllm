# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup autotuning for FlashInfer ReplaySSM CUDA-graph launches."""

from __future__ import annotations

import gc
import json
import math
import statistics
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.distributed.parallel_state import get_world_group
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    FLASHINFER_REPLAYSSM_AUTO_TACTIC,
    FlashInferReplaySSMTactic,
    update_replayssm_ring_trackers,
    use_flashinfer_replayssm_tactic,
)
from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    resolve_flashinfer_autotune_file,
    write_flashinfer_autotune_cache,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_CACHE_SCHEMA_VERSION = 3
_TUNING_FIXTURE = "t1_mixed_history_cycle_v4"
_CACHE_FILE_NAME = "replayssm_autotune_configs.json"
_PRECOMPUTE_MAIN_TACTIC_COUNT = 2
_RELATIVE_TIE_TOLERANCE = 0.001

FLASHINFER_REPLAYSSM_TUNING_CANDIDATES = (
    FLASHINFER_REPLAYSSM_AUTO_TACTIC,
    FlashInferReplaySSMTactic("monolith"),
    FlashInferReplaySSMTactic("two-kernel", 1, 1),
    FlashInferReplaySSMTactic("two-kernel", 1, 2),
    FlashInferReplaySSMTactic("two-kernel", 1, 4),
    FlashInferReplaySSMTactic("two-kernel", 1, 8),
    FlashInferReplaySSMTactic("two-kernel", 1, 16),
    FlashInferReplaySSMTactic("two-kernel", 2, 1),
    FlashInferReplaySSMTactic("two-kernel", 2, 2),
    FlashInferReplaySSMTactic("two-kernel", 2, 4),
    FlashInferReplaySSMTactic("two-kernel", 2, 8),
    FlashInferReplaySSMTactic("two-kernel", 2, 16),
)
_TACTICS_BY_NAME = {
    tactic.name: tactic for tactic in FLASHINFER_REPLAYSSM_TUNING_CANDIDATES
}


def _tactic_from_name(name: str) -> FlashInferReplaySSMTactic | None:
    if tactic := _TACTICS_BY_NAME.get(name):
        return tactic
    for base in FLASHINFER_REPLAYSSM_TUNING_CANDIDATES:
        if base.algorithm != "two-kernel":
            continue
        prefix = f"{base.name}_h"
        if name.startswith(prefix):
            try:
                heads_per_cta = int(name.removeprefix(prefix))
                return replace(base, precompute_heads_per_cta=heads_per_cta)
            except ValueError:
                return None
    return None


def _precompute_heads_per_cta_candidates(heads_per_group: int) -> tuple[int, ...]:
    """Return the distinct HEADS_PER_GROUP >> k launch geometries."""
    if heads_per_group <= 0:
        raise ValueError("heads_per_group must be positive")
    candidates = set()
    value = heads_per_group
    while value:
        candidates.add(value)
        value >>= 1
    return tuple(sorted(candidates))


def _expanded_precompute_tactics(
    main_timings: list[float], heads_per_group: int
) -> tuple[FlashInferReplaySSMTactic, ...]:
    ranked = sorted(
        (
            index
            for index, tactic in enumerate(
                FLASHINFER_REPLAYSSM_TUNING_CANDIDATES
            )
            if tactic.algorithm == "two-kernel"
            and math.isfinite(main_timings[index])
        ),
        key=lambda index: (main_timings[index], index),
    )[:_PRECOMPUTE_MAIN_TACTIC_COUNT]
    return tuple(
        replace(base, precompute_heads_per_cta=heads_per_cta)
        for index in ranked
        for base in (FLASHINFER_REPLAYSSM_TUNING_CANDIDATES[index],)
        for heads_per_cta in _precompute_heads_per_cta_candidates(heads_per_group)
    )


@dataclass
class FlashInferReplaySSMAutotuneResult:
    spec_query_len: int
    tactics: dict[int, FlashInferReplaySSMTactic]

    def tactic_for(
        self, batch_descriptor: BatchDescriptor
    ) -> FlashInferReplaySSMTactic | None:
        if (
            not batch_descriptor.uniform
            or batch_descriptor.num_reqs is None
            or batch_descriptor.num_tokens
            != batch_descriptor.num_reqs * self.spec_query_len
        ):
            return None
        return self.tactics.get(batch_descriptor.num_reqs)


def _make_cache_key(fingerprint: dict[str, Any], batch: int, T: int) -> str:
    return json.dumps(
        {
            **fingerprint,
            "batch_sequences": batch,
            "spec_query_len": T,
            "fixture": _TUNING_FIXTURE,
            "candidate_schema": {
                "main": [
                    tactic.name
                    for tactic in FLASHINFER_REPLAYSSM_TUNING_CANDIDATES
                ],
                "precompute": "top2_main_x_heads_per_group_shift_chain",
                "relative_tie_tolerance": _RELATIVE_TIE_TOLERANCE,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _select_fastest(
    timings: list[float],
    candidates: tuple[FlashInferReplaySSMTactic, ...] = (
        FLASHINFER_REPLAYSSM_TUNING_CANDIDATES
    ),
) -> FlashInferReplaySSMTactic | None:
    if len(timings) != len(candidates):
        raise ValueError("one timing is required for each ReplaySSM tactic")
    finite = [i for i, timing in enumerate(timings) if math.isfinite(timing)]
    if not finite:
        return None
    best_timing = min(timings[i] for i in finite)
    tied = [
        i
        for i in finite
        if timings[i] <= best_timing * (1 + _RELATIVE_TIE_TOLERANCE)
    ]
    winner = min(tied)
    return candidates[winner]


def _load_cache(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    if payload.get("schema_version") != _CACHE_SCHEMA_VERSION:
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return {}
    return {
        key: value
        for key, value in entries.items()
        if isinstance(key, str)
        and isinstance(value, str)
        and _tactic_from_name(value) is not None
    }


def _save_cache(path: Path, entries: dict[str, str]) -> None:
    payload = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "entries": dict(sorted(entries.items())),
    }
    write_flashinfer_autotune_cache(
        path,
        (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode(),
    )


def _find_replayssm_layers(runner: GPUModelRunner) -> tuple[MambaMixer2, ...]:
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2

    return tuple(
        module
        for module in runner.get_model().modules()
        if (
            isinstance(module, MambaMixer2)
            and module.use_replayssm
            and module.mamba_config.backend == MambaBackendEnum.FLASHINFER
        )
    )


def _uniform_capture_batches(runner: GPUModelRunner) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                desc.num_reqs
                for _, descs in runner.cudagraph_dispatcher.get_capture_descs()
                for desc in descs
                if desc.uniform and desc.num_reqs is not None
            }
        )
    )


def _layer_fingerprint(layer: MambaMixer2) -> dict[str, Any]:
    import flashinfer
    from flashinfer.jit import env as flashinfer_jit_env

    _, state, x_cache, dt_cache, B_cache, *_ = layer.kv_cache
    props = torch.cuda.get_device_properties(state.device)
    return {
        "flashinfer_version": flashinfer.__version__,
        "flashinfer_workspace": [
            flashinfer_jit_env.FLASHINFER_WORKSPACE_DIR.parent.name,
            flashinfer_jit_env.FLASHINFER_WORKSPACE_DIR.name,
        ],
        "gpu_name": props.name,
        "gpu_capability": list(torch.cuda.get_device_capability(state.device)),
        "gpu_sm_count": props.multi_processor_count,
        "nheads": state.shape[1],
        "headdim": state.shape[2],
        "dstate": state.shape[3],
        "ngroups": B_cache.shape[1],
        "cache_slots": state.shape[0],
        "physical_ring_len": x_cache.shape[2],
        "state_dtype": str(state.dtype),
        "activation_dtype": str(x_cache.dtype),
        "dt_cache_dtype": str(dt_cache.dtype),
        "state_stride": list(state.stride()),
        "x_cache_stride": list(x_cache.stride()),
        "B_cache_stride": list(B_cache.stride()),
        "dt_cache_stride": list(dt_cache.stride()),
        "A_dtype": str(layer.A.dtype),
        "A_stride": list(layer.A.stride()),
        "D_dtype": str(layer.D.dtype),
        "D_stride": list(layer.D.stride()),
        "dt_bias_dtype": str(layer.dt_bias.dtype),
        "dt_bias_stride": list(layer.dt_bias.stride()),
        "stochastic_rounding": layer.mamba_config.enable_stochastic_rounding,
        "stochastic_rounding_philox_rounds": (
            layer.mamba_config.stochastic_rounding_philox_rounds
        ),
        "tp_size": layer.tp_size,
    }


class _ReplaySSMBenchmark:
    def __init__(self, layer: MambaMixer2, batch: int):
        from flashinfer.mamba import checkpointing_ssu

        _, self.state, self.x_cache, self.dt_cache, self.B_cache, *_ = layer.kv_cache
        if self.state.shape[0] <= batch:
            raise ValueError(
                f"ReplaySSM autotune batch {batch} needs {batch + 1} cache "
                f"slots, but only {self.state.shape[0]} are available"
            )

        self._kernel = checkpointing_ssu
        self.batch = batch
        self.logical_window = self.x_cache.shape[2] - 1
        if self.logical_window <= 0:
            raise ValueError("ReplaySSM history window must be positive")

        device = self.state.device
        activation_dtype = self.x_cache.dtype
        nheads = self.state.shape[1]
        headdim = self.state.shape[2]
        dstate = self.state.shape[3]
        ngroups = self.B_cache.shape[1]
        generator = torch.Generator(device=device)
        generator.manual_seed(0x5253534D + batch)

        self.x = torch.randn(
            batch,
            1,
            nheads,
            headdim,
            dtype=activation_dtype,
            device=device,
            generator=generator,
        )
        dt_base = torch.randn(
            batch,
            1,
            nheads,
            dtype=activation_dtype,
            device=device,
            generator=generator,
        )
        self.dt = dt_base.unsqueeze(-1).expand(batch, 1, nheads, headdim)
        self.B = torch.randn(
            batch,
            1,
            ngroups,
            dstate,
            dtype=activation_dtype,
            device=device,
            generator=generator,
        )
        self.C = torch.randn(
            self.B.shape,
            dtype=activation_dtype,
            device=device,
            generator=generator,
        )
        self.out = torch.empty_like(self.x)
        self.indices = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
        self.ring_start = torch.zeros(
            self.state.shape[0], dtype=torch.int32, device=device
        )
        self.prev_num_accepted = torch.zeros_like(self.ring_start)
        rows = torch.arange(batch, dtype=torch.int32, device=device)
        self.initial_prev_num_accepted = torch.zeros_like(self.prev_num_accepted)
        self.initial_prev_num_accepted[1 : batch + 1] = rows.remainder(
            self.logical_window
        ).add_(1)
        self.initial_ring_start = torch.zeros_like(self.ring_start)
        self.initial_ring_start[1 : batch + 1] = rows.remainder(self.logical_window + 1)

        self.A = (
            layer.A[:, None, ...][:, :, None]
            .expand(-1, headdim, dstate)
            .to(dtype=torch.float32)
        )
        self.D = layer.D[:, None, ...].expand(-1, headdim)
        self.dt_bias = layer.dt_bias[:, None, ...].expand(-1, headdim)
        self.rand_seed = (
            torch.zeros(1, dtype=torch.int64, device=device)
            if layer.mamba_config.enable_stochastic_rounding
            else None
        )
        self.philox_rounds = layer.mamba_config.stochastic_rounding_philox_rounds or 10

        k_old = ((self.logical_window + 7) // 8) * 8
        self.cb_scaled = torch.empty(
            batch, nheads, 32, 8, dtype=activation_dtype, device=device
        )
        self.cumAdt_vec = torch.empty(
            batch, nheads, 16, dtype=torch.float32, device=device
        )
        self.cb_old = torch.empty(
            batch,
            nheads,
            32,
            k_old // 2,
            dtype=activation_dtype,
            device=device,
        )

    def reset(self) -> None:
        for tensor in (self.state, self.x_cache, self.dt_cache, self.B_cache):
            tensor[: self.batch + 1].zero_()
        self.out.zero_()
        self.ring_start.copy_(self.initial_ring_start)
        self.prev_num_accepted.copy_(self.initial_prev_num_accepted)

    def call(self, tactic: FlashInferReplaySSMTactic) -> None:
        self._kernel(
            self.state,
            self.x_cache,
            self.B_cache,
            self.dt_cache,
            self.ring_start,
            self.prev_num_accepted,
            self.x,
            self.dt,
            self.A,
            self.B,
            self.C,
            self.out,
            D=self.D,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            state_batch_indices=self.indices,
            pad_slot_id=0,
            rand_seed=self.rand_seed,
            philox_rounds=self.philox_rounds,
            cb_scaled=self.cb_scaled,
            cumAdt_vec=self.cumAdt_vec,
            cb_old=self.cb_old,
            precompute_heads_per_cta=tactic.precompute_heads_per_cta,
            algorithm=tactic.algorithm,
        )
        update_replayssm_ring_trackers(
            self.ring_start,
            self.prev_num_accepted,
            self.indices,
            logical_window=self.logical_window,
            pad_slot_id=0,
        )

    def benchmark(self, tactic: FlashInferReplaySSMTactic) -> float:
        calls_per_graph = self.logical_window
        graph: torch.cuda.CUDAGraph | None = None
        samples = []
        try:
            with use_flashinfer_replayssm_tactic(tactic):
                self.reset()
                side_stream = torch.cuda.Stream()
                side_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side_stream):
                    for _ in range(3):
                        self.call(tactic)
                torch.cuda.current_stream().wait_stream(side_stream)
                torch.cuda.synchronize()

                self.reset()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(calls_per_graph):
                        self.call(tactic)
                torch.cuda.synchronize()

                for _ in range(3):
                    self.reset()
                    for _ in range(3):
                        graph.replay()
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) / (10 * calls_per_graph))
        finally:
            self.reset()
            del graph
        return statistics.median(samples)


def _aggregate_timings(timings: list[float]) -> list[float]:
    world = get_world_group()
    if world.world_size == 1:
        return timings
    values = torch.tensor(timings, dtype=torch.float64)
    torch.distributed.all_reduce(
        values,
        op=torch.distributed.ReduceOp.MAX,
        group=world.cpu_group,
    )
    return values.tolist()


def _all_ranks_support_tuning(supported: bool) -> bool:
    world = get_world_group()
    if world.world_size == 1:
        return supported
    flag = torch.tensor([int(supported)], dtype=torch.int32)
    torch.distributed.all_reduce(
        flag,
        op=torch.distributed.ReduceOp.MIN,
        group=world.cpu_group,
    )
    return bool(flag.item())


@torch.inference_mode()
def flashinfer_replayssm_autotune_warmup(worker: Worker) -> None:
    runner = worker.model_runner
    runner.flashinfer_replayssm_autotune_result = None
    if worker.vllm_config.kernel_config.enable_flashinfer_autotune is not True:
        return
    if worker.model_config.enforce_eager:
        logger.info_once("Skipping FlashInfer ReplaySSM autotune without CUDA graphs.")
        return
    if getattr(worker, "use_v2_model_runner", False):
        logger.info_once(
            "Skipping FlashInfer ReplaySSM autotune with the V2 model runner."
        )
        return
    if runner.parallel_config.use_ubatching:
        logger.info_once(
            "Skipping FlashInfer ReplaySSM autotune with uniform microbatching."
        )
        return

    layers = _find_replayssm_layers(runner)
    if not _all_ranks_support_tuning(bool(layers)):
        return
    assert layers
    local_layer_fingerprints = None
    try:
        local_layer_fingerprints = tuple(_layer_fingerprint(layer) for layer in layers)
    except Exception:
        logger.warning(
            "Could not fingerprint the FlashInfer ReplaySSM layers.",
            exc_info=True,
        )
    if not _all_ranks_support_tuning(local_layer_fingerprints is not None):
        logger.warning_once(
            "Skipping FlashInfer ReplaySSM autotune because layer "
            "fingerprinting failed on at least one rank."
        )
        return
    assert local_layer_fingerprints is not None
    layers_are_homogeneous = all(
        fingerprint == local_layer_fingerprints[0]
        for fingerprint in local_layer_fingerprints[1:]
    )
    if not _all_ranks_support_tuning(layers_are_homogeneous):
        logger.warning_once(
            "Skipping FlashInfer ReplaySSM autotune because ReplaySSM layer "
            "geometries differ within a rank."
        )
        return
    layer = layers[0]
    T = runner.uniform_decode_query_len
    if T != 1:
        logger.warning_once(
            "Skipping FlashInfer ReplaySSM autotune for T=%d; this vLLM "
            "integration currently supports only T=1.",
            T,
        )
        return

    local_batches = _uniform_capture_batches(runner)
    world = get_world_group()
    batches = world.broadcast_object(
        local_batches if world.rank_in_group == 0 else None, src=0
    )
    if not _all_ranks_support_tuning(local_batches == batches):
        logger.warning_once(
            "Skipping FlashInfer ReplaySSM autotune because CUDA-graph "
            "descriptors differ across ranks."
        )
        return
    if not batches:
        logger.info_once(
            "Skipping FlashInfer ReplaySSM autotune because there are no "
            "uniform FULL CUDA-graph capture descriptors."
        )
        return

    local_fingerprint = local_layer_fingerprints[0]
    fingerprint = world.broadcast_object(
        local_fingerprint if world.rank_in_group == 0 else None, src=0
    )
    if not _all_ranks_support_tuning(local_fingerprint == fingerprint):
        logger.warning_once(
            "Skipping FlashInfer ReplaySSM autotune because Mamba geometry "
            "differs across ranks."
        )
        return

    cache_path = None
    try:
        cache_path = resolve_flashinfer_autotune_file(runner).with_name(
            _CACHE_FILE_NAME
        )
    except Exception:
        logger.warning(
            "Could not resolve the FlashInfer ReplaySSM autotune cache path.",
            exc_info=True,
        )
    if not _all_ranks_support_tuning(cache_path is not None):
        logger.warning_once(
            "Skipping FlashInfer ReplaySSM autotune because the cache path "
            "could not be resolved on every rank."
        )
        return
    assert cache_path is not None
    cached_entries = _load_cache(cache_path) if world.rank_in_group == 0 else None
    cached_entries = world.broadcast_object(cached_entries, src=0)
    assert cached_entries is not None

    selected: dict[int, FlashInferReplaySSMTactic] = {}
    cache_changed = False
    tuning_started = time.perf_counter()
    for batch in batches:
        cache_key = _make_cache_key(fingerprint, batch, T)
        cached_name = cached_entries.get(cache_key)
        cached_tactic = (
            _tactic_from_name(cached_name) if cached_name is not None else None
        )
        if cached_tactic is not None:
            tactic = cached_tactic
            selected[batch] = tactic
            if world.rank_in_group == 0:
                logger.info(
                    "FlashInfer ReplaySSM autotune cache hit for batch %d: %s",
                    batch,
                    tactic.name,
                )
            continue

        benchmark = None
        try:
            benchmark = _ReplaySSMBenchmark(layer, batch)
        except Exception:
            logger.warning(
                "Could not construct the FlashInfer ReplaySSM benchmark for batch %d.",
                batch,
                exc_info=True,
            )
        if not _all_ranks_support_tuning(benchmark is not None):
            logger.warning_once(
                "Skipping FlashInfer ReplaySSM autotune for batch %d because "
                "the benchmark could not be constructed on every rank.",
                batch,
            )
            del benchmark
            gc.collect()
            torch.cuda.empty_cache()
            continue
        assert benchmark is not None

        candidate_count = len(FLASHINFER_REPLAYSSM_TUNING_CANDIDATES)
        shift = batch % candidate_count
        candidate_indices = tuple(range(candidate_count))
        candidate_order = candidate_indices[shift:] + candidate_indices[:shift]
        if (batch // candidate_count) % 2:
            candidate_order = tuple(reversed(candidate_order))
        local_timings = [float("inf")] * candidate_count
        for candidate_index in candidate_order:
            tactic = FLASHINFER_REPLAYSSM_TUNING_CANDIDATES[candidate_index]
            try:
                local_timings[candidate_index] = benchmark.benchmark(tactic)
            except Exception:
                logger.warning(
                    "FlashInfer ReplaySSM tactic %s failed for batch %d.",
                    tactic.name,
                    batch,
                    exc_info=True,
                )
        timings = _aggregate_timings(local_timings)
        _, _, _, _, B_cache, *_ = layer.kv_cache
        heads_per_group = layer.kv_cache[1].shape[1] // B_cache.shape[1]
        expanded_candidates = _expanded_precompute_tactics(
            timings, heads_per_group
        )
        local_expanded_timings = [float("inf")] * len(expanded_candidates)
        expanded_count = len(expanded_candidates)
        if expanded_count:
            expanded_shift = batch % expanded_count
            expanded_order = tuple(range(expanded_count))
            expanded_order = (
                expanded_order[expanded_shift:]
                + expanded_order[:expanded_shift]
            )
            for candidate_index in expanded_order:
                expanded_tactic = expanded_candidates[candidate_index]
                try:
                    local_expanded_timings[candidate_index] = benchmark.benchmark(
                        expanded_tactic
                    )
                except Exception:
                    logger.warning(
                        "FlashInfer ReplaySSM tactic %s failed for batch %d.",
                        expanded_tactic.name,
                        batch,
                        exc_info=True,
                    )
        expanded_timings = _aggregate_timings(local_expanded_timings)
        all_candidates = (
            FLASHINFER_REPLAYSSM_TUNING_CANDIDATES + expanded_candidates
        )
        all_timings = timings + expanded_timings
        tactic = _select_fastest(all_timings, all_candidates)
        if tactic is None:
            logger.warning_once(
                "Every FlashInfer ReplaySSM tactic failed for batch %d; "
                "leaving the default launch policy unchanged.",
                batch,
            )
            del benchmark
            gc.collect()
            torch.cuda.empty_cache()
            continue
        selected[batch] = tactic
        cached_entries[cache_key] = tactic.name
        cache_changed = True
        if world.rank_in_group == 0:
            timing_log = ", ".join(
                f"{candidate.name}={timing:.6f}ms"
                for candidate, timing in zip(
                    all_candidates,
                    all_timings,
                    strict=True,
                )
            )
            logger.info(
                "FlashInfer ReplaySSM autotune selected batch %d: %s (%s)",
                batch,
                tactic.name,
                timing_log,
            )
        del benchmark
        gc.collect()
        torch.cuda.empty_cache()

    if cache_changed and world.rank_in_group == 0:
        try:
            _save_cache(cache_path, cached_entries)
        except Exception:
            logger.warning(
                "Could not save the FlashInfer ReplaySSM autotune cache to %s.",
                cache_path,
                exc_info=True,
            )
    runner.flashinfer_replayssm_autotune_result = FlashInferReplaySSMAutotuneResult(
        T, selected
    )
    if world.rank_in_group == 0:
        logger.info(
            "FlashInfer ReplaySSM autotune prepared %d CUDA-graph batch "
            "tactics in %.2f seconds.",
            len(selected),
            time.perf_counter() - tuning_started,
        )


@contextmanager
def use_flashinfer_replayssm_tactic_for_capture(
    runner: GPUModelRunner,
    batch_descriptor: BatchDescriptor,
):
    result = runner.flashinfer_replayssm_autotune_result
    tactic = result.tactic_for(batch_descriptor) if result is not None else None
    scope = (
        use_flashinfer_replayssm_tactic(tactic) if tactic is not None else nullcontext()
    )
    with scope:
        yield
