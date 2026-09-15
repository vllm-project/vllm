# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 / W4 decode-efficiency (TPOT) heatmap over a batch-size x context-length grid.

Protocol (see README.md):

* one vLLM engine launch per precision row.  The base precision is selected through
  the decision-6 policy spec: ``VLLM_DUAL_PRECISION_POLICY=uniform_w4`` for the W4 row
  and the variable unset (vanilla behaviour) for the BF16 row.  Each row runs in a
  child process of this script so the environment is read by a fresh engine.
* synthetic KV (``synthetic_kv_connector.py``) skips prefill for the long contexts;
* a decode barrier (``sync_prefill.SynchronizedPrefillScheduler``) releases the batch
  only once every request has its first token, so every timed step decodes the full
  batch;
* per-step timing brackets synchronous ``llm_engine.step()`` calls with
  ``cuda.synchronize``; ``warmup_steps`` are discarded inside the same live request and
  ``measurement_steps`` are recorded, ``measurement_repetitions`` times; an untimed
  ``initial_precision_warmup_steps`` generation primes the precision path before the
  first cell;
* results are appended to ``cells.jsonl`` (fsync'd, resumable); ``heatmap.json``
  (matrix form), ``manifest_<precision>.json`` and ``progress.json`` are rewritten
  atomically after every cell.

The module-level helpers (axis parsing, cell ordering, matrix payload, resume status)
have no vLLM dependency so they can be unit-tested on CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
PRECISION_BF16 = "bf16"
PRECISION_INT4 = "int4"
PRECISION_NVFP4 = "nvfp4"
PRECISIONS = (PRECISION_BF16, PRECISION_INT4, PRECISION_NVFP4)
POLICY_ENV = "VLLM_DUAL_PRECISION_POLICY"
UNIFORM_W4_SPEC = "uniform_w4"
CONNECTOR_MODULE = "synthetic_kv_connector"
CONNECTOR_CLASS = "SyntheticKVConnector"


@dataclass(frozen=True)
class GridCell:
    batch_size: int
    seq_len: int
    required_kv_bytes: int
    required_blocks_by_group: tuple[int, ...]


class KVCapacityError(RuntimeError):
    """The current grid cell cannot fit in the initialized KV cache."""


# --------------------------------------------------------------------------- pure
# helpers


def parse_axis(value: str) -> list[int]:
    """Parse either comma-separated integers or an inclusive start:stop[:step] range."""
    value = value.strip()
    if not value:
        raise ValueError("Grid axis cannot be empty")
    if ":" in value:
        if "," in value:
            raise ValueError("Cannot mix comma and range syntax")
        fields = [int(field) for field in value.split(":")]
        if len(fields) not in (2, 3):
            raise ValueError("Range syntax is start:stop[:step]")
        start, stop = fields[:2]
        step = fields[2] if len(fields) == 3 else 1
        if step <= 0 or stop < start:
            raise ValueError("Grid ranges require stop >= start and step > 0")
        values = list(range(start, stop + 1, step))
    else:
        values = [int(field.strip()) for field in value.split(",")]
    if not values or any(item <= 0 for item in values):
        raise ValueError("Grid values must be positive integers")
    return sorted(set(values))


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return rows


def completed_keys(rows: list[dict[str, Any]]) -> set[tuple[int, int, str]]:
    return {
        (int(row["batch_size"]), int(row["seq_len"]), str(row["precision"]))
        for row in rows
        if row.get("status") == "ok"
    }


def preserved_run_status(
    progress: dict[str, Any],
    done: set[tuple[int, int, str]],
    expected: set[tuple[int, int, str]],
    retry_interrupted: bool,
) -> int | None:
    """Return an exit status when a preserved run must not start an engine."""
    status = progress.get("status")
    if status == "capacity_exhausted":
        return 3
    if status == "complete" and expected.issubset(done):
        return 0
    current = progress.get("current_cell")
    if status not in ("running_cell", "interrupted") or not current:
        return None
    key = (
        int(current["batch_size"]),
        int(current["seq_len"]),
        str(current["precision"]),
    )
    if key not in done and not retry_interrupted:
        return 2
    return None


def build_cells(
    batch_sizes: list[int],
    seq_lens: list[int],
    decode_steps: int,
    group_specs: list[tuple[int, int]],
) -> list[GridCell]:
    """Grid cells ordered by estimated resident KV bytes (cheapest first)."""
    cells = []
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            resident_tokens = seq_len + decode_steps + 1
            blocks = tuple(
                batch_size * math.ceil(resident_tokens / block_size)
                for block_size, _ in group_specs
            )
            required_bytes = sum(
                count * page_size
                for count, (_, page_size) in zip(blocks, group_specs, strict=True)
            )
            cells.append(GridCell(batch_size, seq_len, required_bytes, blocks))
    return sorted(
        cells, key=lambda cell: (cell.required_kv_bytes, cell.seq_len, cell.batch_size)
    )


def matrix_payload(
    rows: list[dict[str, Any]],
    batch_sizes: list[int],
    seq_lens: list[int],
    cells: list[GridCell] | None = None,
) -> dict[str, Any]:
    lookup = {
        (int(row["batch_size"]), int(row["seq_len"]), str(row["precision"])): row
        for row in rows
        if row.get("status") == "ok"
    }

    def matrix(precision: str) -> list[list[float | None]]:
        return [
            [
                (
                    float(lookup[(batch, seq, precision)]["median_tpot_ms"])
                    if (batch, seq, precision) in lookup
                    else None
                )
                for seq in seq_lens
            ]
            for batch in batch_sizes
        ]

    bf16 = matrix(PRECISION_BF16)
    int4 = matrix(PRECISION_INT4)
    nvfp4 = matrix(PRECISION_NVFP4)
    required_lookup = {
        (cell.batch_size, cell.seq_len): cell.required_kv_bytes for cell in cells or []
    }
    for row in rows:
        required_lookup.setdefault(
            (int(row["batch_size"]), int(row["seq_len"])),
            int(row["required_kv_bytes"]),
        )
    required_kv_bytes = [
        [required_lookup.get((batch, seq)) for seq in seq_lens] for batch in batch_sizes
    ]
    def speedup_over(quantized: list[list[float | None]]) -> list[list[float | None]]:
        return [
            [
                left / right if left is not None and right not in (None, 0.0) else None
                for left, right in zip(bf16_row, quantized_row, strict=True)
            ]
            for bf16_row, quantized_row in zip(bf16, quantized, strict=True)
        ]

    return {
        "batch_sizes": batch_sizes,
        "seq_lens": seq_lens,
        "required_kv_bytes": required_kv_bytes,
        "bf16_tpot_ms": bf16,
        "int4_tpot_ms": int4,
        "nvfp4_tpot_ms": nvfp4,
        "speedup_bf16_over_int4": speedup_over(int4),
        "speedup_bf16_over_nvfp4": speedup_over(nvfp4),
    }


def median_speedup(
    payload: dict[str, Any], key: str = "speedup_bf16_over_int4"
) -> float | None:
    values = [v for row in payload.get(key, []) for v in row if v is not None]
    return statistics.median(values) if values else None


def is_capacity_failure(error: BaseException) -> bool:
    if isinstance(error, KVCapacityError):
        return True
    message = str(error).lower()
    phrases = (
        "cache blocks",
        "no available memory",
        "not enough kv cache",
        "kv cache capacity",
    )
    return any(phrase in message for phrase in phrases)


def precision_environment(
    precision: str,
    int4_model: str | None,
    nvfp4_model: str | None = None,
    *,
    standalone: bool = False,
) -> dict[str, str | None]:
    """Environment overrides selecting the base precision of one engine launch.

    This is the forced-precision seam.  There are two ways to reach a quantized row:

    * ``standalone=False`` (default, the archived protocol): every engine loads the
      BF16 ``--model`` and the INT4 row runs the decision-6 spec ``uniform_w4`` against
      the INT4 shadow, i.e. inside the dual-precision runtime the scheduler itself uses.
    * ``standalone=True``: each row is vanilla vLLM launched directly on that
      precision's own checkpoint (see ``base_model_for_precision``), so every row is
      symmetric and no dual-precision runtime takes part.

    NVFP4 is always standalone: the dual-precision loader carries no FP4 shadow format,
    so that row is vanilla vLLM on the NVFP4 checkpoint whatever ``standalone`` says.

    ``None`` means "remove from the environment".
    """
    if precision == PRECISION_BF16:
        return {POLICY_ENV: None}
    if precision == PRECISION_NVFP4:
        if not nvfp4_model:
            raise ValueError("--nvfp4-model is required for the nvfp4 precision row")
        return {POLICY_ENV: None}
    if precision == PRECISION_INT4:
        if not int4_model:
            raise ValueError("--int4-model is required for the int4 precision row")
        if standalone:
            return {POLICY_ENV: None}
        return {
            POLICY_ENV: UNIFORM_W4_SPEC,
            "VLLM_DUAL_PRECISION_ROLLOUT": "1",
            "VLLM_DUAL_PRECISION_INT4_MODEL": str(int4_model),
        }
    raise ValueError(f"Unsupported precision: {precision}")


def base_model_for_precision(precision: str, args: argparse.Namespace) -> str:
    """Checkpoint this row's engine loads as its base model.

    Only the standalone rows swap the base checkpoint.  Under the dual-precision
    protocol every row loads ``--model`` and the INT4 weights arrive as a shadow.
    """
    if precision == PRECISION_NVFP4:
        return str(args.nvfp4_model)
    if precision == PRECISION_INT4 and args.standalone_base_precision:
        return str(args.int4_model)
    return str(args.model)


def expected_keys(
    batch_sizes: list[int], seq_lens: list[int], precisions: tuple[str, ...]
) -> set[tuple[int, int, str]]:
    return {(b, s, p) for b in batch_sizes for s in seq_lens for p in precisions}


# --------------------------------------------------------------------------- engine
# side


def make_prompts(tokenizer: Any, batch_size: int, seq_len: int) -> list[Any]:
    from vllm.inputs import TokensPrompt

    vocab_size = int(tokenizer.vocab_size)
    safe_start = 1000 if vocab_size > 2000 else 1
    usable = max(1, vocab_size - safe_start - 1)
    bos_token_id = tokenizer.bos_token_id
    prompts = []
    for request_index in range(batch_size):
        tokens = [
            safe_start + ((position * 131 + request_index * 977) % usable)
            for position in range(seq_len)
        ]
        if bos_token_id is not None:
            tokens[0] = int(bos_token_id)
        prompts.append(TokensPrompt(prompt_token_ids=tokens))
    return prompts


def engine_scheduler(llm: Any):
    core_client = llm.llm_engine.engine_core
    engine_core = getattr(core_client, "engine_core", None)
    if engine_core is None:
        raise RuntimeError("The heatmap requires VLLM_ENABLE_V1_MULTIPROCESSING=0")
    return engine_core, engine_core.scheduler


def set_heatmap_mode(llm: Any, enabled: bool) -> Any:
    """Toggle both synthetic external KV hits and the decode barrier."""
    _, scheduler = engine_scheduler(llm)
    if scheduler.has_unfinished_requests():
        raise RuntimeError("Cannot toggle heatmap mode with unfinished requests")
    toggle = getattr(scheduler.connector, "set_scheduler_enabled", None)
    if toggle is None:
        raise TypeError(
            f"KV connector is not runtime-toggleable: {type(scheduler.connector)!r}"
        )
    toggle(enabled)
    scheduler.set_barrier_enabled(enabled)
    return scheduler


def run_stepwise_generation(
    llm: Any,
    lora_request: Any,
    prompts: list[Any],
    scheduler: Any,
    warmup_steps: int,
    measurement_steps: int,
) -> tuple[list[Any], list[float], float, float]:
    import torch

    from vllm import SamplingParams

    total_decode_steps = warmup_steps + measurement_steps
    params = SamplingParams(
        temperature=0.0,
        max_tokens=total_decode_steps + 1,
        min_tokens=total_decode_steps + 1,
        ignore_eos=True,
    )
    llm.enqueue(prompts, params, lora_request=lora_request, use_tqdm=False)
    releases_before = scheduler.barrier_release_count
    decode_iteration = 0
    iteration_ms: list[float] = []
    outputs = []
    generation_started = time.perf_counter()
    while llm.llm_engine.has_unfinished_requests():
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        step_outputs = llm.llm_engine.step()
        torch.cuda.synchronize()
        step_ms = (time.perf_counter() - step_started) * 1000.0
        outputs.extend(output for output in step_outputs if output.finished)

        if scheduler.barrier_release_count > releases_before:
            if scheduler.barrier_release_count != releases_before + 1:
                raise RuntimeError("Generation crossed more than one barrier")
            if decode_iteration >= warmup_steps:
                iteration_ms.append(step_ms)
            decode_iteration += 1

    generation_wall_seconds = time.perf_counter() - generation_started
    if scheduler.barrier_release_count != releases_before + 1:
        raise RuntimeError("Generation did not cross one synchronized barrier")
    if decode_iteration != total_decode_steps:
        raise RuntimeError(
            f"Expected {total_decode_steps} decode iterations after the barrier, "
            f"observed {decode_iteration}"
        )
    if len(iteration_ms) != measurement_steps:
        raise RuntimeError(
            f"Expected {measurement_steps} measured iterations, "
            f"observed {len(iteration_ms)}"
        )
    return outputs, iteration_ms, generation_wall_seconds, sum(iteration_ms) / 1000.0


def measure_cell(
    llm: Any,
    lora_request: Any,
    tokenizer: Any,
    cell: GridCell,
    precision: str,
    warmup_steps: int,
    measurement_steps: int,
    measurement_repetitions: int,
) -> dict[str, Any]:
    _, scheduler = engine_scheduler(llm)
    prompts = make_prompts(tokenizer, cell.batch_size, cell.seq_len)

    expected = warmup_steps + measurement_steps + 1
    tpot_ms: list[float] = []
    first_token_spreads_ms: list[float] = []
    repetition_median_tpot_ms: list[float] = []
    measurement_wall_seconds = 0.0
    generation_wall_seconds = 0.0
    for _ in range(measurement_repetitions):
        outputs, iteration_ms, repetition_wall_seconds, measured_seconds = (
            run_stepwise_generation(
                llm, lora_request, prompts, scheduler, warmup_steps, measurement_steps
            )
        )
        generation_wall_seconds += repetition_wall_seconds
        measurement_wall_seconds += measured_seconds
        if len(outputs) != cell.batch_size:
            raise RuntimeError(
                f"Expected {cell.batch_size} outputs, received {len(outputs)}"
            )
        first_token_times = []
        for output in outputs:
            if len(output.outputs[0].token_ids) != expected:
                raise RuntimeError("Measurement request ended before the target length")
            metrics = output.metrics
            if metrics is None:
                raise RuntimeError("vLLM request metrics are unavailable")
            first_token_times.append(float(metrics.first_token_ts))
        tpot_ms.extend(iteration_ms)
        repetition_median_tpot_ms.append(statistics.median(iteration_ms))
        first_token_spreads_ms.append(
            (max(first_token_times) - min(first_token_times)) * 1000.0
        )

    return {
        "status": "ok",
        "batch_size": cell.batch_size,
        "seq_len": cell.seq_len,
        "precision": precision,
        "warmup_steps": warmup_steps,
        "measurement_steps": measurement_steps,
        "measurement_repetitions": measurement_repetitions,
        "required_kv_bytes": cell.required_kv_bytes,
        "required_blocks_by_group": list(cell.required_blocks_by_group),
        "external_kv_tokens_per_request": cell.seq_len - 1,
        "synchronized_prefill_barrier": True,
        "barrier_release_count": measurement_repetitions,
        "measured_context_start": cell.seq_len + warmup_steps + 1,
        "measured_context_end": cell.seq_len + warmup_steps + measurement_steps,
        "timing_method": "synchronized_engine_step_wall_time",
        "request_tpot_ms": tpot_ms,
        "repetition_median_tpot_ms": repetition_median_tpot_ms,
        "median_tpot_ms": statistics.median(tpot_ms),
        "p10_tpot_ms": percentile(tpot_ms, 0.10),
        "p90_tpot_ms": percentile(tpot_ms, 0.90),
        "first_token_spread_ms": statistics.median(first_token_spreads_ms),
        "batch_decode_span_ms_per_step": statistics.median(tpot_ms),
        "generation_wall_seconds": generation_wall_seconds,
        "measurement_wall_seconds": measurement_wall_seconds,
        "timestamp": time.time(),
    }


def write_heatmaps(
    output_dir: Path,
    rows: list[dict[str, Any]],
    batch_sizes: list[int],
    seq_lens: list[int],
    cells: list[GridCell] | None = None,
    *,
    render_plots: bool = False,
) -> dict[str, Any]:
    payload = matrix_payload(rows, batch_sizes, seq_lens, cells)
    atomic_write_json(output_dir / "heatmap.json", payload)
    if render_plots:
        from heatmap_plots import write_pngs

        write_pngs(output_dir, payload)
    return payload


def git_sha() -> str | None:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=THIS_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet"], cwd=THIS_DIR, stderr=subprocess.DEVNULL
        ).returncode
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def run_precision_row(args: argparse.Namespace, precision: str) -> int:
    """Measure every cell of one precision row inside this process (one engine
    launch)."""
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    import torch
    from sync_prefill import SynchronizedPrefillScheduler

    from vllm import LLM

    batch_sizes = parse_axis(args.batch_sizes)
    seq_lens = parse_axis(args.seq_lens)
    output_dir: Path = args.output_dir
    cells_path = output_dir / "cells.jsonl"
    progress_path = output_dir / "progress.json"
    manifest_path = output_dir / f"manifest_{precision}.json"
    existing_rows = load_rows(cells_path)
    done = completed_keys(existing_rows)

    requested_max_model_len = (
        max(seq_lens) + args.warmup_steps + args.measurement_steps + 2
    )
    max_model_len = args.max_model_len or requested_max_model_len
    if max_model_len < requested_max_model_len:
        raise ValueError(
            f"max_model_len={max_model_len} is smaller than required "
            f"{requested_max_model_len}"
        )
    manifest: dict[str, Any] = {
        "status": "initializing",
        "precision": precision,
        "git_sha": git_sha(),
        "model": args.model,
        "base_model": base_model_for_precision(precision, args),
        "int4_model": args.int4_model,
        "nvfp4_model": args.nvfp4_model,
        "standalone_base_precision": bool(args.standalone_base_precision),
        "adapter": str(args.adapter) if args.adapter else None,
        "policy_env": dict(
            precision_environment(
                precision,
                args.int4_model,
                args.nvfp4_model,
                standalone=args.standalone_base_precision,
            )
        ),
        "batch_sizes": batch_sizes,
        "seq_lens": seq_lens,
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
        "measurement_repetitions": args.measurement_repetitions,
        "initial_precision_warmup_steps": args.initial_precision_warmup_steps,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": max_model_len,
        "synthetic_kv_float_value": args.synthetic_float_value,
        "synthetic_kv_integer_value": args.synthetic_integer_value,
        "synchronized_prefill_barrier": True,
        "async_scheduling": False,
        "enforce_eager": args.enforce_eager,
        "full_cudagraph_without_torch_compile": (
            args.full_cudagraph_without_torch_compile
        ),
        "started_at": time.time(),
    }
    atomic_write_json(manifest_path, manifest)

    llm_kwargs: dict[str, Any] = {
        "model": base_model_for_precision(precision, args),
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": max_model_len,
        "max_num_seqs": args.max_num_seqs or max(batch_sizes),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "scheduler_cls": SynchronizedPrefillScheduler,
        "async_scheduling": False,
        "enforce_eager": args.enforce_eager,
        "kv_transfer_config": {
            "kv_connector": CONNECTOR_CLASS,
            "kv_connector_module_path": CONNECTOR_MODULE,
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "float_value": args.synthetic_float_value,
                "integer_value": args.synthetic_integer_value,
            },
        },
        "seed": 0,
    }
    if args.language_model_only:
        llm_kwargs["language_model_only"] = True
    if args.adapter:
        llm_kwargs.update(enable_lora=True, max_lora_rank=args.max_lora_rank)
    if args.full_cudagraph_without_torch_compile:
        if args.enforce_eager:
            raise ValueError(
                "--full-cudagraph-without-torch-compile and --enforce-eager "
                "are mutually exclusive"
            )
        llm_kwargs["compilation_config"] = {"mode": 0, "cudagraph_mode": "FULL"}

    engine_started = time.perf_counter()
    llm = LLM(**llm_kwargs)
    manifest["engine_initialization_seconds"] = time.perf_counter() - engine_started
    manifest["gpus"] = [
        {
            "visible_index": index,
            "name": torch.cuda.get_device_name(index),
            "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
        }
        for index in range(torch.cuda.device_count())
    ]
    manifest["status"] = "running"
    atomic_write_json(manifest_path, manifest)

    lora_request = None
    if args.adapter:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("zero_adapter", 1, str(args.adapter))
    tokenizer = llm.get_tokenizer()
    core, scheduler = engine_scheduler(llm)
    if not isinstance(scheduler, SynchronizedPrefillScheduler):
        raise TypeError(f"Unexpected scheduler type: {type(scheduler)!r}")
    set_heatmap_mode(llm, True)
    kv_config = core.scheduler.kv_cache_manager.kv_cache_config
    group_specs = [
        (group.kv_cache_spec.block_size, group.kv_cache_spec.page_size_bytes)
        for group in kv_config.kv_cache_groups
    ]
    cells = build_cells(
        batch_sizes, seq_lens, args.warmup_steps + args.measurement_steps, group_specs
    )
    scan_order = [
        {
            **asdict(cell),
            "required_blocks_by_group": list(cell.required_blocks_by_group),
        }
        for cell in cells
    ]
    manifest["kv_cache_num_blocks"] = kv_config.num_blocks
    manifest["kv_cache_group_specs"] = [
        {"block_size": block_size, "page_size_bytes": page_size}
        for block_size, page_size in group_specs
    ]
    atomic_write_json(manifest_path, manifest)

    # Prime the precision path once before recording the first grid cell: the first
    # request triggers lazy setup that is not part of steady-state decode latency.
    if args.initial_precision_warmup_steps:
        priming_cell = cells[0]
        run_stepwise_generation(
            llm,
            lora_request,
            make_prompts(tokenizer, priming_cell.batch_size, priming_cell.seq_len),
            scheduler,
            warmup_steps=0,
            measurement_steps=args.initial_precision_warmup_steps,
        )

    benchmark_started = time.perf_counter()
    total_cells = len(cells) * len(args.precision_list)
    for cell_index, cell in enumerate(cells):
        key = (cell.batch_size, cell.seq_len, precision)
        if key in done:
            continue
        atomic_write_json(
            progress_path,
            {
                "status": "running_cell",
                "current_cell": {
                    **asdict(cell),
                    "required_blocks_by_group": list(cell.required_blocks_by_group),
                    "precision": precision,
                },
                "cell_index": cell_index,
                "completed_cells": len(done),
                "total_precision_cells": total_cells,
                "scan_order": scan_order,
                "updated_at": time.time(),
            },
        )
        try:
            if any(
                required > kv_config.num_blocks
                for required in cell.required_blocks_by_group
            ):
                raise KVCapacityError(
                    "Cell requires blocks per cache group "
                    f"{cell.required_blocks_by_group}, but only "
                    f"{kv_config.num_blocks} blocks are available per group"
                )
            row = measure_cell(
                llm,
                lora_request,
                tokenizer,
                cell,
                precision,
                args.warmup_steps,
                args.measurement_steps,
                args.measurement_repetitions,
            )
        except Exception as error:
            if not is_capacity_failure(error):
                raise
            row = {
                "status": "capacity_failure",
                "batch_size": cell.batch_size,
                "seq_len": cell.seq_len,
                "precision": precision,
                "required_kv_bytes": cell.required_kv_bytes,
                "required_blocks_by_group": list(cell.required_blocks_by_group),
                "error": f"{type(error).__name__}: {error}",
                "timestamp": time.time(),
            }
        append_jsonl(cells_path, row)
        existing_rows.append(row)
        if row["status"] == "ok":
            done.add(key)
        write_heatmaps(output_dir, existing_rows, batch_sizes, seq_lens, cells)
        atomic_write_json(
            progress_path,
            {
                "status": "running",
                "current_cell": None,
                "completed_cells": len(done),
                "total_precision_cells": total_cells,
                "scan_order": scan_order,
                "updated_at": time.time(),
            },
        )
        manifest["benchmark_seconds"] = time.perf_counter() - benchmark_started
        atomic_write_json(manifest_path, manifest)

    manifest["benchmark_seconds"] = time.perf_counter() - benchmark_started
    manifest["finished_at"] = time.time()
    manifest["status"] = "complete"
    atomic_write_json(manifest_path, manifest)
    return 0


# --------------------------------------------------------------------------- driver


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="BF16 checkpoint path or HF id")
    parser.add_argument(
        "--int4-model", help="INT4 shadow checkpoint (required for the int4 row)"
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        help="zero-delta LoRA adapter (tools/rollout_lora/make_zero_lora.py)",
    )
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument(
        "--batch-sizes", required=True, help="e.g. 1,2,4,8,16,32 or 1:32:1"
    )
    parser.add_argument("--seq-lens", required=True, help="e.g. 1024:8192:1024")
    parser.add_argument(
        "--precisions", default="bf16,int4", help="subset of bf16,int4,nvfp4"
    )
    parser.add_argument(
        "--nvfp4-model", help="NVFP4 checkpoint; required for the nvfp4 precision row"
    )
    parser.add_argument(
        "--standalone-base-precision",
        action="store_true",
        help=(
            "launch the int4 row as vanilla vLLM directly on --int4-model instead of "
            "through the dual-precision runtime, so every row is symmetric "
            "(nvfp4 is always standalone)"
        ),
    )
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measurement-steps", type=int, default=9)
    parser.add_argument("--measurement-repetitions", type=int, default=1)
    parser.add_argument("--initial-precision-warmup-steps", type=int, default=9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--full-cudagraph-without-torch-compile",
        action="store_true",
        help=(
            "compilation mode=0 with FULL cudagraphs (architectures whose LoRA "
            "metadata cannot be traced by torch.compile)"
        ),
    )
    parser.add_argument(
        "--no-language-model-only", dest="language_model_only", action="store_false"
    )
    parser.add_argument("--synthetic-float-value", type=float, default=0.015)
    parser.add_argument("--synthetic-integer-value", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retry-interrupted-cell", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--worker-precision", choices=PRECISIONS, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if (
        args.warmup_steps < 0
        or args.measurement_steps <= 0
        or args.measurement_repetitions <= 0
        or args.initial_precision_warmup_steps < 0
    ):
        raise ValueError(
            "warmup steps must be >= 0; measurement steps and repetitions > 0"
        )
    if args.synthetic_float_value == 0.0 or args.synthetic_integer_value == 0:
        raise ValueError("Synthetic KV fill values must be non-zero")
    precisions = tuple(item.strip() for item in args.precisions.split(","))
    if not precisions or len(set(precisions)) != len(precisions):
        raise ValueError("precisions must be a non-empty list without duplicates")
    if any(precision not in PRECISIONS for precision in precisions):
        raise ValueError(f"precisions must be a subset of {PRECISIONS}")
    args.precision_list = precisions
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_precision:
        return run_precision_row(args, args.worker_precision)

    batch_sizes = parse_axis(args.batch_sizes)
    seq_lens = parse_axis(args.seq_lens)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells_path = args.output_dir / "cells.jsonl"
    progress_path = args.output_dir / "progress.json"
    expected = expected_keys(batch_sizes, seq_lens, args.precision_list)
    done = completed_keys(load_rows(cells_path))
    if progress_path.exists():
        previous = json.loads(progress_path.read_text(encoding="utf-8"))
        status = preserved_run_status(
            previous, done, expected, args.retry_interrupted_cell
        )
        if status == 2:
            current = previous["current_cell"]
            previous["status"] = "interrupted"
            atomic_write_json(progress_path, previous)
            print(
                "Previous run stopped inside cell "
                f"bs={current['batch_size']}, seq={current['seq_len']}, "
                f"precision={current['precision']}; preserved results were left "
                "untouched. Pass --retry-interrupted-cell to retry explicitly.",
                file=sys.stderr,
            )
            return 2
        if status is not None:
            label = "complete" if status == 0 else "capacity exhausted"
            print(f"Preserved benchmark is {label}; no engine was started.")
            return status

    argv_base = [sys.executable, str(Path(__file__).resolve())] + (
        list(argv) if argv is not None else sys.argv[1:]
    )
    for precision in args.precision_list:
        if expected_keys(batch_sizes, seq_lens, (precision,)).issubset(done):
            continue
        env = dict(os.environ)
        row_environment = precision_environment(
            precision,
            args.int4_model,
            args.nvfp4_model,
            standalone=args.standalone_base_precision,
        )
        for key, value in row_environment.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        print(f"[tpot_heatmap] launching engine for precision={precision}", flush=True)
        result = subprocess.run(argv_base + ["--worker-precision", precision], env=env)
        if result.returncode != 0:
            atomic_write_json(
                progress_path,
                {**json.loads(progress_path.read_text()), "status": "interrupted"}
                if progress_path.exists()
                else {"status": "interrupted", "current_cell": None},
            )
            return result.returncode
        done = completed_keys(load_rows(cells_path))

    rows = load_rows(cells_path)
    payload = write_heatmaps(
        args.output_dir, rows, batch_sizes, seq_lens, render_plots=not args.no_plots
    )
    capacity_exhausted = not expected.issubset(done)
    atomic_write_json(
        progress_path,
        {
            "status": "capacity_exhausted" if capacity_exhausted else "complete",
            "current_cell": None,
            "completed_cells": len(done),
            "total_precision_cells": len(expected),
            "updated_at": time.time(),
        },
    )
    atomic_write_json(
        args.output_dir / "manifest.json",
        {
            "status": "capacity_exhausted" if capacity_exhausted else "complete",
            "precisions": list(args.precision_list),
            "model": args.model,
            "int4_model": args.int4_model,
            "nvfp4_model": args.nvfp4_model,
            "standalone_base_precision": bool(args.standalone_base_precision),
            "base_models": {
                precision: base_model_for_precision(precision, args)
                for precision in args.precision_list
            },
            "batch_sizes": batch_sizes,
            "seq_lens": seq_lens,
            "median_speedup_bf16_over_int4": median_speedup(payload),
            "median_speedup_bf16_over_nvfp4": median_speedup(
                payload, "speedup_bf16_over_nvfp4"
            ),
            "git_sha": git_sha(),
            "finished_at": time.time(),
        },
    )
    return 3 if capacity_exhausted else 0


if __name__ == "__main__":
    raise SystemExit(main())
