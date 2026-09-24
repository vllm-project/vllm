# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the windowed hidden-state capture feature with a real DSpark model.

This standalone test depends only on vLLM. It loads a target model and a
DSpark draft model on CUDA, sends request-local capture plans directly to
AsyncLLM, and writes a JSON report.

Example:
    VLLM_USE_V2_MODEL_RUNNER=1 \
      .venv/bin/python \
      tests/special_standalone/vllm_windowed_hidden_capture_gpu.py \
      --target-model /models/Qwen3-8B \
      --draft-model /models/Qwen3-8B-DSpark \
      --tensor-parallel-size 2 --mode both --enforce-perf
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DSPARK_ARCHITECTURES = frozenset(
    {
        "DSparkDraftModel",
        "DSparkV41DraftModel",
        "Gemma4DSparkModel",
        "K3DSparkModel",
        "Qwen3DSparkModel",
        "Qwen3OmniDSparkModel",
    }
)


@dataclass
class PhaseStats:
    tokens: int = 0
    seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float | None:
        return self.tokens / self.seconds if self.seconds > 0 else None


@dataclass
class RunResult:
    request_id: str
    token_ids: list[int]
    capture: Any | None
    skip_reason: str | None
    phases: dict[str, PhaseStats]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--prompt", default="Explain speculative decoding in detail.")
    parser.add_argument("--window-start", type=int, default=16)
    parser.add_argument("--window-end", type=int, default=48)
    parser.add_argument("--min-rows", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--speculative-tokens", type=int, default=7)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--draft-tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--mode", choices=("cudagraph", "eager", "both"), default="cudagraph"
    )
    parser.add_argument("--perf-repeats", type=int, default=3)
    parser.add_argument("--max-regression", type=float, default=0.03)
    parser.add_argument("--enforce-perf", action="store_true")
    parser.add_argument(
        "--hidden-layout",
        choices=("dflash_aux", "dflash_aux_plus_last"),
        default="dflash_aux_plus_last",
    )
    parser.add_argument(
        "--aux-layer-ids",
        help="Comma-separated target layer IDs; auto-detected when omitted.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--report", type=Path, default=Path("vllm_hidden_capture_gpu_report.json")
    )
    args = parser.parse_args()
    if not 0 <= args.window_start < args.window_end <= args.max_tokens:
        parser.error("require 0 <= window-start < window-end <= max-tokens")
    width = args.window_end - args.window_start
    if args.window_end + width > args.max_tokens:
        parser.error("max-tokens must fit two non-overlapping capture windows")
    if not 1 <= args.min_rows <= width:
        parser.error("min-rows must fit inside the capture window")
    if args.speculative_tokens < 1 or args.perf_repeats < 1:
        parser.error("speculative-tokens and perf-repeats must be positive")
    if not 0.0 <= args.max_regression < 1.0:
        parser.error("max-regression must be in [0, 1)")
    return args


def configure_environment() -> None:
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")


def validate_runtime_environment(*, run=subprocess.run) -> str:
    probe = (
        "import vllm; "
        "print('vllm=' + str(vllm.__file__), flush=True); "
        "import vllm._C_stable_libtorch; "
        "print('native_extension=ok', flush=True)"
    )
    completed = run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    if completed.returncode:
        details = "\n".join(
            part.strip()
            for part in (completed.stdout, completed.stderr)
            if part.strip()
        )
        raise RuntimeError(
            "The vLLM registry subprocess cannot import the checkout's native "
            "extension. Install this checkout into one virtual environment and "
            "run the script with that environment's Python:\n"
            "  uv venv --python 3.12\n"
            "  VLLM_USE_PRECOMPILED=1 uv pip install --editable . "
            "--torch-backend=auto\n"
            "  .venv/bin/python "
            "tests/special_standalone/vllm_windowed_hidden_capture_gpu.py ...\n"
            "Do not combine source files from the checkout with a different "
            f"site-packages vLLM installation. Probe output:\n{details}"
        )
    return completed.stdout.strip()


def validate_capture_result_contract(capture_module: Any | None = None) -> None:
    if capture_module is None:
        from vllm.v1 import hidden_state_capture as capture_module

    result_type = capture_module.HiddenStateCaptureResult
    fields = getattr(result_type, "__dataclass_fields__", {})
    missing = {"layer_ids", "includes_final_layer"} - fields.keys()
    if missing:
        raise RuntimeError(
            "The GPU test script and imported vLLM capture result are out of sync: "
            f"{sorted(missing)} missing from {capture_module.__file__}. "
            "Update the vLLM checkout with the result metadata changes and run "
            "this script using that checkout's virtual environment."
        )


def validate_dspark_architectures(architectures: list[str]) -> None:
    normalized = {str(name) for name in architectures}
    if normalized.isdisjoint(_DSPARK_ARCHITECTURES):
        raise ValueError(
            "--draft-model must be a DSpark checkpoint. Expected an architecture "
            "such as 'Qwen3DSparkModel'; got "
            f"{sorted(normalized)!r}. A checkpoint declaring 'DFlashDraftModel' "
            "must be tested with method='dflash', not method='dspark'."
        )


def build_engine_kwargs(
    args: argparse.Namespace,
    *,
    eager: bool,
    speculative_config: dict[str, Any],
) -> dict[str, Any]:
    engine_kwargs = {
        "model": args.target_model,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "async_scheduling": False,
        "enforce_eager": eager,
        "trust_remote_code": args.trust_remote_code,
        "seed": args.seed,
        "speculative_config": speculative_config,
        "disable_log_stats": False,
    }
    if args.max_model_len is not None:
        engine_kwargs["max_model_len"] = args.max_model_len
    return engine_kwargs


def get_value(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if hasattr(value, "get"):
        return value.get(name, default)
    return getattr(value, name, default)


def as_layer_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    return tuple(int(item) for item in value)


def capture_layout(cli_layout: str) -> str:
    return {
        "dflash_aux": "dflash_aux",
        "dflash_aux_plus_last": "aux_final",
    }[cli_layout]


def resolve_aux_layer_ids(
    engine: Any,
    configured: str | None,
    *,
    get_aux_layers: Callable[[Any], tuple[int, ...] | None] | None = None,
) -> tuple[int, ...]:
    if configured:
        return as_layer_ids(configured)
    speculative = get_value(engine.vllm_config, "speculative_config")
    if get_aux_layers is None:
        from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
            get_eagle3_aux_layers_from_config,
        )

        get_aux_layers = get_eagle3_aux_layers_from_config
    layer_ids = get_aux_layers(speculative)
    if layer_ids:
        return tuple(sorted(set(layer_ids)))
    raise AssertionError(
        "DSpark auxiliary layer IDs are unavailable; pass --aux-layer-ids"
    )


def phase_name(position: int, start: int, end: int) -> str:
    if position < start:
        return "before_window"
    if position < end:
        return "in_window"
    return "after_window"


def make_sampling_params(
    args: argparse.Namespace,
    *,
    ignore_eos: bool = True,
    stop_token_ids: list[int] | None = None,
):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    return SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        seed=args.seed,
        ignore_eos=ignore_eos,
        stop_token_ids=stop_token_ids,
        output_kind=RequestOutputKind.CUMULATIVE,
    )


def make_plan(
    request_id: str,
    prompt_len: int,
    start: int,
    end: int,
    min_rows: int,
    aux_layer_ids: tuple[int, ...],
    hidden_layout: str,
):
    from vllm.v1.hidden_state_capture import HiddenStateCapturePlan

    return HiddenStateCapturePlan.from_window(
        request_id=request_id,
        prompt_len=prompt_len,
        start=start,
        end=end,
        collection_id=f"gpu-test:{request_id}",
        coordinate="response",
        min_rows=min_rows,
        aux_layer_ids=aux_layer_ids,
        hidden_layout=capture_layout(hidden_layout),
    )


async def generate(
    engine: Any,
    prompt: Any,
    sampling_params: Any,
    request_id: str,
    *,
    plan: Any | None = None,
    window_start: int,
    window_end: int,
) -> RunResult:
    phases = {
        name: PhaseStats() for name in ("before_window", "in_window", "after_window")
    }
    last_tokens = 0
    last_time = time.perf_counter()
    final_output = None
    async for output in engine.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
        hidden_state_capture=plan,
    ):
        now = time.perf_counter()
        total_tokens = len(output.outputs[0].token_ids) if output.outputs else 0
        count = max(total_tokens - last_tokens, 0)
        if count:
            per_token = (now - last_time) / count
            for position in range(last_tokens, total_tokens):
                stats = phases[phase_name(position, window_start, window_end)]
                stats.tokens += 1
                stats.seconds += per_token
        last_tokens = total_tokens
        last_time = now
        final_output = output
    assert final_output is not None and final_output.finished
    return RunResult(
        request_id=request_id,
        token_ids=list(final_output.outputs[0].token_ids),
        capture=final_output.hidden_state_capture,
        skip_reason=final_output.hidden_capture_skip_reason,
        phases=phases,
    )


def validate_capture(
    result: RunResult,
    plan: Any,
    hidden_size: int,
    response_window_end: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    capture = result.capture
    assert capture is not None, f"capture missing: {result.skip_reason}"
    expected_positions = np.arange(plan.window_start_abs, plan.window_end_abs)
    np.testing.assert_array_equal(capture.hidden_positions, expected_positions)
    assert capture.hidden_states.device.type == "cpu"
    assert capture.hidden_states.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert capture.hidden_states.shape[0] == len(expected_positions)
    expected_parts = len(plan.aux_layer_ids) + (plan.hidden_layout != "dflash_aux")
    assert capture.hidden_states.shape[1] == hidden_size * expected_parts
    assert capture.hidden_position_start == plan.window_start_abs
    assert capture.hidden_position_end == plan.window_end_abs
    assert capture.hidden_window_start == plan.window_start_abs
    assert capture.hidden_window_end == plan.window_end_abs
    assert capture.hidden_layout == plan.hidden_layout
    assert capture.layer_ids == plan.aux_layer_ids
    assert capture.includes_final_layer == (plan.hidden_layout != "dflash_aux")
    assert capture.request_id == result.request_id
    assert capture.collection_id == plan.collection_id
    assert capture.copied_bytes > 0
    assert capture.copy_ms >= 0.0
    assert len(result.token_ids) >= response_window_end
    return {
        "rows": int(capture.hidden_states.shape[0]),
        "width": int(capture.hidden_states.shape[1]),
        "dtype": str(capture.hidden_states.dtype),
        "layout": capture.hidden_layout,
        "layer_ids": list(capture.layer_ids),
        "includes_final_layer": capture.includes_final_layer,
        "position_start": int(capture.hidden_position_start),
        "position_end": int(capture.hidden_position_end),
        "copied_bytes": int(capture.copied_bytes),
        "copy_ms": float(capture.copy_ms),
    }


def median_phase_tps(results: list[RunResult]) -> dict[str, float | None]:
    summary = {}
    for phase in ("before_window", "in_window", "after_window"):
        values = [
            result.phases[phase].tokens_per_second
            for result in results
            if result.phases[phase].tokens_per_second is not None
        ]
        summary[phase] = statistics.median(values) if values else None
    return summary


async def run_mode(args: argparse.Namespace, eager: bool) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig, AutoTokenizer

    from vllm import TokensPrompt
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.hidden_state_capture import hidden_state_capture_capability

    assert torch.cuda.is_available(), "CUDA is unavailable"
    required_gpus = args.tensor_parallel_size * args.pipeline_parallel_size
    assert torch.cuda.device_count() >= required_gpus, (
        f"need {required_gpus} visible GPUs, found {torch.cuda.device_count()}"
    )

    speculative_config = {
        "method": "dspark",
        "model": args.draft_model,
        "num_speculative_tokens": args.speculative_tokens,
        "draft_sample_method": "greedy",
    }
    if args.draft_tensor_parallel_size is not None:
        speculative_config["draft_tensor_parallel_size"] = (
            args.draft_tensor_parallel_size
        )
    draft_config = AutoConfig.from_pretrained(
        args.draft_model, trust_remote_code=args.trust_remote_code
    )
    validate_dspark_architectures(
        list(getattr(draft_config, "architectures", []) or [])
    )
    engine_kwargs = build_engine_kwargs(
        args, eager=eager, speculative_config=speculative_config
    )
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kwargs))
    try:
        capability_error = hidden_state_capture_capability(engine.vllm_config)
        assert capability_error is None, capability_error
        active_speculative_config = get_value(engine.vllm_config, "speculative_config")
        assert get_value(active_speculative_config, "method") == "dspark"
        tokenizer = AutoTokenizer.from_pretrained(
            args.target_model, trust_remote_code=args.trust_remote_code
        )
        prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
        prompt = TokensPrompt(prompt_token_ids=prompt_ids)
        prompt_len = len(prompt_ids)
        model_config = AutoConfig.from_pretrained(
            args.target_model, trust_remote_code=args.trust_remote_code
        )
        text_config = getattr(model_config, "text_config", model_config)
        hidden_size = int(text_config.hidden_size)
        aux_layer_ids = resolve_aux_layer_ids(engine, args.aux_layer_ids)

        await generate(
            engine,
            prompt,
            make_sampling_params(args),
            f"warmup-{uuid.uuid4().hex}",
            window_start=args.window_start,
            window_end=args.window_end,
        )
        baseline = await generate(
            engine,
            prompt,
            make_sampling_params(args),
            f"baseline-{uuid.uuid4().hex}",
            window_start=args.window_start,
            window_end=args.window_end,
        )
        assert baseline.capture is None and baseline.skip_reason is None
        assert baseline.token_ids

        capture_id = f"prefix-cache-{uuid.uuid4().hex}"
        capture_plan = make_plan(
            capture_id,
            prompt_len,
            args.window_start,
            args.window_end,
            args.min_rows,
            aux_layer_ids,
            args.hidden_layout,
        )
        captured = await generate(
            engine,
            prompt,
            make_sampling_params(args),
            capture_id,
            plan=capture_plan,
            window_start=args.window_start,
            window_end=args.window_end,
        )
        assert captured.token_ids == baseline.token_ids
        capture_summary = validate_capture(
            captured, capture_plan, hidden_size, args.window_end
        )

        width = args.window_end - args.window_start
        mixed_requests = []
        for offset in (0, width):
            start = args.window_start + offset
            end = args.window_end + offset
            request_id = f"mixed-{offset}-{uuid.uuid4().hex}"
            plan = make_plan(
                request_id,
                prompt_len,
                start,
                end,
                args.min_rows,
                aux_layer_ids,
                args.hidden_layout,
            )
            mixed_requests.append((request_id, plan, start, end))
        mixed_results = await asyncio.gather(
            *(
                generate(
                    engine,
                    prompt,
                    make_sampling_params(args),
                    request_id,
                    plan=plan,
                    window_start=start,
                    window_end=end,
                )
                for request_id, plan, start, end in mixed_requests
            )
        )
        mixed_summaries = [
            validate_capture(result, item[1], hidden_size, item[3])
            for result, item in zip(mixed_results, mixed_requests, strict=True)
        ]
        assert all(result.token_ids == baseline.token_ids for result in mixed_results)

        early_id = f"early-stop-{uuid.uuid4().hex}"
        early_plan = make_plan(
            early_id,
            prompt_len,
            args.window_start,
            args.window_end,
            args.min_rows,
            aux_layer_ids,
            args.hidden_layout,
        )
        early_result = await generate(
            engine,
            prompt,
            make_sampling_params(
                args, ignore_eos=False, stop_token_ids=[baseline.token_ids[0]]
            ),
            early_id,
            plan=early_plan,
            window_start=args.window_start,
            window_end=args.window_end,
        )
        assert len(early_result.token_ids) < args.window_start
        assert early_result.capture is None
        assert early_result.skip_reason == "ended_before_window"

        baseline_perf = []
        capture_perf = []
        for index in range(args.perf_repeats):
            baseline_perf.append(
                await generate(
                    engine,
                    prompt,
                    make_sampling_params(args),
                    f"perf-base-{index}-{uuid.uuid4().hex}",
                    window_start=args.window_start,
                    window_end=args.window_end,
                )
            )
            request_id = f"perf-capture-{index}-{uuid.uuid4().hex}"
            plan = make_plan(
                request_id,
                prompt_len,
                args.window_start,
                args.window_end,
                args.min_rows,
                aux_layer_ids,
                args.hidden_layout,
            )
            capture_perf.append(
                await generate(
                    engine,
                    prompt,
                    make_sampling_params(args),
                    request_id,
                    plan=plan,
                    window_start=args.window_start,
                    window_end=args.window_end,
                )
            )
            assert baseline_perf[-1].token_ids == capture_perf[-1].token_ids
        baseline_tps = median_phase_tps(baseline_perf)
        capture_tps = median_phase_tps(capture_perf)
        regressions = {}
        for phase in ("before_window", "after_window"):
            base = baseline_tps[phase]
            value = capture_tps[phase]
            regressions[phase] = 1.0 - value / base if base and value else None
            if args.enforce_perf and regressions[phase] is not None:
                assert regressions[phase] <= args.max_regression, (
                    f"{phase} median throughput regression "
                    f"{regressions[phase]:.2%} exceeds {args.max_regression:.2%}"
                )

        return {
            "mode": "eager" if eager else "cudagraph",
            "target_model": args.target_model,
            "draft_model": args.draft_model,
            "speculative_method": "dspark",
            "speculative_tokens": args.speculative_tokens,
            "prompt_tokens": prompt_len,
            "generated_tokens": len(captured.token_ids),
            "aux_layer_ids": list(aux_layer_ids),
            "capture": capture_summary,
            "mixed_batch": mixed_summaries,
            "early_stop": {
                "generated_tokens": len(early_result.token_ids),
                "skip_reason": early_result.skip_reason,
                "copied_bytes": 0,
            },
            "baseline_phase_tokens_per_sec": baseline_tps,
            "capture_phase_tokens_per_sec": capture_tps,
            "phase_regression": regressions,
            "perf_enforced": args.enforce_perf,
        }
    finally:
        engine.shutdown()


async def main_async(args: argparse.Namespace) -> list[dict[str, Any]]:
    modes = {
        "cudagraph": (False,),
        "eager": (True,),
        "both": (False, True),
    }[args.mode]
    reports = []
    for eager in modes:
        reports.append(await run_mode(args, eager))
        if len(modes) > 1:
            gc.collect()
            await asyncio.sleep(1)
    return reports


def main() -> None:
    args = parse_args()
    configure_environment()
    runtime_probe = validate_runtime_environment()
    validate_capture_result_contract()
    report = {
        "status": "passed",
        "runtime_probe": runtime_probe,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "runs": asyncio.run(main_async(args)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"PASS: report written to {args.report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
