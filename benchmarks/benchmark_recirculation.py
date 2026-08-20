# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evaluate fixed Recirculation with reproducible token windows."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_MODEL_REVISION = "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29"
DEFAULT_DATASET_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def build_windows(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
    )
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.dataset_split,
        revision=args.dataset_revision,
        streaming=True,
    )
    windows = []
    for document_index, row in enumerate(dataset):
        token_ids = tokenizer.encode(
            row[args.dataset_text_column],
            add_special_tokens=False,
        )
        for token_offset in range(
            0, len(token_ids) - args.window_size + 1, args.window_size
        ):
            window = token_ids[token_offset : token_offset + args.window_size]
            windows.append(
                {
                    "document_index": document_index,
                    "token_offset": token_offset,
                    "token_ids": window,
                    "sha256": sha256_json(window),
                }
            )
            if len(windows) == args.num_windows:
                break
        if len(windows) == args.num_windows:
            break
    if len(windows) != args.num_windows:
        raise RuntimeError(
            f"Found only {len(windows)} full windows; requested {args.num_windows}"
        )
    return {
        "format_version": 1,
        "dataset": {
            "name": args.dataset,
            "config": args.dataset_config,
            "revision": args.dataset_revision,
            "split": args.dataset_split,
            "text_column": args.dataset_text_column,
        },
        "tokenizer": {
            "name": args.model,
            "revision": args.model_revision,
        },
        "construction": {
            "window_size": args.window_size,
            "add_special_tokens": False,
            "selection": (
                "dataset order; non-overlapping full windows within each document; "
                "incomplete tails skipped"
            ),
        },
        "windows": windows,
    }


def load_or_build_windows(args: argparse.Namespace) -> dict[str, Any]:
    path = args.windows_file
    if path.exists():
        data = json.loads(path.read_text())
        expected = {
            "name": args.dataset,
            "config": args.dataset_config,
            "revision": args.dataset_revision,
            "split": args.dataset_split,
            "text_column": args.dataset_text_column,
        }
        matches = (
            data.get("dataset") == expected
            and data.get("tokenizer")
            == {"name": args.model, "revision": args.model_revision}
            and data.get("construction", {}).get("window_size") == args.window_size
        )
        if matches and len(data.get("windows", [])) >= args.num_windows:
            for window in data["windows"][: args.num_windows]:
                if sha256_json(window["token_ids"]) != window["sha256"]:
                    raise ValueError(f"Corrupt window cache: {path}")
            data["windows"] = data["windows"][: args.num_windows]
            return data

    data = build_windows(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return data


class PeakGpuMemory:
    def __init__(self, device_index: int) -> None:
        import pynvml

        self._pynvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        self.start_bytes = memory.used
        self.peak_bytes = memory.used
        self.total_bytes = memory.total
        self.name = pynvml.nvmlDeviceGetName(self._handle)
        self.driver = pynvml.nvmlSystemGetDriverVersion()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self) -> None:
        while not self._stopped.wait(0.01):
            used = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle).used
            self.peak_bytes = max(self.peak_bytes, used)

    def __enter__(self) -> "PeakGpuMemory":
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stopped.set()
        self._thread.join()
        used = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle).used
        self.peak_bytes = max(self.peak_bytes, used)
        self._pynvml.nvmlShutdown()


def chosen_prompt_nll(output: Any) -> float:
    if output.prompt_logprobs is None or output.prompt_token_ids is None:
        raise RuntimeError("vLLM did not return requested prompt logprobs")
    nll = 0.0
    for token_id, entries in zip(
        output.prompt_token_ids[1:], output.prompt_logprobs[1:]
    ):
        if entries is None or token_id not in entries:
            raise RuntimeError(f"Missing prompt logprob for token {token_id}")
        nll -= entries[token_id].logprob
    return nll


def score_window(llm: Any, token_ids: list[int], seed: int) -> dict[str, float]:
    from vllm import SamplingParams

    prompt_token_ids = token_ids[:-1]
    final_target = token_ids[-1]
    params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        seed=seed,
        prompt_logprobs=0,
        logprobs=1,
        logprob_token_ids=[final_target],
        ignore_eos=True,
    )
    start = time.perf_counter()
    output = llm.generate(
        [{"prompt_token_ids": prompt_token_ids}],
        params,
        use_tqdm=False,
    )[0]
    elapsed = time.perf_counter() - start
    completion_logprobs = output.outputs[0].logprobs
    if not completion_logprobs or final_target not in completion_logprobs[0]:
        raise RuntimeError(f"Missing final target logprob for token {final_target}")
    nll = chosen_prompt_nll(output) - completion_logprobs[0][final_target].logprob
    prefill_latency = output.metrics.first_token_latency if output.metrics else elapsed
    return {
        "negative_log_likelihood": nll,
        "elapsed_s": elapsed,
        "prefill_latency_s": prefill_latency,
    }


def score_windows(llm: Any, windows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    scores = []
    for index, window in enumerate(windows, 1):
        score = score_window(llm, window["token_ids"], seed)
        scores.append(score)
        print(
            f"window {index}/{len(windows)}: "
            f"nll={score['negative_log_likelihood']:.6f}, "
            f"elapsed={score['elapsed_s']:.3f}s",
            flush=True,
        )
    total_nll = sum(score["negative_log_likelihood"] for score in scores)
    scored_tokens = sum(len(window["token_ids"]) - 1 for window in windows)
    return {
        "num_windows": len(windows),
        "scored_tokens": scored_tokens,
        "negative_log_likelihood": total_nll,
        "mean_negative_log_likelihood": total_nll / scored_tokens,
        "perplexity": math.exp(total_nll / scored_tokens),
        "elapsed_s": sum(score["elapsed_s"] for score in scores),
        "mean_prefill_latency_s": sum(score["prefill_latency_s"] for score in scores)
        / len(scores),
        "per_window": scores,
    }


def measure_decode(
    llm: Any,
    token_ids: list[int],
    max_model_len: int,
    decode_tokens: int,
    seed: int,
) -> dict[str, Any]:
    from vllm import SamplingParams

    prompt_tokens = max_model_len - decode_tokens
    prompt_token_ids = token_ids[:prompt_tokens]
    params = SamplingParams(
        temperature=0.0,
        max_tokens=decode_tokens,
        seed=seed,
        ignore_eos=True,
    )
    start = time.perf_counter()
    output = llm.generate(
        [{"prompt_token_ids": prompt_token_ids}],
        params,
        use_tqdm=False,
    )[0]
    elapsed = time.perf_counter() - start
    output_tokens = len(output.outputs[0].token_ids)
    metrics = output.metrics
    decode_elapsed = metrics.last_token_ts - metrics.first_token_ts if metrics else 0.0
    measured_decode_tokens = max(output_tokens - 1, 0)
    return {
        "prompt_tokens": len(prompt_token_ids),
        "output_tokens": output_tokens,
        "elapsed_s": elapsed,
        "prefill_latency_s": metrics.first_token_latency if metrics else None,
        "decode_elapsed_s": decode_elapsed,
        "decode_throughput_tokens_per_s": (
            measured_decode_tokens / decode_elapsed if decode_elapsed > 0 else None
        ),
        "output_text": output.outputs[0].text,
        "output_token_ids": list(output.outputs[0].token_ids),
    }


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def environment_metadata() -> dict[str, Any]:
    import torch

    return {
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "python": platform.python_version(),
        "vllm": importlib.metadata.version("vllm"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "datasets": importlib.metadata.version("datasets"),
    }


def run(args: argparse.Namespace, window_data: dict[str, Any]) -> dict[str, Any]:
    from vllm import LLM

    hf_overrides = None
    if args.mode == "recirculation":
        hf_overrides = {
            "recirculation_config": {
                "source_layer": args.source_layer,
                "destination_layer": args.destination_layer,
                "alpha": args.alpha,
                "ramp_tokens": args.ramp_tokens,
            }
        }
    start = time.perf_counter()
    llm = LLM(
        model=args.model,
        revision=args.model_revision,
        tokenizer_revision=args.model_revision,
        dtype=args.dtype,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        enforce_eager=True,
        enable_prefix_caching=False,
        long_prefill_token_threshold=args.long_prefill_token_threshold,
        hf_overrides=hf_overrides,
        disable_log_stats=False,
    )
    load_s = time.perf_counter() - start
    try:
        windows = window_data["windows"]
        quality = (
            None if args.performance_only else score_windows(llm, windows, args.seed)
        )
        performance = measure_decode(
            llm,
            windows[0]["token_ids"],
            args.max_model_len,
            args.decode_tokens,
            args.seed,
        )
        return {
            "model_load_s": load_s,
            "quality": quality,
            "performance": performance,
        }
    finally:
        llm.llm_engine.engine_core.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "recirculation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows-file", type=Path, required=True)
    parser.add_argument("--num-windows", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--model", default="google/gemma-3-1b-pt")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--dataset", default="allenai/c4")
    parser.add_argument("--dataset-config", default="en")
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--dataset-text-column", default="text")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--source-layer", type=int, default=11)
    parser.add_argument("--destination-layer", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--ramp-tokens", type=int, default=10)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--performance-only", action="store_true")
    args = parser.parse_args()
    if args.window_size != args.max_model_len:
        parser.error("--window-size must equal --max-model-len")
    if not 0 < args.decode_tokens < args.max_model_len:
        parser.error("--decode-tokens must be between 1 and max-model-len - 1")
    return args


def main() -> None:
    args = parse_args()
    window_data = load_or_build_windows(args)
    started = time.perf_counter()
    with PeakGpuMemory(args.device_index) as gpu_memory:
        measurements = run(args, window_data)
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "configuration": {
            "model": args.model,
            "model_revision": args.model_revision,
            "dtype": args.dtype,
            "seed": args.seed,
            "max_model_len": args.max_model_len,
            "max_num_seqs": 1,
            "enforce_eager": True,
            "speculative_decoding": False,
            "prefix_caching": False,
            "long_prefill_token_threshold": args.long_prefill_token_threshold,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "recirculation_config": (
                None
                if args.mode == "baseline"
                else {
                    "source_layer": args.source_layer,
                    "destination_layer": args.destination_layer,
                    "alpha": args.alpha,
                    "beta": None,
                    "ramp_tokens": args.ramp_tokens,
                }
            ),
        },
        "window_spec": {
            key: value for key, value in window_data.items() if key != "windows"
        },
        "window_file": str(args.windows_file),
        "window_file_sha256": sha256_json(window_data),
        "window_hashes": [window["sha256"] for window in window_data["windows"]],
        "environment": environment_metadata(),
        "gpu": {
            "name": gpu_memory.name,
            "driver": gpu_memory.driver,
            "total_bytes": gpu_memory.total_bytes,
            "start_used_bytes": gpu_memory.start_bytes,
            "peak_used_bytes": gpu_memory.peak_bytes,
            "peak_delta_bytes": gpu_memory.peak_bytes - gpu_memory.start_bytes,
        },
        "total_elapsed_s": time.perf_counter() - started,
        **measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
