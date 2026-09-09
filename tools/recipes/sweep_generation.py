# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generate optional vLLM benchmark sweep files from one initial suggestion."""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any

import yaml

from runtime_tuning import WorkloadHints

PROMPTS_PER_CONCURRENCY = 10
MIN_NUM_PROMPTS = 100
MAX_NUM_PROMPTS = 1000


def _num_prompts_for_concurrency(concurrency: int) -> int:
    return min(
        MAX_NUM_PROMPTS,
        max(MIN_NUM_PROMPTS, concurrency * PROMPTS_PER_CONCURRENCY),
    )


def _positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Sweep generation requires a positive initial {key!r} value.")
    return value


def _strict_lower_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << ((value - 1).bit_length() - 1)


def _strict_upper_power_of_two(value: int) -> int:
    return 1 << value.bit_length()


def validate_sweep_workload(workload: WorkloadHints) -> None:
    required = {
        "--input-tokens": workload.input_tokens,
        "--output-tokens": workload.output_tokens,
        "--concurrency": workload.concurrency,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("--generate-sweep requires " + ", ".join(missing) + ".")


def build_serve_params(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a diverse, bounded sweep around the initial suggestion.

    The initial five-point sweep changed only one parameter at a time and
    skipped the useful middle sequence count. Eight directed tuned points cover
    the batch-budget curve at full concurrency plus interactions at 3/4 and 1/2
    of the initial scheduler concurrency. Three additional reference points
    compare those tuned values with vLLM-resolved defaults without paying for a
    full Cartesian grid.
    """
    initial_seqs = _positive_int(config, "max-num-seqs")
    initial_batch = _positive_int(config, "max-num-batched-tokens")

    minimum_batch = initial_seqs
    if config.get("enable-chunked-prefill") is False:
        max_model_len = config.get("max-model-len")
        if (
            isinstance(max_model_len, int)
            and not isinstance(max_model_len, bool)
            and max_model_len > 0
        ):
            minimum_batch = max(minimum_batch, max_model_len)

    lower_seqs = max(1, (initial_seqs + 1) // 2)
    middle_seqs = max(lower_seqs, (3 * initial_seqs + 3) // 4)
    lower_batch = max(
        minimum_batch,
        _strict_lower_power_of_two(initial_batch),
    )
    smaller_batch = max(
        minimum_batch,
        _strict_lower_power_of_two(lower_batch),
    )
    higher_batch = max(
        minimum_batch,
        _strict_upper_power_of_two(initial_batch),
    )

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int | None, int | None]] = set()

    def add(
        name: str,
        max_num_seqs: int | None,
        max_num_batched_tokens: int | None,
    ) -> None:
        if max_num_seqs is not None and max_num_batched_tokens is not None:
            max_num_batched_tokens = max(max_num_batched_tokens, max_num_seqs)
        signature = (max_num_seqs, max_num_batched_tokens)
        if signature in seen:
            return
        seen.add(signature)

        candidate: dict[str, Any] = {"_benchmark_name": name}
        if max_num_seqs is not None:
            candidate["max_num_seqs"] = max_num_seqs
        if max_num_batched_tokens is not None:
            candidate["max_num_batched_tokens"] = max_num_batched_tokens
        candidates.append(candidate)

    # Keep the exact initial suggestion as the measured baseline. Additional
    # values exist only in the optional sweep package.
    add("initial", initial_seqs, initial_batch)
    add("smaller_batch_budget", initial_seqs, smaller_batch)
    add("lower_batch_budget", initial_seqs, lower_batch)
    add("higher_batch_budget", initial_seqs, higher_batch)
    add("middle_seqs_lower_batch", middle_seqs, lower_batch)
    add("middle_seqs_higher_batch", middle_seqs, higher_batch)
    add("lower_seqs_lower_batch", lower_seqs, lower_batch)
    add("lower_seqs_higher_batch", lower_seqs, higher_batch)

    # Default-reference candidates intentionally omit one or both scheduler
    # keys. The sweep server starts from sweep_config.yml, where both keys are
    # removed, so omission lets vLLM resolve its normal runtime default.
    add("vllm_default_max_num_seqs", None, initial_batch)
    add("vllm_default_max_num_batched_tokens", initial_seqs, None)
    add("vllm_defaults", None, None)

    return candidates


def build_bench_params(workload: WorkloadHints) -> list[dict[str, Any]]:
    validate_sweep_workload(workload)
    assert workload.input_tokens is not None
    assert workload.output_tokens is not None
    assert workload.concurrency is not None

    return [
        {
            "_benchmark_name": "user_workload",
            "random_input_len": workload.input_tokens,
            "random_output_len": workload.output_tokens,
            "max_concurrency": workload.concurrency,
            "num_prompts": _num_prompts_for_concurrency(workload.concurrency),
        }
    ]


def _benchmark_models(config: dict[str, Any]) -> tuple[str, str]:
    model = config.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("Sweep generation requires a model in config.yml.")

    served_model_name = config.get("served-model-name")
    if isinstance(served_model_name, str) and served_model_name:
        return served_model_name, model
    if (
        isinstance(served_model_name, list)
        and served_model_name
        and isinstance(served_model_name[0], str)
    ):
        return served_model_name[0], model

    return model, model


def _relative_to(directory: Path, target: str) -> str:
    return os.path.relpath(Path(target).resolve(), directory.resolve())


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_sweep_config(path: Path, config: dict[str, Any]) -> None:
    sweep_config = dict(config)
    sweep_config.pop("max-num-seqs", None)
    sweep_config.pop("max-num-batched-tokens", None)

    body = yaml.safe_dump(
        sweep_config,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    header = (
        "# Generated for runtime sweep only.\n"
        "# Scheduler values are supplied per candidate; omitted values use "
        "vLLM defaults.\n"
    )
    path.write_text(header + body, encoding="utf-8")


def _write_run_script(
    path: Path,
    *,
    config_rel: str,
    env_rel: str,
    request_model: str,
    tokenizer: str,
    workload: WorkloadHints,
) -> None:
    bench_parts = [
        "vllm bench serve",
        "--backend vllm",
        f"--model {shlex.quote(request_model)}",
        f"--tokenizer {shlex.quote(tokenizer)}",
        "--dataset-name random",
        "--request-rate inf",
        "--ignore-eos",
        "--metric-percentiles 99",
    ]

    goodput_pairs: list[str] = []
    if workload.ttft_sla_ms is not None:
        goodput_pairs.append(f"ttft:{workload.ttft_sla_ms:g}")
    if workload.tpot_sla_ms is not None:
        goodput_pairs.append(f"tpot:{workload.tpot_sla_ms:g}")
    if goodput_pairs:
        bench_parts.append(
            "--goodput " + " ".join(shlex.quote(pair) for pair in goodput_pairs)
        )

    bench_cmd = " ".join(bench_parts)
    script = f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"
CONFIG_PATH="${{SCRIPT_DIR}}/{config_rel}"
ENV_PATH="${{SCRIPT_DIR}}/{env_rel}"

source "${{ENV_PATH}}"

vllm bench sweep serve \
  --serve-cmd "vllm serve --config '${{CONFIG_PATH}}'" \
  --bench-cmd "{bench_cmd}" \
  --serve-params "${{SCRIPT_DIR}}/serve_params.json" \
  --bench-params "${{SCRIPT_DIR}}/bench_params.json" \
  --output-dir "${{SCRIPT_DIR}}/results" \
  --experiment-name runtime-tuning \
  "$@"
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def _write_recommend_script(
    path: Path,
    *,
    config_rel: str,
    env_rel: str,
    workload: WorkloadHints,
) -> None:
    template_path = Path(__file__).with_name("sweep_recommendation.py")
    source = template_path.read_text(encoding="utf-8")

    replacements = {
        "DEFAULT_CONFIG_PATH: str | None = None": (
            f"DEFAULT_CONFIG_PATH: str | None = {config_rel!r}"
        ),
        "DEFAULT_ENV_PATH: str | None = None": (
            f"DEFAULT_ENV_PATH: str | None = {env_rel!r}"
        ),
        "DEFAULT_TTFT_SLA_MS: float | None = None": (
            f"DEFAULT_TTFT_SLA_MS: float | None = {workload.ttft_sla_ms!r}"
        ),
        "DEFAULT_TPOT_SLA_MS: float | None = None": (
            f"DEFAULT_TPOT_SLA_MS: float | None = {workload.tpot_sla_ms!r}"
        ),
    }
    for marker, replacement in replacements.items():
        if marker not in source:
            raise ValueError(f"Recommender template marker not found: {marker}")
        source = source.replace(marker, replacement, 1)

    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def _write_guide(path: Path, workload: WorkloadHints) -> None:
    sla_lines: list[str] = []
    if workload.ttft_sla_ms is not None:
        sla_lines.append(f"- TTFT objective: `{workload.ttft_sla_ms:g} ms`")
    if workload.tpot_sla_ms is not None:
        sla_lines.append(f"- TPOT objective: `{workload.tpot_sla_ms:g} ms`")

    sla_text = "\n".join(sla_lines)
    if sla_text:
        sla_text = (
            "\n## Supplied latency objectives\n\n"
            + sla_text
            + "\n\nThe generated benchmark uses these values with vLLM "
            "`--goodput`.\n"
        )

    content = f"""# Optional Runtime Tuning Sweep

The generated `config.yml` is the **single initial suggestion** and can be
deployed directly. The sweep benchmarks nearby values for:

- `max-num-seqs`
- `max-num-batched-tokens`

In addition to nearby tuned values, the sweep includes three vLLM-default
references:

- vLLM default `max-num-seqs` with the initial batch-token budget
- initial `max-num-seqs` with vLLM default `max-num-batched-tokens`
- vLLM defaults for both parameters

The sweep runs every server from a generated `sweep_config.yml` that copies the
initial `config.yml` but removes these two scheduler keys. Tuned candidates add
explicit CLI values; a default-reference candidate simply omits the selected
key, allowing `vllm serve` to resolve its normal runtime default. This avoids
passing the literal string `None` and keeps defaults platform-, world-size-,
model-, and usage-context-aware.

## Run

```bash
./run_sweep.sh --dry-run
./run_sweep.sh
./recommend.py
```

For a quick one-run experiment:

```bash
rm -rf results/runtime-tuning
./run_sweep.sh --num-runs 1
./recommend.py
```

Resume an interrupted sweep:

```bash
./run_sweep.sh --resume
```

By default, vLLM benchmarks each parameter combination three times.

The benchmark request count scales with the supplied workload concurrency using
the same model as vLLM's `PROMPTS_PER_CONCURRENCY` performance-benchmark
control:

```text
num_prompts = min(1000, max(100, concurrency * 10))
```

This targets about ten concurrency turnovers when neither bound applies, keeps
at least 100 requests for percentile/compliance measurements, and caps each run
at 1000 requests. All scheduler candidates use the same request count.
{sla_text}
## Outputs

`recommend.py` writes:

```text
recommended-config.yml
recommendation.json
```

With TTFT/TPOT objectives it requires duration-weighted combined compliance of
at least 99% plus median P99 TTFT/TPOT compliance, then selects highest mean
output-token throughput. If no candidate qualifies, it records a best-effort
candidate but does not write `recommended-config.yml`. Without latency
objectives it selects highest mean output-token throughput. Configurations with
failed requests are excluded.

`recommended-config.yml` copies the initial configuration and changes only
`max-num-seqs` and `max-num-batched-tokens`. If a vLLM-default reference wins,
the corresponding key is removed from `recommended-config.yml`, allowing
`vllm serve` to resolve that parameter from its normal runtime default policy.
"""
    path.write_text(content, encoding="utf-8")


def write_sweep_files(
    output_dir: str,
    *,
    config_path: str,
    env_path: str,
    config: dict[str, Any],
    workload: WorkloadHints,
) -> list[Path]:
    """Write an optional sweep package around the initial config suggestion."""
    validate_sweep_workload(workload)

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    sweep_config = directory / "sweep_config.yml"
    serve_params = directory / "serve_params.json"
    bench_params = directory / "bench_params.json"
    run_script = directory / "run_sweep.sh"
    recommend_script = directory / "recommend.py"
    guide = directory / "SWEEP.md"

    config_rel = _relative_to(directory, config_path)
    sweep_config_rel = _relative_to(directory, str(sweep_config))
    env_rel = _relative_to(directory, env_path)

    _write_sweep_config(sweep_config, config)
    _write_json(serve_params, build_serve_params(config))
    _write_json(bench_params, build_bench_params(workload))
    request_model, tokenizer = _benchmark_models(config)
    _write_run_script(
        run_script,
        config_rel=sweep_config_rel,
        env_rel=env_rel,
        request_model=request_model,
        tokenizer=tokenizer,
        workload=workload,
    )
    _write_recommend_script(
        recommend_script,
        config_rel=config_rel,
        env_rel=env_rel,
        workload=workload,
    )
    _write_guide(guide, workload)

    return [
        sweep_config,
        serve_params,
        bench_params,
        run_script,
        recommend_script,
        guide,
    ]
