# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generate optional vLLM benchmark sweep files from one initial suggestion."""

from __future__ import annotations

import json
import math
import os
import shlex
from pathlib import Path
from typing import Any

import yaml
from runtime_tuning import WorkloadHints

SUPPORTED_TENSOR_PARALLEL_SIZES = frozenset({1, 2, 4, 8})

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
    """Build a bounded batch-budget sweep at the sequence-count lower bound.

    Explicit candidates keep the initial per-replica sequence count. Three
    reference points compare those tuned values with vLLM-resolved defaults.
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
    # Default-reference candidates intentionally omit one or both scheduler
    # keys. The sweep server starts from sweep_config.yml, where both keys are
    # removed, so omission lets vLLM resolve its normal runtime default.
    add("vllm_default_max_num_seqs", None, initial_batch)
    add("vllm_default_max_num_batched_tokens", initial_seqs, None)
    add("vllm_defaults", None, None)

    return candidates


def _scheduler_baseline_for_dp(
    config: dict[str, Any], workload: WorkloadHints, data_parallel_size: int
) -> tuple[int, int]:
    """Return per-replica scheduler settings for a DP layout."""
    assert workload.concurrency is not None
    assert workload.input_tokens is not None
    per_replica_concurrency = math.ceil(workload.concurrency / data_parallel_size)
    output_tokens = workload.output_tokens or 1
    prefills_per_step = max(1.0, per_replica_concurrency / output_tokens)
    if workload.target_qps is not None and workload.tpot_sla_ms is not None:
        per_replica_qps = workload.target_qps / data_parallel_size
        prefills_per_step = max(
            prefills_per_step,
            per_replica_qps * workload.tpot_sla_ms / 1000.0,
        )
    prefills_per_step = min(float(per_replica_concurrency), prefills_per_step)
    batch = max(
        2048,
        per_replica_concurrency,
        per_replica_concurrency + math.ceil(workload.input_tokens * prefills_per_step),
    )
    if config.get("enable-chunked-prefill") is False:
        max_model_len = config.get("max-model-len")
        if isinstance(max_model_len, int) and not isinstance(max_model_len, bool):
            batch = max(batch, max_model_len)
    return per_replica_concurrency, batch


def build_parallel_layout_params(
    config: dict[str, Any], workload: WorkloadHints, numa_node_count: int
) -> list[dict[str, Any]]:
    """Build full-NUMA layouts plus the largest supported TP layout."""
    validate_sweep_workload(workload)
    if numa_node_count <= 1:
        raise ValueError("Parallel-layout sweep requires at least two NUMA nodes.")

    candidates = []
    max_supported_tp = max(
        size for size in SUPPORTED_TENSOR_PARALLEL_SIZES if size <= numa_node_count
    )
    for tensor_parallel_size in range(numa_node_count, 0, -1):
        if tensor_parallel_size not in SUPPORTED_TENSOR_PARALLEL_SIZES:
            continue
        uses_all_numa_nodes = numa_node_count % tensor_parallel_size == 0
        if not uses_all_numa_nodes and tensor_parallel_size != max_supported_tp:
            continue
        data_parallel_size = max(1, numa_node_count // tensor_parallel_size)
        max_num_seqs, max_num_batched_tokens = _scheduler_baseline_for_dp(
            config, workload, data_parallel_size
        )
        name = f"tp{tensor_parallel_size}_dp{data_parallel_size}"
        if not uses_all_numa_nodes:
            used_nodes = tensor_parallel_size * data_parallel_size
            name += f"_numa{used_nodes}of{numa_node_count}"
        candidates.append(
            {
                "_benchmark_name": name,
                "tensor_parallel_size": tensor_parallel_size,
                "data_parallel_size": data_parallel_size,
                "max_num_seqs": max_num_seqs,
                "max_num_batched_tokens": max_num_batched_tokens,
            }
        )
    return candidates


def build_bench_params(workload: WorkloadHints) -> list[dict[str, Any]]:
    validate_sweep_workload(workload)
    assert workload.input_tokens is not None
    assert workload.output_tokens is not None
    assert workload.concurrency is not None

    return [
        {
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
    serve_params_name: str = "serve_params.json",
    experiment_name: str = "runtime-tuning",
    prepare_config_rel: str | None = None,
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
    prepare_config = ""
    if prepare_config_rel is not None:
        prepare_config = f"""
SOURCE_CONFIG_PATH="${{SCRIPT_DIR}}/{prepare_config_rel}"
python3 - "${{SOURCE_CONFIG_PATH}}" "${{CONFIG_PATH}}" <<'PY'
import sys
from pathlib import Path

import yaml

source = Path(sys.argv[1])
target = Path(sys.argv[2])
config = yaml.safe_load(source.read_text(encoding="utf-8"))
if not isinstance(config, dict):
    raise SystemExit(str(source) + " does not contain a YAML configuration object.")
config.pop("max-num-seqs", None)
config.pop("max-num-batched-tokens", None)
target.write_text(
    yaml.safe_dump(
        config,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)
PY

"""

    script = f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"
CONFIG_PATH="${{SCRIPT_DIR}}/{config_rel}"
ENV_PATH="${{SCRIPT_DIR}}/{env_rel}"

source "${{ENV_PATH}}"

# recipe tools may be newer than the installed vLLM package. Only pass the
# resilience flag when this vLLM CLI actually supports it.
CONTINUE_ON_ERROR_ARG=""
if vllm bench sweep serve --help 2>&1 | grep -q -- "--continue-on-error"; then
  CONTINUE_ON_ERROR_ARG="--continue-on-error"
fi

# On older vLLM versions, a failed benchmark aborts the sweep process. Retry the
# interrupted sweep with --resume so one transient run failure does not discard
# completed work or force a manual restart.
SWEEP_RETRY_COUNT="${{VLLM_RECIPE_SWEEP_RETRY_COUNT:-0}}"
MAX_SWEEP_RETRIES="${{VLLM_RECIPE_SWEEP_RETRIES:-2}}"

{prepare_config}vllm bench sweep serve \
  --serve-cmd "vllm serve --config '${{CONFIG_PATH}}'" \
  --bench-cmd "{bench_cmd}" \
  --serve-params "${{SCRIPT_DIR}}/{serve_params_name}" \
  --bench-params "${{SCRIPT_DIR}}/bench_params.json" \
  --output-dir "${{SCRIPT_DIR}}/results" \
  --experiment-name {experiment_name} \
  --warmup-num-prompts {workload.concurrency} \
  ${{CONTINUE_ON_ERROR_ARG}} \
  "$@" || {{
    if [[ "${{SWEEP_RETRY_COUNT}}" -ge "${{MAX_SWEEP_RETRIES}}" ]]; then
      echo "Sweep failed after ${{MAX_SWEEP_RETRIES}} automatic retries." >&2
      exit 1
    fi

    NEXT_RETRY=$((SWEEP_RETRY_COUNT + 1))
    RETRY_ARGS=()
    for arg in "$@"; do
      [[ "$arg" == "--resume" ]] && continue
      RETRY_ARGS+=("$arg")
    done

    EXPERIMENT_DIR="${{SCRIPT_DIR}}/results/{experiment_name}"
    if [[ -d "${{EXPERIMENT_DIR}}" ]]; then
      echo "Sweep command failed;" \
        "retry ${{NEXT_RETRY}}/${{MAX_SWEEP_RETRIES}} with --resume"      
      RETRY_ARGS=(--resume "${{RETRY_ARGS[@]}}")
    else
      echo "Sweep command failed before resumable state was created;" \
        "retry ${{NEXT_RETRY}}/${{MAX_SWEEP_RETRIES}} from the start"      
    fi

    VLLM_RECIPE_SWEEP_RETRY_COUNT="${{NEXT_RETRY}}" exec "$0" "${{RETRY_ARGS[@]}}"
  }}
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def _write_recommend_script(
    path: Path,
    *,
    config_rel: str,
    env_rel: str,
    workload: WorkloadHints,
    results_dir: str = "results/runtime-tuning",
    output_config: str = "recommended-config.yml",
    output_json: str = "recommendation.json",
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
        'DEFAULT_RESULTS_DIR = "results/runtime-tuning"': (
            f"DEFAULT_RESULTS_DIR = {results_dir!r}"
        ),
        'DEFAULT_OUTPUT_CONFIG = "recommended-config.yml"': (
            f"DEFAULT_OUTPUT_CONFIG = {output_config!r}"
        ),
        'DEFAULT_OUTPUT_JSON = "recommendation.json"': (
            f"DEFAULT_OUTPUT_JSON = {output_json!r}"
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
deployed directly. The scheduler sweep keeps explicit `max-num-seqs` at the
per-replica concurrency lower bound while benchmarking nearby values for:

- `max-num-batched-tokens`

It does not test explicit `max-num-seqs` values below per-replica concurrency,
because those values can introduce scheduler queueing and severe TTFT
degradation. The vLLM-default references still test whether an explicit
`max-num-seqs` override is needed.

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

Before measurement, the generated script runs one unmeasured warmup containing
one full concurrency window ({workload.concurrency} prompts). It is saved as
`warmup.json` for auditing but excluded from recommendation statistics. By
default, vLLM then benchmarks each parameter combination three times.

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
at least 99% plus median P99 TTFT/TPOT compliance. Scheduler candidates within
1% of the highest eligible output-token throughput are treated as practically
equivalent, and the recommender prefers fewer explicit scheduler overrides
within that set. Use `--throughput-equivalence-percent 0` for exact
highest-throughput selection. If no candidate qualifies, it records a
best-effort candidate but does not write `recommended-config.yml`. Without
latency objectives it applies the same scheduler equivalence policy.
Configurations with failed requests are excluded. TP/DP comparisons continue
to use exact highest-throughput selection.

`recommended-config.yml` copies the initial configuration and changes only
`max-num-seqs` and `max-num-batched-tokens`. If a vLLM-default reference wins,
the corresponding key is removed from `recommended-config.yml`, allowing
`vllm serve` to resolve that parameter from its normal runtime default policy.
"""
    path.write_text(content, encoding="utf-8")


def _write_post_benchmark_analysis_files(directory: Path) -> list[Path]:
    """Copy standalone visualization helpers into a generated sweep package."""
    source_dir = Path(__file__).resolve().parent
    generated: list[Path] = []
    for source_name, target_name, executable in (
        ("sweep_visualization.py", "visualize.py", True),
        ("requirements.txt", "requirements.txt", False),
        ("VISUALIZATION.md", "VISUALIZATION.md", False),
    ):
        source = source_dir / source_name
        target = directory / target_name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        if executable:
            target.chmod(target.stat().st_mode | 0o111)
        generated.append(target)
    return generated


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
    analysis_files = _write_post_benchmark_analysis_files(directory)

    return [
        sweep_config,
        serve_params,
        bench_params,
        run_script,
        recommend_script,
        guide,
        *analysis_files,
    ]


def write_parallel_layout_sweep_files(
    output_dir: str,
    *,
    config_path: str,
    env_path: str,
    config: dict[str, Any],
    workload: WorkloadHints,
    numa_node_count: int,
) -> list[Path]:
    """Write a standalone NUMA-aware TP/DP sweep package."""
    validate_sweep_workload(workload)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    parallel_params = directory / "parallel_layout_serve_params.json"
    bench_params = directory / "bench_params.json"
    run_parallel = directory / "run_parallel_layout_sweep.sh"
    recommend_parallel = directory / "recommend_parallel_layout.py"
    guide = directory / "SWEEP.md"

    initial_config_rel = _relative_to(directory, config_path)
    selected_config_rel = "parallel-layout-config.yml"
    env_rel = _relative_to(directory, env_path)
    request_model, tokenizer = _benchmark_models(config)

    _write_json(
        parallel_params,
        build_parallel_layout_params(config, workload, numa_node_count),
    )
    _write_json(bench_params, build_bench_params(workload))
    _write_run_script(
        run_parallel,
        config_rel=initial_config_rel,
        env_rel=env_rel,
        request_model=request_model,
        tokenizer=tokenizer,
        workload=workload,
        serve_params_name=parallel_params.name,
        experiment_name="parallel-layout",
    )
    _write_recommend_script(
        recommend_parallel,
        config_rel=initial_config_rel,
        env_rel=env_rel,
        workload=workload,
        results_dir="results/parallel-layout",
        output_config=selected_config_rel,
        output_json="parallel-layout-recommendation.json",
    )

    guide.write_text(
        f"""# Parallel-Layout Sweep

This package tunes only the NUMA-aware TP/DP layout.

Primary candidates use every effective NUMA node:

```text
tensor_parallel_size * data_parallel_size = {numa_node_count}
```

Tensor parallelism is restricted to `1`, `2`, `4`, or `8`. The largest
supported TP not exceeding the NUMA-node count is also included, even when it
leaves some NUMA nodes idle.

Run:

```bash
./run_parallel_layout_sweep.sh --dry-run
./run_parallel_layout_sweep.sh
./recommend_parallel_layout.py
```

The recommender writes:

```text
parallel-layout-config.yml
parallel-layout-recommendation.json
```

This mode stops after TP/DP selection. Use `--generate-full-sweep` when
concurrency and scheduler tuning should follow automatically.
""",
        encoding="utf-8",
    )
    analysis_files = _write_post_benchmark_analysis_files(directory)

    return [
        parallel_params,
        bench_params,
        run_parallel,
        recommend_parallel,
        guide,
        *analysis_files,
    ]


def build_concurrency_bench_params(workload: WorkloadHints) -> list[dict[str, Any]]:
    """Build one workload shape while leaving max_concurrency to Workload Explorer."""
    validate_sweep_workload(workload)
    assert workload.input_tokens is not None
    assert workload.output_tokens is not None
    assert workload.concurrency is not None
    return [
        {
            "_benchmark_name": "user_workload",
            "random_input_len": workload.input_tokens,
            "random_output_len": workload.output_tokens,
            "num_prompts": _num_prompts_for_concurrency(workload.concurrency),
        }
    ]


def _write_concurrency_run_script(
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

# recipe tools may be newer than the installed vLLM package. Only pass the
# resilience flag when this vLLM CLI actually supports it.
CONTINUE_ON_ERROR_ARG=""
if vllm bench sweep serve_workload --help 2>&1 | grep -q -- "--continue-on-error"; then
  CONTINUE_ON_ERROR_ARG="--continue-on-error"
fi

# On older vLLM versions, a failed benchmark aborts the sweep process. Retry the
# interrupted sweep with --resume so one transient run failure does not discard
# completed work or force a manual restart.
SWEEP_RETRY_COUNT="${{VLLM_RECIPE_SWEEP_RETRY_COUNT:-0}}"
MAX_SWEEP_RETRIES="${{VLLM_RECIPE_SWEEP_RETRIES:-2}}"

vllm bench sweep serve_workload \
  --serve-cmd "vllm serve --config '${{CONFIG_PATH}}'" \
  --bench-cmd "{bench_cmd}" \
  --bench-params "${{SCRIPT_DIR}}/concurrency_bench_params.json" \
  --output-dir "${{SCRIPT_DIR}}/results" \
  --experiment-name concurrency-tuning \
  --workload-var max_concurrency \
  --workload-iters 10 \
  --warmup-num-prompts {workload.concurrency} \
  ${{CONTINUE_ON_ERROR_ARG}} \
  "$@" || {{
    if [[ "${{SWEEP_RETRY_COUNT}}" -ge "${{MAX_SWEEP_RETRIES}}" ]]; then
      echo "Sweep failed after ${{MAX_SWEEP_RETRIES}} automatic retries." >&2
      exit 1
    fi

    NEXT_RETRY=$((SWEEP_RETRY_COUNT + 1))
    RETRY_ARGS=()
    for arg in "$@"; do
      [[ "$arg" == "--resume" ]] && continue
      RETRY_ARGS+=("$arg")
    done

    EXPERIMENT_DIR="${{SCRIPT_DIR}}/results/concurrency-tuning"
    if [[ -d "${{EXPERIMENT_DIR}}" ]]; then
      echo "Sweep command failed;" \
        "retry ${{NEXT_RETRY}}/${{MAX_SWEEP_RETRIES}} with --resume"      
      RETRY_ARGS=(--resume "${{RETRY_ARGS[@]}}")
    else
      echo "Sweep command failed before resumable state was created;" \
        "retry ${{NEXT_RETRY}}/${{MAX_SWEEP_RETRIES}} from the start"      
    fi

    VLLM_RECIPE_SWEEP_RETRY_COUNT="${{NEXT_RETRY}}" exec "$0" "${{RETRY_ARGS[@]}}"
  }}
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def _write_concurrency_recommend_script(path: Path, *, workload: WorkloadHints) -> None:
    template_path = Path(__file__).with_name("concurrency_recommendation.py")
    source = template_path.read_text(encoding="utf-8")
    replacements = {
        "DEFAULT_TTFT_SLA_MS: float | None = None": (
            f"DEFAULT_TTFT_SLA_MS: float | None = {workload.ttft_sla_ms!r}"
        ),
        "DEFAULT_TPOT_SLA_MS: float | None = None": (
            f"DEFAULT_TPOT_SLA_MS: float | None = {workload.tpot_sla_ms!r}"
        ),
        "DEFAULT_SEED_CONCURRENCY: int | None = None": (
            f"DEFAULT_SEED_CONCURRENCY: int | None = {workload.concurrency!r}"
        ),
    }
    for marker, replacement in replacements.items():
        if marker not in source:
            raise ValueError(f"Concurrency recommender marker not found: {marker}")
        source = source.replace(marker, replacement, 1)
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def _write_scheduler_prepare_script(path: Path) -> None:
    template_path = Path(__file__).with_name("scheduler_sweep_prepare.py")
    path.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def _write_scheduler_wrapper(path: Path, *, workload: WorkloadHints) -> None:
    assert workload.input_tokens is not None
    assert workload.output_tokens is not None
    args = [
        'python3 "${SCRIPT_DIR}/prepare_scheduler.py"',
        '--config "${SCRIPT_DIR}/parallel-layout-config.yml"',
        '--concurrency-recommendation "${SCRIPT_DIR}/concurrency-recommendation.json"',
        f"--input-tokens {workload.input_tokens}",
        f"--output-tokens {workload.output_tokens}",
    ]
    if workload.tpot_sla_ms is not None:
        args.append(f"--tpot-sla-ms {workload.tpot_sla_ms:g}")
    if workload.target_qps is not None:
        args.append(f"--target-qps {workload.target_qps:g}")
    args.extend(
        [
            '--output-config "${SCRIPT_DIR}/parallel-layout-sweep-config.yml"',
            '--serve-params "${SCRIPT_DIR}/serve_params.json"',
            '--bench-params "${SCRIPT_DIR}/bench_params.json"',
        ]
    )
    prepare_cmd = " \\\n  ".join(args)
    script = f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"

{prepare_cmd}

exec "${{SCRIPT_DIR}}/run_sweep.sh" "$@"
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def _write_full_run_script(path: Path) -> None:
    script = """#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_ARGS=("$@")

"${SCRIPT_DIR}/run_parallel_layout_sweep.sh" "${RUN_ARGS[@]}"
"${SCRIPT_DIR}/recommend_parallel_layout.py"
"${SCRIPT_DIR}/run_concurrency_sweep.sh" "${RUN_ARGS[@]}"
"${SCRIPT_DIR}/recommend_concurrency.py"
"${SCRIPT_DIR}/run_scheduler_sweep.sh" "${RUN_ARGS[@]}"
"${SCRIPT_DIR}/recommend.py"
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def _write_full_guide(path: Path, workload: WorkloadHints) -> None:
    content = f"""# End-to-End Runtime Tuning

This package keeps one fixed workload shape and tunes in dependency order:

```text
TP/DP -> max_concurrency -> scheduler
```

The supplied `--concurrency={workload.concurrency}` is the representative load
used for TP/DP selection and to size the Workload Explorer dataset. Stage 2
selects the measured SLA-feasible `max_concurrency`. Stage 3 recalculates its
scheduler baseline from that concurrency and the selected DP layout.

Run all stages:

```bash
./run_full_sweep.sh
```

Or run each stage explicitly:

```bash
./run_parallel_layout_sweep.sh
./recommend_parallel_layout.py
./run_concurrency_sweep.sh
./recommend_concurrency.py
./run_scheduler_sweep.sh
./recommend.py
```

Generated recipe sweeps use `--continue-on-error`. An isolated warmup or
benchmark invocation failure is recorded and skipped. A combination with no
successful measured runs remains retryable with `--resume`.
"""
    path.write_text(content, encoding="utf-8")


def write_concurrency_sweep_files(
    output_dir: str,
    *,
    config_path: str,
    env_path: str,
    config: dict[str, Any],
    workload: WorkloadHints,
) -> list[Path]:
    """Write a standalone max_concurrency Workload Explorer package."""
    validate_sweep_workload(workload)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    bench_params = directory / "concurrency_bench_params.json"
    run_script = directory / "run_concurrency_sweep.sh"
    recommend_script = directory / "recommend_concurrency.py"

    config_rel = _relative_to(directory, config_path)
    env_rel = _relative_to(directory, env_path)
    request_model, tokenizer = _benchmark_models(config)

    _write_json(bench_params, build_concurrency_bench_params(workload))
    _write_concurrency_run_script(
        run_script,
        config_rel=config_rel,
        env_rel=env_rel,
        request_model=request_model,
        tokenizer=tokenizer,
        workload=workload,
    )
    _write_concurrency_recommend_script(recommend_script, workload=workload)
    analysis_files = _write_post_benchmark_analysis_files(directory)
    return [bench_params, run_script, recommend_script, *analysis_files]


def write_full_sweep_files(
    output_dir: str,
    *,
    config_path: str,
    env_path: str,
    config: dict[str, Any],
    workload: WorkloadHints,
    numa_node_count: int,
) -> list[Path]:
    """Write TP/DP -> concurrency -> scheduler end-to-end tuning artifacts."""
    files = write_parallel_layout_sweep_files(
        output_dir,
        config_path=config_path,
        env_path=env_path,
        config=config,
        workload=workload,
        numa_node_count=numa_node_count,
    )

    directory = Path(output_dir)
    concurrency_bench = directory / "concurrency_bench_params.json"
    run_concurrency = directory / "run_concurrency_sweep.sh"
    recommend_concurrency = directory / "recommend_concurrency.py"
    prepare_scheduler = directory / "prepare_scheduler.py"
    scheduler_runner = directory / "run_sweep.sh"
    recommend_scheduler = directory / "recommend.py"
    run_scheduler = directory / "run_scheduler_sweep.sh"
    run_full = directory / "run_full_sweep.sh"
    guide = directory / "SWEEP.md"

    env_rel = _relative_to(directory, env_path)
    request_model, tokenizer = _benchmark_models(config)

    _write_json(concurrency_bench, build_concurrency_bench_params(workload))
    _write_concurrency_run_script(
        run_concurrency,
        config_rel="parallel-layout-config.yml",
        env_rel=env_rel,
        request_model=request_model,
        tokenizer=tokenizer,
        workload=workload,
    )
    _write_concurrency_recommend_script(recommend_concurrency, workload=workload)

    # Full tuning owns the scheduler stage. The standalone parallel-layout mode
    # intentionally stops after TP/DP selection.
    _write_scheduler_prepare_script(prepare_scheduler)
    _write_run_script(
        scheduler_runner,
        config_rel="parallel-layout-sweep-config.yml",
        env_rel=env_rel,
        request_model=request_model,
        tokenizer=tokenizer,
        workload=workload,
    )
    _write_recommend_script(
        recommend_scheduler,
        config_rel="parallel-layout-config.yml",
        env_rel=env_rel,
        workload=workload,
    )
    _write_scheduler_wrapper(run_scheduler, workload=workload)

    _write_full_run_script(run_full)
    _write_full_guide(guide, workload)

    return [
        *files,
        concurrency_bench,
        run_concurrency,
        recommend_concurrency,
        prepare_scheduler,
        scheduler_runner,
        recommend_scheduler,
        run_scheduler,
        run_full,
    ]
