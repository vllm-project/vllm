# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL integration test runner.

Drives accuracy tests for the NIXL KV-cache connector by:

1. Loading test parameters from YAML config files (see ``configs/``).
2. Launching ``run_accuracy_test.sh`` with the appropriate environment.
3. Collecting pass/fail results and writing a summary.

Subsumes the functionality of ``config_sweep_accuracy_test.sh``.

Usage examples::

    # Run a single config
    python test_config.py tp_qwen_2p2d.yaml

    # Run all configs matching a glob
    python test_config.py 'tp_*'

    # Run all configs (default)
    python test_config.py
"""

import logging
import os
import signal
import subprocess
import sys
import time
import types
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Literal, NamedTuple

import yaml

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = SCRIPT_DIR / "configs"
RUN_ACCURACY_SCRIPT = str(SCRIPT_DIR / "run_accuracy_test.sh")

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ModelConfig(NamedTuple):
    supports_hma: bool = False


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "Qwen/Qwen3-0.6B": ModelConfig(),
    "deepseek-ai/deepseek-vl2-tiny": ModelConfig(),
    "deepseek-ai/deepseek-vl2-small": ModelConfig(),
    "deepseek-ai/DeepSeek-V2-Lite-Chat": ModelConfig(),
    "microsoft/Phi-4-mini-instruct": ModelConfig(),
    "google/gemma-3-4b-it": ModelConfig(supports_hma=True),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8": ModelConfig(supports_hma=True),
    "Qwen/Qwen3.5-35B-A3B": ModelConfig(supports_hma=True),
}


@dataclass
class TestConfig:
    """Parameters for a single accuracy test run.

    Every field corresponds to a key in the YAML config files under
    ``configs/``.  Only ``hf_model`` is required; all others have
    sensible defaults.
    """

    # HuggingFace model name or path (e.g. "Qwen/Qwen3-0.6B").
    hf_model: str

    # How many prefill / decode vllm-serve processes to launch.
    num_prefill_instances: int = 1
    num_decode_instances: int = 1

    # Tensor-parallel size for prefill instances.
    prefill_tp_size: int = 1

    # Parallelism degree for decode instances. Interpreted as
    # --tensor-parallel-size by default, or as --data-parallel-size
    # (with TP fixed to 1) when enable_dp_ep is set.
    decode_parallel_size: int = 1

    # KV-cache block size (tokens per block) for each role.
    prefill_block_size: int = 128
    decode_block_size: int = 128

    # Device that holds the NixlConnector KV transfer buffers.
    # "cuda" keeps them on GPU; "cpu" stages through host memory.
    kv_buffer_device: Literal["cuda", "cpu"] = "cuda"

    # Pack KV blocks from different layers together in the
    # NixlConnector transfer to reduce the number of RDMA ops.
    enable_cross_layer_blocks: bool = False

    # Memory layout for KV cache on the decoder side.
    # "HND" = [num_heads, seq_len, head_dim] (default).
    # "NHD" = [seq_len, num_heads, head_dim] (enables permute).
    decode_kv_layout: Literal["HND", "NHD"] = "HND"

    # Enable the Hybrid Memory Allocator for models that mix
    # attention and SSM layers (e.g. Nemotron-Nano, Jamba).
    use_hma: bool = False

    # Fraction of GPU memory reserved for the KV cache.
    gpu_memory_utilization: float = 0.2

    # Use data-parallel + expert-parallel on the decoder instead
    # of tensor-parallel. Intended for MoE models.
    enable_dp_ep: bool = False

    # Extra CLI flags forwarded verbatim to every vllm-serve process
    # (e.g. ["--max-model-len", "8192", "--attention-backend", "FLASHINFER"]).
    vllm_serve_extra_args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in (
            "num_prefill_instances",
            "num_decode_instances",
            "prefill_tp_size",
            "decode_parallel_size",
            "prefill_block_size",
            "decode_block_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")

        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0.0, 1.0]")

        if self.kv_buffer_device not in ("cuda", "cpu"):
            raise ValueError(
                f"kv_buffer_device must be 'cuda' or 'cpu', "
                f"got '{self.kv_buffer_device}'"
            )

        if self.decode_kv_layout not in ("HND", "NHD"):
            raise ValueError(
                f"decode_kv_layout must be 'HND' or 'NHD', "
                f"got '{self.decode_kv_layout}'"
            )

        if self.hf_model not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{self.hf_model}'. "
                f"Registered models: {sorted(MODEL_REGISTRY)}"
            )

        if self.use_hma:
            reg = MODEL_REGISTRY[self.hf_model]
            if not reg.supports_hma:
                supported = [
                    name for name, m in MODEL_REGISTRY.items() if m.supports_hma
                ]
                raise ValueError(
                    f"use_hma=True is not supported for model "
                    f"'{self.hf_model}'. "
                    f"Models with HMA support: {supported}"
                )

            if self.enable_dp_ep:
                raise ValueError("use_hma=True is incompatible with enable_dp_ep=True")

            if self.prefill_tp_size != self.decode_parallel_size:
                raise ValueError(
                    f"use_hma=True requires prefill_tp_size == "
                    f"decode_parallel_size, got {self.prefill_tp_size} "
                    f"vs {self.decode_parallel_size}"
                )

    def to_env(self) -> dict[str, str]:
        """Build the environment variables that ``run_accuracy_test.sh``
        reads.  The returned dict is meant to be merged into
        ``os.environ`` before calling the script. Optional knobs are
        always set to ensure that inherited parent env values cannot
        change the behavior of a YAML-defined test run."""
        env: dict[str, str] = {
            "MODEL_NAMES": self.hf_model,
            "NUM_PREFILL_INSTANCES": str(self.num_prefill_instances),
            "NUM_DECODE_INSTANCES": str(self.num_decode_instances),
            "PREFILLER_TP_SIZE": str(self.prefill_tp_size),
            "DECODER_TP_SIZE": str(self.decode_parallel_size),
            "PREFILL_BLOCK_SIZE": str(self.prefill_block_size),
            "DECODE_BLOCK_SIZE": str(self.decode_block_size),
            "GPU_MEMORY_UTILIZATION": str(self.gpu_memory_utilization),
            "DECODER_KV_LAYOUT": self.decode_kv_layout,
            "ENABLE_HMA_FLAG": "1" if self.use_hma else "",
            "DP_EP": "1" if self.enable_dp_ep else "",
            "VLLM_SERVE_EXTRA_ARGS": ",".join(self.vllm_serve_extra_args),
        }
        return env

    def to_cli_args(self) -> list[str]:
        """Build the CLI arguments for ``run_accuracy_test.sh``."""
        args: list[str] = []
        if self.kv_buffer_device != "cuda":
            args.extend(["--kv_buffer_device", self.kv_buffer_device])
        if self.enable_cross_layer_blocks:
            args.append("--enable-cross-layers")
        return args


# ---------------------------------------------------------------------------
# Config loading & discovery
# ---------------------------------------------------------------------------


def load_test_config(path: str | Path) -> TestConfig:
    """Load a ``TestConfig`` from a YAML file.

    All configuration must be specified in the YAML file -- environment
    variables are intentionally ignored so that test runs are
    reproducible."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return TestConfig(**data)


def discover_configs(pattern: str = "*.yaml") -> list[Path]:
    """Return YAML config paths matching a glob *pattern*.

    The pattern is matched against filenames inside the configs
    directory.  A ``.yaml`` suffix is appended automatically when
    the pattern doesn't already end with one.  Results are sorted
    alphabetically for deterministic ordering."""
    if not pattern.endswith(".yaml"):
        pattern += ".yaml"
    return sorted(CONFIGS_DIR.glob(pattern))


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

_active_proc: subprocess.Popen[bytes] | None = None


def _forward_sigterm(_signum: int, _frame: types.FrameType | None) -> None:
    """Forward SIGTERM to the running subprocess, then exit.

    The bash script's own ``trap … SIGTERM EXIT`` handles killing
    its ``vllm serve`` background workers."""
    if _active_proc is not None and _active_proc.poll() is None:
        _active_proc.send_signal(signal.SIGTERM)
        _active_proc.wait()
    sys.exit(1)


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------


def _run_test(
    cfg: TestConfig,
    log_file: Path,
) -> None:
    """Execute ``run_accuracy_test.sh`` with the given configuration.

    Stdout and stderr are written to separate files
    (``<stem>.stdout`` and ``<stem>.stderr``) so that only the
    runner's own progress is shown on the terminal.

    Uses ``Popen`` (not ``subprocess.run``) so that on SIGINT the
    bash script receives the signal via the shared process group and
    its ``trap 'kill $(jobs -pr)' SIGINT SIGTERM EXIT`` can clean up
    the ``vllm serve`` workers gracefully.  SIGTERM is forwarded
    explicitly by :func:`_forward_sigterm`."""
    global _active_proc
    env = {**os.environ, **cfg.to_env()}
    cmd = ["bash", RUN_ACCURACY_SCRIPT, *cfg.to_cli_args()]

    log_file.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = log_file.with_suffix(".stdout")
    stderr_path = log_file.with_suffix(".stderr")
    with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
        proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=err)
        _active_proc = proc
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.wait()
            raise
        finally:
            _active_proc = None

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


@dataclass
class TestResult:
    config_name: str
    log_path: Path
    status: str = "pending"
    elapsed_secs: float = 0.0
    error: str | None = None


def _run_one(
    cfg: TestConfig,
    label: str,
    log_file: Path,
) -> TestResult:
    """Run a single config, returning a :class:`TestResult`."""
    logger.info(
        "[START] %s  (model=%s, P-TP=%d, D-TP=%d)",
        label,
        cfg.hf_model,
        cfg.prefill_tp_size,
        cfg.decode_parallel_size,
    )
    result = TestResult(config_name=label, log_path=log_file)
    t0 = time.monotonic()
    try:
        _run_test(cfg, log_file=log_file)
        result.status = "passed"
    except Exception as exc:
        result.status = "failed"
        result.error = str(exc)
    finally:
        result.elapsed_secs = time.monotonic() - t0

    if result.status == "passed":
        logger.info("[PASS] %s completed in %.1fs", label, result.elapsed_secs)
    else:
        logger.error(
            "[FAIL] %s failed after %.1fs: %s", label, result.elapsed_secs, result.error
        )
    return result


def _log_summary(results: list[TestResult]) -> None:
    """Write a pass/fail summary table to the logger."""
    passed = [r for r in results if r.status == "passed"]
    failed = [r for r in results if r.status == "failed"]

    lines: list[str] = ["", "=" * 72, "SUMMARY", "=" * 72]
    for r in results:
        tag = "PASS" if r.status == "passed" else "FAIL"
        elapsed = f"{r.elapsed_secs:.1f}s"
        stem = r.log_path.with_suffix("")
        log_info = f"  logs: {stem}.{{stdout,stderr}}"
        lines.append(f"  [{tag}]  {r.config_name:<40} {elapsed:>8}{log_info}")
        if r.error:
            lines.append(f"         error: {r.error}")
    lines.append("-" * 72)
    lines.append(
        f"  Passed: {len(passed)}  |  Failed: {len(failed)}  |  Total: {len(results)}"
    )
    lines.append("=" * 72)
    logger.info("\n".join(lines))


def run_sweep(
    configs: dict[str, TestConfig],
    *,
    log_dir: Path,
) -> list[TestResult]:
    """Run *configs* sequentially.

    *configs* maps a human-readable label (typically the YAML
    filename) to a fully resolved :class:`TestConfig`.

    All configs are executed regardless of individual failures.
    Returns a list of :class:`TestResult` with per-config outcomes."""
    results: list[TestResult] = []

    logger.info("Running %d config(s)", len(configs))
    logger.info("Logs: %s", log_dir)

    for label, cfg in configs.items():
        log_file = log_dir / f"{Path(label).stem}.log"
        results.append(_run_one(cfg, label, log_file))

    _log_summary(results)
    return results


# ---------------------------------------------------------------------------
# Logging setup & CLI entry point
# ---------------------------------------------------------------------------


def _setup_logging(log_dir: Path) -> None:
    """Configure the module logger with a file handler and a stream handler.

    The file handler writes to ``<log_dir>/runner.log`` so that all
    progress and summary output is persisted alongside per-test logs."""
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "runner.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)


if __name__ == "__main__":
    import argparse

    signal.signal(signal.SIGTERM, _forward_sigterm)

    parser = argparse.ArgumentParser(
        description="Run nixl integration accuracy tests from YAML configs."
    )

    parser.add_argument(
        "pattern",
        nargs="?",
        default="*",
        metavar="PATTERN",
        help="Glob pattern to match YAML config files in the configs/ "
        "directory (e.g. 'tp_*', 'hybrid_*', 'tp_qwen_2p2d.yaml'). "
        "'.yaml' is appended automatically if omitted. "
        "Defaults to '*' (all configs).",
    )

    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory for per-config log files. "
        "Defaults to logs/<timestamp> next to this script.",
    )
    parser.add_argument(
        "--extra-vllm-args",
        type=str,
        default="",
        metavar="ARGS",
        help="Extra arguments forwarded to every vllm serve process, "
        "merged with any vllm_serve_extra_args in the YAML config. "
        "Use = syntax: --extra-vllm-args='--attention-backend FLASHINFER'",
    )

    args = parser.parse_args()

    cli_extra_args: list[str] = (
        args.extra_vllm_args.split() if args.extra_vllm_args else []
    )

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = SCRIPT_DIR / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    _setup_logging(log_dir)

    config_paths = discover_configs(args.pattern)
    if not config_paths:
        parser.error(f"No configs matching '{args.pattern}' in {CONFIGS_DIR}")

    configs: dict[str, TestConfig] = {}
    for path in config_paths:
        cfg = load_test_config(path)
        if cli_extra_args:
            cfg = replace(
                cfg,
                vllm_serve_extra_args=cfg.vllm_serve_extra_args + cli_extra_args,
            )
        configs[path.name] = cfg

    try:
        results = run_sweep(configs, log_dir=log_dir)
    except KeyboardInterrupt:
        sys.exit(130)

    if any(r.status == "failed" for r in results):
        sys.exit(1)
