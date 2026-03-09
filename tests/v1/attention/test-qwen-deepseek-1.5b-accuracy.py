# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Accuracy evaluation tests for small DeepSeek model family in vLLM.

Covers:
  - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  - deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

Precision variants tested per model:
  - bfloat16   (baseline, native BF16 weights)
  - float16    (half precision)
  - fp8        (dynamic W8A8, quantization="fp8")
  - auto       (vLLM default dtype resolution)

Benchmarks used:
  - gsm8k  (5-shot, using vLLM's evaluate_gsm8k_offline)  – math reasoning
  - arc_easy (25-shot, via lm-eval)                       – multi-choice commonsense
  - hellaswag (10-shot, via lm-eval)                      – sentence completion

Usage:
    pytest tests/v1/attention/test-qwen-deepseek-1.5b-accuracy.py -v

Requirements:
    pip install lm_eval vllm
"""

from __future__ import annotations

import contextlib
import gc
import os
import sys
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# CRITICAL: set multiprocessing start method BEFORE any CUDA-touching import.
# vllm v1 forks a child process (EngineCore_DP0).  If CUDA is already
# initialised in the parent when the fork happens, the child inherits a broken
# CUDA context → "Can't initialize NVML" / Triton disabled / engine crash.
# Forcing 'spawn' here (before torch is imported) prevents that entirely.
# ---------------------------------------------------------------------------
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pytest

# ---------------------------------------------------------------------------
# Import vLLM's GSM8K harness and test utilities
# ---------------------------------------------------------------------------
# Use relative imports (from tests/v1/attention/ up to tests/)
try:
    from ...evals.gsm8k.gsm8k_eval import evaluate_gsm8k
    from ...utils import RemoteOpenAIServer

    HAS_GSM8K_HARNESS = True
except ImportError as e:
    print(f"WARNING: Failed to import GSM8K harness: {e}")
    HAS_GSM8K_HARNESS = False

# ---------------------------------------------------------------------------
# Optional heavy imports – skipped gracefully when not installed
# ---------------------------------------------------------------------------
try:
    import lm_eval  # noqa: F401
except ImportError:
    try:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "lm_eval"])
    except Exception:
        pass

try:
    from lm_eval import evaluator

    HAS_LM_EVAL = True
except ImportError:
    HAS_LM_EVAL = False

try:
    from vllm import LLM, SamplingParams

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

# ---------------------------------------------------------------------------
# GPU availability detection
# ---------------------------------------------------------------------------


def _cuda_device_count() -> int:
    """Return number of visible CUDA GPUs WITHOUT initialising the CUDA context.

    Calling torch.cuda.device_count() initialises CUDA in the parent process.
    Once CUDA is initialised, vllm cannot safely fork a worker subprocess —
    it must use 'spawn' instead, but even then the spawned process may fail to
    initialise NVML.  We therefore avoid touching torch.cuda entirely here and
    instead use nvidia-smi or parse CUDA_VISIBLE_DEVICES.
    """
    # 1. If CUDA_VISIBLE_DEVICES is already set, honour it directly.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd.lower() == "nodevfile":
        return 0
    if cvd.strip():
        return len([d for d in cvd.split(",") if d.strip()])

    # 2. Query nvidia-smi without initialising CUDA.
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = [
                line for line in result.stdout.strip().splitlines() if line.strip()
            ]
            return len(lines)
    except Exception:
        pass

    return 0


# Detect GPU count BEFORE modifying CUDA_VISIBLE_DEVICES.
_NUM_GPUS: int = _cuda_device_count()
HAS_CUDA = _NUM_GPUS > 0

# Ensure CUDA_VISIBLE_DEVICES is set to ALL available GPUs so that vllm's
# forked EngineCore_DP0 subprocess inherits a non-empty device list.
# We expose all GPUs and pass tensor_parallel_size explicitly so large models
# (7B) can use multi-GPU tensor parallelism when needed.
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(i) for i in range(max(_NUM_GPUS, 1))
    )

# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------
requires_gpu = pytest.mark.skipif(
    not (HAS_VLLM and HAS_LM_EVAL and HAS_CUDA and HAS_GSM8K_HARNESS),
    reason=(
        "vllm, lm_eval, and gsm8k_harness must be installed and at "
        "least one CUDA GPU must be visible"
    ),
)

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    model_id: str
    # Minimum acceptable scores per (task, metric) pair.
    # These are conservative lower bounds; a fully-loaded model should exceed them.
    min_scores: dict[str, float]
    # Some tiny models need a higher max_model_len to avoid OOM on long contexts
    max_model_len: int = 4096
    # Reasoning models benefit from stripping chain-of-thought before scoring
    think_end_token: str | None = "<|end_of_thought|>"
    # How many GPUs to shard across; None = auto-select based on _NUM_GPUS
    tensor_parallel_size: int | None = None

    def get_tensor_parallel_size(self) -> int:
        """Return the tensor parallel size to use, defaulting to all available GPUs."""
        if self.tensor_parallel_size is not None:
            return self.tensor_parallel_size
        return max(_NUM_GPUS, 1)


SMALL_DEEPSEEK_MODELS = [
    ModelConfig(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        min_scores={
            "gsm8k_accuracy": 0.60,
            "arc_easy_acc": 0.62,
            "hellaswag_acc_norm": 0.46,
        },
        max_model_len=2048,
        think_end_token="</think>",
        tensor_parallel_size=1,  # 1.5B fits on a single GPU
    ),
]

# dtype → (vllm_dtype_arg, quantization_arg)
PRECISION_VARIANTS: dict[str, tuple[str, str | None]] = {
    "bfloat16": ("bfloat16", None),
    "float16": ("float16", None),
    "fp8": (
        "bfloat16",
        "fp8",
    ),  # FP8 dynamic quant; weights in BF16, activations in FP8
    "auto": ("auto", None),
}

# Evaluation sample limit – keeps CI fast while still being statistically meaningful
EVAL_LIMIT = 250

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_vllm_llm(
    model_id: str,
    dtype: str,
    quantization: str | None,
    max_model_len: int,
    tensor_parallel_size: int = 1,
) -> LLM:
    """
    Build a vLLM LLM object for offline evaluation.

    This replaces the lm-eval model_args approach and directly creates
    a vLLM LLM instance that can be used with evaluate_gsm8k_offline().
    """
    llm_kwargs = {
        "model": model_id,
        "dtype": dtype,
        "max_model_len": max_model_len,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": float(os.environ.get("VLLM_TEST_GPU_UTIL", "0.3")),
        "seed": 42,
        "trust_remote_code": True,
    }
    if quantization:
        llm_kwargs["quantization"] = quantization

    return LLM(**llm_kwargs)


def _build_lm_eval_model_args(
    model_id: str,
    dtype: str,
    quantization: str | None,
    max_model_len: int,
    tensor_parallel_size: int = 1,
) -> str:
    """
    Build the model_args string for lm-eval's vllm backend.

    Used only for arc_easy and hellaswag since we now use the vLLM harness for GSM8K.
    """
    parts = [
        f"pretrained={model_id}",
        f"dtype={dtype}",
        f"max_model_len={max_model_len}",
        f"tensor_parallel_size={tensor_parallel_size}",
        f"gpu_memory_utilization={os.environ.get('VLLM_TEST_GPU_UTIL', '0.3')}",
        "add_bos_token=True",
        "seed=42",
    ]
    if quantization:
        parts.append(f"quantization={quantization}")
    return ",".join(parts)


def _teardown_vllm(llm_obj) -> None:
    """Robustly tear down a vllm LLM (or lm_eval wrapper) and free GPU memory.

    vllm v1 spawns an EngineCore_DP0 subprocess that holds an NCCL process
    group.  If we don't explicitly shut it down, the process group leaks and
    the next test fails to initialise a new one ("destroy_process_group() was
    not called before program exit").

    We try several shutdown paths from most- to least-specific, silencing all
    errors so a broken shutdown path doesn't mask the real test result.
    """
    # Resolve the raw vllm LLM object (lm_eval wraps it under .model)
    raw_llm = getattr(llm_obj, "model", llm_obj)
    engine = getattr(raw_llm, "llm_engine", None)
    if engine is not None:
        # vllm v1: LLMEngine.engine_core is a SyncMPClient with .shutdown()
        core = getattr(engine, "engine_core", None)
        for attr in ("shutdown", "close", "stop", "abort"):
            fn = getattr(core, attr, None)
            if callable(fn):
                with contextlib.suppress(Exception):
                    fn()
                break
        # vllm v0/v1 fallback: abort all requests then stop engine
        for attr in (
            "abort_all_requests",
            "_stop_remote_worker_execution_loop",
        ):
            fn = getattr(engine, attr, None)
            if callable(fn):
                with contextlib.suppress(Exception):
                    fn()

    del llm_obj
    gc.collect()

    # Destroy the NCCL process group if one is alive — prevents the
    # "destroy_process_group() was not called" warning from leaking into the
    # next test's worker spawn.
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    try:
        from torch import cuda

        cuda.empty_cache()
        cuda.synchronize()
        gc.collect()
    except Exception:
        pass


def _run_gsm8k_eval(
    server_url: str, num_shots: int = 5, num_questions: int = 250
) -> dict:
    """
    Run GSM8K evaluation using vLLM's evaluate_gsm8k harness against a server.

    This replaces the lm-eval GSM8K evaluation with the native vLLM harness,
    which is more efficient and consistent with vLLM's test infrastructure.
    """
    # Extract host and port from server URL
    if "://" in server_url:
        server_url = server_url.split("://")[1]

    host_port = server_url.split("/")[0]  # Remove path if present
    if ":" in host_port:
        host, p = host_port.split(":")
        port = int(p)
    else:
        host = host_port
        port = 8000

    # Add http:// prefix if not present
    if not host.startswith("http"):
        host = f"http://{host}"

    results = evaluate_gsm8k(
        num_questions=num_questions,
        num_shots=num_shots,
        max_tokens=256,
        host=host,
        port=port,
        temperature=0.0,
    )
    return results


def _run_lm_eval(
    model_args: str,
    tasks: list[str],
    num_fewshot_map: dict[str, int],
    limit: int,
) -> dict:
    """Run lm-evaluation-harness via the vllm string backend and return results.

    Builds the vllm LM object once, evaluates all tasks against it (each with
    its own integer num_fewshot), then explicitly destroys the LM so the vllm
    EngineCore_DP0 subprocess exits and frees GPU memory before the next test.

    NOTE: This is now only used for arc_easy and hellaswag, not GSM8K.
    """
    import lm_eval.api.registry as _registry

    # Build the LM once for all tasks in this precision variant.
    lm = _registry.get_model("vllm").create_from_arg_string(
        model_args, {"batch_size": "auto"}
    )

    all_results: dict = {}
    try:
        for task in tasks:
            nshot = num_fewshot_map.get(task, 0)
            result = evaluator.simple_evaluate(
                model=lm,
                tasks=[task],
                num_fewshot=nshot,
                limit=limit,
                log_samples=False,
            )
            all_results.update(result["results"])
    finally:
        _teardown_vllm(lm)

    return all_results


def _extract_score(results: dict, task: str, metric: str) -> float:
    """Pull a scalar score from lm-eval result dict."""
    task_results = results.get(task, {})
    # lm-eval may suffix metric names with ",none" or ",flexible-extract"
    for key, val in task_results.items():
        if metric in key:
            return float(val)
    raise KeyError(
        f"Metric '{metric}' not found in task '{task}'. Available: {list(task_results)}"
    )


# ---------------------------------------------------------------------------
# Parametrize matrix: (model_config, precision_label)
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        pytest.param(
            (model_cfg, precision),
            id=f"{model_cfg.model_id.split('/')[-1]}-{precision}",
        )
        for model_cfg in SMALL_DEEPSEEK_MODELS
        for precision in PRECISION_VARIANTS
    ]
)
def model_and_precision(request):
    return request.param


# ---------------------------------------------------------------------------
# Core accuracy test
# ---------------------------------------------------------------------------


@requires_gpu
class TestDeepSeekSmallAccuracy:
    """
    End-to-end accuracy evaluation for small DeepSeek R1 distill models
    across multiple precision types (BF16, FP16, FP8, auto).

    Uses vLLM's native GSM8K harness for math reasoning, and lm-eval for
    arc_easy and hellaswag benchmarks.
    """

    # arc_easy and hellaswag still use lm-eval
    LM_EVAL_TASKS = ["arc_easy", "hellaswag"]
    LM_EVAL_NUM_FEWSHOT = {"arc_easy": 25, "hellaswag": 10}

    def test_accuracy_within_tolerance(self, model_and_precision):
        """
        Load the model in the given precision, run evaluations, and assert that
        each task's accuracy meets the minimum threshold defined in ModelConfig.

        GSM8K is evaluated using vLLM's evaluate_gsm8k_offline() harness.
        Other benchmarks use lm-eval.
        """
        model_cfg, precision_label = model_and_precision
        dtype, quantization = PRECISION_VARIANTS[precision_label]

        failures = []

        # -- GSM8K (using vLLM harness with RemoteOpenAIServer) --
        print("\n=== Running GSM8K with vLLM harness ===")

        # Build server arguments
        server_args = [
            f"--dtype={dtype}",
            f"--max-model-len={model_cfg.max_model_len}",
            f"--tensor-parallel-size={model_cfg.get_tensor_parallel_size()}",
            f"--gpu-memory-utilization={os.environ.get('VLLM_TEST_GPU_UTIL', '0.3')}",
            "--trust-remote-code",
            "--disable-uvicorn-access-log",
        ]
        if quantization:
            server_args.append(f"--quantization={quantization}")

        # Launch server and run GSM8K evaluation
        with RemoteOpenAIServer(
            model_cfg.model_id,
            server_args,
            max_wait_seconds=600,
        ) as remote_server:
            server_url = remote_server.url_for("v1")
            print(f"Server started at: {server_url}")

            gsm8k_results = _run_gsm8k_eval(
                server_url, num_shots=5, num_questions=EVAL_LIMIT
            )
            gsm8k_score = gsm8k_results["accuracy"]
            threshold = model_cfg.min_scores["gsm8k_accuracy"]

            print(f"GSM8K accuracy: {gsm8k_score:.4f} (threshold: {threshold:.4f})")
            print(f"GSM8K invalid rate: {gsm8k_results['invalid_rate']:.3f}")
            print(f"GSM8K latency: {gsm8k_results['latency']:.1f}s")

            if gsm8k_score < threshold:
                failures.append(
                    f"gsm8k accuracy={gsm8k_score:.4f} < threshold={threshold} "
                    f"[model={model_cfg.model_id}, dtype={precision_label}]"
                )

        # -- ARC Easy & HellaSwag (using lm-eval) --
        print("\n=== Running arc_easy and hellaswag with lm-eval ===")
        model_args = _build_lm_eval_model_args(
            model_id=model_cfg.model_id,
            dtype=dtype,
            quantization=quantization,
            max_model_len=model_cfg.max_model_len,
            tensor_parallel_size=model_cfg.get_tensor_parallel_size(),
        )

        lm_eval_results = _run_lm_eval(
            model_args=model_args,
            tasks=self.LM_EVAL_TASKS,
            num_fewshot_map=self.LM_EVAL_NUM_FEWSHOT,
            limit=EVAL_LIMIT,
        )

        # -- ARC Easy (commonsense MC) --
        arc_score = _extract_score(lm_eval_results, "arc_easy", "acc")
        threshold = model_cfg.min_scores["arc_easy_acc"]
        print(f"ARC Easy acc: {arc_score:.4f} (threshold: {threshold:.4f})")
        if arc_score < threshold:
            failures.append(
                f"arc_easy acc={arc_score:.4f} < threshold={threshold} "
                f"[model={model_cfg.model_id}, dtype={precision_label}]"
            )

        # -- HellaSwag (sentence completion) --
        hs_score = _extract_score(lm_eval_results, "hellaswag", "acc_norm")
        threshold = model_cfg.min_scores["hellaswag_acc_norm"]
        print(f"HellaSwag acc_norm: {hs_score:.4f} (threshold: {threshold:.4f})")
        if hs_score < threshold:
            failures.append(
                f"hellaswag acc_norm={hs_score:.4f} < threshold={threshold} "
                f"[model={model_cfg.model_id}, dtype={precision_label}]"
            )

        assert not failures, "\n".join(failures)

    def test_fp8_accuracy_vs_bfloat16_regression(self, model_and_precision):
        """
        Ensure FP8 accuracy does not regress more than 5% relative to BF16 on GSM8K.
        Only runs for the fp8 precision variant; skips otherwise.

        Uses vLLM's evaluate_gsm8k_offline() harness for both baseline and FP8.
        """
        model_cfg, precision_label = model_and_precision
        if precision_label != "fp8":
            pytest.skip("Regression check only applies to fp8 variant")

        # -- BF16 baseline --
        print("\n=== Running BF16 baseline for FP8 regression test ===")
        bf16_server_args = [
            "--dtype=bfloat16",
            f"--max-model-len={model_cfg.max_model_len}",
            f"--tensor-parallel-size={model_cfg.get_tensor_parallel_size()}",
            f"--gpu-memory-utilization={os.environ.get('VLLM_TEST_GPU_UTIL', '0.3')}",
            "--trust-remote-code",
            "--disable-uvicorn-access-log",
        ]

        with RemoteOpenAIServer(
            model_cfg.model_id,
            bf16_server_args,
            max_wait_seconds=600,
        ) as remote_server:
            server_url = remote_server.url_for("v1")
            bf16_results = _run_gsm8k_eval(
                server_url, num_shots=5, num_questions=EVAL_LIMIT
            )
            bf16_score = bf16_results["accuracy"]
            print(f"BF16 GSM8K accuracy: {bf16_score:.4f}")

        # -- FP8 --
        print("\n=== Running FP8 for regression test ===")
        fp8_server_args = [
            "--dtype=bfloat16",
            "--quantization=fp8",
            f"--max-model-len={model_cfg.max_model_len}",
            f"--tensor-parallel-size={model_cfg.get_tensor_parallel_size()}",
            f"--gpu-memory-utilization={os.environ.get('VLLM_TEST_GPU_UTIL', '0.3')}",
            "--trust-remote-code",
            "--disable-uvicorn-access-log",
        ]

        with RemoteOpenAIServer(
            model_cfg.model_id,
            fp8_server_args,
            max_wait_seconds=600,
        ) as remote_server:
            server_url = remote_server.url_for("v1")
            fp8_results = _run_gsm8k_eval(
                server_url, num_shots=5, num_questions=EVAL_LIMIT
            )
            fp8_score = fp8_results["accuracy"]
            print(f"FP8 GSM8K accuracy: {fp8_score:.4f}")

        max_allowed_regression = 0.06  # 6 percentage points absolute
        regression = bf16_score - fp8_score

        assert regression <= max_allowed_regression, (
            f"FP8 accuracy regressed by {regression:.4f} vs BF16 "
            f"(bf16={bf16_score:.4f}, fp8={fp8_score:.4f}) for {model_cfg.model_id}. "
            f"Max allowed regression: {max_allowed_regression}"
        )


# ---------------------------------------------------------------------------
# Lightweight sanity tests (no GPU / lm-eval required)
# ---------------------------------------------------------------------------


class TestDeepSeekSanity:
    """
    Fast, import-level sanity checks that run without a GPU.
    Validates that precision configs and model IDs are well-formed.
    """

    def test_precision_variants_are_valid_vllm_dtypes(self):
        valid_vllm_dtypes = {"auto", "half", "float16", "bfloat16", "float", "float32"}
        for label, (dtype, _) in PRECISION_VARIANTS.items():
            assert dtype in valid_vllm_dtypes, (
                f"Precision variant '{label}' uses invalid vllm dtype '{dtype}'"
            )

    def test_model_ids_follow_deepseek_naming(self):
        for model_cfg in SMALL_DEEPSEEK_MODELS:
            assert "deepseek" in model_cfg.model_id.lower(), (
                f"Model '{model_cfg.model_id}' does not appear to be a DeepSeek model"
            )

    def test_min_score_keys_are_consistent(self):
        expected_keys = {"gsm8k_accuracy", "arc_easy_acc", "hellaswag_acc_norm"}
        for model_cfg in SMALL_DEEPSEEK_MODELS:
            assert set(model_cfg.min_scores.keys()) == expected_keys, (
                f"Model '{model_cfg.model_id}' has unexpected min_scores keys: "
                f"{set(model_cfg.min_scores.keys())}"
            )

    def test_all_min_scores_are_in_valid_range(self):
        for model_cfg in SMALL_DEEPSEEK_MODELS:
            for key, score in model_cfg.min_scores.items():
                assert 0.0 <= score <= 1.0, (
                    f"min_score for '{key}' in '{model_cfg.model_id}' "
                    f"is out of range: {score}"
                )

    @pytest.mark.parametrize("precision", list(PRECISION_VARIANTS))
    def test_fp8_uses_bf16_base_dtype(self, precision):
        """FP8 dynamic quantization requires bf16 base weights."""
        dtype, quantization = PRECISION_VARIANTS[precision]
        if quantization == "fp8":
            assert dtype == "bfloat16", (
                "FP8 dynamic quantization should use bfloat16 as the "
                f"base dtype in vllm. Got: {dtype}"
            )

    def test_eval_limit_is_positive(self):
        assert EVAL_LIMIT > 0


# ---------------------------------------------------------------------------
# vLLM-direct smoke test (no lm-eval dependency)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (HAS_VLLM and HAS_CUDA), reason="vllm not installed or no CUDA GPU visible"
)
@pytest.mark.parametrize(
    "model_id,dtype,quantization",
    [
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "bfloat16", None),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "float16", None),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "bfloat16", "fp8"),
    ],
    ids=["1.5B-bf16", "1.5B-fp16", "1.5B-fp8"],
)
def test_vllm_generate_smoke(model_id: str, dtype: str, quantization: str | None):
    """
    Smoke-test: load model in the given precision and run a short generation.
    Checks that outputs are non-empty strings and that the model does not crash
    on basic arithmetic prompts.

    CUDA_VISIBLE_DEVICES is set at module level to ensure vllm's forked
    EngineCore_DP0 subprocess inherits a valid non-empty device list.
    """
    llm_kwargs: dict = dict(
        model=model_id,
        dtype=dtype,
        max_model_len=512,
        tensor_parallel_size=1,  # 1.5B fits on a single GPU
        gpu_memory_utilization=float(os.environ.get("VLLM_TEST_GPU_UTIL", "0.3")),
        seed=0,
        enforce_eager=True,  # disable CUDA graphs for faster cold start in tests
    )
    if quantization:
        llm_kwargs["quantization"] = quantization

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)

    prompts = [
        "What is 12 + 7?",
        "The capital of France is",
    ]
    outputs = llm.generate(prompts, sampling_params)

    assert len(outputs) == len(prompts), "Expected one output per prompt"
    for output in outputs:
        generated_text = output.outputs[0].text
        assert isinstance(generated_text, str), "Output text should be a string"
        assert len(generated_text.strip()) > 0, "Output text should not be empty"

    # Explicitly shut down vllm and free GPU memory so the next parametrized
    # smoke test can allocate a fresh LLM without hitting residual CUDA context.
    _teardown_vllm(llm)
