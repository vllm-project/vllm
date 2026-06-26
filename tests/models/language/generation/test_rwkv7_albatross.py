# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RWKV7 dynamic correctness checks against the Albatross reference path.

These tests are opt-in because they require an Albatross checkout, a matching
Albatross `.pth` checkpoint, CUDA, and a vLLM-loadable checkpoint with the same
weights. Use `tests/models/language/generation/run_rwkv7_albatross.sh` to load
the local environment and fail fast when required model paths are missing. They
default to eager and decode CUDAGraph vLLM execution. RWKV7 does not support
torch.compile. The CUDAGraph mode is limited to the Albatross-style fixed-buffer
decode path. These tests intentionally do not test registry/import presence.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import urlparse

import pytest
import requests
import torch

from vllm import SamplingParams
from vllm.config.compilation import CUDAGraphMode

TOP_K = int(os.environ.get("RWKV7_ALBATROSS_TOP_K", "20"))
LOGPROB_ATOL = float(os.environ.get("RWKV7_ALBATROSS_LOGPROB_ATOL", "0.20"))
LOSS_MEAN_ATOL = float(os.environ.get("RWKV7_ALBATROSS_LOSS_MEAN_ATOL", "0.03"))
LOSS_MAX_ATOL = float(os.environ.get("RWKV7_ALBATROSS_LOSS_MAX_ATOL", "0.25"))
MAX_MODEL_LEN = int(os.environ.get("RWKV7_ALBATROSS_MAX_MODEL_LEN", "1024"))
GPU_MEMORY_UTILIZATION = float(
    os.environ.get("RWKV7_ALBATROSS_GPU_MEMORY_UTILIZATION", "0.70")
)
ENABLE_FLASHINFER_AUTOTUNE = (
    os.environ.get("RWKV7_ALBATROSS_ENABLE_FLASHINFER_AUTOTUNE", "0") == "1"
)
EXECUTION_MODE_NAMES = tuple(
    mode.strip()
    for mode in os.environ.get(
        "RWKV7_ALBATROSS_EXECUTION_MODES", "eager,cudagraph"
    ).split(",")
    if mode.strip()
)

PROMPTS = [
    {
        "name": "english",
        "text": "User: Explain why recurrent state matters in RWKV.\nAssistant:",
    },
    {
        "name": "chinese",
        "text": "用户：用两句话解释 RWKV 的 state 为什么不能串请求。\n助手：",
    },
    {
        "name": "code",
        "text": (
            "def rwkv_state_debug(requests):\n    for request in requests:\n        "
        ),
    },
    {
        "name": "math",
        "text": "Question: If x + 2 = 7, what is x?\nAnswer:",
    },
    {
        "name": "long_context",
        "text": "RWKV state alignment check. " * 80 + "\nConclusion:",
    },
]

MMLU_STYLE_CASES = [
    {
        "name": "math_choice",
        "prompt": (
            "User: You are a very talented expert in elementary math. "
            "Answer this question:\n"
            "What is 2 + 2?\n"
            "A. 3\n"
            "B. 4\n"
            "C. 5\n"
            "D. 6\n\n"
            "Assistant: The answer is"
        ),
        "choices": [" A", " B", " C", " D"],
    },
    {
        "name": "science_choice",
        "prompt": (
            "User: You are a very talented expert in science. "
            "Answer this question:\n"
            "Which object orbits the Earth?\n"
            "A. The Moon\n"
            "B. Mars\n"
            "C. The Sun\n"
            "D. Venus\n\n"
            "Assistant: The answer is"
        ),
        "choices": [" A", " B", " C", " D"],
    },
]


@dataclass(frozen=True)
class AlbatrossSettings:
    root: Path
    impl_dir: Path
    checkpoint: Path
    vllm_model: str
    max_model_len: int


@dataclass(frozen=True)
class ParallelSettings:
    tensor_parallel_size: int
    pipeline_parallel_size: int


def _positive_int_from_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _parallel_settings_from_env() -> ParallelSettings:
    return ParallelSettings(
        tensor_parallel_size=_positive_int_from_env(
            "RWKV7_ALBATROSS_TENSOR_PARALLEL_SIZE", 1
        ),
        pipeline_parallel_size=_positive_int_from_env(
            "RWKV7_ALBATROSS_PIPELINE_PARALLEL_SIZE", 1
        ),
    )


PARALLEL_SETTINGS = _parallel_settings_from_env()


@dataclass(frozen=True)
class ExecutionMode:
    name: str
    enforce_eager: bool
    env: Mapping[str, str]


def _execution_modes() -> tuple[Any, ...]:
    modes: list[Any] = []
    for mode in EXECUTION_MODE_NAMES:
        if mode == "eager":
            modes.append(
                pytest.param(
                    ExecutionMode(
                        name=mode,
                        enforce_eager=True,
                        env={},
                    ),
                    id=mode,
                )
            )
        elif mode in ("none", "cudagraph"):
            modes.append(
                pytest.param(
                    ExecutionMode(
                        name=mode,
                        enforce_eager=False,
                        env={"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"},
                    ),
                    id=mode,
                )
            )
        else:
            msg = (
                "RWKV7_ALBATROSS_EXECUTION_MODES only supports "
                f"'eager', 'none', and 'cudagraph', got {mode!r}"
            )
            raise ValueError(msg)
    if not modes:
        raise ValueError("RWKV7_ALBATROSS_EXECUTION_MODES cannot be empty")
    return tuple(modes)


@pytest.fixture(
    scope="module",
    params=_execution_modes(),
)
def rwkv7_execution_mode(request: pytest.FixtureRequest) -> ExecutionMode:
    return cast(ExecutionMode, request.param)


@contextmanager
def _temporary_env(overrides: Mapping[str, str]) -> Iterator[None]:
    old_values = {name: os.environ.get(name) for name in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for name, value in old_values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _require_cuda_devices(parallel_settings: ParallelSettings) -> None:
    required = (
        parallel_settings.tensor_parallel_size
        * parallel_settings.pipeline_parallel_size
    )
    available = torch.accelerator.device_count()
    if available < required:
        pytest.fail(
            "RWKV7 Albatross alignment requires at least "
            f"{required} CUDA devices for tensor_parallel_size="
            f"{parallel_settings.tensor_parallel_size} and "
            f"pipeline_parallel_size={parallel_settings.pipeline_parallel_size}; "
            f"only {available} CUDA device(s) are visible."
        )


def test_rwkv7_parallel_settings_default_to_single_rank(monkeypatch) -> None:
    monkeypatch.delenv("RWKV7_ALBATROSS_TENSOR_PARALLEL_SIZE", raising=False)
    monkeypatch.delenv("RWKV7_ALBATROSS_PIPELINE_PARALLEL_SIZE", raising=False)

    assert _parallel_settings_from_env() == ParallelSettings(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )


def test_rwkv7_parallel_settings_read_from_env(monkeypatch) -> None:
    monkeypatch.setenv("RWKV7_ALBATROSS_TENSOR_PARALLEL_SIZE", "2")
    monkeypatch.setenv("RWKV7_ALBATROSS_PIPELINE_PARALLEL_SIZE", "3")

    assert _parallel_settings_from_env() == ParallelSettings(
        tensor_parallel_size=2,
        pipeline_parallel_size=3,
    )


def test_rwkv7_vllm_outputs_passes_parallel_settings_to_runner(monkeypatch) -> None:
    settings = AlbatrossSettings(
        root=Path("/unused"),
        impl_dir=Path("/unused/impl"),
        checkpoint=Path("/unused/model.pth"),
        vllm_model="/unused/model.pth",
        max_model_len=128,
    )
    execution_mode = ExecutionMode(
        name="none",
        enforce_eager=False,
        env={"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"},
    )
    parallel_settings = ParallelSettings(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )
    monkeypatch.setattr(sys.modules[__name__], "PARALLEL_SETTINGS", parallel_settings)
    runner_calls: list[dict[str, Any]] = []

    class FakeLLM:
        def generate(self, prompts: list[dict[str, list[int]]], sampling_params):
            return [(prompts, sampling_params)]

    class FakeRunner:
        def __enter__(self):
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def get_llm(self) -> FakeLLM:
            return FakeLLM()

    def fake_vllm_runner(model: str, **kwargs: Any) -> FakeRunner:
        runner_calls.append({"model": model, "kwargs": kwargs})
        return FakeRunner()

    _vllm_outputs(
        fake_vllm_runner,
        settings,
        execution_mode,
        [[1, 2, 3]],
        max_tokens=1,
    )

    assert runner_calls[0]["kwargs"]["tensor_parallel_size"] == 2
    assert runner_calls[0]["kwargs"]["pipeline_parallel_size"] == 1


def test_rwkv7_vllm_outputs_disables_cudagraph_for_pipeline_parallel(
    monkeypatch,
) -> None:
    settings = AlbatrossSettings(
        root=Path("/unused"),
        impl_dir=Path("/unused/impl"),
        checkpoint=Path("/unused/model.pth"),
        vllm_model="/unused/model.pth",
        max_model_len=128,
    )
    execution_mode = ExecutionMode(
        name="cudagraph",
        enforce_eager=False,
        env={"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"},
    )
    parallel_settings = ParallelSettings(
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
    )
    monkeypatch.setattr(sys.modules[__name__], "PARALLEL_SETTINGS", parallel_settings)
    runner_calls: list[dict[str, Any]] = []

    class FakeLLM:
        def generate(self, prompts: list[dict[str, list[int]]], sampling_params):
            return [(prompts, sampling_params)]

    class FakeRunner:
        def __enter__(self):
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def get_llm(self) -> FakeLLM:
            return FakeLLM()

    def fake_vllm_runner(model: str, **kwargs: Any) -> FakeRunner:
        runner_calls.append({"model": model, "kwargs": kwargs})
        return FakeRunner()

    _vllm_outputs(
        fake_vllm_runner,
        settings,
        execution_mode,
        [[1, 2, 3]],
        max_tokens=1,
    )

    kwargs = runner_calls[0]["kwargs"]
    assert kwargs["distributed_executor_backend"] == "mp"
    assert kwargs["compilation_config"] == {
        "cudagraph_mode": CUDAGraphMode.NONE,
        "cudagraph_capture_sizes": [],
    }


def test_rwkv7_server_args_include_parallel_settings() -> None:
    settings = AlbatrossSettings(
        root=Path("/unused"),
        impl_dir=Path("/unused/impl"),
        checkpoint=Path("/unused/model.pth"),
        vllm_model="/unused/model.pth",
        max_model_len=128,
    )
    execution_mode = ExecutionMode(name="cudagraph", enforce_eager=False, env={})
    parallel_settings = ParallelSettings(
        tensor_parallel_size=2,
        pipeline_parallel_size=3,
    )

    server_args = _server_args(settings, execution_mode, parallel_settings, port=1234)

    assert server_args[server_args.index("--tensor-parallel-size") + 1] == "2"
    assert server_args[server_args.index("--pipeline-parallel-size") + 1] == "3"


def test_rwkv7_albatross_settings_preserve_http_model_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "albatross"
    impl_dir = root / "faster3a_2605"
    impl_dir.mkdir(parents=True)
    checkpoint = tmp_path / "rwkv7-g1g-1.5b-20260526-ctx8192.pth"
    checkpoint.touch()
    model_url = (
        "https://huggingface.co/BlinkDL/rwkv7-g1/blob/main/"
        "rwkv7-g1g-1.5b-20260526-ctx8192.pth?download=1"
    )

    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.accelerator,
        "current_accelerator",
        lambda: SimpleNamespace(type="cuda"),
    )
    monkeypatch.setattr(sys.modules[__name__], "_require_cuda_devices", lambda _: None)
    monkeypatch.setenv("ALBATROSS_ROOT", str(root))
    monkeypatch.setenv("ALBATROSS_IMPL", "faster3a_2605")
    monkeypatch.setenv("ALBATROSS_PTH", str(checkpoint))
    monkeypatch.setenv("VLLM_RWKV7_MODEL", model_url)
    monkeypatch.setenv("VLLM_RWKV7_WKV_MODE", "fp32io16")

    settings = rwkv7_albatross_settings.__wrapped__()

    assert settings.vllm_model == model_url


@pytest.fixture(scope="module")
def rwkv7_albatross_settings() -> AlbatrossSettings:
    if (
        not torch.accelerator.is_available()
        or torch.accelerator.current_accelerator().type != "cuda"
    ):
        pytest.skip("CUDA is required for RWKV7 Albatross alignment tests")
    _require_cuda_devices(PARALLEL_SETTINGS)
    if os.environ.get("VLLM_RWKV7_WKV_MODE", "fp32io16") != "fp32io16":
        pytest.skip("RWKV7 Albatross alignment requires VLLM_RWKV7_WKV_MODE=fp32io16")

    root = Path(
        os.environ.get(
            "ALBATROSS_ROOT", str(Path.home() / "Projects/MachineLearning/albatross")
        )
    ).expanduser()
    impl = os.environ.get("ALBATROSS_IMPL", "faster3a_2605")
    impl_dir = root / impl
    checkpoint = Path(os.environ.get("ALBATROSS_PTH", "")).expanduser()
    vllm_model_env = os.environ.get("VLLM_RWKV7_MODEL", "")
    parsed_vllm_model = urlparse(vllm_model_env)
    is_vllm_model_url = parsed_vllm_model.scheme in ("http", "https")
    vllm_model = None if is_vllm_model_url else Path(vllm_model_env).expanduser()

    if not impl_dir.is_dir():
        pytest.skip(f"Albatross implementation directory not found: {impl_dir}")
    if not checkpoint.is_file():
        pytest.skip("Set ALBATROSS_PTH to a matching RWKV7 .pth checkpoint")
    if not vllm_model_env:
        pytest.skip("Set VLLM_RWKV7_MODEL to the matching vLLM checkpoint")
    if not is_vllm_model_url and vllm_model is not None and not vllm_model.exists():
        pytest.skip(f"vLLM checkpoint path not found: {vllm_model}")

    return AlbatrossSettings(
        root=root,
        impl_dir=impl_dir,
        checkpoint=checkpoint,
        vllm_model=vllm_model_env if is_vllm_model_url else str(vllm_model),
        max_model_len=MAX_MODEL_LEN,
    )


@pytest.fixture(scope="module")
def albatross_oracle(
    rwkv7_albatross_settings: AlbatrossSettings,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    work_dir = tmp_path_factory.mktemp("rwkv7_albatross")
    request_path = work_dir / "request.json"
    response_path = work_dir / "response.json"
    request_path.write_text(
        json.dumps(
            {
                "prompts": PROMPTS,
                "mmlu_style_cases": MMLU_STYLE_CASES,
                "max_new_tokens": 32,
                "top_k": TOP_K,
            }
        ),
        encoding="utf-8",
    )

    code = _albatross_oracle_code()
    env = os.environ.copy()
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(rwkv7_albatross_settings.impl_dir),
            str(rwkv7_albatross_settings.checkpoint),
            str(request_path),
            str(response_path),
        ],
        cwd=rwkv7_albatross_settings.impl_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Albatross oracle failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return json.loads(response_path.read_text(encoding="utf-8"))


def _albatross_oracle_code() -> str:
    return textwrap.dedent(
        r"""
        import json
        import sys

        import torch
        import torch.nn.functional as F
        from rwkv.utils import PIPELINE

        impl_dir, checkpoint, request_path, response_path = sys.argv[1:5]
        sys.path.insert(0, impl_dir)

        import rwkv7_fast_v3a as v3a

        with open(request_path, "r", encoding="utf-8") as f:
            request = json.load(f)

        v3a.MODEL_PATH = checkpoint
        v3a.WKV_MODE = "fp32io16"
        v3a.EMB_DEVICE = "cpu"
        v3a.RKV_MODE = "off"
        v3a.CMIX_SPARSE = "no-fc"
        v3a.LOWRANK_WEIGHT = "both"
        v3a.ORIG_LINEAR_GROUPS = v3a.parse_orig_linear_groups(
            "att_c2c,ffn_key,head"
        )
        v3a.load_extensions(v3a.WKV_MODE)
        model = v3a.RWKV7()
        tokenizer = PIPELINE(model, "rwkv_vocab_v20230424")
        token_device = "cpu" if model.emb_cpu else "cuda"

        def top_logprobs(logits, top_k):
            logprobs = F.log_softmax(logits.float(), dim=-1)
            values, ids = torch.topk(logprobs, k=top_k)
            return [
                [int(token_id), float(value)]
                for token_id, value in zip(ids.detach().cpu(), values.detach().cpu())
            ]

        def prompt_losses(ids):
            if len(ids) < 2:
                return []
            state = model.zero_state(1)
            tokens = torch.tensor(
                ids[:-1], dtype=torch.long, device=token_device
            ).view(1, -1)
            targets = torch.tensor(ids[1:], dtype=torch.long, device="cuda")
            logits = model.forward_all_logits(tokens, state).squeeze(0).float()
            return [
                float(x)
                for x in F.cross_entropy(logits, targets, reduction="none")
                .detach()
                .cpu()
            ]

        def greedy(ids, max_new_tokens):
            state = model.zero_state(1)
            tokens = torch.tensor(
                ids, dtype=torch.long, device=token_device
            ).view(1, -1)
            logits = model.forward(tokens, state).view(-1)
            out = []
            for _ in range(max_new_tokens):
                token_id = int(torch.argmax(logits.float()).item())
                out.append(token_id)
                token = torch.tensor(
                    [[token_id]], dtype=torch.long, device=token_device
                )
                logits = model.forward(token, state).view(-1)
            return out

        def choice_logprobs(logits, choice_ids):
            logprobs = F.log_softmax(logits.float(), dim=-1)
            return [
                [int(token_id), float(logprobs[int(token_id)].detach().cpu())]
                for token_id in choice_ids
            ]

        cases = []
        for item in request["prompts"]:
            ids = tokenizer.encode(item["text"])
            state = model.zero_state(1)
            tokens = torch.tensor(
                ids, dtype=torch.long, device=token_device
            ).view(1, -1)
            logits = model.forward(tokens, state).view(-1)
            losses = prompt_losses(ids)
            cases.append(
                {
                    "name": item["name"],
                    "prompt": item["text"],
                    "prompt_token_ids": ids,
                    "next_top_logprobs": top_logprobs(logits, request["top_k"]),
                    "greedy_token_ids": greedy(ids, request["max_new_tokens"]),
                    "prompt_losses": losses,
                    "prompt_mean_loss": (
                        sum(losses) / len(losses) if len(losses) > 0 else 0.0
                    ),
                }
            )

        mmlu_cases = []
        for item in request["mmlu_style_cases"]:
            ids = tokenizer.encode(item["prompt"])
            choice_token_ids = [tokenizer.encode(choice) for choice in item["choices"]]
            if not all(len(choice) == 1 for choice in choice_token_ids):
                raise RuntimeError(f"MMLU choices must be single tokens: {item}")
            choice_token_ids = [choice[0] for choice in choice_token_ids]
            state = model.zero_state(1)
            tokens = torch.tensor(
                ids, dtype=torch.long, device=token_device
            ).view(1, -1)
            logits = model.forward(tokens, state).view(-1)
            choices = choice_logprobs(logits, choice_token_ids)
            pred = max(choices, key=lambda item: item[1])[0]
            mmlu_cases.append(
                {
                    "name": item["name"],
                    "prompt_token_ids": ids,
                    "choice_token_ids": choice_token_ids,
                    "choice_logprobs": choices,
                    "pred_token_id": pred,
                }
            )

        with open(response_path, "w", encoding="utf-8") as f:
            json.dump({"cases": cases, "mmlu_style_cases": mmlu_cases}, f)
        """
    )


def _vllm_outputs(
    vllm_runner,
    settings: AlbatrossSettings,
    execution_mode: ExecutionMode,
    prompt_token_ids: list[list[int]],
    *,
    max_tokens: int,
    logprobs: int | None = None,
    prompt_logprobs: int | None = None,
    logprob_token_ids: list[int] | None = None,
    enable_chunked_prefill: bool = False,
    max_num_batched_tokens: int | None = None,
) -> list[Any]:
    kwargs: dict[str, Any] = {
        "enforce_eager": execution_mode.enforce_eager,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "kernel_config": {
            "enable_flashinfer_autotune": ENABLE_FLASHINFER_AUTOTUNE,
        },
        "max_model_len": settings.max_model_len,
        "enable_chunked_prefill": enable_chunked_prefill,
        "tensor_parallel_size": PARALLEL_SETTINGS.tensor_parallel_size,
        "pipeline_parallel_size": PARALLEL_SETTINGS.pipeline_parallel_size,
    }
    if PARALLEL_SETTINGS.pipeline_parallel_size > 1:
        kwargs["distributed_executor_backend"] = "mp"
        kwargs["compilation_config"] = {
            "cudagraph_mode": CUDAGraphMode.NONE,
            "cudagraph_capture_sizes": [],
        }
    if max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    if execution_mode.name == "none":
        kwargs["compilation_config"] = {
            "cudagraph_mode": CUDAGraphMode.NONE,
            "cudagraph_capture_sizes": [],
        }

    with (
        _temporary_env(execution_mode.env),
        vllm_runner(settings.vllm_model, **kwargs) as runner,
    ):
        llm = runner.get_llm()
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            logprob_token_ids=logprob_token_ids,
        )
        return llm.generate(
            [{"prompt_token_ids": ids} for ids in prompt_token_ids],
            sampling_params=sampling_params,
        )


def _sorted_logprobs(
    sample_logprobs: Mapping[int, Any],
) -> list[tuple[int, float]]:
    return sorted(
        (
            (int(token_id), float(logprob.logprob))
            for token_id, logprob in sample_logprobs.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )


def _assert_next_token_logprobs_close(
    *,
    case: dict[str, Any],
    vllm_logprobs: Mapping[int, Any],
) -> None:
    albatross_top = [
        (int(token_id), float(value)) for token_id, value in case["next_top_logprobs"]
    ]
    vllm_top = _sorted_logprobs(vllm_logprobs)

    assert vllm_top[0][0] == albatross_top[0][0], (
        f"{case['name']}: top-1 mismatch: "
        f"albatross={albatross_top[:5]} vllm={vllm_top[:5]}"
    )

    albatross_top10 = {token_id for token_id, _ in albatross_top[:10]}
    vllm_top_ids = {token_id for token_id, _ in vllm_top}
    missing = albatross_top10 - vllm_top_ids
    assert not missing, (
        f"{case['name']}: vLLM logprobs missed Albatross top tokens: "
        f"missing={sorted(missing)} albatross={albatross_top[:10]} "
        f"vllm={vllm_top[:10]}"
    )

    vllm_by_id = dict(vllm_top)
    for token_id, expected in albatross_top[:10]:
        actual = vllm_by_id[token_id]
        assert abs(actual - expected) <= LOGPROB_ATOL, (
            f"{case['name']}: logprob mismatch for token {token_id}: "
            f"albatross={expected:.6f} vllm={actual:.6f}"
        )


def test_rwkv7_next_token_logprobs_match_albatross(
    vllm_runner,
    rwkv7_albatross_settings: AlbatrossSettings,
    rwkv7_execution_mode: ExecutionMode,
    albatross_oracle: dict[str, Any],
) -> None:
    cases = albatross_oracle["cases"]
    outputs = _vllm_outputs(
        vllm_runner,
        rwkv7_albatross_settings,
        rwkv7_execution_mode,
        [case["prompt_token_ids"] for case in cases],
        max_tokens=1,
        logprobs=TOP_K,
    )

    for case, output in zip(cases, outputs):
        sample = output.outputs[0]
        assert list(sample.token_ids) == [case["greedy_token_ids"][0]]
        assert sample.logprobs is not None
        _assert_next_token_logprobs_close(
            case=case,
            vllm_logprobs=cast(Mapping[int, Any], sample.logprobs[0]),
        )


def test_rwkv7_greedy_generation_matches_albatross(
    vllm_runner,
    rwkv7_albatross_settings: AlbatrossSettings,
    rwkv7_execution_mode: ExecutionMode,
    albatross_oracle: dict[str, Any],
) -> None:
    cases = albatross_oracle["cases"]
    outputs = _vllm_outputs(
        vllm_runner,
        rwkv7_albatross_settings,
        rwkv7_execution_mode,
        [case["prompt_token_ids"] for case in cases],
        max_tokens=32,
    )

    for case, output in zip(cases, outputs):
        actual = list(output.outputs[0].token_ids)
        assert actual == case["greedy_token_ids"], (
            f"{case['name']}: greedy token mismatch\n"
            f"albatross={case['greedy_token_ids']}\nvllm={actual}"
        )


def test_rwkv7_prompt_loss_matches_albatross(
    vllm_runner,
    rwkv7_albatross_settings: AlbatrossSettings,
    rwkv7_execution_mode: ExecutionMode,
    albatross_oracle: dict[str, Any],
) -> None:
    cases = albatross_oracle["cases"]
    outputs = _vllm_outputs(
        vllm_runner,
        rwkv7_albatross_settings,
        rwkv7_execution_mode,
        [case["prompt_token_ids"] for case in cases],
        max_tokens=1,
        prompt_logprobs=0,
    )

    for case, output in zip(cases, outputs):
        prompt_logprobs = output.prompt_logprobs
        assert prompt_logprobs is not None
        assert prompt_logprobs[0] is None
        actual_losses = []
        for token_id, logprob_by_token in zip(
            case["prompt_token_ids"][1:],
            prompt_logprobs[1:],
        ):
            assert logprob_by_token is not None
            actual_losses.append(-float(logprob_by_token[int(token_id)].logprob))

        expected_losses = [float(x) for x in case["prompt_losses"]]
        assert len(actual_losses) == len(expected_losses)
        actual_mean = sum(actual_losses) / len(actual_losses)
        expected_mean = case["prompt_mean_loss"]
        assert abs(actual_mean - expected_mean) <= LOSS_MEAN_ATOL, (
            f"{case['name']}: mean loss mismatch: "
            f"albatross={expected_mean:.6f} vllm={actual_mean:.6f}"
        )
        max_diff = max(
            abs(actual - expected)
            for actual, expected in zip(actual_losses, expected_losses)
        )
        assert max_diff <= LOSS_MAX_ATOL, (
            f"{case['name']}: per-position loss diff too high: max_diff={max_diff:.6f}"
        )


def test_rwkv7_mmlu_style_choice_logits_match_albatross(
    vllm_runner,
    rwkv7_albatross_settings: AlbatrossSettings,
    rwkv7_execution_mode: ExecutionMode,
    albatross_oracle: dict[str, Any],
) -> None:
    cases = albatross_oracle["mmlu_style_cases"]
    choice_token_ids = cases[0]["choice_token_ids"]
    outputs = _vllm_outputs(
        vllm_runner,
        rwkv7_albatross_settings,
        rwkv7_execution_mode,
        [case["prompt_token_ids"] for case in cases],
        max_tokens=1,
        logprobs=len(choice_token_ids),
        logprob_token_ids=choice_token_ids,
    )

    for case, output in zip(cases, outputs):
        sample = output.outputs[0]
        assert sample.logprobs is not None
        vllm_logprobs = {
            int(token_id): float(logprob.logprob)
            for token_id, logprob in sample.logprobs[0].items()
        }
        expected = {
            int(token_id): float(logprob)
            for token_id, logprob in case["choice_logprobs"]
        }
        actual_pred = max(
            (token_id for token_id in case["choice_token_ids"]),
            key=lambda token_id: vllm_logprobs[token_id],
        )
        assert actual_pred == case["pred_token_id"], (
            f"{case['name']}: MMLU-style prediction mismatch: "
            f"albatross={case['pred_token_id']} vllm={actual_pred}"
        )
        for token_id, expected_logprob in expected.items():
            actual_logprob = vllm_logprobs[token_id]
            assert abs(actual_logprob - expected_logprob) <= LOGPROB_ATOL, (
                f"{case['name']}: choice logprob mismatch for token {token_id}: "
                f"albatross={expected_logprob:.6f} vllm={actual_logprob:.6f}"
            )


def test_rwkv7_chunked_prefill_matches_albatross(
    vllm_runner,
    rwkv7_albatross_settings: AlbatrossSettings,
    rwkv7_execution_mode: ExecutionMode,
    albatross_oracle: dict[str, Any],
) -> None:
    cases = albatross_oracle["cases"]
    long_cases = [case for case in cases if case["name"] == "long_context"]
    outputs = _vllm_outputs(
        vllm_runner,
        rwkv7_albatross_settings,
        rwkv7_execution_mode,
        [case["prompt_token_ids"] for case in long_cases],
        max_tokens=1,
        logprobs=TOP_K,
        enable_chunked_prefill=True,
        max_num_batched_tokens=64,
    )

    for case, output in zip(long_cases, outputs):
        sample = output.outputs[0]
        assert list(sample.token_ids) == [case["greedy_token_ids"][0]]
        assert sample.logprobs is not None
        _assert_next_token_logprobs_close(
            case=case,
            vllm_logprobs=cast(Mapping[int, Any], sample.logprobs[0]),
        )


def test_rwkv7_continuous_batching_preserves_albatross_outputs(
    vllm_runner,
    rwkv7_albatross_settings: AlbatrossSettings,
    rwkv7_execution_mode: ExecutionMode,
    albatross_oracle: dict[str, Any],
) -> None:
    cases = albatross_oracle["cases"][:3]
    target_case = cases[0]
    solo_outputs = _vllm_outputs(
        vllm_runner,
        rwkv7_albatross_settings,
        rwkv7_execution_mode,
        [target_case["prompt_token_ids"]],
        max_tokens=16,
    )
    batched_outputs = _vllm_outputs(
        vllm_runner,
        rwkv7_albatross_settings,
        rwkv7_execution_mode,
        [case["prompt_token_ids"] for case in reversed(cases)],
        max_tokens=16,
    )

    solo_actual = list(solo_outputs[0].outputs[0].token_ids)
    assert solo_actual == target_case["greedy_token_ids"][:16]
    for case, output in zip(reversed(cases), batched_outputs):
        actual = list(output.outputs[0].token_ids)
        expected = case["greedy_token_ids"][:16]
        assert actual == expected
        if case["name"] == target_case["name"]:
            assert actual == solo_actual


def _server_args(
    settings: AlbatrossSettings,
    execution_mode: ExecutionMode,
    parallel_settings: ParallelSettings,
    port: int,
) -> list[str]:
    args = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        settings.vllm_model,
        "--port",
        str(port),
        "--max-model-len",
        str(settings.max_model_len),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--tensor-parallel-size",
        str(parallel_settings.tensor_parallel_size),
        "--pipeline-parallel-size",
        str(parallel_settings.pipeline_parallel_size),
        "--trust-remote-code",
    ]
    if parallel_settings.pipeline_parallel_size > 1:
        args.extend(["--distributed-executor-backend", "mp"])
    if execution_mode.enforce_eager:
        args.append("--enforce-eager")
    if ENABLE_FLASHINFER_AUTOTUNE:
        args.append("--enable-flashinfer-autotune")
    else:
        args.append("--no-enable-flashinfer-autotune")
    return args


def test_rwkv7_openai_server_matches_albatross_greedy(
    rwkv7_albatross_settings: AlbatrossSettings,
    rwkv7_execution_mode: ExecutionMode,
    albatross_oracle: dict[str, Any],
) -> None:
    if os.environ.get("RWKV7_RUN_SERVER_ALIGNMENT") != "1":
        pytest.skip("Set RWKV7_RUN_SERVER_ALIGNMENT=1 to run the server test")

    case = albatross_oracle["cases"][0]
    port = _free_port()
    server_args = _server_args(
        rwkv7_albatross_settings,
        rwkv7_execution_mode,
        PARALLEL_SETTINGS,
        port,
    )
    with tempfile.NamedTemporaryFile(
        "w+", encoding="utf-8", prefix="rwkv7-vllm-server-", suffix=".log"
    ) as server_log:
        server_log_path = Path(server_log.name)
        with _temporary_env(rwkv7_execution_mode.env):
            env = os.environ.copy()
            env.setdefault("VLLM_RWKV7_WKV_MODE", "fp32io16")
            env.setdefault("VLLM_RWKV7_EMB_DEVICE", "cpu")
            process = subprocess.Popen(
                server_args,
                env=env,
                text=True,
                stdout=server_log,
                stderr=subprocess.STDOUT,
            )
        try:
            _wait_for_server(port, process, server_log_path)
            response = requests.post(
                f"http://127.0.0.1:{port}/v1/completions",
                json={
                    "model": rwkv7_albatross_settings.vllm_model,
                    "prompt": case["prompt"],
                    "temperature": 0,
                    "max_tokens": 16,
                    "logprobs": 1,
                    "return_token_ids": True,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            token_ids = data["choices"][0]["token_ids"]
            assert token_ids == case["greedy_token_ids"][:16]
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(port: int, process: subprocess.Popen[str], log_path: Path) -> None:
    deadline = time.monotonic() + 180
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = _read_server_log(log_path)
            raise RuntimeError(f"vLLM server exited early:\n{output}")
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            time.sleep(2)
    output = _read_server_log(log_path)
    raise TimeoutError(f"Timed out waiting for vLLM server:\n{output}")


def _read_server_log(log_path: Path) -> str:
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"<failed to read vLLM server log {log_path}: {exc}>"
