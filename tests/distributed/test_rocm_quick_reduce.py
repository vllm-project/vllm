# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import importlib
import multiprocessing as mp
import os
import queue
import traceback
from functools import lru_cache
from types import SimpleNamespace
from typing import Literal
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download

import vllm.envs as envs
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="ROCm-only quick-reduce tests",
)

MB = 1024 * 1024
WORLD_SIZE = 2
QUANT_LEVELS = ["FP", "INT8", "INT6", "INT4"]


def _log(message: str) -> None:
    print(f"[rocm_quick_reduce] {message}", flush=True)


def _reload_envs():
    return importlib.reload(envs)


def _make_quick_allreduce(
    *,
    disabled: bool = False,
    world_size: int = 2,
    quant_level: str = "FP",
    use_fp16_kernels: bool = False,
    qr_max_size: int = 64 * MB,
):
    from vllm.distributed.device_communicators.quick_all_reduce import (
        QuickAllReduce,
        QuickReduceRegime,
    )

    qar = QuickAllReduce.__new__(QuickAllReduce)
    qar.disabled = disabled
    qar.world_size = world_size
    qar.use_fp16_kernels = use_fp16_kernels
    qar.qr_quant_level = QuickReduceRegime[quant_level]
    qar.qr_max_size = qr_max_size
    return qar


def _quick_allreduce_worker(
    rank: int,
    port: int,
    quant_level: str,
    dtype_name: str,
    cast_bf16: bool,
):
    os.environ["VLLM_ROCM_QUICK_REDUCE_QUANTIZATION"] = quant_level
    os.environ["VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"] = "1" if cast_bf16 else "0"
    _log(
        f"worker start: rank={rank} quant={quant_level} "
        f"dtype={dtype_name} cast_bf16={cast_bf16}"
    )

    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD_SIZE,
    )

    qar = None
    try:
        from vllm.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )

        qar = QuickAllReduce(group=dist.GroupMember.WORLD, device=rank)
        assert not qar.disabled

        num_elements = 8 * MB if dtype_name == "float16" else 4 * MB

        dtype = getattr(torch, dtype_name)
        inp = torch.ones(num_elements, dtype=dtype, device=device)

        assert qar.should_quick_allreduce(inp)
        if cast_bf16:
            assert qar.use_fp16_kernels

        out = qar.quick_all_reduce(inp)
        assert torch.allclose(out, inp * WORLD_SIZE, atol=2.5, rtol=0.1)
        _log(
            f"worker complete: rank={rank} quant={quant_level} "
            f"dtype={dtype_name} num_elements={num_elements} "
            f"use_fp16_kernels={qar.use_fp16_kernels}"
        )
    finally:
        if qar is not None:
            qar.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_two_gpu_quick_allreduce_test(
    *,
    quant_level: str,
    dtype_name: str,
    cast_bf16: bool,
):
    _log(
        f"launch 2-GPU case: quant={quant_level} "
        f"dtype={dtype_name} cast_bf16={cast_bf16}"
    )
    ctx = mp.get_context("spawn")
    port = get_open_port()
    procs = []

    for rank in range(WORLD_SIZE):
        proc = ctx.Process(
            target=_quick_allreduce_worker,
            args=(rank, port, quant_level, dtype_name, cast_bf16),
        )
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0, f"worker exited with code {proc.exitcode}"
    _log(
        f"finished 2-GPU case: quant={quant_level} "
        f"dtype={dtype_name} cast_bf16={cast_bf16}"
    )


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
E2E_PREFILL_TOKENS = 1024
E2E_MAX_MODEL_LEN = 1536
E2E_GPU_MEMORY_UTILIZATION = 0.3
E2E_KV_CACHE_MEMORY_BYTES = 2 << 30

_BACKGROUND_LINE = (
    "Background filler: this archived operations memo repeats a routine status "
    "line so the distributed test uses a realistically long prefill."
)
_BACKGROUND_BLOCK = " ".join([_BACKGROUND_LINE] * 48)


def _build_prompt(*, fact_block: str, question: str) -> str:
    return (
        "Read the archived operations memo below. Most of the memo is filler. "
        "Use only the fact block near the end when answering.\n"
        f"{_BACKGROUND_BLOCK}\n"
        "Fact block:\n"
        f"{fact_block}\n"
        f"Question: {question}\n"
        "Answer in one short sentence."
    )


E2E_PROMPTS = [
    _build_prompt(
        fact_block=(
            "- Festival city: Oslo\n- Mascot animal: otter\n- Welcome drink: tea"
        ),
        question="Which city hosts the festival, and what animal is the mascot?",
    ),
    _build_prompt(
        fact_block=(
            "- Meeting day: Tuesday\n"
            "- Planned snack: apricot cake\n"
            "- Backup room: Cedar"
        ),
        question="What day is the meeting, and what snack is planned?",
    ),
]
RECORDED_RESPONSE_TEXTS = (
    " The city hosting the festival is Oslo, and the mascot is an otter.",
    " The meeting is on Tuesday and the snack planned is apricot cake.",
)
REQUIRED_WORDS = (("oslo", "otter"), ("tuesday", "apricot"))


def _log_prompt_summaries() -> None:
    for i, prompt in enumerate(E2E_PROMPTS):
        prompt_lines = prompt.splitlines()
        fact_block = [line for line in prompt_lines if line.startswith("- ")]
        fact_summary = "; ".join(line.removeprefix("- ") for line in fact_block)
        _log(f"prompt {i} facts: {fact_summary}")


@lru_cache(maxsize=1)
def _get_model_path() -> str:
    try:
        path = snapshot_download(repo_id=MODEL_NAME, local_files_only=True)
        _log(f"using cached model snapshot: {path}")
        return path
    except Exception:
        path = snapshot_download(repo_id=MODEL_NAME)
        _log(f"downloaded model snapshot: {path}")
        return path


def _get_hidden_size(model_config) -> int:
    hidden_size = getattr(model_config, "hidden_size", None)
    if hidden_size is None and hasattr(model_config, "text_config"):
        hidden_size = getattr(model_config.text_config, "hidden_size", None)
    assert isinstance(hidden_size, int)
    return hidden_size


def _check_tp_allreduce_uses_quick_reduce(
    self,
    num_tokens: int,
    dtype_name: str = "float16",
) -> dict[str, int | bool]:
    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
    from vllm.distributed.parallel_state import get_tp_group

    assert self.device is not None
    qr_comm = get_tp_group().device_communicator.qr_comm
    assert qr_comm is not None
    assert not qr_comm.disabled

    hidden_size = _get_hidden_size(self.model_runner.model.config)
    dtype = getattr(torch, dtype_name)
    sample = torch.full(
        (num_tokens, hidden_size),
        fill_value=float(self.rank + 1),
        dtype=dtype,
        device=self.device,
    )

    assert qr_comm.should_quick_allreduce(sample)

    expected = sample.clone()
    reduced = tensor_model_parallel_all_reduce(sample)
    dist.all_reduce(expected, group=get_tp_group().device_group)
    torch.testing.assert_close(reduced, expected, atol=2.5, rtol=0.1)

    stats = {
        "rank": self.rank,
        "hidden_size": hidden_size,
        "num_tokens": num_tokens,
        "use_fp16_kernels": qr_comm.use_fp16_kernels,
    }
    _log(
        "worker quick-reduce check: "
        f"rank={self.rank} hidden_size={hidden_size} "
        f"num_tokens={num_tokens} use_fp16_kernels={qr_comm.use_fp16_kernels}"
    )
    return stats


def _check_quick_reduce_disabled(self) -> int:
    from vllm.distributed.parallel_state import get_tp_group

    qr_comm = get_tp_group().device_communicator.qr_comm
    assert qr_comm is not None
    assert qr_comm.disabled
    _log(f"worker confirmed quick reduce is disabled: rank={self.rank}")
    return self.rank


def _collect_generations(outputs) -> list[tuple[tuple[int, ...], str]]:
    return [
        (tuple(output.outputs[0].token_ids), output.outputs[0].text)
        for output in outputs
    ]


def _shutdown_llm(llm: LLM | None) -> None:
    if llm is None:
        cleanup_dist_env_and_memory()
        return

    with contextlib.suppress(Exception):
        llm.llm_engine.engine_core.shutdown()

    del llm
    cleanup_dist_env_and_memory()


def _log_generations(
    label: str,
    generations: list[tuple[tuple[int, ...], str]],
) -> None:
    for i, (token_ids, text) in enumerate(generations):
        _log(f"{label} prompt {i} token ids: {list(token_ids)}")
        _log(f"{label} prompt {i} text: {text!r}")


def _assert_required_words(
    label: str,
    generations: list[tuple[tuple[int, ...], str]],
) -> None:
    for i, (_, text) in enumerate(generations):
        lowered = text.lower()
        missing = [word for word in REQUIRED_WORDS[i] if word not in lowered]
        assert not missing, (
            f"{label} prompt {i} is missing required words {missing}. "
            f"Observed text: {text!r}"
        )


def _collect_soft_mismatches(
    baseline_generations: list[tuple[tuple[int, ...], str]],
    quick_reduce_generations: list[tuple[tuple[int, ...], str]],
) -> list[str]:
    mismatches = []

    for i, (_, text) in enumerate(baseline_generations):
        expected = RECORDED_RESPONSE_TEXTS[i]
        if text != expected:
            mismatches.append(
                f"baseline prompt {i} drifted from the recorded response.\n"
                f"expected={expected!r}\nactual={text!r}"
            )

    for i, (_, text) in enumerate(quick_reduce_generations):
        expected = RECORDED_RESPONSE_TEXTS[i]
        if text != expected:
            mismatches.append(
                f"quick-reduce prompt {i} drifted from the recorded response.\n"
                f"expected={expected!r}\nactual={text!r}"
            )

    for i, ((_, baseline_text), (_, quick_reduce_text)) in enumerate(
        zip(baseline_generations, quick_reduce_generations)
    ):
        if baseline_text != quick_reduce_text:
            mismatches.append(
                f"baseline and quick-reduce responses differ for prompt {i}.\n"
                f"baseline={baseline_text!r}\nquick_reduce={quick_reduce_text!r}"
            )

    return mismatches


def _run_generation(
    *,
    backend: Literal["mp", "ray"],
    quant_mode: str,
    expect_quick_reduce: bool,
) -> list[tuple[tuple[int, ...], str]]:
    llm = None
    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        m.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", quant_mode)
        model_path = _get_model_path()
        _log(
            f"starting generation: backend={backend} quant={quant_mode} "
            f"gpu_memory_utilization={E2E_GPU_MEMORY_UTILIZATION} "
            f"kv_cache_bytes={E2E_KV_CACHE_MEMORY_BYTES} model={model_path}"
        )

        try:
            llm = LLM(
                model=model_path,
                tokenizer=model_path,
                tensor_parallel_size=2,
                distributed_executor_backend=backend,
                dtype="half",
                enforce_eager=True,
                max_model_len=E2E_MAX_MODEL_LEN,
                max_num_seqs=len(E2E_PROMPTS),
                gpu_memory_utilization=E2E_GPU_MEMORY_UTILIZATION,
                kv_cache_memory_bytes=E2E_KV_CACHE_MEMORY_BYTES,
                seed=0,
            )

            if not expect_quick_reduce:
                assert llm.collective_rpc(_check_quick_reduce_disabled) == [0, 1]

            if expect_quick_reduce:
                worker_stats = llm.collective_rpc(
                    _check_tp_allreduce_uses_quick_reduce,
                    args=(E2E_PREFILL_TOKENS,),
                )
                assert [stat["rank"] for stat in worker_stats] == [0, 1]
                worker_summary = "; ".join(
                    "rank={rank} hidden_size={hidden_size} num_tokens={num_tokens} "
                    "use_fp16_kernels={use_fp16_kernels}".format(**stat)
                    for stat in worker_stats
                )
                _log(f"{backend} quick-reduce worker checks: {worker_summary}")

            outputs = llm.generate(
                E2E_PROMPTS,
                SamplingParams(
                    temperature=0.0,
                    max_tokens=20,
                    stop=["\nAnswer:", " Answer:"],
                ),
                use_tqdm=False,
            )
            generations = _collect_generations(outputs)
            assert all(text.strip() for _, text in generations)
            _log_generations(f"{backend} {quant_mode}", generations)
            return generations
        finally:
            _shutdown_llm(llm)


def _run_quick_reduce_llm_e2e_in_subprocess(
    *,
    backend: Literal["mp", "ray"],
) -> str | None:
    _log(f"running LLM e2e: backend={backend}")
    _log_prompt_summaries()
    baseline_outputs = _run_generation(
        backend=backend,
        quant_mode="NONE",
        expect_quick_reduce=False,
    )
    quick_reduce_outputs = _run_generation(
        backend=backend,
        quant_mode="FP",
        expect_quick_reduce=True,
    )

    _assert_required_words("baseline", baseline_outputs)
    _assert_required_words("quick-reduce", quick_reduce_outputs)

    mismatches = _collect_soft_mismatches(baseline_outputs, quick_reduce_outputs)
    if mismatches:
        details = "\n\n".join(mismatches)
        _log(f"soft response mismatch:\n{details}")
        return details

    _log(f"LLM e2e backend={backend} matched the recorded responses exactly")
    return None


def _quick_reduce_llm_e2e_worker(
    result_queue: mp.Queue,
    backend: Literal["mp", "ray"],
) -> None:
    try:
        xfail_reason = _run_quick_reduce_llm_e2e_in_subprocess(backend=backend)
    except Exception:
        result_queue.put({"status": "error", "reason": traceback.format_exc()})
        raise
    else:
        if xfail_reason is not None:
            result_queue.put({"status": "xfail", "reason": xfail_reason})
        else:
            result_queue.put({"status": "ok"})


def run_quick_reduce_llm_e2e(
    *,
    backend: Literal["mp", "ray"],
) -> None:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_quick_reduce_llm_e2e_worker,
        args=(result_queue, backend),
    )
    proc.start()
    proc.join(timeout=600)

    try:
        result = result_queue.get(timeout=5)
    except queue.Empty as exc:
        if proc.exitcode != 0:
            raise AssertionError(
                f"quick-reduce llm e2e subprocess failed for backend={backend} "
                f"with exit code {proc.exitcode} and produced no result"
            ) from exc
        raise AssertionError(
            f"quick-reduce llm e2e subprocess produced no result for backend={backend}"
        ) from exc

    if result["status"] == "xfail":
        pytest.xfail(result["reason"])
    if result["status"] == "error":
        raise AssertionError(
            f"quick-reduce llm e2e subprocess failed for backend={backend}:\n"
            f"{result['reason']}"
        )

    assert proc.exitcode == 0, (
        f"quick-reduce llm e2e subprocess failed for backend={backend} "
        f"with exit code {proc.exitcode}"
    )


def test_quick_reduce_regime_values():
    from vllm.distributed.device_communicators.quick_all_reduce import QuickReduceRegime

    assert QuickReduceRegime.FP.value == 0
    assert QuickReduceRegime.INT8.value == 1
    assert QuickReduceRegime.INT6.value == 2
    assert QuickReduceRegime.INT4.value == 3
    assert QuickReduceRegime.NONE.value == 4


def test_quick_reduce_regime_names():
    from vllm.distributed.device_communicators.quick_all_reduce import QuickReduceRegime

    assert set(QuickReduceRegime.__members__) == {"FP", "INT8", "INT6", "INT4", "NONE"}


@pytest.mark.parametrize("quant_level", QUANT_LEVELS + ["NONE"])
def test_quick_reduce_quantization_env_var(monkeypatch, quant_level):
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", quant_level)

    reloaded_envs = _reload_envs()
    assert quant_level == reloaded_envs.VLLM_ROCM_QUICK_REDUCE_QUANTIZATION


def test_quick_reduce_quantization_default(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", raising=False)

    reloaded_envs = _reload_envs()
    assert reloaded_envs.VLLM_ROCM_QUICK_REDUCE_QUANTIZATION == "NONE"


@pytest.mark.parametrize("cast_bf16", [True, False])
def test_quick_reduce_cast_bf16_to_fp16_env_var(monkeypatch, cast_bf16):
    monkeypatch.setenv(
        "VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", "1" if cast_bf16 else "0"
    )

    reloaded_envs = _reload_envs()
    assert reloaded_envs.VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16 is cast_bf16


def test_quick_reduce_cast_bf16_to_fp16_default(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", raising=False)

    reloaded_envs = _reload_envs()
    assert reloaded_envs.VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16 is True


@pytest.mark.parametrize("max_mb", [128, 512, 2048, None])
def test_quick_reduce_max_size_env_var(monkeypatch, max_mb):
    if max_mb is None:
        monkeypatch.delenv("VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", raising=False)
    else:
        monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", str(max_mb))

    reloaded_envs = _reload_envs()
    assert max_mb == reloaded_envs.VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB


def test_quick_reduce_max_size_default(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", raising=False)

    reloaded_envs = _reload_envs()
    assert reloaded_envs.VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB is None


@pytest.mark.parametrize(
    ("gcn_arch_name", "expected"),
    [
        ("gfx942", True),
        ("gfx950", True),
        ("gfx90a", False),
        ("", False),
    ],
)
def test_quick_allreduce_rocm_arch_available(gcn_arch_name, expected):
    from vllm.distributed.device_communicators.quick_all_reduce import QuickAllReduce

    qar = QuickAllReduce.__new__(QuickAllReduce)
    qar.disabled = True

    with (
        patch(
            "vllm.distributed.device_communicators.quick_all_reduce.current_platform."
            "is_rocm",
            return_value=True,
        ),
        patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(gcnArchName=gcn_arch_name),
        ),
    ):
        assert qar._rocm_arch_available() is expected


def test_quick_allreduce_rocm_arch_available_handles_probe_failure():
    from vllm.distributed.device_communicators.quick_all_reduce import QuickAllReduce

    qar = QuickAllReduce.__new__(QuickAllReduce)
    qar.disabled = True

    with (
        patch(
            "vllm.distributed.device_communicators.quick_all_reduce.current_platform."
            "is_rocm",
            return_value=True,
        ),
        patch("torch.cuda.get_device_properties", side_effect=RuntimeError),
    ):
        assert qar._rocm_arch_available() is False


def test_quick_allreduce_rejects_disabled():
    qar = _make_quick_allreduce(disabled=True)

    inp = torch.zeros(1024, dtype=torch.float16)
    assert qar.should_quick_allreduce(inp) is False


def test_quick_allreduce_rejects_unsupported_dtype():
    qar = _make_quick_allreduce()

    inp = torch.zeros(1024 * 1024, dtype=torch.float32)
    assert qar.should_quick_allreduce(inp) is False


def test_quick_allreduce_rejects_non_aligned_input():
    qar = _make_quick_allreduce()

    inp = torch.zeros(5, dtype=torch.float16)
    assert qar.should_quick_allreduce(inp) is False


def test_quick_allreduce_rejects_non_contiguous_input():
    qar = _make_quick_allreduce()

    inp = torch.zeros((1024, 1024), dtype=torch.float16)[:, ::2]
    assert qar.should_quick_allreduce(inp) is False


def test_quick_allreduce_rejects_input_smaller_than_threshold():
    qar = _make_quick_allreduce()

    inp = torch.zeros((MB // 2) - 8, dtype=torch.float16)
    assert qar.should_quick_allreduce(inp) is False


def test_quick_allreduce_accepts_input_at_threshold():
    qar = _make_quick_allreduce()

    inp = torch.zeros(MB // 2, dtype=torch.float16)
    assert qar.should_quick_allreduce(inp) is True


def test_quick_allreduce_rejects_input_larger_than_max_size():
    qar = _make_quick_allreduce(qr_max_size=1 * MB)

    inp = torch.zeros(MB, dtype=torch.float16)
    assert qar.should_quick_allreduce(inp) is False


def test_quick_allreduce_bf16_uses_fp16_threshold_when_cast_enabled():
    inp = torch.zeros(MB // 2, dtype=torch.bfloat16)

    without_cast = _make_quick_allreduce(use_fp16_kernels=False)
    with_cast = _make_quick_allreduce(use_fp16_kernels=True)

    assert without_cast.should_quick_allreduce(inp) is False
    assert with_cast.should_quick_allreduce(inp) is True


def test_quick_allreduce_supported_world_sizes():
    from vllm.distributed.device_communicators.quick_all_reduce import QuickAllReduce

    assert QuickAllReduce._SUPPORTED_WORLD_SIZES == [2, 4, 8]


def test_quick_allreduce_supported_dtypes():
    from vllm.distributed.device_communicators.quick_all_reduce import QuickAllReduce

    assert [torch.float16, torch.bfloat16] == QuickAllReduce._SUPPORTED_DTYPES


def test_quick_allreduce_min_size_table():
    from vllm.distributed.device_communicators.quick_all_reduce import QuickAllReduce

    for dtype in [torch.float16, torch.bfloat16]:
        for world_size in QuickAllReduce._SUPPORTED_WORLD_SIZES:
            min_sizes = QuickAllReduce._QR_MIN_SIZE[(dtype, world_size)]
            assert len(min_sizes) == 4
            assert all(size > 0 for size in min_sizes)


def test_qr_max_size():
    from vllm import _custom_ops as ops

    max_size = ops.qr_max_size()
    assert isinstance(max_size, int)
    assert max_size > 0


@pytest.mark.skipif(
    current_platform.device_count() < WORLD_SIZE,
    reason="requires 2 ROCm GPUs",
)
@pytest.mark.parametrize("quant_level", QUANT_LEVELS)
def test_quick_allreduce_two_gpu_correctness(quant_level):
    _log(f"two-GPU correctness case: quant={quant_level}")
    _run_two_gpu_quick_allreduce_test(
        quant_level=quant_level,
        dtype_name="float16",
        cast_bf16=False,
    )


@pytest.mark.skipif(
    current_platform.device_count() < WORLD_SIZE,
    reason="requires 2 ROCm GPUs",
)
def test_quick_allreduce_bf16_cast_mode():
    _log("BF16 cast case")
    _run_two_gpu_quick_allreduce_test(
        quant_level="FP",
        dtype_name="bfloat16",
        cast_bf16=True,
    )


@pytest.mark.skipif(
    current_platform.device_count() < WORLD_SIZE,
    reason="requires 2 ROCm GPUs",
)
def test_quick_allreduce_llm_e2e():
    _log("LLM e2e case: backend=mp")
    run_quick_reduce_llm_e2e(backend="mp")


@pytest.mark.skipif(
    current_platform.device_count() < WORLD_SIZE,
    reason="requires 2 ROCm GPUs",
)
def test_quick_allreduce_llm_e2e_ray():
    _log("LLM e2e case: backend=ray")
    run_quick_reduce_llm_e2e(backend="ray")
