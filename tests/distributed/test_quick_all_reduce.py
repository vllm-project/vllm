# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import random

import pytest
import ray
import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce  # noqa
from vllm.distributed.device_communicators.quick_all_reduce import (
    KB,
    MB,
    QuickAllReduce,
    QuickReduceRegime,
)
from vllm.distributed.parallel_state import get_tp_group, graph_capture
from vllm.envs import disable_envs_cache
from vllm.platforms import current_platform

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
    set_random_seed,
)


def on_gfx942() -> bool:
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx942 as rocm_on_gfx942

        return rocm_on_gfx942()
    return False


set_random_seed(42)
_test_size_rng = random.Random(44)
# Size over 8MB is sufficient for custom quick allreduce.
test_sizes = [
    _test_size_rng.randint(8 * 1024 * 1024, 10 * 1024 * 1024) for _ in range(8)
]
for i, v in enumerate(test_sizes):
    test_sizes[i] -= v % 8


def _assert_quickreduce(fa, inp):
    assert fa is not None
    assert not fa.disabled
    assert fa.should_quick_allreduce(inp)


@pytest.fixture
def envs_cache_disabled():
    disable_envs_cache()
    yield
    disable_envs_cache()


def _make_quick_allreduce_for_test(
    min_size_mb: int | None = None,
    quantization_min_size: int | None = None,
) -> QuickAllReduce:
    quick_reduce = QuickAllReduce.__new__(QuickAllReduce)
    quick_reduce.disabled = False
    quick_reduce.qr_max_size = 16 * MB
    quick_reduce.qr_min_size = min_size_mb * MB if min_size_mb is not None else None
    quick_reduce.qr_quant_level = QuickReduceRegime.INT4
    quick_reduce.qr_quantization_min_size = quantization_min_size
    quick_reduce.use_fp16_kernels = False
    quick_reduce.world_size = 2
    return quick_reduce


def test_should_quick_allreduce_uses_builtin_min_size_when_unset():
    quick_reduce = _make_quick_allreduce_for_test(min_size_mb=None)

    below_builtin_min = torch.empty(MB // 4, dtype=torch.float16)
    at_builtin_min = torch.empty(MB // 2, dtype=torch.float16)

    assert not quick_reduce.should_quick_allreduce(below_builtin_min)
    assert quick_reduce.should_quick_allreduce(at_builtin_min)


def test_should_quick_allreduce_uses_min_size_override():
    quick_reduce = _make_quick_allreduce_for_test(min_size_mb=0)

    below_builtin_min = torch.empty(8, dtype=torch.float16)

    assert quick_reduce.should_quick_allreduce(below_builtin_min)


def test_quick_allreduce_min_size_env_unset(
    monkeypatch: pytest.MonkeyPatch,
    envs_cache_disabled,
):
    monkeypatch.delenv("VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB", raising=False)

    assert QuickAllReduce._get_qr_min_size(qr_max_size=16 * MB) is None


def test_quick_allreduce_min_size_env_converts_mb_to_bytes(
    monkeypatch: pytest.MonkeyPatch,
    envs_cache_disabled,
):
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB", "4")

    assert QuickAllReduce._get_qr_min_size(qr_max_size=16 * MB) == 4 * MB


def test_quick_allreduce_min_size_env_rejects_negative(
    monkeypatch: pytest.MonkeyPatch,
    envs_cache_disabled,
):
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB", "-1")

    with pytest.raises(ValueError, match="must be non-negative"):
        QuickAllReduce._get_qr_min_size(qr_max_size=16 * MB)


def test_quick_allreduce_min_size_env_allows_equal_to_max(
    monkeypatch: pytest.MonkeyPatch,
    envs_cache_disabled,
):
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB", "16")

    assert QuickAllReduce._get_qr_min_size(qr_max_size=16 * MB) == 16 * MB


def test_quick_allreduce_min_size_env_rejects_larger_than_max(
    monkeypatch: pytest.MonkeyPatch,
    envs_cache_disabled,
):
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB", "17")

    with pytest.raises(ValueError, match="effective QuickReduce max size"):
        QuickAllReduce._get_qr_min_size(qr_max_size=16 * MB)


def test_quick_allreduce_quantization_min_size_env_unset(
    monkeypatch: pytest.MonkeyPatch,
    envs_cache_disabled,
):
    monkeypatch.delenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB", raising=False)

    assert QuickAllReduce._get_qr_quantization_min_size() is None


def test_quick_allreduce_quantization_min_size_env_converts_kb_to_bytes(
    monkeypatch: pytest.MonkeyPatch,
    envs_cache_disabled,
):
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB", "2048")

    assert QuickAllReduce._get_qr_quantization_min_size() == 2048 * KB


def test_quick_allreduce_quantization_min_size_env_rejects_negative(
    monkeypatch: pytest.MonkeyPatch,
    envs_cache_disabled,
):
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB", "-1")

    with pytest.raises(ValueError, match="must be non-negative"):
        QuickAllReduce._get_qr_quantization_min_size()


def test_quick_allreduce_quantization_min_size_unset_uses_configured_codec():
    quick_reduce = _make_quick_allreduce_for_test(quantization_min_size=None)
    inp = torch.empty(8, dtype=torch.float16)

    assert quick_reduce._get_qr_quant_level(inp) == QuickReduceRegime.INT4.value


def test_quick_allreduce_quantization_min_size_uses_fp_below_threshold():
    quick_reduce = _make_quick_allreduce_for_test(quantization_min_size=2048)
    inp = torch.empty(1024 // 2, dtype=torch.float16)

    assert quick_reduce._get_qr_quant_level(inp) == QuickReduceRegime.FP.value


def test_quick_allreduce_quantization_min_size_uses_configured_codec_at_threshold():
    quick_reduce = _make_quick_allreduce_for_test(quantization_min_size=2048)
    inp = torch.empty(2048 // 2, dtype=torch.float16)

    assert quick_reduce._get_qr_quant_level(inp) == QuickReduceRegime.INT4.value


def test_quick_allreduce_quantization_min_size_does_not_change_eligibility():
    quick_reduce = _make_quick_allreduce_for_test(quantization_min_size=2 * MB)

    below_builtin_min = torch.empty(MB // 4, dtype=torch.float16)
    at_builtin_min = torch.empty(MB // 2, dtype=torch.float16)

    assert not quick_reduce.should_quick_allreduce(below_builtin_min)
    assert quick_reduce.should_quick_allreduce(at_builtin_min)


def test_quick_allreduce_passes_dynamic_quant_level(
    monkeypatch: pytest.MonkeyPatch,
):
    quick_reduce = _make_quick_allreduce_for_test(quantization_min_size=2 * KB)
    quick_reduce._ptr = object()
    inp = torch.empty(KB // 2, dtype=torch.float16)
    called_quant_level = None

    def fake_qr_all_reduce(
        fa,
        inp,
        out,
        quant_level,
        cast_bf2half,
    ):
        nonlocal called_quant_level
        called_quant_level = quant_level

    monkeypatch.setattr(ops, "qr_all_reduce", fake_qr_all_reduce)

    quick_reduce.quick_all_reduce(inp)

    assert called_quant_level == QuickReduceRegime.FP.value


@ray.remote(num_gpus=1, max_calls=1)
def graph_quickreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group().device_group
        fa = get_tp_group().device_communicator.qr_comm

        # A small all_reduce for warmup.
        # this is needed because device communicators might be created lazily
        # (e.g. NCCL). This will ensure that the communicator is initialized
        # before any communication happens, so that this group can be used for
        # graph capture immediately.
        data = torch.zeros(1)
        data = data.to(device=device)
        torch.distributed.all_reduce(data, group=group)
        torch.accelerator.synchronize()
        del data

        # we use the first group to communicate once
        # and the second group to communicate twice
        # and so on
        # this is used to demonstrate that each group can
        # communicate independently
        num_communication = rank // tp_size + 1

        for sz in test_sizes:
            for dtype in [torch.float16, torch.bfloat16]:
                with graph_capture(device=device) as graph_capture_context:
                    device_idx = torch.accelerator.current_device_index()
                    inp1 = torch.randint(1, 23, (sz,), dtype=dtype, device=device_idx)
                    inp2 = torch.randint(-23, 1, (sz,), dtype=dtype, device=device_idx)
                    _assert_quickreduce(fa, inp1)
                    _assert_quickreduce(fa, inp2)

                    torch.accelerator.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                        for _ in range(num_communication):
                            out1 = tensor_model_parallel_all_reduce(inp1)
                            dist.all_reduce(inp1, group=group)
                            out2 = tensor_model_parallel_all_reduce(inp2)
                            dist.all_reduce(inp2, group=group)
                graph.replay()
                torch.testing.assert_close(out1, inp1, atol=2.5, rtol=0.1)
                torch.testing.assert_close(out2, inp2, atol=2.5, rtol=0.1)


@ray.remote(num_gpus=1, max_calls=1)
def eager_quickreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)

        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

        # Size over 8MB is sufficient for custom quick allreduce.
        sz = 16 * 1024 * 1024
        fa = get_tp_group().device_communicator.qr_comm
        inp = torch.tensor(
            [1.0 * ((i) % 23) for i in range(sz)], dtype=torch.float16, device=device
        )
        _assert_quickreduce(fa, inp)
        out = fa.quick_all_reduce(inp)
        torch.testing.assert_close(out, inp * tp_size, atol=2.5, rtol=0.1)

        inp = torch.tensor(
            [1.0 * ((i) % 23) for i in range(sz)], dtype=torch.bfloat16, device=device
        )
        _assert_quickreduce(fa, inp)
        out = fa.quick_all_reduce(inp)
        torch.testing.assert_close(out, inp * tp_size, atol=2.5, rtol=0.1)


@ray.remote(num_gpus=1, max_calls=1)
def bf16_cast_quickreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        m.setenv("VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", "1")
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

        sz = 16 * 1024 * 1024
        fa = get_tp_group().device_communicator.qr_comm
        inp = torch.tensor(
            [1.0 * (i % 23) for i in range(sz)], dtype=torch.bfloat16, device=device
        )
        _assert_quickreduce(fa, inp)
        assert fa.use_fp16_kernels
        out = fa.quick_all_reduce(inp)
        torch.testing.assert_close(out, inp * tp_size, atol=2.5, rtol=0.1)


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="only test quick allreduce for rocm"
)
@pytest.mark.parametrize("quant_mode", ["FP", "INT8", "INT6", "INT4", "INT3"])
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
@pytest.mark.parametrize("test_target", [graph_quickreduce, eager_quickreduce])
def test_custom_quick_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pipeline_parallel_size,
    test_target,
    quant_mode,
):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    if test_target is graph_quickreduce and on_gfx942():
        pytest.xfail(
            "CUDA graph capture with quick reduce hits "
            "hipErrorStreamCaptureInvalidated on gfx942"
        )

    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", quant_mode)

    multi_process_parallel(monkeypatch, tp_size, pipeline_parallel_size, test_target)


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="only test quick allreduce for rocm"
)
def test_custom_quick_allreduce_bf16_cast(monkeypatch: pytest.MonkeyPatch):
    if torch.accelerator.device_count() < 2:
        pytest.skip("Not enough GPUs to run the test.")
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", "FP")
    multi_process_parallel(monkeypatch, 2, 1, bf16_cast_quickreduce)


def qr_variable_input(rank, world_size):
    """
    When the tensor parallelism is set to 4 or 8, frequent changes
    in the input shape can cause QuickReduce to hang (this issue
    has been observed with the gpt_oss model).
    """
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    qr_max_size = None  # MB
    _ptr = ops.init_custom_qr(rank, world_size, qr_max_size)
    ranks = []
    for i in range(world_size):
        ranks.append(i)
    if envs.VLLM_DISTRIBUTED_USE_SPLIT_GROUP:
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            init_method="tcp://127.0.0.1:29500",
            rank=rank,
            world_size=world_size,
            device_id=device,
        )
    else:
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:29500",
            rank=rank,
            world_size=world_size,
        )
    if envs.VLLM_DISTRIBUTED_USE_SPLIT_GROUP:
        cpu_group = torch.distributed.split_group(
            split_ranks=[ranks], backend="cpu:gloo,cuda:nccl"
        )
    else:
        cpu_group = torch.distributed.new_group(ranks, backend="nccl")

    handle = ops.qr_get_handle(_ptr)
    world_size = dist.get_world_size(group=cpu_group)
    handles = [None] * world_size
    dist.all_gather_object(handles, handle, group=cpu_group)
    ops.qr_open_handles(_ptr, handles)

    num = 1
    s1 = 1024
    while num < 50000:  # 50000 is sufficient to identify issues.
        dtype = torch.float16
        device_idx = torch.accelerator.current_device_index()
        if num % 2 == 0:
            s2 = 1024
            inp1 = torch.zeros((s1, s2), dtype=dtype, device=device_idx)
        else:
            s2 = 2048
            inp1 = torch.ones((s1, s2), dtype=dtype, device=device_idx)
        result = torch.empty_like(inp1)
        # FP = 0 INT8 = 1 INT6 = 2 INT4 = 3 INT3 = 4
        ops.qr_all_reduce(_ptr, inp1, result, 3, cast_bf2half=True)
        try:
            if inp1[0, 0] == 0:
                assert torch.all(result == 0)
            else:
                assert torch.all(result == world_size)
        except AssertionError:
            print("Assertion failed! Allreduce results are incorrect.")
            raise
        num += 1


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="only test quick allreduce for rocm"
)
@pytest.mark.parametrize("tp_size", [4, 8])
@pytest.mark.parametrize("pipeline_parallel_size", [1])
def test_custom_quick_allreduce_variable_input(tp_size, pipeline_parallel_size):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")

    multiprocessing.set_start_method("spawn", force=True)
    # 60s is enough
    timeout = 60
    processes = []
    for rank in range(tp_size):
        p = multiprocessing.Process(target=qr_variable_input, args=(rank, tp_size))
        p.start()
        processes.append((rank, p))
    for rank, p in processes:
        p.join(timeout=timeout)
        if p.is_alive():
            for r, proc in processes:
                if proc.is_alive():
                    proc.terminate()
                    proc.join()
            raise RuntimeError(f"QuickReduce hang detected after {timeout} seconds!")


if __name__ == "__main__":
    test_custom_quick_allreduce_variable_input(tp_size=4, pipeline_parallel_size=1)
