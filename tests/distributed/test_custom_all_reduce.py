# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce  # noqa
from vllm.distributed.device_communicators import custom_all_reduce as car
from vllm.distributed.parallel_state import get_tp_group, graph_capture

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

random.seed(42)
test_sizes = [random.randint(1024, 2048 * 1024) for _ in range(8)]
for i, v in enumerate(test_sizes):
    test_sizes[i] -= v % 8


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (torch.float32, True),
        (torch.float16, True),
        (torch.bfloat16, True),
        (torch.int8, False),
        (torch.float8_e4m3fn, False),
    ],
)
def test_custom_allreduce_filters_dtype(
    dtype: torch.dtype,
    expected: bool,
) -> None:
    communicator = car.CustomAllreduce.__new__(car.CustomAllreduce)
    communicator.disabled = False
    communicator._ptr = 0
    communicator.world_size = 2
    communicator.max_size = 1024

    assert communicator.should_custom_ar(torch.empty(16, dtype=dtype)) is expected


@pytest.mark.parametrize(
    ("major", "local_multicast", "expected"),
    [
        (8, True, False),
        (9, True, False),
        (10, False, False),
        (10, True, True),
    ],
)
def test_cross_node_mnnvl_gate_checks_generation_and_multicast(
    monkeypatch,
    major,
    local_multicast,
    expected,
):
    def has_device_capability(capability, device_id):
        assert capability == 100
        assert device_id == 3
        return major >= 10

    monkeypatch.setattr(
        car.current_platform,
        "has_device_capability",
        has_device_capability,
    )
    monkeypatch.setattr(
        car,
        "_has_local_multicast_support",
        lambda _device: local_multicast,
    )
    monkeypatch.setattr(car.dist, "all_reduce", lambda *_args, **_kwargs: None)

    assert car._group_can_attempt_mnnvl(object(), torch.device("cuda:3")) is expected


def test_cross_node_mnnvl_gate_requires_support_on_every_rank(monkeypatch):
    monkeypatch.setattr(
        car.current_platform,
        "has_device_capability",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        car,
        "_has_local_multicast_support",
        lambda _device: True,
    )

    def report_unsupported_peer(support, **_kwargs):
        support.zero_()

    monkeypatch.setattr(car.dist, "all_reduce", report_unsupported_peer)

    assert not car._group_can_attempt_mnnvl(object(), torch.device("cuda:0"))


def test_local_multicast_support_rejects_non_cuda(monkeypatch):
    monkeypatch.setattr(car.current_platform, "is_cuda", lambda: False)

    assert not car._has_local_multicast_support(torch.device("cuda:0"))


@pytest.mark.parametrize(
    ("world_size", "device_capability", "local_multicast", "expected"),
    [
        (2, (10, 0), True, True),
        (4, (10, 3), True, True),
        (8, (10, 0), True, True),
        (8, (10, 3), True, True),
        (6, (10, 3), True, False),
        (8, (10, 1), True, False),
        (8, (9, 0), True, False),
        (8, (10, 3), False, False),
    ],
)
def test_mnnvl_multimem_reduce_scatter_platform_gate(
    monkeypatch,
    world_size,
    device_capability,
    local_multicast,
    expected,
):
    def is_device_capability(capability, device_id):
        assert capability in ((10, 0), (10, 3))
        assert device_id == 3
        return device_capability == capability

    monkeypatch.setattr(
        car.current_platform,
        "is_device_capability",
        is_device_capability,
    )
    monkeypatch.setattr(
        car,
        "_has_local_multicast_support",
        lambda _device: local_multicast,
    )

    supported = car._supports_mnnvl_multimem_reduce_scatter(
        torch.device("cuda:3"), world_size
    )
    assert supported is expected


@pytest.mark.parametrize(
    (
        "message_bytes",
        "multimem_ptr",
        "multimem_initialized",
        "batch_invariant",
        "expected",
    ),
    [
        (16 * 1024 * 1024, 1, True, False, "mnnvl_lamport"),
        (16 * 1024 * 1024 + 128, 1, True, False, "mnnvl_multimem"),
        (64 * 1024 * 1024, 1, True, False, "mnnvl_multimem"),
        (64 * 1024 * 1024 + 128, 1, True, False, None),
        (32 * 1024 * 1024, 0, True, False, None),
        (32 * 1024 * 1024, 0, False, False, "mnnvl_multimem"),
        (8 * 1024 * 1024, 1, True, True, "mnnvl_lamport"),
        (32 * 1024 * 1024, 1, True, True, None),
    ],
)
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_mnnvl_reduce_scatter_backend_gate(
    monkeypatch,
    world_size,
    message_bytes,
    multimem_ptr,
    multimem_initialized,
    batch_invariant,
    expected,
):
    monkeypatch.setattr(car.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(car.envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    communicator = car.CustomAllreduce.__new__(car.CustomAllreduce)
    communicator.disabled = False
    communicator._ptr = 0
    communicator.world_size = world_size
    communicator.mnnvl_only = False
    communicator.fully_connected = True
    communicator.mnnvl_multicast_ptr = 1
    communicator.mnnvl_multimem_rs_supported = True
    communicator.mnnvl_multimem_rs_initialized = multimem_initialized
    communicator.mnnvl_multimem_rs_multicast_ptr = multimem_ptr
    communicator.max_mnnvl_reduce_scatter_size = 16 * 1024 * 1024
    communicator.max_mnnvl_multimem_reduce_scatter_size = 64 * 1024 * 1024
    communicator.max_reduce_scatter_size = 16 * 1024 * 1024
    inp = torch.empty(
        (world_size, message_bytes // torch.bfloat16.itemsize // world_size),
        dtype=torch.bfloat16,
    )

    assert inp.nbytes == message_bytes
    assert communicator._select_reduce_scatter_backend(inp) == expected
    assert communicator.should_custom_reduce_scatter(inp) is (expected is not None)
    assert communicator.should_mnnvl_multimem_reduce_scatter(inp) is (
        expected == "mnnvl_multimem"
    )


def test_mnnvl_multimem_reduce_scatter_skips_rendezvous_after_peer_alloc_failure(
    monkeypatch,
):
    events = []

    class FakeSymmMem:
        @staticmethod
        def empty(*_args, **_kwargs):
            events.append("empty")
            return torch.empty(1, dtype=torch.uint8)

        @staticmethod
        def rendezvous(*_args, **_kwargs):
            events.append("rendezvous")
            return None

    def report_peer_allocation_failure(group_value, **_kwargs):
        events.append("all_reduce")
        assert group_value.item() == 1
        group_value.zero_()

    monkeypatch.setattr(car, "torch_symm_mem", FakeSymmMem)
    monkeypatch.setattr(car.dist, "all_reduce", report_peer_allocation_failure)
    warnings = []
    monkeypatch.setattr(
        car.logger,
        "warning_once",
        lambda message, *_args, **_kwargs: warnings.append(message),
    )

    communicator = car.CustomAllreduce.__new__(car.CustomAllreduce)
    communicator.disabled = True
    communicator._ptr = 0
    communicator.group = object()
    communicator.device = torch.device("cuda:0")
    communicator.max_mnnvl_multimem_reduce_scatter_size = 64 * 1024 * 1024
    communicator.mnnvl_multimem_rs_supported = True
    communicator.mnnvl_multimem_rs_initialized = False
    communicator.mnnvl_multimem_rs_buffer = None
    communicator.mnnvl_multimem_rs_multicast_ptr = 0

    communicator._init_mnnvl_multimem_reduce_scatter_buffer()

    assert events == ["empty", "all_reduce"]
    assert communicator.mnnvl_multimem_rs_initialized
    assert communicator.mnnvl_multimem_rs_buffer is None
    assert communicator.mnnvl_multimem_rs_multicast_ptr == 0
    assert warnings == [
        "MNNVL multimem reduce-scatter symmetric-memory allocation "
        "failed on at least one rank; falling back to NCCL."
    ]


def test_mnnvl_multimem_reduce_scatter_warns_on_rendezvous_failure(monkeypatch):
    events = []

    class FakeSymmMem:
        @staticmethod
        def empty(*_args, **_kwargs):
            events.append("empty")
            return torch.empty(1, dtype=torch.uint8)

        @staticmethod
        def rendezvous(*_args, **_kwargs):
            events.append("rendezvous")
            raise RuntimeError("rendezvous failed")

    def preserve_local_result(_group_value, **_kwargs):
        events.append("all_reduce")

    warnings = []
    monkeypatch.setattr(car, "torch_symm_mem", FakeSymmMem)
    monkeypatch.setattr(car.dist, "all_reduce", preserve_local_result)
    monkeypatch.setattr(
        car.logger,
        "warning_once",
        lambda message, *_args, **_kwargs: warnings.append(message),
    )

    communicator = car.CustomAllreduce.__new__(car.CustomAllreduce)
    communicator.disabled = True
    communicator._ptr = 0
    communicator.group = type("Group", (), {"group_name": "test"})()
    communicator.device = torch.device("cuda:0")
    communicator.max_mnnvl_multimem_reduce_scatter_size = 64 * 1024 * 1024
    communicator.mnnvl_multimem_rs_supported = True
    communicator.mnnvl_multimem_rs_initialized = False
    communicator.mnnvl_multimem_rs_buffer = None
    communicator.mnnvl_multimem_rs_multicast_ptr = 0

    communicator._init_mnnvl_multimem_reduce_scatter_buffer()

    assert events == ["empty", "all_reduce", "rendezvous", "all_reduce"]
    assert communicator.mnnvl_multimem_rs_initialized
    assert communicator.mnnvl_multimem_rs_buffer is None
    assert communicator.mnnvl_multimem_rs_multicast_ptr == 0
    assert warnings == [
        "MNNVL multimem reduce-scatter symmetric-memory rendezvous "
        "failed on at least one rank; falling back to NCCL."
    ]


@ray.remote(num_gpus=1, max_calls=1)
def graph_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group().device_group

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
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                with graph_capture(device=device) as graph_capture_context:
                    # use integers so result matches NCCL exactly
                    device_idx = torch.accelerator.current_device_index()
                    inp1 = torch.randint(1, 16, (sz,), dtype=dtype, device=device_idx)
                    inp2 = torch.randint(1, 16, (sz,), dtype=dtype, device=device_idx)

                    torch.accelerator.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                        for i in range(num_communication):
                            out1 = tensor_model_parallel_all_reduce(inp1)
                            # the input buffer is immediately modified to test
                            # synchronization
                            dist.all_reduce(inp1, group=group)
                            out2 = tensor_model_parallel_all_reduce(inp2)
                            dist.all_reduce(inp2, group=group)
                graph.replay()
                torch.testing.assert_close(out1, inp1)
                torch.testing.assert_close(out2, inp2)


@ray.remote(num_gpus=1, max_calls=1)
def eager_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

        # we use the first group to communicate once
        # and the second group to communicate twice
        # and so on
        # this is used to demonstrate that each group can
        # communicate independently
        num_communication = rank // tp_size + 1
        sz = 1024
        fa = get_tp_group().device_communicator.ca_comm
        inp = torch.ones(sz, dtype=torch.float32, device=device)
        out = inp
        for _ in range(num_communication):
            out = fa.all_reduce(out, registered=False)
        torch.testing.assert_close(out, inp * (tp_size**num_communication))

        inp = torch.ones(sz * 4, dtype=torch.bfloat16, device=device)
        out = inp
        for _ in range(num_communication):
            out = fa.all_reduce(out, registered=False)
        torch.testing.assert_close(out, inp * (tp_size**num_communication))


@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
@pytest.mark.parametrize("test_target", [eager_allreduce, graph_allreduce])
def test_custom_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pipeline_parallel_size,
    test_target,
):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    multi_process_parallel(monkeypatch, tp_size, pipeline_parallel_size, test_target)
