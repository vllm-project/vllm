# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce  # noqa
from vllm.distributed.device_communicators import custom_all_reduce as car
from vllm.distributed.parallel_state import get_tp_group, graph_capture
from vllm.forward_context import BatchDescriptor, override_forward_context
from vllm.platforms import current_platform

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

random.seed(42)
test_sizes = [random.randint(1024, 2048 * 1024) for _ in range(8)]
for i, v in enumerate(test_sizes):
    test_sizes[i] -= v % 8


@ray.remote(num_gpus=1, max_calls=1)
def _all_reduce_mhc(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm.model_executor.kernels.mhc.tilelang import (
        mhc_post_tilelang,
        mhc_pre_delayed_tilelang,
    )

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        comm = get_tp_group().device_communicator.ca_comm
        assert comm is not None and comm.mnnvl_lamport_ag_multicast_ptr
        fn = torch.zeros(24, 20480, device=device)
        scale = torch.ones(3, device=device)
        base = torch.zeros(24, device=device)

        # Changing shapes and interleaving all-gather exercise the shared
        # Lamport stages, including cleanup of a larger previous payload.
        def run(n):
            torch.manual_seed(42 + rank)
            x = torch.randn(n, 5120, device=device, dtype=torch.bfloat16)
            # Packed +0/-0 pairs collide with the Lamport sentinel.
            x[:, :16] = 0
            x[:, 9:16:2] = -0.0
            torch.manual_seed(123)
            residual = torch.randn(n, 4, 5120, device=device, dtype=torch.bfloat16)
            post = torch.rand(n, 4, device=device)
            comb = torch.randn(n, 4, 4, device=device) * 0.1
            pre = torch.rand(n, 4, device=device)
            weight = torch.randn(5120, device=device, dtype=torch.bfloat16)
            output = torch.empty_like(residual)
            normalized = torch.empty_like(x)

            def fused():
                torch.ops._C_custom_ar.all_reduce_mhc(
                    x,
                    residual,
                    post,
                    comb,
                    pre,
                    weight,
                    output,
                    normalized,
                    comm.mnnvl_lamport_ag_local_ptr,
                    comm.mnnvl_lamport_ag_multicast_ptr,
                    comm.mnnvl_lamport_epochs[0],
                    rank,
                    comm.mnnvl_buffer_size,
                    1e-6,
                )

            def check():
                gathered = comm.custom_all_gather(x)
                assert gathered is not None
                peers = gathered.view(tp_size, n, 5120).float()
                reduced = peers[0].clone()
                for peer in peers[1:]:
                    reduced.add_(peer)
                expected = mhc_post_tilelang(
                    reduced.bfloat16(), residual, post.unsqueeze(-1), comb
                )
                expected_norm = mhc_pre_delayed_tilelang(
                    expected,
                    fn,
                    scale,
                    base,
                    1e-6,
                    1e-6,
                    1e-6,
                    2.0,
                    20,
                    pre_mix=pre,
                    norm_weight=weight,
                )[2]
                torch.testing.assert_close(output, expected, rtol=0.008, atol=1e-6)
                torch.testing.assert_close(
                    normalized, expected_norm, rtol=0.008, atol=0.008
                )

            fused()
            check()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(6):
                    fused()
            for _ in range(20):
                graph.replay()
            check()
            x.mul_(0.5)
            residual.mul_(2)
            graph.replay()
            check()

        # Cover fixed Q6 and shrinking/growing adaptive verification batches.
        for n in (1, 6, 12, 8, 16, 3, 5, 2, 4, 1):
            run(n)


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="Requires SM100"
)
def test_all_reduce_mhc_preserves_bf16_boundaries_and_graph_replay(monkeypatch):
    if torch.accelerator.device_count() < 4:
        pytest.skip("Requires four GPUs with NVLink multicast")
    multi_process_parallel(monkeypatch, 4, 1, _all_reduce_mhc)


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
    monkeypatch.setattr(car.ops, "meta_size", lambda: 128)
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
    monkeypatch.setattr(car.ops, "meta_size", lambda: 128)
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


def test_mnnvl_multimem_reduce_scatter_initializes_signals(monkeypatch):
    events = []
    buffers = []

    class FakeHandle:
        multicast_ptr = 0x3000

    class FakeSymmMem:
        @staticmethod
        def empty(size, **_kwargs):
            events.append(("empty", size))
            buffer = torch.ones(size, dtype=torch.uint8)
            buffers.append(buffer)
            return buffer

        @staticmethod
        def rendezvous(*_args, **_kwargs):
            events.append(("rendezvous", None))
            return FakeHandle()

    def preserve_local_result(*_args, **_kwargs):
        events.append(("all_reduce", None))

    monkeypatch.setattr(car, "torch_symm_mem", FakeSymmMem)
    monkeypatch.setattr(car.ops, "meta_size", lambda: 128)
    monkeypatch.setattr(
        car.torch.accelerator,
        "synchronize",
        lambda: events.append(("synchronize", None)),
    )
    monkeypatch.setattr(car.dist, "all_reduce", preserve_local_result)

    communicator = car.CustomAllreduce.__new__(car.CustomAllreduce)
    communicator.disabled = True
    communicator._ptr = 0
    communicator.group = type("Group", (), {"group_name": "test"})()
    communicator.device = torch.device("cpu")
    communicator.max_mnnvl_multimem_reduce_scatter_size = 129
    communicator.mnnvl_multimem_rs_supported = True
    communicator.mnnvl_multimem_rs_initialized = False
    communicator.mnnvl_multimem_rs_buffer = None
    communicator.mnnvl_multimem_rs_multicast_ptr = 0

    communicator._init_mnnvl_multimem_reduce_scatter_buffer()

    assert events == [
        ("empty", 257),
        ("all_reduce", None),
        ("rendezvous", None),
        ("synchronize", None),
        ("all_reduce", None),
    ]
    assert torch.all(buffers[0][:128] == 0)
    assert torch.all(buffers[0][128:] == 1)
    assert communicator.mnnvl_multimem_rs_buffer_size == 129
    assert communicator.mnnvl_multimem_rs_local_ptr == buffers[0].data_ptr() + 128
    assert communicator.mnnvl_multimem_rs_multicast_ptr == 0x3080


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


@pytest.mark.parametrize(
    (
        "arch",
        "shape",
        "dtype",
        "capturing",
        "batch_kind",
        "batch_size",
        "expected",
    ),
    [
        ("gfx1201", (128, 8192), torch.bfloat16, True, "decode", 128, True),
        ("gfx1201", (1, 8), torch.float16, True, "decode", 1, True),
        ("gfx1201", (1, 8), torch.float16, False, "decode", 1, False),
        ("gfx1201", (129, 8), torch.float16, True, "decode", 129, False),
        ("gfx1201", (1, 8200), torch.float16, True, "decode", 1, False),
        ("gfx1201", (0, 8), torch.float16, True, "decode", 0, False),
        ("gfx1201", (1, 7), torch.float16, True, "decode", 1, False),
        ("gfx1201", (8,), torch.float16, True, "decode", 8, False),
        ("gfx1201", (1, 8), torch.float32, True, "decode", 1, False),
        ("gfx1201", (40, 2048), torch.bfloat16, True, "none", 40, False),
        ("gfx1201", (40, 2048), torch.bfloat16, True, "prefill", 40, False),
        ("gfx1201", (16, 2048), torch.bfloat16, True, "mixed", 16, False),
        ("gfx1201", (16, 2048), torch.bfloat16, True, "decode", 8, False),
        ("gfx1100", (16, 8184), torch.bfloat16, True, "decode", 16, True),
        ("gfx1100", (16, 8192), torch.bfloat16, True, "decode", 16, False),
    ],
)
def test_rdna_routes_only_supported_decode_graphs(
    monkeypatch, arch, shape, dtype, capturing, batch_kind, batch_size, expected
):
    from vllm.distributed.device_communicators.rdna_custom_all_reduce import (
        RdnaCustomAllreduce,
    )

    comm = RdnaCustomAllreduce.__new__(RdnaCustomAllreduce)
    comm.disabled = False
    comm.device = torch.device("cpu")
    comm.arch = arch
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    if batch_kind == "none":
        forward_context = None
    else:
        forward_context = SimpleNamespace(
            batch_descriptor=BatchDescriptor(
                num_tokens=batch_size,
                num_reqs=batch_size if batch_kind != "prefill" else 1,
                uniform=batch_kind == "decode",
            )
        )
    with override_forward_context(forward_context):
        assert comm.should_custom_ar(torch.empty(shape, dtype=dtype)) is expected


def test_rdna_option_mismatch_disables_whole_group(monkeypatch):
    from vllm.distributed.device_communicators.rdna_custom_all_reduce import (
        RdnaCustomAllreduce,
    )

    monkeypatch.setattr(dist, "get_rank", lambda _: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda _: 2)
    monkeypatch.setattr(
        RdnaCustomAllreduce, "_gather", lambda *_: [(True, True), (False, False)]
    )
    allocate = Mock(side_effect=AssertionError("must not allocate GPU memory"))
    monkeypatch.setattr(torch, "empty", allocate)
    comm = RdnaCustomAllreduce(object(), torch.device("cuda:0"), enabled=True)
    assert comm.disabled
    allocate.assert_not_called()


@ray.remote(num_gpus=1, max_calls=1)
def rdna_graph_allreduce(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    import torch

    from vllm.distributed.device_communicators.rdna_custom_all_reduce import (
        RdnaCustomAllreduce,
    )
    from vllm.distributed.parallel_state import destroy_model_parallel

    monkeypatch.setenv("VLLM_ROCM_USE_RDNA_ALL_REDUCE", "1")
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    group = get_tp_group()
    backend = group.device_communicator.rdna_ar_comm
    assert backend is not None and not backend.disabled
    assert group.device_communicator.ca_comm is None
    assert group.device_communicator.qr_comm is None
    eager = torch.full((1, 8), rank + 1, device=device, dtype=torch.bfloat16)
    assert backend.custom_all_reduce(eager) is None
    torch.testing.assert_close(
        tensor_model_parallel_all_reduce(eager),
        torch.full_like(eager, tp_size * (tp_size + 1) / 2),
    )

    for dtype in (torch.bfloat16, torch.float16):
        # Different rank alignment must not produce divergent routing.
        storage = torch.empty(2049, device=device, dtype=dtype)
        inp = storage[rank % 2 : rank % 2 + 2048].view(1, 2048)
        large = torch.full((129, 8), rank + 1, device=device, dtype=dtype)
        inp.fill_(rank + 1)
        decode_context = SimpleNamespace(
            batch_descriptor=BatchDescriptor(
                num_tokens=1,
                num_reqs=1,
                uniform=True,
            )
        )
        with (
            override_forward_context(decode_context),
            graph_capture(device=device) as capture,
        ):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture.stream):
                assert backend.should_custom_ar(inp)
                out = tensor_model_parallel_all_reduce(inp)

        fallback_context = SimpleNamespace(
            batch_descriptor=BatchDescriptor(
                num_tokens=129,
                num_reqs=129,
                uniform=True,
            )
        )
        with (
            override_forward_context(fallback_context),
            graph_capture(device=device) as capture,
        ):
            fallback_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(fallback_graph, stream=capture.stream):
                assert not backend.should_custom_ar(large)
                fallback = tensor_model_parallel_all_reduce(large)
        for replay in range(3):
            if tp_size == 4:
                # Exposes the old rank-dependent FP32 accumulation order.
                base = 2.0 ** (80 if dtype == torch.bfloat16 else 15)
                levels = (base, -base, torch.finfo(dtype).eps, torch.finfo(dtype).eps)
                inp.fill_(levels[rank] if rank < 2 else levels[rank] * (replay + 1))
                expected = 2 * torch.finfo(dtype).eps * (replay + 1)
                if backend.arch == "gfx1100":
                    # RDNA3's pairwise path rounds each pair to the input dtype.
                    expected = float(torch.tensor(expected, dtype=dtype))
            else:
                inp.fill_((rank + 1) * (replay + 1))
                expected = 3 * (replay + 1)
            saved = inp.clone()
            graph.replay()
            fallback_graph.replay()
            torch.testing.assert_close(
                out, torch.full_like(out, expected), rtol=0, atol=0
            )
            torch.testing.assert_close(inp, saved, rtol=0, atol=0)
            torch.testing.assert_close(
                fallback, torch.full_like(large, tp_size * (tp_size + 1) / 2)
            )
        del graph
        del fallback_graph
    backend.close()
    for operation in ("allocate_shared_buffer_and_handle", "open_mem_handle"):
        original = getattr(torch.ops._rdna_custom_ar, operation)

        def fail_one_rank(*args, original=original):
            if rank == 0:
                raise RuntimeError("injected IPC initialization failure")
            return original(*args)

        with monkeypatch.context() as patch:
            patch.setattr(torch.ops._rdna_custom_ar, operation, fail_one_rank)
            failed = RdnaCustomAllreduce(group.cpu_group, device, enabled=True)
        assert failed.disabled and failed._closed
        assert failed._shared == 0 and not failed._opened
    dist.barrier(group=group.cpu_group)
    destroy_model_parallel()


@pytest.mark.parametrize("tp_size", [2, 4])
def test_rdna_vllm_graph_and_fallback(monkeypatch, tp_size):
    from vllm.platforms import current_platform

    if not current_platform.is_rocm() or torch.accelerator.device_count() < tp_size:
        pytest.skip("Requires RDNA GPUs with peer access")
    arches = [
        torch.cuda.get_device_properties(i).gcnArchName.split(":")[0]
        for i in range(tp_size)
    ]
    if len(set(arches)) != 1 or arches[0] not in ("gfx1100", "gfx1201"):
        pytest.skip("Requires homogeneous gfx1100 or gfx1201 GPUs")
    multi_process_parallel(monkeypatch, tp_size, 1, rdna_graph_allreduce)
