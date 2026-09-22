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
    communicator.world_size = 2
    communicator.max_size = 1024
    communicator._ptr = 0

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


def _order_sensitive_inputs(
    numel: int, dtype: torch.dtype, world_size: int, seed: int
) -> torch.Tensor:
    """Per-rank inputs, shape [numel, world_size], whose sum depends on the
    order the ranks are accumulated in.

    Each element holds +B, -B and small terms t in a random rank order, with t
    below the float32 accumulator's resolution at B: adding t to B loses it,
    cancelling +B and -B first keeps it, and the difference survives the
    downcast. Ordinary random fp16/bf16 data cannot detect a changed order,
    because its float32 partial sums are exact.
    """
    log2_big, log2_tiny = {
        torch.float16: (13, -13),
        torch.bfloat16: (20, -8),
        torch.float32: (20, -8),
    }[dtype]
    gen = torch.Generator().manual_seed(seed)
    big = (torch.rand(numel, generator=gen, dtype=torch.float64) + 1) * 2.0**log2_big
    tiny = torch.rand(numel, world_size - 2, generator=gen, dtype=torch.float64) + 1
    sign = torch.randint(0, 2, tiny.shape, generator=gen) * 2 - 1
    tiny *= sign * 2.0**log2_tiny
    terms = torch.cat([big[:, None], -big[:, None], tiny], dim=1)
    order = torch.argsort(torch.rand(numel, world_size, generator=gen), dim=1)
    return torch.gather(terms, 1, order).to(dtype)


@ray.remote(num_gpus=1, max_calls=1)
def one_and_two_stage_allreduce(
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
        fa = get_tp_group().device_communicator.ca_comm
        assert fa is not None and not fa.disabled, "custom all-reduce is disabled"

        # Sizes are counts of 16-byte packed elements. They straddle the
        # 512 KiB 1-stage/2-stage threshold, and all but one leave a remainder
        # when divided by the world size, giving the last rank a larger part.
        for dtype, bits in [
            (torch.float16, torch.int16),
            (torch.bfloat16, torch.int16),
            (torch.float32, torch.int32),
        ]:
            for packed in [1021, 32771, 65536, 262139]:
                numel = packed * (16 // torch.empty((), dtype=dtype).element_size())
                case = f"{dtype}, {packed * 16} bytes"

                ints = torch.randint(
                    -8, 9, (numel, tp_size), generator=torch.Generator().manual_seed(0)
                ).to(dtype)
                exact = ints.sum(dim=1).to(device)
                inputs = _order_sensitive_inputs(numel, dtype, tp_size, seed=packed)
                outs = {}
                for algo in ["1stage", "2stage"]:
                    m.setenv("VLLM_CUSTOM_ALLREDUCE_ALGO", algo)
                    summed = fa.all_reduce(
                        ints[:, rank].contiguous().to(device), registered=False
                    )
                    torch.testing.assert_close(summed, exact, rtol=0, atol=0)
                    outs[algo] = fa.all_reduce(
                        inputs[:, rank].contiguous().to(device), registered=False
                    )
                torch.accelerator.synchronize()

                differ = outs["1stage"].view(bits) != outs["2stage"].view(bits)
                assert not differ.any(), (
                    f"{case}: 2-stage differs from 1-stage in "
                    f"{int(differ.sum())}/{numel} elements"
                )


def test_one_and_two_stage_allreduce_bitwise_identical(
    monkeypatch: pytest.MonkeyPatch,
):
    """Both kernels accumulate ranks in the same order, so which one the size
    heuristic selects cannot change the result's bits.

    Needs 4 ranks: at 2, the 2-stage owner rotation only turns a+b into b+a,
    which is commutative, so no 2-rank run can tell the orders apart.
    """
    tp_size = 4
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    if not car.custom_ar:
        pytest.skip("Custom all-reduce ops are not available.")
    physical_ids = [
        current_platform.visible_device_id_to_physical_device_id(i)
        for i in range(tp_size)
    ]
    if not current_platform.is_fully_connected(physical_ids):
        pytest.skip("Custom all-reduce needs fully connected GPUs above 2 ranks.")
    multi_process_parallel(monkeypatch, tp_size, 1, one_and_two_stage_allreduce)
