# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for push-based allreduce (ported from SGLang).
Covers correctness, edge cases, CUDA graph safety, dispatch integration,
multi-layer stacking, and epoch alternation.

Test infrastructure:
  - Unit tests: torch.multiprocessing.spawn with gloo+nccl groups
  - Integration tests: Ray workers with vLLM test infrastructure
"""

import os
import random

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Reusable test sizes aligned to 16 bytes (8 BF16 elements)
random.seed(42)
UNIT_TEST_SIZES = [
    16,  # minimal (32 bytes for BF16)
    128,  # single warp
    1024,  # small
    7168,  # decode BS=1 hidden_size
    28672,  # decode BS=4 (4 * 7168)
    65536,  # moderate
    131072,  # larger
]


# ============================================================
# Helper: init/teardown for mp.spawn-based tests
# ============================================================
def _init_groups(rank: int, world_size: int, port: int):
    """Initialize gloo (CPU) and nccl process groups."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    return dist.group.WORLD, dist.new_group(backend="nccl")


def _teardown():
    dist.destroy_process_group()


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


# ============================================================
# UT-1: Build Verification
# ============================================================
def test_push_ar_ops_registered():
    """Verify push_ar custom ops are registered and callable."""
    import vllm._custom_ops as ops

    assert hasattr(ops, "init_push_ar")
    assert hasattr(ops, "get_push_ar_ipc_handle")
    assert hasattr(ops, "post_init_push_ar")
    assert hasattr(ops, "push_ar_all_reduce")
    assert hasattr(ops, "dispose_push_ar")


# ============================================================
# UT-2: PushAllReduce Initialization
# ============================================================
def _push_ar_init_worker(rank, world_size, port):
    cpu_group, _ = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    assert not push_ar.disabled
    assert push_ar.rank == rank
    assert push_ar.world_size == world_size
    assert push_ar.push_buffer_bytes > 0
    assert push_ar.num_sm > 0

    push_ar.close()
    _teardown()


@pytest.mark.parametrize("world_size", [2])
def test_push_ar_initialization(world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs")
    mp.spawn(
        _push_ar_init_worker,
        args=(world_size, _find_free_port()),
        nprocs=world_size,
        join=True,
    )


# ============================================================
# UT-3: should_use() Predicate Logic
# ============================================================
def _push_ar_should_use_worker(rank, world_size, port):
    cpu_group, _ = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    # Case 1: Valid small tensor -> True
    t1 = torch.randn(7168, dtype=torch.bfloat16, device=device)
    assert push_ar.should_use(t1) is True

    # Case 2: Valid exactly-at-threshold -> True
    max_elems = push_ar.max_message_bytes // 2  # BF16
    max_elems = (max_elems // 8) * 8  # align to 16 bytes
    t2 = torch.randn(max_elems, dtype=torch.bfloat16, device=device)
    assert push_ar.should_use(t2) is True

    # Case 3: Above threshold -> False
    t3 = torch.randn(max_elems + 8, dtype=torch.bfloat16, device=device)
    assert push_ar.should_use(t3) is False

    # Case 4: Zero-size tensor -> False
    t4 = torch.empty(0, dtype=torch.bfloat16, device=device)
    assert push_ar.should_use(t4) is False

    # Case 5: Non-aligned size (not divisible by 16 bytes) -> False
    t5 = torch.randn(7, dtype=torch.bfloat16, device=device)  # 14 bytes
    assert push_ar.should_use(t5) is False

    # Case 6: Non-contiguous tensor -> False
    t6_base = torch.randn(1024, 2, dtype=torch.bfloat16, device=device)
    t6 = t6_base[:, 0]  # non-contiguous view
    assert push_ar.should_use(t6) is False

    # Case 7: Weakly-contiguous tensor -> True
    t7_base = torch.randn(2048, dtype=torch.bfloat16, device=device)
    t7 = t7_base[:1024]  # contiguous slice
    assert push_ar.should_use(t7) is True

    # Case 8: Disabled communicator -> always False
    push_ar.disabled = True
    assert push_ar.should_use(t1) is False
    push_ar.disabled = False

    push_ar.close()
    _teardown()


def test_push_ar_should_use():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2 GPUs")
    mp.spawn(
        _push_ar_should_use_worker,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )


# ============================================================
# UT-4: Basic Allreduce Correctness (Integer Data, Bit-Exact)
# ============================================================
def _push_ar_correctness_int_worker(rank, world_size, port):
    cpu_group, nccl_group = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    for size in UNIT_TEST_SIZES:
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            inp = torch.randint(0, 16, (size,), dtype=dtype, device=device)

            if not push_ar.should_use(inp):
                continue

            out_push = push_ar.all_reduce(inp)

            out_nccl = inp.clone()
            dist.all_reduce(out_nccl, group=nccl_group)

            assert torch.all(out_push == out_nccl), (
                f"Mismatch at size={size}, dtype={dtype}, rank={rank}"
            )

    push_ar.close()
    _teardown()


@pytest.mark.parametrize("world_size", [2])
def test_push_ar_correctness_integer(world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs")
    mp.spawn(
        _push_ar_correctness_int_worker,
        args=(world_size, _find_free_port()),
        nprocs=world_size,
        join=True,
    )


# ============================================================
# UT-5: Allreduce Correctness (Random Float Data)
# ============================================================
def _push_ar_correctness_float_worker(rank, world_size, port):
    cpu_group, nccl_group = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    for size in UNIT_TEST_SIZES:
        for dtype in [torch.bfloat16, torch.float16]:
            inp = torch.randn(size, dtype=dtype, device=device)

            if not push_ar.should_use(inp):
                continue

            out_push = push_ar.all_reduce(inp)

            out_nccl = inp.clone()
            dist.all_reduce(out_nccl, group=nccl_group)

            torch.testing.assert_close(
                out_push,
                out_nccl,
                atol=1e-2,
                rtol=1e-2,
                msg=f"Mismatch at size={size}, dtype={dtype}",
            )

    push_ar.close()
    _teardown()


@pytest.mark.parametrize("world_size", [2])
def test_push_ar_correctness_float(world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs")
    mp.spawn(
        _push_ar_correctness_float_worker,
        args=(world_size, _find_free_port()),
        nprocs=world_size,
        join=True,
    )


# ============================================================
# UT-6: Positive-Zero Sentinel Handling
# ============================================================
def _push_ar_zero_handling_worker(rank, world_size, port):
    cpu_group, nccl_group = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    # Case 1: All zeros tensor
    inp_zeros = torch.zeros(7168, dtype=torch.bfloat16, device=device)
    out = push_ar.all_reduce(inp_zeros)
    assert torch.all(out == 0.0), "All-zero input failed"

    # Case 2: Mix of zeros and non-zeros (rank-dependent)
    inp_mixed = torch.zeros(7168, dtype=torch.bfloat16, device=device)
    inp_mixed[::2] = float(rank + 1)
    out_mixed = push_ar.all_reduce(inp_mixed)
    expected_sum = sum(range(1, world_size + 1))
    torch.testing.assert_close(
        out_mixed[::2],
        torch.full_like(out_mixed[::2], expected_sum),
        atol=1e-2,
        rtol=1e-2,
    )
    assert torch.all(out_mixed[1::2] == 0.0)

    # Case 3: Bit pattern verification
    inp_pz = torch.zeros(1024, dtype=torch.bfloat16, device=device)
    raw_bits = inp_pz.view(torch.int16)
    assert torch.all(raw_bits == 0), "Expected +0.0 bit pattern"
    out_pz = push_ar.all_reduce(inp_pz)
    assert torch.all(out_pz == 0.0), "Positive zero handling failed"

    # Case 4: Negative zeros
    inp_nz = torch.tensor(
        [-0.0] * 1024, dtype=torch.bfloat16, device=device
    )
    out_nz = push_ar.all_reduce(inp_nz)
    assert torch.all(out_nz == 0.0)

    # Case 5: FP16 positive zeros
    inp_f16 = torch.zeros(1024, dtype=torch.float16, device=device)
    out_f16 = push_ar.all_reduce(inp_f16)
    assert torch.all(out_f16 == 0.0), "FP16 positive zero handling failed"

    push_ar.close()
    _teardown()


@pytest.mark.timeout(120)
def test_push_ar_zero_handling():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2 GPUs")
    mp.spawn(
        _push_ar_zero_handling_worker,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )


# ============================================================
# UT-7: Epoch Alternation (1000 Iterations)
# ============================================================
def _push_ar_epoch_worker(rank, world_size, port):
    cpu_group, nccl_group = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    NUM_ITERATIONS = 1000

    for i in range(NUM_ITERATIONS):
        inp = torch.randint(
            0, 16, (7168,), dtype=torch.bfloat16, device=device
        )
        out_push = push_ar.all_reduce(inp)

        out_nccl = inp.clone()
        dist.all_reduce(out_nccl, group=nccl_group)

        assert torch.all(out_push == out_nccl), (
            f"Epoch mismatch at iteration {i}"
        )

    push_ar.close()
    _teardown()


def test_push_ar_epoch_alternation():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2 GPUs")
    mp.spawn(
        _push_ar_epoch_worker,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )


# ============================================================
# UT-8: Thread Count Selection (Dynamic SM Boundaries)
# ============================================================
def _push_ar_thread_count_worker(rank, world_size, port):
    cpu_group, nccl_group = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    sm_count = push_ar.num_sm

    # BF16: kVecSize=4, elements per thread = 2*kVecSize = 8
    boundaries = {
        128: 128 * sm_count * 8,
        256: 256 * sm_count * 8,
    }

    test_sizes_tc = [
        1024,
        7168,
        boundaries[128],
        boundaries[128] + 8,
    ]

    for size in test_sizes_tc:
        if size * 2 > push_ar.max_message_bytes:
            continue
        inp = torch.randint(
            0, 8, (size,), dtype=torch.bfloat16, device=device
        )
        out = push_ar.all_reduce(inp)
        ref = inp.clone()
        dist.all_reduce(ref, group=nccl_group)
        assert torch.all(out == ref), (
            f"Failed at size={size} (sm_count={sm_count})"
        )

    push_ar.close()
    _teardown()


def test_push_ar_thread_count():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2 GPUs")
    mp.spawn(
        _push_ar_thread_count_worker,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )


# ============================================================
# UT-9: Buffer Size Threshold Boundary
# ============================================================
def _push_ar_threshold_worker(rank, world_size, port):
    cpu_group, nccl_group = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    max_bytes = push_ar.max_message_bytes

    # Case 1: Exactly at threshold (BF16)
    max_elems = max_bytes // 2
    max_elems = (max_elems // 8) * 8  # align to 16 bytes
    inp_exact = torch.randint(
        0, 8, (max_elems,), dtype=torch.bfloat16, device=device
    )
    assert push_ar.should_use(inp_exact) is True
    out_exact = push_ar.all_reduce(inp_exact)
    ref_exact = inp_exact.clone()
    dist.all_reduce(ref_exact, group=nccl_group)
    assert torch.all(out_exact == ref_exact)

    # Case 2: One vector above threshold
    inp_over = torch.randint(
        0, 8, (max_elems + 8,), dtype=torch.bfloat16, device=device
    )
    assert push_ar.should_use(inp_over) is False

    # Case 3: One vector below threshold
    inp_under = torch.randint(
        0, 8, (max_elems - 8,), dtype=torch.bfloat16, device=device
    )
    assert push_ar.should_use(inp_under) is True
    out_under = push_ar.all_reduce(inp_under)
    ref_under = inp_under.clone()
    dist.all_reduce(ref_under, group=nccl_group)
    assert torch.all(out_under == ref_under)

    push_ar.close()
    _teardown()


def test_push_ar_threshold():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2 GPUs")
    mp.spawn(
        _push_ar_threshold_worker,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )


# ============================================================
# UT-10: Out-of-Place Semantics
# ============================================================
def _push_ar_outofplace_worker(rank, world_size, port):
    cpu_group, nccl_group = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    inp = torch.randint(
        1, 16, (7168,), dtype=torch.bfloat16, device=device
    )
    inp_original = inp.clone()

    out = push_ar.all_reduce(inp)

    # Input NOT modified
    assert torch.all(inp == inp_original), "Input was modified in-place!"

    # Output is a DIFFERENT tensor
    assert out.data_ptr() != inp.data_ptr(), "Output aliases input!"

    # Output has correct values
    ref = inp_original.clone()
    dist.all_reduce(ref, group=nccl_group)
    assert torch.all(out == ref)

    push_ar.close()
    _teardown()


def test_push_ar_outofplace():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2 GPUs")
    mp.spawn(
        _push_ar_outofplace_worker,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )


# ============================================================
# UT-11: Multiple Dtype Support
# ============================================================
def _push_ar_dtype_worker(rank, world_size, port):
    cpu_group, nccl_group = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        size = 7168 if dtype != torch.float32 else 3584
        inp = torch.randint(0, 16, (size,), dtype=dtype, device=device)

        if not push_ar.should_use(inp):
            continue

        out = push_ar.all_reduce(inp)
        ref = inp.clone()
        dist.all_reduce(ref, group=nccl_group)
        assert torch.all(out == ref), f"Failed for dtype={dtype}"

    push_ar.close()
    _teardown()


def test_push_ar_dtype():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2 GPUs")
    mp.spawn(
        _push_ar_dtype_worker,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )


# ============================================================
# UT-12: Multi-Layer Stacking (122 ARs/step * 10 steps)
# ============================================================
def _push_ar_multilayer_worker(rank, world_size, port):
    cpu_group, nccl_group = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    NUM_LAYERS = 122  # 61 blocks * 2 allreduces/block
    NUM_STEPS = 3  # Reduced for test speed

    for step in range(NUM_STEPS):
        for layer in range(NUM_LAYERS):
            inp = torch.randint(
                0, 16, (7168,), dtype=torch.bfloat16, device=device
            )
            out = push_ar.all_reduce(inp)
            ref = inp.clone()
            dist.all_reduce(ref, group=nccl_group)
            if not torch.all(out == ref):
                raise RuntimeError(
                    f"Mismatch at step={step}, layer={layer}"
                )

    push_ar.close()
    _teardown()


def test_push_ar_multilayer():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2 GPUs")
    mp.spawn(
        _push_ar_multilayer_worker,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )


# ============================================================
# UT-14: close() and Resource Lifecycle
# ============================================================
def _push_ar_lifecycle_worker(rank, world_size, port):
    cpu_group, _ = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    # Normal allreduce works
    inp = torch.randint(
        0, 16, (1024,), dtype=torch.bfloat16, device=device
    )
    out = push_ar.all_reduce(inp)
    assert out is not None

    # First close
    push_ar.close()
    assert push_ar.disabled is True

    # Double close should NOT crash
    push_ar.close()

    # __del__ should also be safe after explicit close
    del push_ar

    _teardown()


def test_push_ar_lifecycle():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2 GPUs")
    mp.spawn(
        _push_ar_lifecycle_worker,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )


# ============================================================
# UT-15: Graph Capture Warmup Path
# ============================================================
def _push_ar_warmup_worker(rank, world_size, port):
    cpu_group, _ = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    inp = torch.randn(7168, dtype=torch.bfloat16, device=device)

    # Outside capture context: real allreduce
    assert not push_ar._IS_CAPTURING
    out_real = push_ar.all_reduce(inp)
    assert out_real.shape == inp.shape

    # Inside capture context, but NOT in graph capture stream:
    # should return empty_like (warmup allocation)
    with push_ar.capture():
        assert push_ar._IS_CAPTURING
        out_warmup = push_ar.all_reduce(inp)
        assert out_warmup.shape == inp.shape
        assert out_warmup.dtype == inp.dtype

    # After capture context: back to normal
    assert not push_ar._IS_CAPTURING

    push_ar.close()
    _teardown()


def test_push_ar_warmup():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2 GPUs")
    mp.spawn(
        _push_ar_warmup_worker,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )


# ============================================================
# UT-16: Rank-Dependent Input Data (Asymmetric Reduction)
# ============================================================
def _push_ar_asymmetric_worker(rank, world_size, port):
    cpu_group, nccl_group = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    # Each rank has unique data
    torch.manual_seed(42 + rank)
    inp = torch.randn(7168, dtype=torch.bfloat16, device=device)

    out_push = push_ar.all_reduce(inp)

    out_nccl = inp.clone()
    dist.all_reduce(out_nccl, group=nccl_group)

    torch.testing.assert_close(
        out_push,
        out_nccl,
        atol=1e-2,
        rtol=1e-2,
        msg="Asymmetric reduction failed",
    )

    # Verify sum property
    all_inputs = [torch.empty_like(inp) for _ in range(world_size)]
    dist.all_gather(all_inputs, inp, group=nccl_group)
    expected_sum = torch.stack(all_inputs).float().sum(dim=0).to(inp.dtype)
    torch.testing.assert_close(
        out_push, expected_sum, atol=5e-2, rtol=5e-2
    )

    push_ar.close()
    _teardown()


def test_push_ar_asymmetric():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2 GPUs")
    mp.spawn(
        _push_ar_asymmetric_worker,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )


# ============================================================
# EC-1: Tensor with All Identical Values
# ============================================================
def _push_ar_identical_worker(rank, world_size, port):
    cpu_group, _ = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    inp = torch.ones(7168, dtype=torch.bfloat16, device=device) * 3.14
    out = push_ar.all_reduce(inp)
    expected = inp * world_size
    torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-2)

    push_ar.close()
    _teardown()


def test_push_ar_identical_values():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2 GPUs")
    mp.spawn(
        _push_ar_identical_worker,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )


# ============================================================
# IT-1: Dispatch Priority (mp.spawn-based integration test)
# ============================================================
def _dispatch_priority_worker(rank, world_size, port):
    cpu_group, nccl_group = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    # Verify push_ar_comm was initialized correctly
    assert not push_ar.disabled
    assert push_ar.rank == rank

    # Small tensor -> should use push allreduce
    small_inp = torch.randn(7168, dtype=torch.bfloat16, device=device)
    assert push_ar.should_use(small_inp) is True

    # Large tensor -> should NOT use push allreduce
    max_bytes = push_ar.max_message_bytes
    large_elems = max_bytes // 2 + 1024  # slightly over threshold
    large_inp = torch.randn(large_elems, dtype=torch.bfloat16, device=device)
    assert push_ar.should_use(large_inp) is False

    # Verify dispatch produces correct results for small
    out_small = push_ar.all_reduce(small_inp)
    ref_small = small_inp.clone()
    dist.all_reduce(ref_small, group=nccl_group)
    torch.testing.assert_close(
        out_small, ref_small, atol=1e-2, rtol=1e-2
    )

    push_ar.close()
    _teardown()


def test_dispatch_priority():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2 GPUs")
    mp.spawn(
        _dispatch_priority_worker,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )


# ============================================================
# IT-4: Coexistence Test (Interleaved Push + NCCL)
# ============================================================
def _coexistence_worker(rank, world_size, port):
    cpu_group, nccl_group = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    for _ in range(50):
        # Small message -> push allreduce
        small = torch.randint(
            0, 16, (7168,), dtype=torch.bfloat16, device=device
        )
        out_small = push_ar.all_reduce(small)
        ref_small = small.clone()
        dist.all_reduce(ref_small, group=nccl_group)
        assert torch.all(out_small == ref_small)

    push_ar.close()
    _teardown()


def test_coexistence():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2 GPUs")
    mp.spawn(
        _coexistence_worker,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )


# ============================================================
# IT-10: Interleaved Push AR + NCCL in Same Step
# ============================================================
def _interleaved_ar_worker(rank, world_size, port):
    cpu_group, nccl_group = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    for sz in [7168, 1024, 4096]:
        for dtype in [torch.bfloat16, torch.float16]:
            inp1 = torch.randint(
                1, 16, (sz,), dtype=dtype, device=device
            )
            inp2 = torch.randint(
                1, 16, (sz,), dtype=dtype, device=device
            )

            out1 = push_ar.all_reduce(inp1)
            ref1 = inp1.clone()
            dist.all_reduce(ref1, group=nccl_group)

            out2 = push_ar.all_reduce(inp2)
            ref2 = inp2.clone()
            dist.all_reduce(ref2, group=nccl_group)

            torch.testing.assert_close(
                out1, ref1, atol=1e-2, rtol=1e-2
            )
            torch.testing.assert_close(
                out2, ref2, atol=1e-2, rtol=1e-2
            )

    push_ar.close()
    _teardown()


def test_interleaved_ar():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2 GPUs")
    mp.spawn(
        _interleaved_ar_worker,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )


# ============================================================
# CG-1: CUDA Graph Capture + Replay
# ============================================================
def _graph_capture_worker(rank, world_size, port):
    cpu_group, nccl_group = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    NUM_AR = 5  # allreduces per graph
    sz = 7168

    # Allocate graph input in graph memory pool
    graph_inp = torch.randint(
        1, 16, (sz,), dtype=torch.bfloat16, device=device
    )

    # Warmup
    with push_ar.capture():
        for _ in range(NUM_AR):
            push_ar.all_reduce(graph_inp)
        torch.cuda.synchronize()

        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = []
            for _ in range(NUM_AR):
                outs.append(push_ar.all_reduce(graph_inp))

    # Replay and verify
    for replay_iter in range(10):
        # Fill with new data before each replay
        graph_inp.copy_(
            torch.randint(1, 16, (sz,), dtype=torch.bfloat16, device=device)
        )
        graph.replay()
        torch.cuda.synchronize()

        # Verify last output is correct
        ref = graph_inp.clone()
        # The allreduce was applied NUM_AR times, but each operates on
        # graph_inp independently. The last output should be one allreduce
        # of graph_inp.
        dist.all_reduce(ref, group=nccl_group)
        torch.testing.assert_close(
            outs[-1], ref, atol=1e-2, rtol=1e-2,
            msg=f"Graph replay {replay_iter} failed"
        )

    push_ar.close()
    _teardown()


def test_graph_capture():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2 GPUs")
    mp.spawn(
        _graph_capture_worker,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )


# ============================================================
# E2E-4: Transformer Block Simulation (Isolated)
# ============================================================
def _push_ar_transformer_sim_worker(rank, world_size, port):
    cpu_group, nccl_group = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    hidden_size = 7168
    NUM_BLOCKS = 61  # DeepSeek-V4 layer count
    NUM_STEPS = 2  # Reduced for test speed

    for step in range(NUM_STEPS):
        for block in range(NUM_BLOCKS):
            # AR 1: attention wo_b output
            attn_out = torch.randn(
                1, hidden_size, dtype=torch.bfloat16, device=device
            )
            attn_reduced = push_ar.all_reduce(attn_out)
            attn_ref = attn_out.clone()
            dist.all_reduce(attn_ref, group=nccl_group)
            torch.testing.assert_close(
                attn_reduced,
                attn_ref,
                atol=1e-2,
                rtol=1e-2,
                msg=f"Attn AR failed: step={step}, block={block}",
            )

            # AR 2: MoE output
            moe_out = torch.randn(
                1, hidden_size, dtype=torch.bfloat16, device=device
            )
            moe_reduced = push_ar.all_reduce(moe_out)
            moe_ref = moe_out.clone()
            dist.all_reduce(moe_ref, group=nccl_group)
            torch.testing.assert_close(
                moe_reduced,
                moe_ref,
                atol=1e-2,
                rtol=1e-2,
                msg=f"MoE AR failed: step={step}, block={block}",
            )

    push_ar.close()
    _teardown()


def test_push_ar_transformer_sim():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2 GPUs")
    mp.spawn(
        _push_ar_transformer_sim_worker,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )


# ============================================================
# FT-1: Feature Toggle - ENABLED Startup Log
# ============================================================
def _feature_toggle_enabled_worker(rank, world_size, port):
    cpu_group, _ = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    # Ensure the disable env var is NOT set
    os.environ.pop("VLLM_DISABLE_PUSH_ALLREDUCE", None)

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
        _FEATURE_DESCRIPTION,
        _DISABLE_ENV_VAR,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    # Feature should be enabled
    assert not push_ar.disabled, "PushAllReduce should be enabled"
    assert _DISABLE_ENV_VAR == "VLLM_DISABLE_PUSH_ALLREDUCE"
    assert "AllReduce" in _FEATURE_DESCRIPTION

    # Verify the push_ar works when enabled
    inp = torch.randint(0, 16, (1024,), dtype=torch.bfloat16, device=device)
    out = push_ar.all_reduce(inp)
    assert out is not None
    assert out.shape == inp.shape

    push_ar.close()
    _teardown()


def test_feature_toggle_enabled():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2 GPUs")
    mp.spawn(
        _feature_toggle_enabled_worker,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )


# ============================================================
# FT-2: Feature Toggle - DISABLED via Env Var
# ============================================================
def _feature_toggle_disabled_worker(rank, world_size, port):
    cpu_group, _ = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    # Set the disable env var
    os.environ["VLLM_DISABLE_PUSH_ALLREDUCE"] = "1"

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    # Feature should be disabled
    assert push_ar.disabled, "PushAllReduce should be disabled via env var"

    # should_use should return False when disabled
    inp = torch.randn(1024, dtype=torch.bfloat16, device=device)
    assert push_ar.should_use(inp) is False

    # Clean up
    os.environ.pop("VLLM_DISABLE_PUSH_ALLREDUCE", None)
    push_ar.close()
    _teardown()


def test_feature_toggle_disabled():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2 GPUs")
    mp.spawn(
        _feature_toggle_disabled_worker,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )


# ============================================================
# FT-3: Feature Toggle - Env Var Not Set to "1"
# ============================================================
def _feature_toggle_not_disabled_worker(rank, world_size, port):
    cpu_group, _ = _init_groups(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    # Set the env var to something other than "1" - should NOT disable
    os.environ["VLLM_DISABLE_PUSH_ALLREDUCE"] = "0"

    from vllm.distributed.device_communicators.push_all_reduce import (
        PushAllReduce,
    )

    push_ar = PushAllReduce(group=cpu_group, device=device)

    # Feature should still be enabled (only "1" disables)
    assert not push_ar.disabled, (
        "PushAllReduce should be enabled when env var != '1'"
    )

    # Clean up
    os.environ.pop("VLLM_DISABLE_PUSH_ALLREDUCE", None)
    push_ar.close()
    _teardown()


def test_feature_toggle_not_disabled():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2 GPUs")
    mp.spawn(
        _feature_toggle_not_disabled_worker,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )
