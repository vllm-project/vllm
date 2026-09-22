# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate production Lamport cleanup from a reachable one-epoch skew.

The fast rank publishes epoch N+1 before the slow rank runs epoch N cleanup.
Peer slots are prefilled only so each shipping kernel can finish independently;
the test then checks that slow-rank cleanup retained the real N+1 publication.
The production kernels are also seeded at the final two uint32 values to verify
that stage selection remains bounded and never reuses a stage across wrap.

Each rank's local buffer contains three all-gather stages followed by three
reduce-scatter stages::

    all-gather:     [stage 0][stage 1][stage 2]
    reduce-scatter: [stage 0][stage 1][stage 2]

Each stage contains one payload slot per source rank.
"""

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_tp_group
from vllm.platforms import current_platform

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

_SENTINEL = -2147483648
_ROWS = 32
_HIDDEN = 8
_NUM_STAGES = 3
_NUM_RANKS = 2
_FAST_RANK = 0
_SLOW_RANK = 1
_PACK_BYTES = 16
_UINT32_MODULUS = 1 << 32
_UINT32_MAX = _UINT32_MODULUS - 1
_ALL_GATHER_EPOCH_ROW = 0
_REDUCE_SCATTER_EPOCH_ROW = 1
_REDUCE_SCATTER_REGION_STAGE_OFFSET = 3


def _supports_multimem():
    capability = current_platform.get_device_capability()
    return (
        current_platform.is_cuda() and capability is not None and capability.major >= 9
    )


def _payload(value, device):
    return torch.full((_ROWS, _HIDDEN), value, dtype=torch.bfloat16, device=device)


def _payload_bytes(value):
    return value.contiguous().view(torch.uint8).flatten()


def _int32_bits(value):
    """Represent an unsigned 32-bit value in the signed epoch tensor."""
    value %= _UINT32_MODULUS
    return value if value < (1 << 31) else value - _UINT32_MODULUS


def _epoch_value(epoch):
    return int(epoch.item()) % _UINT32_MODULUS


def _stage_slot(buffer, stage_size, region_offset, stage, src_rank, slot_size):
    offset = region_offset + stage * stage_size + src_rank * slot_size
    return buffer.narrow(0, offset, slot_size)


def _prepare_rank_state(
    fa, epoch_row, region_offset, epoch, stage, peer_rank, peer_payload
):
    """Reset this rank's replica and seed the peer payload needed to finish."""
    buffer = fa.mnnvl_buffer
    all_epochs = fa.mnnvl_lamport_epochs
    assert buffer is not None and all_epochs is not None
    slot_size = peer_payload.nbytes
    slot = _stage_slot(
        buffer, fa.mnnvl_buffer_size, region_offset, stage, peer_rank, slot_size
    )
    buffer.view(torch.int32).fill_(_SENTINEL)
    epochs = all_epochs[epoch_row]
    # epochs[0] selects the stage, epochs[1] is the CTA completion counter,
    # and epochs[2:5] record the valid pack count for each stage.
    epochs.zero_()
    epochs[0] = epoch
    epochs[2:5].fill_(slot_size * _NUM_RANKS // _PACK_BYTES)
    slot.copy_(_payload_bytes(peer_payload))
    return buffer, slot_size


def _all_ranks_agree(checks, device):
    """Give every rank the same result so one failure cannot strand a peer."""
    values = torch.tensor(checks, dtype=torch.int32, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.MIN)
    return [bool(value) for value in values.cpu().tolist()]


def _run_all_gather_retention_case(fa, rank, local_rank, base_epoch, device):
    slow_epoch = base_epoch
    fast_epoch = base_epoch + 1
    slow_stage = slow_epoch % _NUM_STAGES
    fast_stage = fast_epoch % _NUM_STAGES
    fast_local = _payload(11, device)
    fast_peer = _payload(21, device)
    slow_peer = _payload(31, device)
    slow_local = _payload(41, device)

    if rank == _FAST_RANK:
        buffer, slot_size = _prepare_rank_state(
            fa,
            _ALL_GATHER_EPOCH_ROW,
            0,
            fast_epoch,
            fast_stage,
            _SLOW_RANK,
            fast_peer,
        )
    else:
        buffer, slot_size = _prepare_rank_state(
            fa,
            _ALL_GATHER_EPOCH_ROW,
            0,
            slow_epoch,
            slow_stage,
            _FAST_RANK,
            slow_peer,
        )
    torch.accelerator.synchronize()
    dist.barrier(device_ids=[local_rank])

    # Fast rank publishes N+1 through the shipping all-gather kernel.
    fast_output_ok = True
    if rank == _FAST_RANK:
        output = fa.custom_all_gather(fast_local)
        assert output is not None
        torch.accelerator.synchronize()
        fast_output_ok = torch.equal(output, torch.cat((fast_local, fast_peer)))
    dist.barrier(device_ids=[local_rank])

    # Slow rank verifies that publication, then runs shipping epoch-N cleanup.
    publication_ok = True
    slow_output_ok = True
    retained = True
    if rank == _SLOW_RANK:
        target = _stage_slot(
            buffer,
            fa.mnnvl_buffer_size,
            0,
            fast_stage,
            _FAST_RANK,
            slot_size,
        )
        publication_ok = torch.equal(target, _payload_bytes(fast_local))
        output = fa.custom_all_gather(slow_local)
        assert output is not None
        torch.accelerator.synchronize()
        slow_output_ok = torch.equal(output, torch.cat((slow_peer, slow_local)))
        retained = torch.equal(target, _payload_bytes(fast_local))

    checks = _all_ranks_agree(
        [fast_output_ok, publication_ok, slow_output_ok, retained], device
    )
    assert all(checks[:3]), "production all-gather setup did not reach target state"
    assert checks[3], "all-gather cleanup erased the next-stage publication"


def _run_reduce_scatter_retention_case(fa, rank, local_rank, base_epoch, device):
    slow_epoch = base_epoch
    fast_epoch = base_epoch + 1
    slow_stage = slow_epoch % _NUM_STAGES
    fast_stage = fast_epoch % _NUM_STAGES
    region_offset = _REDUCE_SCATTER_REGION_STAGE_OFFSET * fa.mnnvl_buffer_size

    fast_chunk_0 = _payload(11, device)
    fast_chunk_1 = _payload(12, device)
    fast_peer = _payload(21, device)
    slow_chunk_0 = _payload(31, device)
    slow_chunk_1 = _payload(32, device)
    slow_peer = _payload(41, device)
    fast_input = torch.cat((fast_chunk_0, fast_chunk_1))
    slow_input = torch.cat((slow_chunk_0, slow_chunk_1))

    if rank == _FAST_RANK:
        buffer, slot_size = _prepare_rank_state(
            fa,
            _REDUCE_SCATTER_EPOCH_ROW,
            region_offset,
            fast_epoch,
            fast_stage,
            _SLOW_RANK,
            fast_peer,
        )
    else:
        buffer, slot_size = _prepare_rank_state(
            fa,
            _REDUCE_SCATTER_EPOCH_ROW,
            region_offset,
            slow_epoch,
            slow_stage,
            _FAST_RANK,
            slow_peer,
        )
    torch.accelerator.synchronize()
    dist.barrier(device_ids=[local_rank])

    # Fast rank publishes its N+1 contribution through shipping reduce-scatter.
    fast_output_ok = True
    if rank == _FAST_RANK:
        output = fa.custom_reduce_scatter(fast_input)
        assert output is not None
        torch.accelerator.synchronize()
        fast_output_ok = torch.equal(output, fast_chunk_0 + fast_peer)
    dist.barrier(device_ids=[local_rank])

    # Slow rank verifies that contribution, then runs shipping epoch-N cleanup.
    publication_ok = True
    slow_output_ok = True
    retained = True
    if rank == _SLOW_RANK:
        target = _stage_slot(
            buffer,
            fa.mnnvl_buffer_size,
            region_offset,
            fast_stage,
            _FAST_RANK,
            slot_size,
        )
        publication_ok = torch.equal(target, _payload_bytes(fast_chunk_1))
        output = fa.custom_reduce_scatter(slow_input)
        assert output is not None
        torch.accelerator.synchronize()
        slow_output_ok = torch.equal(output, slow_peer + slow_chunk_1)
        retained = torch.equal(target, _payload_bytes(fast_chunk_1))

    checks = _all_ranks_agree(
        [fast_output_ok, publication_ok, slow_output_ok, retained], device
    )
    assert all(checks[:3]), "production reduce-scatter setup did not reach target state"
    assert checks[3], "reduce-scatter cleanup erased the next-stage publication"


def _run_epoch_wrap_case(fa, rank, local_rank, seed_epoch, device):
    """Run both shipping collectives across the uint32 wrap boundary."""
    buffer = fa.mnnvl_buffer
    all_epochs = fa.mnnvl_lamport_epochs
    assert buffer is not None and all_epochs is not None
    buffer.view(torch.int32).fill_(_SENTINEL)
    all_epochs.zero_()
    all_epochs[:, 0] = _int32_bits(seed_epoch)
    torch.accelerator.synchronize()
    dist.barrier(device_ids=[local_rank])

    ag_input = _payload(rank + 1, device)
    ag_output = fa.custom_all_gather(ag_input)
    assert ag_output is not None

    rs_input = torch.cat(
        (_payload(rank + 1, device), _payload((rank + 1) * 10, device))
    )
    rs_output = fa.custom_reduce_scatter(rs_input)
    assert rs_output is not None
    torch.accelerator.synchronize()

    next_stage = (seed_epoch % _NUM_STAGES + 1) % _NUM_STAGES
    checks = _all_ranks_agree(
        [
            torch.equal(
                ag_output,
                torch.cat((_payload(1, device), _payload(2, device))),
            ),
            torch.equal(rs_output, _payload(3 if rank == 0 else 30, device)),
            _epoch_value(all_epochs[_ALL_GATHER_EPOCH_ROW, 0]) == next_stage,
            _epoch_value(all_epochs[_REDUCE_SCATTER_EPOCH_ROW, 0]) == next_stage,
        ],
        device,
    )
    assert all(checks), "Lamport stages did not advance safely across uint32 wrap"


@ray.remote(num_gpus=1, max_calls=1)
def _run_stage_cleanup_test(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.setenv("VLLM_ALLREDUCE_USE_SYMM_MEM", "1")
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)

        fa = get_tp_group().device_communicator.ca_comm
        assert fa is not None and not fa.disabled
        assert fa.mnnvl_multicast_ptr

        for base_epoch in range(_NUM_STAGES):
            _run_all_gather_retention_case(fa, rank, rank, base_epoch, device)
            _run_reduce_scatter_retention_case(fa, rank, rank, base_epoch, device)
        # Old unbounded metadata stores UINT32_MAX and then repeats stage 0.
        for seed_epoch in (_UINT32_MAX - 1, _UINT32_MAX):
            _run_epoch_wrap_case(fa, rank, rank, seed_epoch, device)


@pytest.mark.skipif(
    not _supports_multimem(),
    reason="MNNVL Lamport collectives require an SM90 or newer NVIDIA GPU.",
)
def test_mnnvl_lamport_stage_cleanup(monkeypatch: pytest.MonkeyPatch):
    if torch.accelerator.device_count() < 2:
        pytest.skip("Need at least two GPUs to run the test.")
    multi_process_parallel(monkeypatch, 2, 1, _run_stage_cleanup_test)
