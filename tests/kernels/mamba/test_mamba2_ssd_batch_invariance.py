# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariance properties of the Mamba2 SSD (chunked scan) kernels.

The output rows of a sequence must not depend on which other sequences share
the varlen call (packing invariance) or on how many tokens follow (prefix
invariance), and under VLLM_BATCH_INVARIANT the kernels must run one fixed
Triton configuration instead of autotuning.
"""

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers.mamba.ops import (
    ssd_bmm,
    ssd_chunk_scan,
    ssd_chunk_state,
    ssd_state_passing,
)
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.model_executor.layers.mamba.ops.triton_helpers import (
    batch_invariant_autotune_configs,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton

DEVICE = current_platform.device_type
CHUNK = 64
NHEADS, HEAD_DIM, NGROUPS, DSTATE = 8, 64, 1, 64


def test_batch_invariant_mode_pins_one_config(monkeypatch):
    configs = [triton.Config({"BLOCK_SIZE": 64}), triton.Config({"BLOCK_SIZE": 128})]
    pinned = triton.Config({"BLOCK_SIZE": 256})
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    assert batch_invariant_autotune_configs(configs, pinned) is configs
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    assert batch_invariant_autotune_configs(configs, pinned) == [pinned]


# (autotuned kernel, its pinned configuration, number of default candidates)
SSD_KERNELS = [
    (ssd_bmm._bmm_chunk_fwd_kernel, ssd_bmm._BATCH_INVARIANT_CONFIG, 9),
    (
        ssd_chunk_state._chunk_cumsum_fwd_kernel,
        ssd_chunk_state._CUMSUM_BATCH_INVARIANT_CONFIG,
        6,
    ),
    (
        ssd_chunk_state._chunk_state_fwd_kernel,
        ssd_chunk_state._CHUNK_STATE_BATCH_INVARIANT_CONFIG,
        14,
    ),
    (ssd_chunk_scan._chunk_scan_fwd_kernel, ssd_chunk_scan._BATCH_INVARIANT_CONFIG, 23),
    (
        ssd_state_passing._state_passing_fwd_kernel,
        ssd_state_passing._BATCH_INVARIANT_CONFIG,
        6,
    ),
]


def _key(config):
    return (tuple(sorted(config.kwargs.items())), config.num_warps, config.num_stages)


@pytest.mark.parametrize("kernel,pinned,num_default", SSD_KERNELS)
def test_ssd_kernel_candidates(kernel, pinned, num_default):
    """The pinned configuration is one of the default candidates, and the
    default candidate list is untouched: full size, no duplicates. The lists
    are fixed at import, so which branch runs depends on the environment."""
    keys = [_key(c) for c in kernel.configs]
    if envs.VLLM_BATCH_INVARIANT:
        assert keys == [_key(pinned)]
    else:
        assert len(keys) == num_default
        assert len(set(keys)) == num_default
        assert keys.count(_key(pinned)) == 1


def _params(device):
    torch.manual_seed(0)
    # Realistic Mamba2 initialisation: slow heads keep state across chunks.
    A = -(torch.rand(NHEADS, device=device) * 15 + 1)
    dt_target = torch.exp(
        torch.rand(NHEADS, device=device)
        * (torch.log(torch.tensor(0.1)) - torch.log(torch.tensor(1e-3)))
        + torch.log(torch.tensor(1e-3))
    )
    dt_bias = dt_target + torch.log(-torch.expm1(-dt_target))
    D = torch.ones(NHEADS, device=device)
    return A, dt_bias, D


def _sequence(length, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    dtype = torch.bfloat16
    x = torch.randn(length, NHEADS, HEAD_DIM, device=device, generator=gen).to(dtype)
    dt = (0.5 * torch.randn(length, NHEADS, device=device, generator=gen)).to(dtype)
    B = torch.randn(length, NGROUPS, DSTATE, device=device, generator=gen).to(dtype)
    C = torch.randn(length, NGROUPS, DSTATE, device=device, generator=gen).to(dtype)
    return x, dt, B, C


def _scan(seqs, A, dt_bias, D):
    """One varlen chunked-scan call over the packed sequences; returns the
    per-sequence output rows and final states."""
    device = seqs[0][0].device
    x, dt, B, C = (torch.cat([s[i] for s in seqs]) for i in range(4))
    lens = [s[0].shape[0] for s in seqs]
    cu_seqlens: list[int] = [0]
    cu_chunk: list[int] = []
    seq_idx: list[int] = []
    last_chunk: list[int] = []
    offset = 0
    for i, length in enumerate(lens):
        n_chunks = -(-length // CHUNK)
        cu_chunk.extend(offset + k * CHUNK for k in range(n_chunks))
        seq_idx.extend([i] * n_chunks)
        last_chunk.append(len(cu_chunk) - 1)
        offset += length
        cu_seqlens.append(offset)
    cu_chunk.append(offset)
    i32 = lambda v: torch.tensor(v, dtype=torch.int32, device=device)  # noqa: E731
    out = torch.empty_like(x)
    states = mamba_chunk_scan_combined_varlen(
        x,
        dt,
        A,
        B,
        C,
        chunk_size=CHUNK,
        cu_seqlens=i32(cu_seqlens),
        cu_chunk_seqlens=i32(cu_chunk),
        last_chunk_indices=i32(last_chunk),
        seq_idx=i32(seq_idx),
        out=out,
        D=D,
        z=None,
        dt_bias=dt_bias,
        initial_states=None,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        state_dtype=torch.float32,
    )
    outs = [out[a:b] for a, b in zip(cu_seqlens, cu_seqlens[1:])]
    return outs, states


def _bits_equal(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Mamba2 SSD Triton kernels require a CUDA-alike device.",
)
def test_ssd_is_packing_invariant():
    """A sequence's rows and final state do not depend on its batch mates."""
    device = torch.device(DEVICE)
    A, dt_bias, D = _params(device)
    needle = _sequence(203, device, seed=1)
    fillers = [_sequence(90, device, seed=2), _sequence(333, device, seed=3)]
    alone_out, alone_state = _scan([needle], A, dt_bias, D)
    packed_out, packed_state = _scan([fillers[0], needle, fillers[1]], A, dt_bias, D)
    assert _bits_equal(alone_out[0], packed_out[1])
    assert torch.equal(alone_state[0], packed_state[1])


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Mamba2 SSD Triton kernels require a CUDA-alike device.",
)
def test_ssd_is_prefix_invariant():
    """The rows of a prefix do not depend on the tokens that follow it."""
    device = torch.device(DEVICE)
    A, dt_bias, D = _params(device)
    x, dt, B, C = _sequence(400, device, seed=4)
    full_out, _ = _scan([(x, dt, B, C)], A, dt_bias, D)
    for n in (37, CHUNK, 150, 2 * CHUNK + 5):
        prefix_out, _ = _scan([(x[:n], dt[:n], B[:n], C[:n])], A, dt_bias, D)
        assert _bits_equal(prefix_out[0], full_out[0][:n]), n
