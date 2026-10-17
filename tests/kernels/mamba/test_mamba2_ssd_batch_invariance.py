# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Under VLLM_BATCH_INVARIANT the Mamba2 SSD (chunked scan) kernels run one fixed
Triton configuration instead of autotuning, and the rows of a prefix must not
depend on the tokens that follow it."""

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
from vllm.model_executor.layers.mamba.ops.triton_helpers import pin_autotune_config
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

CHUNK = 64
NHEADS, HEAD_DIM, NGROUPS, DSTATE = 8, 64, 1, 64


@triton.autotune(
    configs=[triton.Config({"BLOCK": 64}), triton.Config({"BLOCK": 128})],
    key=[],
)
@triton.jit
def _dummy_kernel(x_ptr, BLOCK: tl.constexpr):
    pass


def test_pin_autotune_config_follows_the_environment(monkeypatch):
    pinned = triton.Config({"BLOCK": 256})
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    pin_autotune_config(_dummy_kernel, pinned)
    assert len(_dummy_kernel.configs) == 2
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    pin_autotune_config(_dummy_kernel, pinned)
    assert _dummy_kernel.configs == [pinned]


# (autotuned kernel, its pinned configuration)
SSD_KERNELS = [
    (ssd_bmm._bmm_chunk_fwd_kernel, ssd_bmm._BATCH_INVARIANT_CONFIG),
    (
        ssd_chunk_state._chunk_cumsum_fwd_kernel,
        ssd_chunk_state._CUMSUM_BATCH_INVARIANT_CONFIG,
    ),
    (
        ssd_chunk_state._chunk_state_fwd_kernel,
        ssd_chunk_state._CHUNK_STATE_BATCH_INVARIANT_CONFIG,
    ),
    (ssd_chunk_scan._chunk_scan_fwd_kernel, ssd_chunk_scan._BATCH_INVARIANT_CONFIG),
    (
        ssd_state_passing._state_passing_fwd_kernel,
        ssd_state_passing._BATCH_INVARIANT_CONFIG,
    ),
]


def _key(config):
    return (tuple(sorted(config.kwargs.items())), config.num_warps, config.num_stages)


@pytest.mark.parametrize("kernel,pinned", SSD_KERNELS)
def test_ssd_kernel_configs(kernel, pinned):
    """Pinned at import under VLLM_BATCH_INVARIANT; otherwise the pinned tile is
    one of the untouched default candidates."""
    keys = [_key(c) for c in kernel.configs]
    if envs.VLLM_BATCH_INVARIANT:
        assert keys == [_key(pinned)]
    else:
        assert len(keys) > 1 and keys.count(_key(pinned)) == 1


def _scan(x, dt, B, C, A, dt_bias, D):
    length = x.shape[0]
    n_chunks = -(-length // CHUNK)
    i32 = lambda v: torch.tensor(v, dtype=torch.int32, device=x.device)  # noqa: E731
    out = torch.empty_like(x)
    mamba_chunk_scan_combined_varlen(
        x,
        dt,
        A,
        B,
        C,
        chunk_size=CHUNK,
        cu_seqlens=i32([0, length]),
        cu_chunk_seqlens=i32([k * CHUNK for k in range(n_chunks)] + [length]),
        last_chunk_indices=i32([n_chunks - 1]),
        seq_idx=i32([0] * n_chunks),
        out=out,
        D=D,
        z=None,
        dt_bias=dt_bias,
        initial_states=None,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        state_dtype=torch.float32,
    )
    return out


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Mamba2 SSD Triton kernels require a CUDA-alike device.",
)
def test_ssd_is_prefix_invariant():
    """The rows of a prefix do not depend on the tokens that follow it."""
    device = torch.device(current_platform.device_type)
    gen = torch.Generator(device=device).manual_seed(0)
    A = -(torch.rand(NHEADS, device=device, generator=gen) * 15 + 1)
    dt_bias = torch.log(
        torch.expm1(torch.rand(NHEADS, device=device, generator=gen) * 0.09 + 0.01)
    )
    D = torch.ones(NHEADS, device=device)
    bf = torch.bfloat16
    x = torch.randn(400, NHEADS, HEAD_DIM, device=device, generator=gen).to(bf)
    dt = (0.5 * torch.randn(400, NHEADS, device=device, generator=gen)).to(bf)
    B = torch.randn(400, NGROUPS, DSTATE, device=device, generator=gen).to(bf)
    C = torch.randn(400, NGROUPS, DSTATE, device=device, generator=gen).to(bf)
    full = _scan(x, dt, B, C, A, dt_bias, D)
    for n in (37, CHUNK, 150, 2 * CHUNK + 5):
        prefix = _scan(x[:n], dt[:n], B[:n], C[:n], A, dt_bias, D)
        assert torch.equal(prefix.view(torch.int16), full[:n].view(torch.int16)), n
