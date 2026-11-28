# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDP prefill (the mixer's `gdp_prefill`) and FlashInfer's decode/verify
kernel in the mixer's argument layout, against the recurrence: per token the
state decays once, takes `num_householder` beta-gated delta steps, and is read
out with the query."""

import pytest
import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.models.nemotron_shortcut.nvidia.gdp import (
    NemotronShortcutGDPMixer,
    gdp_prefill,
    missing_gdp_kernels,
)
from vllm.platforms import current_platform

from .utils import build, make_config

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not any(current_platform.is_device_capability(cap) for cap in (90, 100, 103))
    or bool(missing_gdp_kernels()),
    reason="needs FlashInfer's GDP kernels on SM90 or SM10x",
)
KEY, VALUE = 128, 64  # the head widths the decode kernel serves


@pytest.fixture(autouse=True)
def _forward_context(dist_init, workspace_init):
    with torch.inference_mode(), set_forward_context(None, get_current_vllm_config()):
        yield


def _mixer(heads: int) -> NemotronShortcutGDPMixer:
    config = make_config(
        gdp_num_heads=heads,
        gdp_num_groups=heads,
        gdp_head_dim=VALUE,
        gdp_key_head_dim=KEY,
    )
    return build(lambda: NemotronShortcutGDPMixer(config, prefix="gdp"))


def _normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)


def _step(mixer, state, q, k, v, a, b, l2norm: bool = True):
    """One token of the recurrence over `state` [rows, heads, VALUE, KEY]."""
    norm = _normalize if l2norm else torch.Tensor.float
    decay = (-mixer.A_log.exp() * F.softplus(a.float() + mixer.dt_bias)).exp()
    state *= decay[:, :, None, None]
    for j in range(mixer.num_householder):
        key = norm(k[:, j])
        residual = v[:, j].float() - (state * key[:, :, None]).sum(-1)
        beta = b[:, j].float().sigmoid()[:, :, None]
        state += (beta * residual)[:, :, :, None] * key[:, :, None]
    return (state * norm(q)[:, :, None]).sum(-1) / KEY**0.5


@pytest.mark.parametrize("heads", [4, 10, 24])
@pytest.mark.parametrize("width", [1, 4, 8, 16])
@pytest.mark.parametrize("batch", [1, 8])
def test_decode_kernel_matches_recurrence_and_scatter(width, batch, heads):
    """Decode and verify keep every accepted token's state snapshot, skip
    unallocated (-1) slots and leave the other slots alone, also in a graph."""
    from flashinfer.cake_gdp import cake_gated_delta_product_mtp

    mixer = _mixer(heads)
    device = mixer.A_log.device
    torch.manual_seed(123)
    heads, n = mixer.num_heads, mixer.num_householder
    q = torch.randn(batch, width, heads, KEY, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, width, n, heads, KEY, device=device, dtype=q.dtype)
    v = torch.randn(batch, width, n, heads, VALUE, device=device, dtype=q.dtype)
    a = torch.randn(batch, width, heads, device=device, dtype=q.dtype)
    b = torch.randn(batch, width, n, heads, device=device, dtype=q.dtype)
    pool = torch.randn(batch * (width + 1) + 1, heads, VALUE, KEY, device=device)
    original = pool.clone()
    read = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
    slots = torch.arange(
        batch + 1, batch * (width + 1) + 1, device=device, dtype=torch.int32
    ).reshape(batch, width)
    slots[-1, -1] = -1  # an unallocated snapshot, as in a ragged window
    state = original[read.long()].clone()
    expected_pool = original.clone()
    expected = torch.empty_like(v[:, :, 0], dtype=torch.float32)
    for t in range(width):
        expected[:, t] = _step(
            mixer, state, q[:, t], k[:, t], v[:, t], a[:, t], b[:, t]
        )
        valid = slots[:, t] >= 0
        expected_pool[slots[valid, t].long()] = state[valid]

    def run(out):
        return cake_gated_delta_product_mtp(
            q,
            k,
            v,
            pool,
            read,
            mixer.A_log.float(),
            a,
            mixer.dt_bias.float(),
            b,
            output=out,
            ssm_state_indices=slots,
            disable_state_update=False,
            use_qk_l2norm=True,
        )

    out = torch.full_like(v[:, :, 0], float("nan"))
    actual, returned_pool = run(out)
    assert actual.data_ptr() == out.data_ptr()
    assert returned_pool.data_ptr() == pool.data_ptr()
    torch.testing.assert_close(actual.float(), expected, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(pool, expected_pool, atol=2e-4, rtol=2e-3)
    torch.testing.assert_close(pool[0], original[0], atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(out)
    pool.copy_(original)
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(out.float(), expected, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(pool, expected_pool, atol=2e-4, rtol=2e-3)


@pytest.mark.parametrize("heads", [4, 24])
@pytest.mark.parametrize("lengths", [(1,), (7, 130), (64, 1, 200)])
def test_prefill_kernel_matches_recurrence(lengths, heads):
    """Prefill over packed sequences of different lengths, from initial
    states, with each token's Householder steps on dim 1."""
    mixer = _mixer(heads)
    device = mixer.A_log.device
    torch.manual_seed(7)
    heads, n, tokens = mixer.num_heads, mixer.num_householder, sum(lengths)
    q = _normalize(torch.randn(tokens, heads, KEY, device=device)).bfloat16()
    k = _normalize(torch.randn(tokens, n, heads, KEY, device=device)).bfloat16()
    v = torch.randn(tokens, n, heads, VALUE, device=device, dtype=torch.bfloat16)
    a = torch.randn(tokens, heads, device=device, dtype=torch.bfloat16)
    b = torch.randn(tokens, n, heads, device=device, dtype=torch.bfloat16)
    initial = torch.randn(len(lengths), heads, VALUE, KEY, device=device) * 0.1
    starts_cpu = torch.tensor([0, *lengths], device="cpu").cumsum(0, dtype=torch.int32)
    starts = starts_cpu.to(device)
    decay = -mixer.A_log.float().exp() * F.softplus(a.float() + mixer.dt_bias)
    out, last = gdp_prefill(
        q,
        k,
        v,
        decay,
        b.float().sigmoid(),
        initial.clone(),
        starts,
        cu_seqlens_cpu=starts_cpu,
    )
    expected = torch.empty(tokens, heads, VALUE, device=device)
    for request, (start, stop) in enumerate(zip(starts[:-1], starts[1:])):
        state = initial[request : request + 1].clone()
        for t in range(int(start), int(stop)):
            row = slice(t, t + 1)
            expected[row] = _step(
                mixer, state, q[row], k[row], v[row], a[row], b[row], l2norm=False
            )
        # The kernel sums in chunks, the reference token by token: over 200
        # tokens a few state entries drift by ~2.5e-3.
        torch.testing.assert_close(last[request], state[0], atol=5e-3, rtol=2e-2)
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)
