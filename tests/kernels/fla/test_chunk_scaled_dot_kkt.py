# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every autotune config of chunk_scaled_dot_kkt_fwd_kernel is deterministic.

With beta applied inside the K loop, the configs with BK < K and
num_warps >= 4 gave different, wrong results from run to run on Hopper
(#65385).
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
    chunk_scaled_dot_kkt_fwd_kernel,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="needs a GPU"
)

BT = 64
REPEATS = 5


def _autotuner():
    fn = chunk_scaled_dot_kkt_fwd_kernel
    while not hasattr(fn, "configs"):
        fn = fn.fn
    return fn


def _reference(k, g, beta, H):
    """beta_i * <k_i, k_j> * exp(g_i - g_j) for j < i within each chunk."""
    B, T, Hg, K = k.shape
    NT = T // BT
    kc = k.float().repeat_interleave(H // Hg, dim=2).view(B, NT, BT, H, K)
    A = torch.einsum("bnihk,bnjhk->bnihj", kc, kc)
    gc = g.float().view(B, NT, BT, H)
    A = A * torch.exp(gc[..., None] - gc.permute(0, 1, 3, 2)[:, :, None])
    A = A * beta.float().view(B, NT, BT, H)[..., None]
    mask = torch.ones(BT, BT, device=k.device).tril(-1)
    A = A * mask[:, None, :]
    return A.reshape(B, T, H, BT)


@pytest.mark.parametrize("K", [128, 256])
@torch.inference_mode()
def test_every_config_is_deterministic(K, monkeypatch):
    B, T, H, Hg = 1, 4096, 32, 16
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(0)
    k = torch.randn(B, T, Hg, K, device=device, generator=gen, dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(k, dim=-1)
    beta = torch.rand(B, T, H, device=device, generator=gen, dtype=torch.bfloat16)
    # Chunk-local cumulative log decay, as chunk_local_cumsum produces it.
    g = torch.rand(B, T // BT, BT, H, device=device, generator=gen) * -0.05
    g = g.cumsum(2).view(B, T, H)
    ref = _reference(k, g, beta, H)

    autotuner = _autotuner()
    failures = []
    for config in list(autotuner.configs):
        monkeypatch.setattr(autotuner, "configs", [config])
        monkeypatch.setattr(autotuner, "cache", {})
        outs = [
            chunk_scaled_dot_kkt_fwd(k=k, g=g, beta=beta, chunk_size=BT)
            for _ in range(REPEATS)
        ]
        if not all(torch.equal(outs[0], out) for out in outs[1:]):
            failures.append(f"{config}: differs between runs")
            continue
        rel_err = ((outs[0] - ref).norm() / ref.norm()).item()
        if rel_err > 1e-5:
            failures.append(f"{config}: relative error {rel_err:.2e}")
    assert not failures, "\n".join(failures)
