# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm V tile of chunk_gated_delta_rule_fwd_h follows the batch."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import chunk_delta_h
from vllm.utils.platform_utils import num_compute_units

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="BV follows occupancy on ROCm only"
)

BT, K, V = 64, 128, 128


@pytest.mark.parametrize(
    "num_seq_heads, num_cus, bv",
    [
        (1, 256, 16),
        (32, 256, 16),
        (48, 256, 32),
        (64, 256, 32),
        (96, 256, 64),
        (128, 256, 64),
        (4096, 256, 64),
        (32, 304, 16),
        (64, 304, 32),
        (152, 304, 64),
    ],
)
def test_bv_from_occupancy(num_seq_heads: int, num_cus: int, bv: int):
    assert chunk_delta_h._chunk_delta_h_bv(V, num_seq_heads, num_cus) == bv


def _inputs(seqlens: list[int], H: int, seed: int = 0):
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(seed)
    T, Hg = sum(seqlens), max(H // 2, 1)

    def randn(*shape, scale=1.0, dtype=torch.bfloat16):
        x = torch.randn(*shape, device=device, generator=gen, dtype=dtype)
        return x.mul_(scale)

    cu_seqlens = torch.tensor(seqlens, device=device).cumsum(0)
    cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0)).to(torch.int32)
    # g is a per-chunk cumulative log decay, as chunk_local_cumsum produces it.
    g = torch.empty(1, T, H, device=device)
    for bos, eos in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        for c in range(bos, eos, BT):
            n = min(BT, eos - c)
            step = torch.rand(n, H, device=device, generator=gen) * -0.05
            g[0, c : c + n] = step.cumsum(0)
    return dict(
        k=randn(1, T, Hg, K, scale=0.09),
        w=randn(1, T, H, K, scale=0.05),
        u=randn(1, T, H, V),
        g=g,
        initial_state=randn(len(seqlens), H, V, K, scale=0.1, dtype=torch.float32),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )


class _LaunchRecorder:
    """Records the BV every launch of the chunk-state kernel ran with."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.tuner = kernel
        while not hasattr(self.tuner, "configs"):
            self.tuner = self.tuner.fn
        self.bvs: list[int] = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            out = self.kernel[grid](*args, **kwargs)
            bv = kwargs.get("BV", self.tuner.best_config.kwargs.get("BV"))
            self.bvs.append(bv)
            return out

        return launch


@pytest.mark.parametrize("H", [4, 32])
@torch.inference_mode()
def test_bv_is_not_frozen_by_first_launch(monkeypatch, H: int):
    """A large batch tunes first, as the profile run does; a later
    single-sequence prefill must still get a narrower V tile, with each
    tile tuned on its own and with unchanged outputs."""
    if num_compute_units(0) < 2 * H:
        pytest.skip("device too small for this head count")
    recorder = _LaunchRecorder(
        chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    )
    monkeypatch.setattr(recorder.tuner, "cache", {})
    monkeypatch.setattr(
        chunk_delta_h, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", recorder
    )

    many = max(2 * num_compute_units(0) // H, 2)
    chunk_delta_h.chunk_gated_delta_rule_fwd_h(**_inputs([BT] * many, H))
    one = _inputs([5 * BT + 17], H, seed=1)
    h, v_new, ht = chunk_delta_h.chunk_gated_delta_rule_fwd_h(**one)

    bv_many, bv_one = recorder.bvs
    assert bv_one < bv_many
    # One tuning per tile width, not one shared by every width.
    assert len(recorder.tuner.cache) == 2

    # The tile only splits V, so the outputs match the widest tile bit for bit.
    monkeypatch.setattr(chunk_delta_h, "_chunk_delta_h_bv", lambda *args: bv_many)
    h_ref, v_new_ref, ht_ref = chunk_delta_h.chunk_gated_delta_rule_fwd_h(**one)
    assert recorder.bvs[-1] == bv_many
    torch.testing.assert_close(h, h_ref, atol=0, rtol=0)
    torch.testing.assert_close(v_new, v_new_ref, atol=0, rtol=0)
    torch.testing.assert_close(ht, ht_ref, atol=0, rtol=0)
