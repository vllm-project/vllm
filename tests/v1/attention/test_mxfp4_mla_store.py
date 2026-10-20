# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mxfp4_mla store kernel must write exactly what AITER's MXFP4 quantizer
produces, into the right rows of the paged cache."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and torch.cuda.is_available()),
    reason="mxfp4_mla is ROCm-only",
)

LATENT, ROW = 512, 272
UNTOUCHED = 0xCD


def test_store_matches_aiter():
    """Bit-exact against ``per_1x32_f4_quant`` on the layout the backend hands
    the kernel: blocks strided past ``block_size`` rows by other layers' pages,
    padded slots, and a latent that is a slice of a wider buffer."""
    pytest.importorskip("aiter")
    quant = pytest.importorskip("aiter.ops.quant")
    from vllm.v1.attention.ops.mxfp4_mla_store import store_mxfp4_mla

    set_random_seed(0)
    num_blocks, block_size, gap = 8, 8, 5
    backing = torch.full(
        (num_blocks, block_size + gap, ROW), UNTOUCHED, dtype=torch.uint8
    ).cuda()
    cache = backing[:, :block_size]

    # 64 tokens, two of them padding; two slots are never written.
    num_slots = num_blocks * block_size
    wide = torch.randn(num_slots, 2 * LATENT, dtype=torch.bfloat16)
    wide[:, ::97] *= 8.0
    latent = wide.cuda()[:, :LATENT]
    written = torch.randperm(num_slots)[: num_slots - 2]
    slots = torch.cat([written, torch.tensor([-1, -1])])

    store_mxfp4_mla(latent, slots.cuda(), cache)

    packed, scales = quant.per_1x32_f4_quant(latent.float().cpu(), pack_dim=-1)
    want = torch.cat(
        [
            packed.contiguous().view(torch.uint8).reshape(len(slots), -1),
            scales.contiguous().view(torch.uint8).reshape(len(slots), -1),
        ],
        dim=1,
    )
    got = backing.cpu()
    for token, slot in enumerate(written.tolist()):
        block, offset = divmod(slot, block_size)
        assert torch.equal(got[block, offset], want[token]), f"slot {slot}"
    for slot in set(range(num_slots)) - set(written.tolist()):
        block, offset = divmod(slot, block_size)
        assert (got[block, offset] == UNTOUCHED).all(), f"slot {slot} was written"
    assert (got[:, block_size:] == UNTOUCHED).all(), "a write landed between blocks"
