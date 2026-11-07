# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.sample.logprob import _ranks_kernel

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="requires a GPU or TRITON_INTERPRET=1",
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _run(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    n, vocab = logits.shape
    out = torch.zeros(n, dtype=torch.int64, device=logits.device)
    _ranks_kernel[(n,)](
        out,
        logits,
        logits.stride(0),
        token_ids,
        vocab,
        BLOCK_SIZE=8192,
        TAIL_BLOCK_SIZE=64 if current_platform.is_cpu() else 8192,
    )
    return out


@pytest.mark.parametrize("batch", [1, 5, 16])
@pytest.mark.parametrize("vocab", [1, 100, 8191, 8192, 8193, 16385, 50272])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("mode", ["normal", "ties", "neginf"])
def test_ranks_kernel_matches_torch(batch, vocab, dtype, mode):
    torch.manual_seed(0)
    buf = torch.randn(batch, vocab + 37)  # padded buffer -> strided view
    if mode == "ties":
        buf = buf.round()
    logits = buf.to(dtype)[:, :vocab].to(DEVICE)
    if mode == "neginf":
        mask = torch.rand(batch, vocab) < 0.5
        mask[:, 0] = False  # keep a finite entry in every row
        logits[mask.to(DEVICE)] = float("-inf")
    finite = torch.isfinite(logits.float()).float()
    token_ids = torch.multinomial(finite, 1).squeeze(1).to(torch.int64)

    got = _run(logits, token_ids)
    ref = (logits >= logits.gather(1, token_ids[:, None])).sum(1)
    assert torch.equal(got, ref)
