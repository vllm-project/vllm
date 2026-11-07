# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer as qsa_indexer_ops
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

requires_qsa_kernels = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="QSA kernels require CUDA and Triton",
)

DEVICE = "cuda"
PAGE_SIZE = 16
PAGES = 6
HEADS = 4
HEAD_DIM = 32


@requires_qsa_kernels
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "rows_per_request",
    [(200,), (96,) * 4, (130, 3, 0, 70), (300,) + (1, 2, 3, 4) * 25],
    ids=["one_request", "equal_chunks", "uneven", "mixed_step"],
)
def test_qsa_prefill_sm120_config_is_bit_equal_to_the_default_config(
    monkeypatch: pytest.MonkeyPatch,
    cache_dtype: torch.dtype,
    rows_per_request: tuple[int, ...],
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(len(rows_per_request))
    requests, total = len(rows_per_request), sum(rows_per_request)
    q = torch.randn((total, HEADS, HEAD_DIM), generator=generator)
    cache = torch.randn((requests * PAGES, PAGE_SIZE, 1, HEAD_DIM), generator=generator)
    table = torch.randperm(requests * PAGES, generator=generator).view(requests, PAGES)
    starts = torch.tensor([0, *rows_per_request]).cumsum(0) + 3
    visible = torch.randint(
        1, PAGES * PAGE_SIZE + 1, (total,), generator=generator, dtype=torch.int32
    )
    args = (
        q.to(cache_dtype).to(DEVICE),
        cache.to(cache_dtype).to(DEVICE),
        table.to(torch.int32).to(DEVICE),
        starts.to(torch.int32).to(DEVICE),
        visible.to(DEVICE),
    )
    width = PAGES * PAGE_SIZE

    def run(is_sm120: bool) -> torch.Tensor:
        monkeypatch.setattr(qsa_indexer_ops, "_is_sm120", lambda: is_sm120)
        workspace = torch.full((total * width,), -1.0, device=DEVICE)
        logits = qsa_indexer_ops._prefill_logits(
            *args, workspace, max(rows_per_request), width, 0, total
        )
        return logits.clone()

    assert (
        qsa_indexer_ops._SM120_PREFILL_CONFIG[cache_dtype]
        != qsa_indexer_ops._PREFILL_CONFIG[cache_dtype]
    )
    assert torch.equal(run(True), run(False))
    assert bool((run(False) != -1.0).any())
