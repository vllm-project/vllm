# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused unit tests for the GLM decode top-k dispatch."""

import pytest
import torch

from vllm.models.glm5next.nvidia.sparse_indexer import _use_cooperative_topk
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only dispatch"
)


@pytest.mark.parametrize("num_rows", [1, 64], ids=["1row", "64rows"])
@torch.inference_mode()
def test_cooperative_topk_is_selected_for_supported_batch(num_rows: int):
    logits = torch.empty(num_rows, 8192, dtype=torch.float32, device="cuda")
    assert _use_cooperative_topk(logits, 512, num_rows)


@pytest.mark.parametrize(
    ("num_rows", "select_k", "stride_padding"),
    [
        (65, 512, 0),
        (1, 256, 0),
        (1, 512, 1),
    ],
    ids=["too_many_rows", "unsupported_k", "unaligned_stride"],
)
@torch.inference_mode()
def test_cooperative_topk_falls_back_on_unsupported_batch(
    num_rows: int, select_k: int, stride_padding: int
):
    logits = torch.empty(
        num_rows, 8192 + stride_padding, dtype=torch.float32, device="cuda"
    )
    if stride_padding:
        logits = logits[:, :8192]
    assert not _use_cooperative_topk(logits, select_k, num_rows)
