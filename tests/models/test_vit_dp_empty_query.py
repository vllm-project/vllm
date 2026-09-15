# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for DP ViT empty-query handling (#52654)."""

import pytest
import torch

from vllm.model_executor.models.vision import get_load_balance_assignment

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_five_images_four_gpus_assignment():
    """Greedy balancer gives 2/1/1/1 for five equal images on TP=4."""
    sizes = [100, 100, 100, 100, 100]
    shuffle, counts, grouped = get_load_balance_assignment(sizes, num_gpus=4)
    assert counts == [2, 1, 1, 1]
    assert grouped == [200, 100, 100, 100]
    assert sorted(shuffle) == list(range(5))
    assert sum(counts) == 5


def test_flashinfer_wrapper_skips_empty_query():
    """0-token queries must not reach FlashInfer's stable-ABI empty_like."""
    from vllm.v1.attention.ops.vit_attn_wrappers import flashinfer_wrapper

    q = torch.empty(0, 2, 4, 8)
    workspace = torch.zeros(8, dtype=torch.uint8)
    out = flashinfer_wrapper(q, q, q, 1.0, workspace, o_data_type=torch.float32)
    assert out.shape == q.shape
    assert out.dtype == torch.float32
