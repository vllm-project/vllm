# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.models.minimax_m3.common.ops import index_topk


@pytest.mark.parametrize(
    "platform,capability,query_len,seq_len,heads,expected_split",
    [
        pytest.param("cuda", (12, 0), 1239, 1239, 1, 2, id="sm120-split2"),
        pytest.param("cuda", (9, 0), 1239, 1239, 1, 1, id="sm90-unsplit"),
        pytest.param("cuda", (10, 0), 1239, 1239, 1, 1, id="sm100-unsplit"),
        pytest.param("cuda", (12, 1), 1239, 1239, 1, 1, id="sm121-unsplit"),
        pytest.param("rocm", None, 1239, 1239, 1, 1, id="rocm-unsplit"),
        pytest.param("cpu", None, 1239, 1239, 1, 1, id="cpu-unsplit"),
        pytest.param("cuda", (12, 0), 1, 1, 1, 1, id="sm120-single-block"),
        pytest.param("cuda", (12, 0), 1600, 1600, 2, 1, id="sm120-grid-sufficient"),
    ],
)
def test_prefill_split_dispatch_is_limited_to_sm120(
    monkeypatch, platform, capability, query_len, seq_len, heads, expected_split
):
    platform_probe = SimpleNamespace(
        is_cuda=MagicMock(return_value=platform == "cuda"),
        is_device_capability=MagicMock(side_effect=lambda wanted: wanted == capability),
    )
    kernel = MagicMock()
    monkeypatch.setattr(index_topk, "current_platform", platform_probe)
    monkeypatch.setattr(index_topk, "_index_block_score_kernel", kernel)
    query = torch.empty((query_len, heads, 128), dtype=torch.bfloat16, device="cpu")
    cache = torch.empty((1, 128, 128), dtype=torch.bfloat16, device="cpu")
    blocks = (seq_len + 127) // 128
    score = index_topk.minimax_m3_index_score(
        query,
        cache,
        torch.zeros((1, blocks), dtype=torch.int32, device="cpu"),
        torch.tensor([0, query_len], dtype=torch.int32, device="cpu"),
        torch.tensor([seq_len], dtype=torch.int32, device="cpu"),
        torch.tensor([seq_len - query_len], dtype=torch.int32, device="cpu"),
        query_len,
        seq_len,
        heads,
    )
    kernel.__getitem__.assert_called_once_with(
        ((query_len + 63) // 64, heads, expected_split)
    )
    launch = kernel.__getitem__.return_value
    launch.assert_called_once()
    assert launch.call_args.kwargs["USE_SPLIT_K"] is (expected_split > 1)
    assert launch.call_args.kwargs["BLOCK_SIZE_Q"] == 64
    assert launch.call_args.kwargs["BLOCK_SIZE_K"] == 128
    assert score.shape == (heads, query_len, ((blocks + 15) // 16) * 16)
    assert score.dtype == torch.float32 and score.device == query.device
    if platform != "cuda":
        platform_probe.is_device_capability.assert_not_called()
