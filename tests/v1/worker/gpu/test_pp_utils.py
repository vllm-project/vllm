# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from vllm.v1.worker.gpu.pp_utils import (
    PendingRecv,
    PPHandler,
    _pad_sampled_token_ids,
)
from vllm.v1.worker.gpu.spec_decode.dspark import utils as dspark_utils


def test_only_cuda_kimi_k3_enables_dspark_pipeline_parallelism(monkeypatch):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="kimi_k3")
        )
    )
    monkeypatch.setattr(dspark_utils.current_platform, "is_cuda", lambda: True)

    assert dspark_utils.supports_dspark_pipeline_parallelism(vllm_config)

    vllm_config.model_config.hf_config.model_type = "qwen3"
    assert not dspark_utils.supports_dspark_pipeline_parallelism(vllm_config)

    vllm_config.model_config.hf_config.model_type = "kimi_k3"
    monkeypatch.setattr(dspark_utils.current_platform, "is_cuda", lambda: False)
    assert not dspark_utils.supports_dspark_pipeline_parallelism(vllm_config)


def test_pad_sampled_token_ids_to_pp_receive_width():
    sampled = torch.tensor([[10], [20]], dtype=torch.int64)

    padded = _pad_sampled_token_ids(sampled, max_sample_len=3)

    torch.testing.assert_close(
        padded,
        torch.tensor([[10, -1, -1], [20, -1, -1]], dtype=torch.int64),
    )


def test_pp_handler_returns_relayed_draft_tokens():
    handler = object.__new__(PPHandler)
    handler.queue = deque()
    handler.req_idx_gen_np = np.zeros(2, dtype=np.int32)
    handler.device = torch.device("cpu")
    handler.main_stream = Mock()

    sampled_tokens = torch.tensor([[10, 11], [20, 21]])
    draft_tokens = torch.tensor([[11], [21]])
    idx_mapping = torch.tensor([0, 1])
    handler.queue.append(
        PendingRecv(
            event=Mock(),
            sampled_tokens=sampled_tokens,
            num_sampled=torch.ones(2, dtype=torch.int32),
            num_rejected=torch.zeros(2, dtype=torch.int32),
            idx_mapping=idx_mapping,
            idx_mapping_np=np.array([0, 1]),
            need_sampled_mask=np.array([True, True]),
            gen_at_receive_np=np.array([0, 0]),
            draft_tokens=draft_tokens,
        )
    )

    output = handler.get_prev_sampled_outputs()

    assert output is not None
    assert output["draft_tokens"] is draft_tokens
    handler.main_stream.wait_event.assert_called_once()
