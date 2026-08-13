# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.pp_utils import PendingRecv, PPHandler


@pytest.mark.skip_global_cleanup
def test_get_prev_sampled_outputs_splits_draft_tokens():
    handler = PPHandler.__new__(PPHandler)
    handler.max_sample_len = 3
    handler.draft_token_width = 2
    handler.main_stream = Mock()
    handler.req_idx_gen_np = np.zeros(4, dtype=np.int32)

    token_payload = torch.tensor([[10, 11, 12, 20, 21], [30, 31, 32, 40, 41]])
    slot = PendingRecv(
        event=Mock(),
        token_payload=token_payload,
        num_sampled=torch.tensor([1, 2]),
        num_rejected=torch.tensor([0, 1]),
        idx_mapping=torch.tensor([1, 3]),
        idx_mapping_np=np.array([1, 3]),
        need_sampled_mask=np.array([True, True]),
        gen_at_receive_np=np.array([0, 0]),
    )
    handler.queue = deque([slot])

    outputs = handler.get_prev_sampled_outputs()

    assert outputs is not None
    torch.testing.assert_close(outputs["sampled_tokens"], token_payload[:, :3])
    torch.testing.assert_close(outputs["draft_tokens"], token_payload[:, 3:])
    torch.testing.assert_close(outputs["draft_idx_mapping"], torch.tensor([1, 3]))
    handler.main_stream.wait_event.assert_called_once_with(slot.event)


@pytest.mark.skip_global_cleanup
def test_postprocess_sampled_updates_draft_tokens():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.is_last_pp_rank = False
    runner.req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=Mock()),
        last_sampled_tokens=Mock(),
        all_token_ids=SimpleNamespace(gpu=Mock()),
        total_len=SimpleNamespace(gpu=Mock()),
        draft_tokens=torch.zeros(4, 2, dtype=torch.int64),
    )
    runner.model_state = Mock()
    draft_tokens = torch.tensor([[20, 21], [40, 41]])

    with patch.object(model_runner_module, "post_update") as mock_post_update:
        runner.postprocess_sampled(
            idx_mapping=torch.tensor([1, 3]),
            sampled_tokens=torch.tensor([[10, 11, 12], [30, 31, 32]]),
            num_sampled=torch.tensor([1, 2]),
            num_rejected=torch.tensor([0, 1]),
            draft_tokens=draft_tokens,
            draft_idx_mapping=torch.tensor([1, 3]),
        )

    mock_post_update.assert_called_once()
    torch.testing.assert_close(runner.req_states.draft_tokens[[1, 3]], draft_tokens)
