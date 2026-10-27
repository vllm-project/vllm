# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``Qwen4ExpMTP.get_top_tokens`` must pick the same draft token
``compute_logits`` would. A typo delegating to the wrong head or tensor
(e.g. ``self.model`` instead of ``self.lm_head``) would otherwise silently
return a different draft token instead of failing.
"""

from unittest import mock

import pytest
import torch

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip(
        "get_top_tokens is implemented on the ROCm Qwen4Exp AMD MTP drafter only.",
        allow_module_level=True,
    )

from vllm.models.qwen4_exp.amd.mtp import Qwen4ExpMTP  # noqa: E402

_EMBEDDING = "vllm.model_executor.layers.vocab_parallel_embedding"


def test_get_top_tokens_matches_compute_logits():
    vocab_size, hidden_size = 256, 16

    # A single vocab shard, so the reduction returns the local argmax without
    # touching the process group.
    with (
        mock.patch(f"{_EMBEDDING}.get_tensor_model_parallel_rank", return_value=0),
        mock.patch(
            f"{_EMBEDDING}.get_tensor_model_parallel_world_size", return_value=1
        ),
    ):
        lm_head = ParallelLMHead(vocab_size, hidden_size, params_dtype=torch.bfloat16)
    with torch.no_grad():
        lm_head.weight.normal_(0.0, 0.02, generator=torch.Generator().manual_seed(7))

    mtp = object.__new__(Qwen4ExpMTP)
    mtp.lm_head = lm_head
    mtp.logits_processor = LogitsProcessor(vocab_size)

    hidden_states = torch.randn(16, hidden_size, dtype=torch.bfloat16)
    logits = mtp.compute_logits(hidden_states)
    assert logits is not None
    top = mtp.get_top_tokens(hidden_states)

    assert torch.equal(top, logits.argmax(dim=-1))
