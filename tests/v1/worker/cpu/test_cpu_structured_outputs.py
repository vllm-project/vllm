# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.cpu.structured_outputs import CPUStructuredOutputsWorker

pytestmark = pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")


def _worker(mask_stride: int = 4) -> CPUStructuredOutputsWorker:
    return CPUStructuredOutputsWorker(
        max_num_logits=8,
        vocab_size=10,
        device=torch.device("cpu"),
        mask_stride=mask_stride,
        num_bonus_tokens=1,
    )


def test_cpu_grammar_mask_applies_to_mapped_rows_only() -> None:
    input_batch = SimpleNamespace(
        req_ids=["plain", "grammar"],
        cu_num_logits_np=np.array([0, 1, 3], dtype=np.int32),
        cu_num_logits=torch.tensor([0, 1, 3], dtype=torch.int32),
        num_draft_tokens_per_req=None,
    )
    grammar_bitmask = np.array([[(1 << 1) | (1 << 4)], [1 << 2]], dtype=np.int32)
    logits = torch.zeros((3, 10))

    _worker().apply_grammar_bitmask(logits, input_batch, ["grammar"], grammar_bitmask)

    assert torch.isfinite(logits[0]).all()
    assert torch.isfinite(logits[1]).nonzero().flatten().tolist() == [1, 4]
    assert torch.isfinite(logits[2]).nonzero().flatten().tolist() == [2]


def test_cpu_grammar_mask_ignores_trimmed_draft_rows() -> None:
    input_batch = SimpleNamespace(
        req_ids=["grammar"],
        cu_num_logits_np=np.array([0, 3], dtype=np.int32),
        cu_num_logits=torch.tensor([0, 1], dtype=torch.int32),
        num_draft_tokens_per_req=np.array([2], dtype=np.int32),
    )
    grammar_bitmask = np.array([[1 << 3], [1 << 4], [1 << 5]], dtype=np.int32)
    logits = torch.zeros((1, 10), dtype=torch.bfloat16)

    _worker().apply_grammar_bitmask(logits, input_batch, ["grammar"], grammar_bitmask)

    assert torch.isfinite(logits[0]).nonzero().flatten().tolist() == [3]


def test_cpu_grammar_mask_handles_word_boundaries_and_sign_bit() -> None:
    input_batch = SimpleNamespace(
        req_ids=["grammar"],
        cu_num_logits_np=np.array([0, 1], dtype=np.int32),
        cu_num_logits=torch.tensor([0, 1], dtype=torch.int32),
        num_draft_tokens_per_req=None,
    )
    packed = np.array([[(1 << 0) | (1 << 31), 1 << 0, 1 << 5]], dtype=np.uint32).view(
        np.int32
    )
    logits = torch.zeros((1, 70))

    _worker().apply_grammar_bitmask(logits, input_batch, ["grammar"], packed)

    assert torch.isfinite(logits[0]).nonzero().flatten().tolist() == [0, 31, 32, 69]
