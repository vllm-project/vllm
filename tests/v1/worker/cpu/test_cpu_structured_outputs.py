# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.platforms import CpuArchEnum, current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.worker.cpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.cpu.structured_outputs import CPUStructuredOutputsWorker

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu()
    or HAS_TRITON
    or current_platform.get_cpu_architecture() == CpuArchEnum.X86,
    reason="requires a non-x86 CPU platform without Triton",
)


def _worker(vocab_size: int) -> CPUStructuredOutputsWorker:
    return CPUStructuredOutputsWorker(
        max_num_logits=8,
        vocab_size=vocab_size,
        device=torch.device("cpu"),
        mask_stride=1,
    )


def _input_batch(req_ids: list[str]) -> InputBatch:
    buffers = InputBuffers(
        max_num_reqs=len(req_ids),
        max_num_tokens=len(req_ids),
        device=torch.device("cpu"),
    )
    batch = InputBatch.make_dummy(len(req_ids), len(req_ids), buffers)
    batch.req_ids = req_ids
    return batch


def test_cpu_grammar_mask_applies_to_mapped_rows_only() -> None:
    input_batch = _input_batch(["plain", "grammar"])
    grammar_bitmask = np.array([[(1 << 1) | (1 << 4)]], dtype=np.int32)
    logits = torch.zeros((2, 10))

    _worker(10).apply_grammar_bitmask(logits, input_batch, ["grammar"], grammar_bitmask)

    assert torch.isfinite(logits[0]).all()
    assert torch.isfinite(logits[1]).nonzero().flatten().tolist() == [1, 4]


def test_cpu_grammar_mask_handles_word_boundaries_and_sign_bit() -> None:
    input_batch = _input_batch(["grammar"])
    packed = np.array([[(1 << 0) | (1 << 31), 1 << 0, 1 << 5]], dtype=np.uint32).view(
        np.int32
    )
    logits = torch.zeros((1, 70))

    _worker(70).apply_grammar_bitmask(logits, input_batch, ["grammar"], packed)

    assert torch.isfinite(logits[0]).nonzero().flatten().tolist() == [0, 31, 32, 69]
