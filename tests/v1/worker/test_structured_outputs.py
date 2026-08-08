# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_structured_outputs_worker_applies_cpu_bitmask(monkeypatch, dtype):
    def fail_if_cuda_stream_is_created():
        pytest.fail("CPU structured outputs must not create a CUDA stream")

    monkeypatch.setattr(torch.cuda, "Stream", fail_if_cuda_stream_is_created)

    worker = StructuredOutputsWorker(
        max_num_logits=3, vocab_size=64, device=torch.device("cpu")
    )
    input_batch = SimpleNamespace(
        req_ids=["unconstrained", "structured"],
        cu_num_logits_np=np.array([0, 1, 3], dtype=np.int32),
    )
    logits = torch.zeros((3, 64), dtype=dtype)
    grammar_bitmask = np.zeros((2, 2), dtype=np.int32)

    worker.apply_grammar_bitmask(
        logits,
        input_batch,
        grammar_req_ids=["structured"],
        grammar_bitmask=grammar_bitmask,
    )

    assert torch.isfinite(logits[0]).all()
    assert torch.isneginf(logits[1:]).all()
