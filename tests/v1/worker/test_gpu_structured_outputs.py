# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker

pytestmark = pytest.mark.cpu_test


def test_explicit_xgrammar_backend_uses_v2_logit_mapping(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Stream", lambda: object())
    worker = StructuredOutputsWorker(
        max_num_logits=4,
        vocab_size=32,
        device=torch.device("cpu"),
        mask_stride=2,
        num_bonus_tokens=1,
        bitmask_backend="torch_native",
    )
    captured: dict[str, object] = {}

    def apply_token_bitmask_inplace(logits, bitmask, *, indices, backend):
        captured["logits"] = logits
        captured["bitmask"] = bitmask.clone()
        captured["indices"] = indices.clone()
        captured["backend"] = backend

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.structured_outputs.xgr",
        SimpleNamespace(apply_token_bitmask_inplace=apply_token_bitmask_inplace),
    )

    logits = torch.zeros((2, 32), dtype=torch.float32)
    bitmask = torch.tensor([[1], [2], [4], [8]], dtype=torch.int32)
    mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    cu_num_logits = torch.tensor([0, 1, 2], dtype=torch.int32)

    worker._apply_xgrammar_bitmask(logits, bitmask, mapping, cu_num_logits)

    assert captured["logits"] is logits
    assert captured["backend"] == "torch_native"
    torch.testing.assert_close(
        captured["indices"], torch.tensor([0, 1], dtype=torch.int32)
    )
    torch.testing.assert_close(
        captured["bitmask"], torch.tensor([[1], [4]], dtype=torch.int32)
    )
