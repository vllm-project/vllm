# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models.qwen3_next import ChunkGatedDeltaRule
from vllm.platforms import current_platform


def test_chunk_gated_delta_rule_uses_cuda_backend_on_sm90_plus(monkeypatch):
    # Simulate Blackwell-like behavior where capability is > 90 but not exactly 90.
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_platform, "has_device_capability", lambda _: True)

    op = ChunkGatedDeltaRule()

    assert op._forward_method == op.forward_cuda


def test_chunk_gated_delta_rule_uses_native_backend_below_sm90(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_platform, "has_device_capability", lambda _: False)

    op = ChunkGatedDeltaRule()

    assert op._forward_method == op.forward_native
