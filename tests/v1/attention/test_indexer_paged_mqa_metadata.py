# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.attention.backends.mla import indexer as indexer_mod


def test_paged_mqa_metadata_requires_supported_arch_and_32_64_states(monkeypatch):
    monkeypatch.setattr(indexer_mod.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(indexer_mod, "is_deep_gemm_supported", lambda: True)
    assert indexer_mod._should_build_paged_mqa_logits_metadata(64) is True
    assert indexer_mod._should_build_paged_mqa_logits_metadata(32) is True
    assert indexer_mod._should_build_paged_mqa_logits_metadata(2) is False

    monkeypatch.setattr(indexer_mod, "is_deep_gemm_supported", lambda: False)
    assert indexer_mod._should_build_paged_mqa_logits_metadata(64) is False


def test_paged_mqa_metadata_skipped_when_not_cuda(monkeypatch):
    monkeypatch.setattr(indexer_mod.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(indexer_mod, "is_deep_gemm_supported", lambda: True)
    assert indexer_mod._should_build_paged_mqa_logits_metadata(64) is False
