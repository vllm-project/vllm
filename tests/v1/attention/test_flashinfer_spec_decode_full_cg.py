# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavioral tests for FlashInfer spec-decode FULL cudagraph support (#50885).

Covers:
1. Capability probe true/false against fast_decode_plan signature.
2. get_cudagraph_support: native multi-token path, causal fallback, SM90/SM12x.
3. Decode wrapper cache keyed by (batch_size, q_len_per_req).
4. fast_plan_decode forwards q_len_per_req (and errors when unsupported).
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("FlashInfer backend requires a CUDA platform.", allow_module_level=True)

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends import flashinfer as flashinfer_backend
from vllm.v1.attention.backends.flashinfer import (
    FlashInferMetadataBuilder,
    fast_plan_decode,
    flashinfer_supports_uniform_multi_token_decode,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

pytestmark = pytest.mark.cpu_test


def _vllm_config(*, use_non_causal: bool = False, dcp: int = 1) -> MagicMock:
    """Minimal VllmConfig stand-in (no HF download / hybrid model imports)."""
    cfg = MagicMock()
    cfg.attention_config.use_non_causal = use_non_causal
    cfg.parallel_config.decode_context_parallel_size = dcp
    cfg.model_config.get_num_attention_heads.return_value = 32
    return cfg


def _kv_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.float16,
    )


def _platform_not_sm90_or_sm12x(monkeypatch):
    monkeypatch.setattr(
        flashinfer_backend.current_platform,
        "is_device_capability",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        flashinfer_backend.current_platform,
        "is_device_capability_family",
        lambda *args, **kwargs: False,
    )


# ---------------------------------------------------------------------------
# 1. Capability probe true/false
# ---------------------------------------------------------------------------


def test_capability_probe_true_when_signature_has_q_len(monkeypatch):
    flashinfer_supports_uniform_multi_token_decode.cache_clear()

    def fake_sig(_fn):
        return inspect.Signature(
            [
                inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter(
                    "q_len_per_req",
                    inspect.Parameter.KEYWORD_ONLY,
                    default=1,
                ),
            ]
        )

    monkeypatch.setattr(flashinfer_backend.inspect, "signature", fake_sig)
    assert flashinfer_supports_uniform_multi_token_decode() is True
    flashinfer_supports_uniform_multi_token_decode.cache_clear()


def test_capability_probe_false_when_signature_lacks_q_len(monkeypatch):
    flashinfer_supports_uniform_multi_token_decode.cache_clear()

    def fake_sig(_fn):
        return inspect.Signature(
            [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        )

    monkeypatch.setattr(flashinfer_backend.inspect, "signature", fake_sig)
    assert flashinfer_supports_uniform_multi_token_decode() is False
    flashinfer_supports_uniform_multi_token_decode.cache_clear()


def test_capability_probe_matches_installed_flashinfer():
    """Live install: probe must agree with flashinfer.decode.fast_decode_plan."""
    flashinfer_supports_uniform_multi_token_decode.cache_clear()
    from flashinfer.decode import fast_decode_plan as fi_plan

    expected = "q_len_per_req" in inspect.signature(fi_plan).parameters
    assert flashinfer_supports_uniform_multi_token_decode() is expected


# ---------------------------------------------------------------------------
# 2. get_cudagraph_support paths
# ---------------------------------------------------------------------------


def test_cudagraph_support_sm90_is_single_token(monkeypatch):
    vllm_config = _vllm_config()
    monkeypatch.setattr(
        flashinfer_backend.current_platform,
        "is_device_capability",
        lambda cap, *a, **k: cap == 90,
    )
    support = FlashInferMetadataBuilder.get_cudagraph_support(vllm_config, _kv_spec())
    assert support == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE


def test_cudagraph_support_native_multi_token_returns_uniform_batch(monkeypatch):
    """No trtllm; FlashInfer multi-token supported; causal → UNIFORM_BATCH."""
    vllm_config = _vllm_config(use_non_causal=False)
    _platform_not_sm90_or_sm12x(monkeypatch)
    monkeypatch.setattr(
        flashinfer_backend, "can_use_trtllm_attention", lambda *a, **k: False
    )
    monkeypatch.setattr(
        flashinfer_backend,
        "flashinfer_supports_uniform_multi_token_decode",
        lambda: True,
    )
    support = FlashInferMetadataBuilder.get_cudagraph_support(vllm_config, _kv_spec())
    assert support == AttentionCGSupport.UNIFORM_BATCH


def test_cudagraph_support_native_multi_token_non_causal_falls_back(monkeypatch):
    """Native multi-token path requires causal attention."""
    vllm_config = _vllm_config(use_non_causal=True)
    _platform_not_sm90_or_sm12x(monkeypatch)
    monkeypatch.setattr(
        flashinfer_backend, "can_use_trtllm_attention", lambda *a, **k: False
    )
    monkeypatch.setattr(
        flashinfer_backend,
        "flashinfer_supports_uniform_multi_token_decode",
        lambda: True,
    )
    support = FlashInferMetadataBuilder.get_cudagraph_support(vllm_config, _kv_spec())
    assert support == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE


def test_cudagraph_support_without_capability_or_trtllm_is_single_token(monkeypatch):
    vllm_config = _vllm_config(use_non_causal=False)
    _platform_not_sm90_or_sm12x(monkeypatch)
    monkeypatch.setattr(
        flashinfer_backend, "can_use_trtllm_attention", lambda *a, **k: False
    )
    monkeypatch.setattr(
        flashinfer_backend,
        "flashinfer_supports_uniform_multi_token_decode",
        lambda: False,
    )
    support = FlashInferMetadataBuilder.get_cudagraph_support(vllm_config, _kv_spec())
    assert support == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE


def test_cudagraph_support_sm12x_trtllm_allows_non_causal(monkeypatch):
    """SM12x XQA: has_trtllm + is_sm12x → UNIFORM_BATCH even if non-causal."""
    vllm_config = _vllm_config(use_non_causal=True, dcp=1)
    monkeypatch.setattr(
        flashinfer_backend.current_platform,
        "is_device_capability",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        flashinfer_backend.current_platform,
        "is_device_capability_family",
        lambda fam, *a, **k: fam == 120,
    )
    monkeypatch.setattr(
        flashinfer_backend, "can_use_trtllm_attention", lambda *a, **k: True
    )
    monkeypatch.setattr(
        flashinfer_backend,
        "flashinfer_supports_uniform_multi_token_decode",
        lambda: True,
    )
    support = FlashInferMetadataBuilder.get_cudagraph_support(vllm_config, _kv_spec())
    assert support == AttentionCGSupport.UNIFORM_BATCH


# ---------------------------------------------------------------------------
# 3. Wrapper keys (batch_size, q_len_per_req)
# ---------------------------------------------------------------------------


def test_get_decode_wrapper_keys_by_batch_and_q_len(monkeypatch):
    """Distinct (batch, q_len) pairs must not share a cudagraph decode wrapper."""
    created: list[tuple] = []

    class FakeWrapper:
        def __init__(self, *args, **kwargs):
            created.append((args, kwargs))

    monkeypatch.setattr(
        flashinfer_backend, "BatchDecodeWithPagedKVCacheWrapper", FakeWrapper
    )
    monkeypatch.setattr(flashinfer_backend, "get_kv_cache_layout", lambda: "NHD")

    builder = object.__new__(FlashInferMetadataBuilder)

    class Buf:
        def __init__(self):
            self.gpu = torch.zeros(32, dtype=torch.int32)

    builder.paged_kv_indptr = Buf()
    builder.paged_kv_indices = Buf()
    builder.paged_kv_last_page_len = Buf()
    builder._workspace_buffer = torch.zeros(1024, dtype=torch.uint8)
    builder.is_kvcache_nvfp4 = False
    builder._decode_wrappers_cudagraph = {}
    builder._decode_wrapper = None

    w1 = builder._get_decode_wrapper(4, use_cudagraph=True, q_len_per_req=1)
    w2 = builder._get_decode_wrapper(4, use_cudagraph=True, q_len_per_req=3)
    w1b = builder._get_decode_wrapper(4, use_cudagraph=True, q_len_per_req=1)
    w3 = builder._get_decode_wrapper(8, use_cudagraph=True, q_len_per_req=1)

    assert w1 is w1b
    assert w1 is not w2
    assert w1 is not w3
    assert set(builder._decode_wrappers_cudagraph.keys()) == {
        (4, 1),
        (4, 3),
        (8, 1),
    }
    assert len(created) == 3


# ---------------------------------------------------------------------------
# 4. fast_plan_decode plan args
# ---------------------------------------------------------------------------


def test_fast_plan_decode_forwards_q_len_when_supported(monkeypatch):
    flashinfer_supports_uniform_multi_token_decode.cache_clear()
    monkeypatch.setattr(
        flashinfer_backend,
        "flashinfer_supports_uniform_multi_token_decode",
        lambda: True,
    )

    planned: dict = {}

    class FakeDecodeWrapper:
        is_cuda_graph_enabled = False

        def plan(self, **kwargs):
            planned.update(kwargs)

    fast_plan_decode(
        FakeDecodeWrapper(),
        indptr_cpu=torch.zeros(3, dtype=torch.int32),
        indices=torch.zeros(4, dtype=torch.int32),
        last_page_len_cpu=torch.ones(2, dtype=torch.int32),
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=64,
        page_size=16,
        q_len_per_req=3,
    )
    assert planned.get("q_len_per_req") == 3


def test_fast_plan_decode_omits_q_len_when_unsupported_and_q_len_one(monkeypatch):
    flashinfer_supports_uniform_multi_token_decode.cache_clear()
    monkeypatch.setattr(
        flashinfer_backend,
        "flashinfer_supports_uniform_multi_token_decode",
        lambda: False,
    )

    planned: dict = {}

    class FakeDecodeWrapper:
        is_cuda_graph_enabled = False

        def plan(self, **kwargs):
            planned.update(kwargs)

    fast_plan_decode(
        FakeDecodeWrapper(),
        indptr_cpu=torch.zeros(2, dtype=torch.int32),
        indices=torch.zeros(1, dtype=torch.int32),
        last_page_len_cpu=torch.ones(1, dtype=torch.int32),
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=64,
        page_size=16,
        q_len_per_req=1,
    )
    assert "q_len_per_req" not in planned


def test_fast_plan_decode_raises_when_multi_token_unsupported(monkeypatch):
    flashinfer_supports_uniform_multi_token_decode.cache_clear()
    monkeypatch.setattr(
        flashinfer_backend,
        "flashinfer_supports_uniform_multi_token_decode",
        lambda: False,
    )

    class FakeDecodeWrapper:
        is_cuda_graph_enabled = False

        def plan(self, **kwargs):
            raise AssertionError("plan should not be called")

    with pytest.raises(RuntimeError, match="uniform multi-token decode"):
        fast_plan_decode(
            FakeDecodeWrapper(),
            indptr_cpu=torch.zeros(2, dtype=torch.int32),
            indices=torch.zeros(1, dtype=torch.int32),
            last_page_len_cpu=torch.ones(1, dtype=torch.int32),
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=64,
            page_size=16,
            q_len_per_req=3,
        )


def test_fast_plan_decode_has_q_len_per_req_param():
    sig = inspect.signature(fast_plan_decode)
    assert "q_len_per_req" in sig.parameters
    assert sig.parameters["q_len_per_req"].default == 1
