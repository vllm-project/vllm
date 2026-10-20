# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end checks that ``backend_per_kind`` selects the requested attention
backend for each KV-cache group at runtime.

Uses ``google/gemma-3-1b-it``, which interleaves full-attention and
sliding-window layers, so the model produces separate ``full_attention`` and
``sliding_window`` KV-cache groups.
"""

import pytest

from vllm import LLM
from vllm.config.attention import AttentionConfig
from vllm.platforms import current_platform

MODEL = "google/gemma-3-1b-it"


def _collect_group_backends(worker) -> list[tuple[str, str]]:
    """Runs on the worker: returns (spec_kind, backend_name) per attn group."""
    from vllm.v1.kv_cache_interface import get_kv_cache_spec_kind

    out: list[tuple[str, str]] = []
    for kv_group in worker.model_runner.attn_groups:
        for attn_group in kv_group:
            kind = get_kv_cache_spec_kind(attn_group.kv_cache_spec)
            out.append((kind.value, attn_group.backend.get_name()))
    return out


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="backend names are CUDA-specific"
)
@pytest.mark.parametrize(
    "backend_per_kind",
    [
        {"full_attention": "FLASH_ATTN", "sliding_window": "TRITON_ATTN"},
        # Swapped, to prove the mapping is causal rather than the default.
        {"full_attention": "TRITON_ATTN", "sliding_window": "FLASH_ATTN"},
    ],
)
def test_backend_per_kind_splits_groups(backend_per_kind, monkeypatch):
    # collective_rpc ships the callable to the EngineCore subprocess; the
    # secure msgpack encoder can't serialize functions, so opt into the
    # pickle fallback (same pattern as test_pooling_chunked_prefill).
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = LLM(
        model=MODEL,
        attention_config=AttentionConfig(backend_per_kind=backend_per_kind),
        enforce_eager=True,
        max_model_len=2048,
        gpu_memory_utilization=0.4,
    )

    group_backends = llm.llm_engine.collective_rpc(_collect_group_backends)[0]
    kinds = {kind for kind, _ in group_backends}

    # gemma3 must actually split into both kinds for this test to be meaningful.
    assert "full_attention" in kinds
    assert "sliding_window" in kinds

    for kind, backend_name in group_backends:
        if kind in backend_per_kind:
            assert backend_name == backend_per_kind[kind], (
                f"{kind} group used {backend_name}, expected {backend_per_kind[kind]}"
            )


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="backend names are CUDA-specific"
)
def test_backend_per_kind_overrides_global_backend(monkeypatch):
    """A per-kind entry wins over the global ``backend`` for its kind; other
    kinds fall back to the global backend."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = LLM(
        model=MODEL,
        attention_config=AttentionConfig(
            backend="FLASH_ATTN",
            backend_per_kind={"sliding_window": "TRITON_ATTN"},
        ),
        enforce_eager=True,
        max_model_len=2048,
        gpu_memory_utilization=0.4,
    )

    group_backends = llm.llm_engine.collective_rpc(_collect_group_backends)[0]

    for kind, backend_name in group_backends:
        if kind == "sliding_window":
            assert backend_name == "TRITON_ATTN"
        elif kind == "full_attention":
            assert backend_name == "FLASH_ATTN"
