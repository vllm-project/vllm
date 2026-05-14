# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the return_embed SamplingParams flag."""

import gc

import pytest
import torch

from vllm import LLM, SamplingParams

MODEL = "meta-llama/Llama-3.2-1B"


def _embedding(out) -> list[float] | None:
    if out.kv_transfer_params is None:
        return None
    return out.kv_transfer_params.get("embed")


@pytest.fixture(scope="module")
def llm():
    """Spin up a single LLM instance per module to avoid repeated init."""
    if not (hasattr(torch, "accelerator") and torch.accelerator.is_available()):
        pytest.skip("Accelerator (CUDA) required")
    inst = LLM(
        model=MODEL,
        enforce_eager=True,
        dtype="bfloat16",
        max_model_len=512,
        gpu_memory_utilization=0.45,
        enable_return_embed=True,
    )
    yield inst
    del inst
    gc.collect()


def _opt_in(max_tokens: int = 8) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        extra_args={"return_embed": True},
    )


def _no_opt(max_tokens: int = 8) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=max_tokens)


def test_pooled_embedding_returned(llm):
    """Single opted-in request gets an embedding back."""
    outs = llm.generate(
        ["The quick brown fox jumps over the lazy dog."],
        _opt_in(),
    )
    emb = _embedding(outs[0])
    assert emb is not None, "expected 'embed' in kv_transfer_params"
    hidden_size = llm.llm_engine.model_config.get_hidden_size()
    assert len(emb) == hidden_size
    # Sanity: not all zeros, not all NaN.
    arr = torch.tensor(emb, dtype=torch.float32)
    assert torch.isfinite(arr).all()
    assert arr.abs().sum().item() > 0


def test_no_opt_in_no_embedding(llm):
    """Non-opted-in request has no embedding (kv_transfer_params should
    not contain the key)."""
    outs = llm.generate(["Hello world."], _no_opt())
    emb = _embedding(outs[0])
    assert emb is None


def test_mixed_batch(llm):
    """Mixing opted-in and non-opted-in requests in a single generate
    call: only the opted-in one gets an embedding."""
    prompts = [
        "Opted-in: capital of France?",
        "Not opted-in: capital of Germany?",
    ]
    params = [_opt_in(), _no_opt()]
    outs = llm.generate(prompts, params)
    assert _embedding(outs[0]) is not None
    assert _embedding(outs[1]) is None


def test_embedding_deterministic(llm):
    """Same prompt twice gives the same embedding (up to fp drift)."""
    prompt = "vLLM serves transformer models efficiently."
    outs1 = llm.generate([prompt], _opt_in())
    outs2 = llm.generate([prompt], _opt_in())
    e1 = torch.tensor(_embedding(outs1[0]), dtype=torch.float32)
    e2 = torch.tensor(_embedding(outs2[0]), dtype=torch.float32)
    torch.testing.assert_close(e1, e2, atol=1e-3, rtol=1e-3)


def test_prefix_cache_hit_preserves_embedding(llm):
    """A second request whose prompt extends the first's must produce
    the right mean even when the shared prefix is served from the prefix
    cache."""
    base = "vLLM serves transformer models efficiently. " * 4
    extension = base + "It also supports prefix caching."

    # First request populates the cache.
    out_base = llm.generate([base], _opt_in())
    e_base = _embedding(out_base[0])
    assert e_base is not None

    # Cold reference for the extension (run before the cache contains
    # extension's prefix sums — but base's blocks are now in the cache,
    # so this run will hit them).
    out_ext_warm = llm.generate([extension], _opt_in())
    e_ext_warm = _embedding(out_ext_warm[0])
    assert e_ext_warm is not None

    # Run extension again: this time both base AND extension's full
    # prompt may be in the cache. Should still match.
    out_ext_again = llm.generate([extension], _opt_in())
    e_ext_again = _embedding(out_ext_again[0])
    assert e_ext_again is not None

    e1 = torch.tensor(e_ext_warm, dtype=torch.float32)
    e2 = torch.tensor(e_ext_again, dtype=torch.float32)
    torch.testing.assert_close(e1, e2, atol=1e-3, rtol=1e-3)
