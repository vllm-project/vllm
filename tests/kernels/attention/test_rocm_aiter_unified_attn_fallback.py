# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the AITER unified-attention LDS-overflow fallback.

Covers the two pieces that can silently regress without hardware: the
exception predicate (which must survive Triton version churn) and the
latch/reroute dispatch in ``forward``. No GPU and no aiter required.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
    RocmAiterUnifiedAttentionImpl,
    _is_lds_overflow_error,
)
from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadata

# The literal Triton emits for this failure, from the #48723 report.
REAL_TRITON_MESSAGE = (
    "out of resource: shared memory, Required: 65792, Hardware limit: 65536. "
    "Reducing block sizes or `num_stages` may help."
)


def test_matches_real_triton_shared_memory_message():
    assert _is_lds_overflow_error(RuntimeError(REAL_TRITON_MESSAGE))


@pytest.mark.parametrize(
    "message",
    [
        "out of resource: registers, Required: 100, Hardware limit: 64",
        "shared memory access fault at 0x0",
    ],
)
def test_rejects_single_marker_messages(message):
    """Both markers are required, so a one-marker error is not swallowed."""
    assert not _is_lds_overflow_error(RuntimeError(message))


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("bad shape"),
        RuntimeError("HIP error: invalid device function"),
    ],
)
def test_rejects_unrelated_errors(exc):
    assert not _is_lds_overflow_error(exc)


def test_matches_typed_triton_error_when_class_is_importable():
    """The isinstance branch fires even if the message text changes."""
    triton_errors = pytest.importorskip("triton.runtime.errors")
    exc = triton_errors.OutOfResources(65792, 65536, "shared memory")
    assert _is_lds_overflow_error(exc)


NUM_BLOCKS = 2
NUM_KV_HEADS = 1
BLOCK_SIZE = 16
HEAD_SIZE = 8
NUM_TOKENS = 2


def _make_impl() -> RocmAiterUnifiedAttentionImpl:
    """Build an Impl without running __init__, which imports aiter."""
    impl = object.__new__(RocmAiterUnifiedAttentionImpl)
    impl.attn_type = AttentionType.DECODER
    impl.head_size = HEAD_SIZE
    impl.scale = 1.0
    impl.kv_cache_dtype = "auto"
    impl.fp8_dtype = torch.float8_e4m3fn
    impl.alibi_slopes = None
    impl.sliding_window = None
    impl.logits_soft_cap = None
    impl.sinks = None
    impl._aiter_lds_overflow = False
    return impl


def _make_metadata(causal: bool) -> RocmAttentionMetadata:
    return RocmAttentionMetadata(
        num_actual_tokens=NUM_TOKENS,
        max_query_len=1,
        query_start_loc=torch.tensor([0, NUM_TOKENS], dtype=torch.int32),
        max_seq_len=1024,
        seq_lens=torch.tensor([NUM_TOKENS], dtype=torch.int32),
        block_table=torch.zeros((1, NUM_BLOCKS), dtype=torch.int32),
        slot_mapping=torch.zeros(NUM_TOKENS, dtype=torch.int64),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        causal=causal,
    )


def _drive_forward(impl, causal=True):
    """Call forward with minimal CPU tensors; returns the output tensor."""
    query = torch.zeros((NUM_TOKENS, 1, HEAD_SIZE))
    kv_cache = torch.zeros((NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, 2 * HEAD_SIZE))
    output = torch.zeros((NUM_TOKENS, HEAD_SIZE))
    scale = torch.tensor(1.0)
    layer = SimpleNamespace(_q_scale=scale, _k_scale=scale, _v_scale=scale)
    return impl.forward(
        layer=layer,
        query=query,
        key=torch.zeros(0),
        value=torch.zeros(0),
        kv_cache=kv_cache,
        attn_metadata=_make_metadata(causal),
        output=output,
    )


def _overflow_raiser(calls):
    def _raise(**kwargs):
        calls.append(kwargs)
        raise RuntimeError(REAL_TRITON_MESSAGE)

    return _raise


def _recorder(calls):
    def _record(*args, **kwargs):
        calls.append((args, kwargs))

    return _record


def test_first_causal_forward_catches_overflow_and_latches():
    impl = _make_impl()
    aiter_calls: list = []
    fallback_calls: list = []
    impl.unified_attention = _overflow_raiser(aiter_calls)
    impl._forward_triton_unified = _recorder(fallback_calls)

    _drive_forward(impl)

    assert len(aiter_calls) == 1
    assert impl._aiter_lds_overflow is True
    assert len(fallback_calls) == 1


def test_latched_impl_skips_aiter_on_later_forwards():
    impl = _make_impl()
    aiter_calls: list = []
    fallback_calls: list = []
    impl.unified_attention = _overflow_raiser(aiter_calls)
    impl._forward_triton_unified = _recorder(fallback_calls)

    _drive_forward(impl)
    _drive_forward(impl)

    assert len(aiter_calls) == 1, "latch must short-circuit the aiter call"
    assert len(fallback_calls) == 2


def test_non_causal_forward_never_calls_aiter():
    impl = _make_impl()
    aiter_calls: list = []
    fallback_calls: list = []
    impl.unified_attention = _overflow_raiser(aiter_calls)
    impl._forward_triton_unified = _recorder(fallback_calls)

    _drive_forward(impl, causal=False)

    assert aiter_calls == []
    assert len(fallback_calls) == 1
    assert impl._aiter_lds_overflow is False


def test_unrelated_error_propagates_and_does_not_latch():
    impl = _make_impl()
    fallback_calls: list = []

    def _raise_unrelated(**kwargs):
        raise RuntimeError("HIP error: invalid device function")

    impl.unified_attention = _raise_unrelated
    impl._forward_triton_unified = _recorder(fallback_calls)

    with pytest.raises(RuntimeError, match="invalid device function"):
        _drive_forward(impl)

    assert impl._aiter_lds_overflow is False
    assert fallback_calls == []
