# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the static-FP8 MLA prefill fast path.

These tests target the helper-method surface added in PR #40304
(``mha_support_kernel_quant``, the cached
``_prefill_o_scale_float``, and the asserts in non-trtllm
``_run_prefill_new_tokens_*`` methods) without spinning up a real
attention backend or GPU kernel. They run on any platform.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import MLACommonImpl
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform


def _stub_impl(
    *,
    routes_to_trtllm: bool,
    prefill_o_scale_float: float | None = None,
):
    """Build a minimal stand-in for an ``MLACommonImpl`` instance.

    Only the attributes touched by the methods under test are populated;
    we deliberately avoid running ``MLACommonImpl.__init__`` so the test
    has no GPU/backend dependencies.
    """
    impl = SimpleNamespace()
    # ``_run_prefill_new_tokens`` is what ``mha_support_kernel_quant``
    # compares against to know which backend was bound at init.
    impl._run_prefill_new_tokens_trtllm_ragged = MagicMock(
        name="_run_prefill_new_tokens_trtllm_ragged"
    )
    if routes_to_trtllm:
        impl._run_prefill_new_tokens = impl._run_prefill_new_tokens_trtllm_ragged
    else:
        impl._run_prefill_new_tokens = MagicMock(name="_run_prefill_new_tokens_other")
    impl._prefill_o_scale_float = prefill_o_scale_float
    return impl


def _prefill_meta(*, chunked: bool, q_data_type: torch.dtype = torch.bfloat16):
    """Bare-minimum stand-in for ``MLACommonPrefillMetadata``.

    ``q_data_type`` defaults to BF16 — the common case where the model's
    FP8 GEMMs dequantize before attention. Pass ``current_platform.fp8_dtype()``
    to simulate FP8 prefill attention (kv_cache_dtype=fp8 +
    use_prefill_query_quantization=True).
    """
    return SimpleNamespace(
        chunked_context=MagicMock() if chunked else None,
        q_data_type=q_data_type,
    )


# ---------------------------------------------------------------------------
# mha_support_kernel_quant gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "quant_key, routes_to_trtllm, prefill_meta, expected",
    [
        # The supported case: trtllm-ragged + static FP8 + no chunked context
        (kFp8StaticTensorSym, True, _prefill_meta(chunked=False), True),
        # Wrong backend -> no fast path even with static FP8
        (kFp8StaticTensorSym, False, _prefill_meta(chunked=False), False),
        # Chunked context disqualifies the fast path
        (kFp8StaticTensorSym, True, _prefill_meta(chunked=True), False),
        # ``prefill_metadata is None`` (e.g. profile/dummy run) disqualifies
        (kFp8StaticTensorSym, True, None, False),
        # FP8 prefill attention disqualifies: flashinfer's wrapper rejects
        # (FP8 query + FP8 out) for trtllm_ragged_attention_deepseek.
        (
            kFp8StaticTensorSym,
            True,
            _prefill_meta(chunked=False, q_data_type=current_platform.fp8_dtype()),
            False,
        ),
        # Per-group FP8 not yet supported by any prefill kernel
        (kFp8Dynamic128Sym, True, _prefill_meta(chunked=False), False),
        # NVFP4 not yet plumbed through the ragged trtllm Python wrapper
        (kNvfp4Dynamic, True, _prefill_meta(chunked=False), False),
    ],
)
def test_mha_support_kernel_quant_gate(
    quant_key, routes_to_trtllm, prefill_meta, expected
):
    impl = _stub_impl(routes_to_trtllm=routes_to_trtllm)
    actual = MLACommonImpl.mha_support_kernel_quant(impl, quant_key, prefill_meta)
    assert actual is expected


# ---------------------------------------------------------------------------
# Asserts in non-trtllm prefill backends
# ---------------------------------------------------------------------------


_NON_TRTLLM_BACKENDS = [
    MLACommonImpl._run_prefill_new_tokens_fa,
    MLACommonImpl._run_prefill_new_tokens_fi,
    MLACommonImpl._run_prefill_new_tokens_cudnn,
]


@pytest.mark.parametrize("method", _NON_TRTLLM_BACKENDS)
def test_non_trtllm_backends_reject_output_scale(method):
    """Backends without an in-kernel FP8 fast path must refuse it loudly."""
    impl = _stub_impl(routes_to_trtllm=False)
    sentinel_scale = MagicMock(name="output_scale")
    with pytest.raises(AssertionError, match="quant output"):
        method(
            impl,
            prefill=MagicMock(),
            q=MagicMock(),
            k=MagicMock(),
            v=MagicMock(),
            return_softmax_lse=False,
            output_scale=sentinel_scale,
        )


@pytest.mark.parametrize("method", _NON_TRTLLM_BACKENDS)
def test_non_trtllm_backends_reject_preallocated_out(method):
    impl = _stub_impl(routes_to_trtllm=False)
    sentinel_out = MagicMock(name="out")
    with pytest.raises(AssertionError, match="preallocated out"):
        method(
            impl,
            prefill=MagicMock(),
            q=MagicMock(),
            k=MagicMock(),
            v=MagicMock(),
            return_softmax_lse=False,
            out=sentinel_out,
        )


# ---------------------------------------------------------------------------
# _prefill_o_scale_float caching in the trtllm-ragged path
# ---------------------------------------------------------------------------


def _trtllm_ragged_inputs():
    """Build minimal tensor inputs that satisfy the trtllm wrapper's checks
    until the kernel call (which we mock).
    """
    device = torch.device("cpu")
    q = torch.zeros((2, 1, 4), dtype=torch.bfloat16, device=device)
    k = torch.zeros((2, 1, 4), dtype=torch.bfloat16, device=device)
    v = torch.zeros((2, 1, 4), dtype=torch.bfloat16, device=device)
    prefill = SimpleNamespace(
        query_seq_lens=torch.tensor([1, 1], dtype=torch.int32, device=device),
        workspace_buffer=torch.zeros(1, dtype=torch.uint8, device=device),
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
        output_dtype=torch.bfloat16,
    )
    return prefill, q, k, v


def _trtllm_impl():
    """Stand-in impl with just the fields the trtllm-ragged path reads."""
    impl = _stub_impl(routes_to_trtllm=True)
    impl.scale = 0.125
    return impl


def test_prefill_o_scale_float_cached_after_first_call():
    """First call samples ``output_scale`` to a host scalar; the second
    must reuse the cached value without touching the GPU again — this is
    what makes the path safe inside a captured cudagraph.
    """
    impl = _trtllm_impl()
    prefill, q, k, v = _trtllm_ragged_inputs()

    output_scale = MagicMock(spec=torch.Tensor)
    output_scale.cpu.return_value.item.return_value = 4.0

    fake_kernel = MagicMock(name="trtllm_ragged_attention_deepseek")
    fake_kernel.return_value = torch.zeros_like(q)

    with patch(
        "flashinfer.prefill.trtllm_ragged_attention_deepseek",
        fake_kernel,
    ):
        MLACommonImpl._run_prefill_new_tokens_trtllm_ragged(
            impl,
            prefill,
            q,
            k,
            v,
            return_softmax_lse=False,
            output_scale=output_scale,
        )
        MLACommonImpl._run_prefill_new_tokens_trtllm_ragged(
            impl,
            prefill,
            q,
            k,
            v,
            return_softmax_lse=False,
            output_scale=output_scale,
        )

    # The host scalar conversion (and the GPU sync it implies) must happen
    # exactly once across the two calls.
    assert output_scale.cpu.call_count == 1
    assert impl._prefill_o_scale_float == 4.0

    # The kernel must receive the inverted, cached scalar both times.
    assert fake_kernel.call_count == 2
    for call in fake_kernel.call_args_list:
        assert call.kwargs["bmm2_scale"] == pytest.approx(0.25)


def test_prefill_o_scale_float_unset_when_no_output_scale():
    """When the slow path is taken (no output_scale), the cache must stay
    untouched and ``bmm2_scale`` must be 1.0.
    """
    impl = _trtllm_impl()
    prefill, q, k, v = _trtllm_ragged_inputs()

    fake_kernel = MagicMock(name="trtllm_ragged_attention_deepseek")
    fake_kernel.return_value = torch.zeros_like(q)

    with patch(
        "flashinfer.prefill.trtllm_ragged_attention_deepseek",
        fake_kernel,
    ):
        MLACommonImpl._run_prefill_new_tokens_trtllm_ragged(
            impl,
            prefill,
            q,
            k,
            v,
            return_softmax_lse=False,
        )

    assert impl._prefill_o_scale_float is None
    assert fake_kernel.call_args.kwargs["bmm2_scale"] == 1.0


def test_prefill_o_scale_float_preserves_warmup_value():
    """A pre-populated cache (e.g. set during warmup) must NOT be re-read
    from ``output_scale`` even on a subsequent call.
    """
    impl = _trtllm_impl()
    impl._prefill_o_scale_float = 8.0  # simulates value cached at warmup
    prefill, q, k, v = _trtllm_ragged_inputs()

    output_scale = MagicMock(spec=torch.Tensor)
    output_scale.cpu.return_value.item.return_value = 999.0  # would be wrong

    fake_kernel = MagicMock(name="trtllm_ragged_attention_deepseek")
    fake_kernel.return_value = torch.zeros_like(q)

    with patch(
        "flashinfer.prefill.trtllm_ragged_attention_deepseek",
        fake_kernel,
    ):
        MLACommonImpl._run_prefill_new_tokens_trtllm_ragged(
            impl,
            prefill,
            q,
            k,
            v,
            return_softmax_lse=False,
            output_scale=output_scale,
        )

    output_scale.cpu.assert_not_called()
    assert impl._prefill_o_scale_float == 8.0
    assert fake_kernel.call_args.kwargs["bmm2_scale"] == pytest.approx(0.125)
