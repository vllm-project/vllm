# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout dispatch tests for ``QwenGatedDeltaNetAttention._forward_core_rocm``.

The AITER fused reshape+conv kernel learned Qwen3.5's flat ``[q|k|v|z]``
packing in https://github.com/ROCm/aiter/pull/3251, so flat-layout models take
the decode fast path too instead of falling back to the generic path.

The dispatch tests run host-side with recording stubs. The prefill numerical
tests exercise the production ChunkGatedDeltaRule wrapper on ROCm against a
token-wise Torch reference, including ragged state and output-buffer layouts.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

PREFIX = "model.layers.0.linear_attn"
H = 2  # num key heads
HV = 4  # num value heads
K = 8  # head_k_dim
V = 8  # head_v_dim


def _make_metadata(
    *,
    num_prefills: int = 0,
    num_decodes: int = 2,
    spec_sequence_masks: torch.Tensor | None = None,
):
    """Only the fields the dispatch condition reads carry meaningful values."""
    return GDNAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefills,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=num_prefills + num_decodes,
        spec_sequence_masks=spec_sequence_masks,
    )


def _make_layer(gqa_interleaved_layout: bool):
    """Stub layer running the real ``_forward_core_rocm``, recording dispatch."""
    layer = types.SimpleNamespace()
    layer.prefix = PREFIX
    layer.gqa_interleaved_layout = gqa_interleaved_layout
    layer.qkvz_layout = "interleaved" if gqa_interleaved_layout else "flat"

    layer.calls = []
    layer._forward_core_decode_aiter = lambda **kw: layer.calls.append("aiter")
    layer._forward_core = lambda **kw: layer.calls.append("generic")
    layer.prepare_gdn_attention_core_inputs = lambda qkvz, ba, n: (
        torch.zeros(n, H * K * 2 + HV * V),
        torch.zeros(n, HV, V),
        torch.zeros(n, HV),
        torch.zeros(n, HV),
    )
    layer._forward_core_rocm = types.MethodType(
        QwenGatedDeltaNetAttention._forward_core_rocm, layer
    )
    return layer


def _run(layer, meta) -> str:
    num_tokens = max(meta.num_actual_tokens, 1)
    ctx = types.SimpleNamespace(attn_metadata={PREFIX: meta})
    with patch.object(qwen_gdn_linear_attn, "get_forward_context", return_value=ctx):
        layer._forward_core_rocm(
            qkvz=torch.zeros(num_tokens, 2 * H * K + 2 * HV * V),
            ba=torch.zeros(num_tokens, 2 * HV),
            z_out=torch.zeros(num_tokens, HV, V),
            core_attn_out=torch.zeros(num_tokens, HV, V),
        )
    assert len(layer.calls) == 1
    return layer.calls[0]


@pytest.mark.parametrize("gqa_interleaved_layout", [True, False])
def test_pure_decode_takes_the_fast_path(gqa_interleaved_layout: bool) -> None:
    """Both packings reach the fast path; flat used to be guarded out."""
    layer = _make_layer(gqa_interleaved_layout)
    assert _run(layer, _make_metadata()) == "aiter"


@pytest.mark.parametrize("gqa_interleaved_layout", [True, False])
def test_layout_string_matches_the_packing(gqa_interleaved_layout: bool) -> None:
    """The value handed to the kernel's ``qkvz_layout`` parameter."""
    layer = _make_layer(gqa_interleaved_layout)
    expected = "interleaved" if gqa_interleaved_layout else "flat"
    assert layer.qkvz_layout == expected


@pytest.mark.parametrize("gqa_interleaved_layout", [True, False])
@pytest.mark.parametrize(
    "meta_kwargs",
    [
        {"num_prefills": 1},
        {"num_decodes": 0},
        {"spec_sequence_masks": torch.zeros(1, dtype=torch.bool)},
    ],
    ids=["has_prefill", "no_decode", "spec_decode"],
)
def test_non_pure_decode_batches_use_the_generic_path(
    gqa_interleaved_layout: bool, meta_kwargs: dict
) -> None:
    layer = _make_layer(gqa_interleaved_layout)
    assert _run(layer, _make_metadata(**meta_kwargs)) == "generic"


def _prefill_reference(q, k, v, g, beta, initial_state, lengths):
    """Independent token-wise recurrence in the layer's [value, key] layout."""
    q, k, v, g, beta, initial_state = (
        x.cpu().float() for x in (q, k, v, g, beta, initial_state)
    )
    q = q.repeat_interleave(v.shape[2] // q.shape[2], dim=2)
    k = k.repeat_interleave(v.shape[2] // k.shape[2], dim=2)
    q = q * q.shape[-1] ** -0.5
    output = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        start = 0
        for sequence, length in enumerate(lengths):
            state = initial_state[sequence].clone()
            for token in range(start, start + length):
                state = state * g[0, token].exp()[:, None, None]
                key = k[0, token]
                prediction = (state * key[:, None, :]).sum(dim=-1)
                delta = (v[0, token] - prediction) * beta[0, token, :, None]
                state = state + delta[:, :, None] * key[:, None, :]
                output[0, token] = (state * q[0, token, :, None, :]).sum(dim=-1)
            final_state[sequence] = state
            start += length
    finally:
        torch.set_num_threads(previous_threads)
    return output, final_state


@pytest.mark.parametrize(
    "state_dtype,correlated",
    [(torch.bfloat16, False), (torch.float32, False), (torch.float32, True)],
    ids=["ragged-bf16-state", "ragged-fp32-state", "correlated-keys"],
)
@torch.inference_mode()
def test_rocm_prefill_numerics_and_output_layout(state_dtype, correlated):
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        ChunkGatedDeltaRule,
    )
    from vllm.third_party.flash_linear_attention.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )

    if not torch.cuda.is_available():
        pytest.skip("ROCm prefill numerics require a GPU")
    # Only the config is needed for production backend selection; no weights.
    config = VllmConfig(
        model_config=ModelConfig(
            model="Qwen/Qwen3.5-0.8B",
            revision="2fc06364715b967f1860aea9cf38778875588b17",
            max_model_len=1024,
        )
    )
    with set_current_vllm_config(config):
        prefill = ChunkGatedDeltaRule()
    assert prefill.gdn_prefill_backend == "triton"
    assert prefill._forward_method == prefill.forward_native

    torch.manual_seed(1234)
    lengths = [64] if correlated else [1, 63, 64, 65, 129]
    tokens, key_heads, value_heads, dim = sum(lengths), 4, 8, 128
    q = torch.randn(1, tokens, key_heads, dim, device="cuda")
    q = torch.nn.functional.normalize(q, dim=-1).to(torch.bfloat16)
    k = torch.randn_like(q)
    k = torch.nn.functional.normalize(k.float(), dim=-1).to(q.dtype)
    v = torch.randn(1, tokens, value_heads, dim, device="cuda", dtype=q.dtype)
    g = -torch.rand(1, tokens, value_heads, device="cuda") * 0.1
    beta = torch.rand_like(g)
    state = (torch.randn(len(lengths), value_heads, dim, dim, device="cuda") * 0.05).to(
        state_dtype
    )
    if correlated:
        q.zero_()
        k.zero_()
        q[..., 0] = 1
        k[..., 0] = 1
        g.zero_()
        beta.fill_(0.9)
        state.zero_()
    expected, expected_state = _prefill_reference(q, k, v, g, beta, state, lengths)
    initial_state_copy = state.clone()
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=prepare_chunk_indices(cu_seqlens, 64),
        chunk_offsets=prepare_chunk_offsets(cu_seqlens, 64),
        use_qk_l2norm_in_kernel=False,
    )
    storage = torch.full(
        (tokens + 3, value_heads, dim), float("nan"), device="cuda", dtype=q.dtype
    )
    output, final_state = prefill(**kwargs, core_attn_out=storage[:tokens])
    assert output.data_ptr() == storage.data_ptr()
    assert output.shape == v.shape
    assert final_state.shape == state.shape
    assert torch.isnan(storage[tokens:]).all()
    torch.testing.assert_close(state, initial_state_copy, atol=0, rtol=0)

    # Preserve the upstream prefill suite's numerical budgets. The reference
    # here is a token-wise recurrence, independent of the chunk implementation.
    output_error = (output.cpu().float() - expected).abs()
    state_error = (final_state.cpu().float() - expected_state).abs()
    assert torch.isfinite(output).all() and torch.isfinite(final_state).all()
    assert output_error.max() < (1e-1 if correlated else 2e-3)
    assert output_error.mean() < (2e-3 if correlated else 6e-5)
    assert state_error.max() < (2e-1 if correlated else 2e-2)
    assert state_error.mean() < 6e-4

    no_buffer_output, no_buffer_state = prefill(**kwargs)
    torch.testing.assert_close(no_buffer_output, output, atol=0, rtol=0)
    torch.testing.assert_close(no_buffer_state, final_state, atol=0, rtol=0)
