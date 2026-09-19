# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from itertools import accumulate
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm.models.qwen4_exp.cpu.model_state import Qwen4ExpModelState
from vllm.models.qwen4_exp.cpu.ngram_embedding import (
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEFp8EmbeddingMethod,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)
from vllm.models.qwen4_exp.cpu.ops.ple import ple_gate, ple_ngram_ids
from vllm.models.qwen4_exp.cpu.ple_layer import Qwen4ExpPLELayer
from vllm.models.qwen4_exp.cpu.runtime import has_active_triton_cpu_backend
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

from .test_ple import (
    _ConvBatchCase,
    _make_conv_case,
    _make_conv_metadata,
    _ngram_hash_params,
    _reference_ngram_ids,
    _short_conv_dilated_dispatch_pytorch,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu(),
    reason="Qwen4Exp CPU PLE tests require a CPU platform",
)
requires_triton_cpu = pytest.mark.skipif(
    not has_active_triton_cpu_backend(),
    reason="Qwen4Exp PLE kernels require an active Triton-CPU backend",
)


def test_cpu_model_state_returns_active_ngram_views() -> None:
    model_state = object.__new__(Qwen4ExpModelState)
    model_state.uses_ngram_embedding = True
    model_state.ngram_context_len = 3
    model_state.ngram_eos_token_id = 99
    model_state.ngram_context = torch.empty((8, 3), dtype=torch.int32)
    model_state.ngram_context_offsets = torch.arange(-3, 0, dtype=torch.int64)
    model_state.ple_query_start_loc = torch.empty(9, dtype=torch.int32)

    input_batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=3,
        idx_mapping=torch.tensor([1, 0]),
        query_start_loc=torch.tensor([0, 2, 3, 3], dtype=torch.int32),
    )
    req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([3, 1])),
        all_token_ids=SimpleNamespace(
            gpu=torch.tensor([[1, 2, 3, 4], [20, 21, 22, 23]], dtype=torch.int32)
        ),
    )

    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_inputs = model_state.prepare_inputs(input_batch, req_states)

    torch.testing.assert_close(
        model_inputs["query_start_loc"],
        torch.tensor([0, 2, 3, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(
        model_inputs["ngram_context"],
        torch.tensor(
            [[99, 99, 20], [1, 2, 3], [99, 99, 99]],
            dtype=torch.int32,
        ),
    )


def test_cpu_ple_fp8_dequantizes_only_selected_rows() -> None:
    layer = nn.Module()
    layer.register_parameter(
        "weight_scale",
        nn.Parameter(torch.tensor([0.25]), requires_grad=False),
    )
    quantized = torch.tensor(
        [[16.0, 32.0], [1.0, 2.0]],
        dtype=torch.float8_e4m3fn,
    )

    output = Qwen4ExpPLEFp8EmbeddingMethod().dequantize(
        layer,
        quantized,
        torch.bfloat16,
    )

    assert quantized.dtype == torch.float8_e4m3fn
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(
        output,
        torch.tensor([[4.0, 8.0], [0.25, 0.5]], dtype=torch.bfloat16),
    )


def test_cpu_ple_bf16_unquantized_embedding_preserves_values() -> None:
    method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(
        None,
        "model.layers.1.ple.ple_embedding.ngram_embedding",
    )
    assert isinstance(method, Qwen4ExpPLEUnquantizedEmbeddingMethod)
    layer = nn.Module()
    layer.register_parameter(
        "weight",
        nn.Parameter(
            torch.tensor(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        ),
    )

    selected = method.embedding(layer, torch.tensor([1, 0]))
    output = method.dequantize(layer, selected, torch.bfloat16)

    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(
        output,
        torch.tensor([[3.0, 4.0], [1.0, 2.0]], dtype=torch.bfloat16),
    )


@requires_triton_cpu
def test_cpu_ple_ngram_ids_match_reference_and_reuse_output() -> None:
    query_lens = [1, 33, 0, 2]
    query_start_loc = torch.tensor([0, *accumulate(query_lens)], dtype=torch.int32)
    input_ids = torch.arange(
        1_000_000_000,
        1_000_000_000 + sum(query_lens),
        dtype=torch.int32,
    )
    input_ids[[5, 32]] = 251
    ngram_context = torch.tensor(
        [
            [1_000_000_036, 1_000_000_037],
            [1_000_000_038, 1_000_000_039],
            [1_000_000_040, 1_000_000_041],
            [1_000_000_042, 251],
        ],
        dtype=torch.int32,
    )
    params = _ngram_hash_params(torch.device("cpu"), ngram_context.shape[1])

    expected = _reference_ngram_ids(input_ids, query_start_loc, ngram_context, **params)
    output = torch.empty_like(expected)
    actual = ple_ngram_ids(
        input_ids,
        query_start_loc,
        ngram_context,
        output=output,
        **params,
    )

    assert actual.data_ptr() == output.data_ptr()
    assert torch.equal(actual, expected)


@requires_triton_cpu
def test_cpu_ple_gate_matches_torch_reference() -> None:
    torch.manual_seed(7)
    num_tokens, hc_count, hidden_size = 3, 4, 64
    key_value = torch.randn(
        num_tokens,
        (hc_count + 1) * hidden_size,
        dtype=torch.bfloat16,
    )
    key = key_value[:, : hc_count * hidden_size]
    value = key_value[:, hc_count * hidden_size :]
    hidden = torch.randn(
        num_tokens,
        hc_count * hidden_size,
        dtype=torch.bfloat16,
    )
    norm_key = torch.randn(hc_count * hidden_size, dtype=torch.bfloat16)
    norm_query = torch.randn(hc_count * hidden_size, dtype=torch.bfloat16)
    norm_conv = torch.randn(hc_count * hidden_size, dtype=torch.bfloat16)

    def grouped_norm(inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        grouped = inputs.float().unflatten(-1, (hc_count, hidden_size))
        variance = grouped.square().mean(dim=-1, keepdim=True)
        normalized = grouped * torch.rsqrt(variance + 1e-6)
        return (normalized.flatten(-2) * (1.0 + weight.float())).bfloat16()

    actual_gated, actual_normed = ple_gate(
        key,
        value,
        hidden,
        norm_key,
        norm_query,
        norm_conv,
        1e-6,
    )
    normalized_key = grouped_norm(key, norm_key).unflatten(-1, (hc_count, hidden_size))
    normalized_query = grouped_norm(hidden, norm_query).unflatten(
        -1, (hc_count, hidden_size)
    )
    dot = (normalized_key * normalized_query).sum(-1, keepdim=True)
    dot = (dot / math.sqrt(hidden_size)).bfloat16()
    gate = torch.sigmoid(dot.sign() * dot.abs().clamp_min(1e-6).sqrt()).bfloat16()
    expected_gated = (gate * value.unsqueeze(-2)).flatten(-2)
    expected_normed = grouped_norm(expected_gated, norm_conv)

    torch.testing.assert_close(actual_gated, expected_gated)
    torch.testing.assert_close(actual_normed, expected_normed)


@requires_triton_cpu
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            _ConvBatchCase(num_decodes=2, channels=64),
            id="decode",
        ),
        pytest.param(
            _ConvBatchCase(prefill_query_lens=(5, 0, 7), channels=64),
            id="prefill",
        ),
    ],
)
def test_cpu_ple_short_conv_matches_shared_reference(
    case: _ConvBatchCase,
) -> None:
    device = torch.device("cpu")
    metadata, num_real_tokens = _make_conv_metadata(case, device)
    module = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(module)
    module.conv_state_len = (case.kernel_size - 1) * case.dilation
    module.short_conv_dilation = case.dilation
    rng, state_reference, conv_state, weights = _make_conv_case(
        device,
        seed=num_real_tokens + case.channels,
        channels=case.channels,
        kernel_size=case.kernel_size,
        dilation=case.dilation,
        state_layout="DS",
        spec_query_len=case.spec_query_len,
    )
    inputs = torch.randn(
        metadata.num_actual_tokens,
        case.channels,
        dtype=torch.bfloat16,
        generator=rng,
    )
    residual = torch.randn(
        inputs.shape,
        dtype=torch.bfloat16,
        generator=rng,
    )
    actual = residual.clone()
    expected = residual.clone()
    null_state = conv_state[NULL_BLOCK_ID].clone()

    module._short_conv_dilated_dispatch(
        inputs=inputs,
        residual=actual,
        metadata=metadata,
        conv_state=conv_state,
        conv_weights=weights,
    )
    _short_conv_dilated_dispatch_pytorch(
        inputs=inputs,
        residual=expected,
        metadata=metadata,
        conv_state=state_reference,
        conv_weights=weights,
        conv_state_len=module.conv_state_len,
        dilation=module.short_conv_dilation,
    )

    assert torch.equal(actual[:num_real_tokens], expected[:num_real_tokens])
    assert torch.equal(conv_state, state_reference)
    assert torch.equal(conv_state[NULL_BLOCK_ID], null_state)
    assert torch.equal(actual[num_real_tokens:], residual[num_real_tokens:])
