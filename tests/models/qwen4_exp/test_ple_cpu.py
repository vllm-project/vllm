# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
from torch import nn

from vllm.models.qwen4_exp.cpu.ngram_embedding import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLEFp8EmbeddingMethod,
)
from vllm.models.qwen4_exp.cpu.ops.ple import ple_gate
from vllm.models.qwen4_exp.cpu.ple_layer import Qwen4ExpPLELayer
from vllm.platforms import current_platform
from vllm.triton_utils import has_active_triton_cpu_backend
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from .test_ple import (
    _ConvBatchCase,
    _make_conv_case,
    _make_conv_metadata,
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


def test_cpu_ple_uses_torch_ngram_id_generation() -> None:
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(module)
    module.ngram_size = 3
    module.heads_per_ngram = 1
    module.ngram_heads = 2
    module.eos_token_id = 99
    module.register_buffer(
        "layer_multipliers",
        torch.tensor([1, 10, 100], dtype=torch.long),
    )
    module.register_buffer(
        "ngram_heads_vocab_sizes",
        torch.tensor([1000, 1000], dtype=torch.long),
    )
    module.register_buffer(
        "ngram_heads_offsets",
        torch.tensor([0, 1000], dtype=torch.long),
    )

    actual = module.compute_ngram_ids(
        input_ids=torch.tensor([5, 6], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        ngram_context=torch.tensor([[3, 4]], dtype=torch.int32),
    )

    assert actual.tolist() == [[45, 1257], [52, 1420]]


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
        pytest.param(
            _ConvBatchCase(
                spec_query_lens=(1, 3),
                num_accepted=(1, 2),
                channels=64,
                spec_query_len=3,
                graph_padding=2,
            ),
            id="spec",
        ),
        pytest.param(
            _ConvBatchCase(
                spec_query_lens=(2,),
                num_accepted=(1,),
                num_decodes=1,
                prefill_query_lens=(4,),
                channels=64,
                spec_query_len=2,
            ),
            id="mixed",
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
