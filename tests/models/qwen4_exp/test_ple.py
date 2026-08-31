# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import vllm.model_executor.layers.vocab_parallel_embedding as embedding_module
import vllm.model_executor.parameter as parameter_module
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.models.qwen4_exp.common.ple import (
    PLEShardOverlap,
    compute_ple_shard_overlap,
    copy_ple_embedding_shard_,
)
from vllm.models.qwen4_exp.nvidia.ple_layer import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLEFp8EmbeddingMethod,
    Qwen4ExpPLELayer,
    _get_ple_embedding_quant_method,
)


def _make_ngram_embedding_for_load_test() -> Qwen4ExpNGramEmbedding:
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(module)
    module.split_ngram_parts = 2
    module.register_buffer("layer_multipliers", torch.zeros(1, dtype=torch.long))
    module.register_buffer("ngram_heads_offsets", torch.zeros(1, dtype=torch.long))
    module.register_buffer("ngram_heads_vocab_sizes", torch.zeros(1, dtype=torch.long))
    module.ngram_embedding = SimpleNamespace(
        org_vocab_size=8,
        embedding_dim=2,
        weight=nn.Parameter(torch.full((4, 2), -1.0)),
        shard_indices=SimpleNamespace(
            org_vocab_start_index=2,
            org_vocab_end_index=6,
        ),
    )
    _set_test_embedding_weight_loader(module.ngram_embedding)
    return module


def _set_test_embedding_weight_loader(embedding) -> None:
    embedding.weight.weight_loader = partial(
        copy_ple_embedding_shard_,
        tp_start=embedding.shard_indices.org_vocab_start_index,
        tp_end=embedding.shard_indices.org_vocab_end_index,
    )


def _make_fp8_ngram_embedding_for_load_test() -> Qwen4ExpNGramEmbedding:
    module = _make_ngram_embedding_for_load_test()
    embedding = nn.Module()
    embedding.org_vocab_size = 8
    embedding.embedding_dim = 2
    embedding.shard_indices = SimpleNamespace(
        org_vocab_start_index=2,
        org_vocab_end_index=6,
    )
    embedding.register_parameter(
        "weight",
        nn.Parameter(
            torch.full((4, 2), -1.0).to(torch.float8_e4m3fn),
            requires_grad=False,
        ),
    )
    embedding.register_parameter(
        "weight_scale",
        nn.Parameter(torch.zeros(1, dtype=torch.bfloat16), requires_grad=False),
    )
    _set_test_embedding_weight_loader(embedding)
    module.ngram_embedding = embedding
    return module


def test_ple_shard_overlap_and_copy() -> None:
    overlap = compute_ple_shard_overlap(
        checkpoint_start=2, checkpoint_rows=5, tp_start=4, tp_end=8
    )
    assert overlap == PLEShardOverlap(source_start=2, destination_start=0, row_count=3)

    destination = torch.full((4, 2), -1.0)
    loaded = torch.arange(10, dtype=torch.float64).reshape(5, 2)
    copied = copy_ple_embedding_shard_(
        destination,
        loaded,
        checkpoint_start=2,
        tp_start=4,
        tp_end=8,
    )

    assert copied == 3
    torch.testing.assert_close(destination[:3], loaded[2:5].float())
    torch.testing.assert_close(destination[3], torch.tensor([-1.0, -1.0]))


def test_ple_shard_copy_is_a_noop_without_overlap() -> None:
    destination = torch.ones(4, 2)
    copied = copy_ple_embedding_shard_(
        destination,
        torch.zeros(2, 2),
        checkpoint_start=10,
        tp_start=4,
        tp_end=8,
    )

    assert copied == 0
    assert torch.equal(destination, torch.ones_like(destination))


def test_ngram_embedding_loads_shards_and_ignores_legacy_token_lookup() -> None:
    module = _make_ngram_embedding_for_load_test()
    shard_0 = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    shard_1 = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)

    loaded = module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", shard_0),
            ("ngram_embedding.shard_1.weight", shard_1),
            ("token_lookup", torch.tensor([2, 1, 0])),
        ]
    )

    assert loaded == {"ngram_embedding.weight"}
    torch.testing.assert_close(
        module.ngram_embedding.weight,
        torch.cat((shard_0[2:4], shard_1[0:2])),
    )


def test_ngram_embedding_loads_fp8_shards_and_global_scale() -> None:
    module = _make_fp8_ngram_embedding_for_load_test()
    shard_0 = torch.arange(8, dtype=torch.float32).reshape(4, 2).to(torch.float8_e4m3fn)
    shard_1 = (
        torch.arange(8, 16, dtype=torch.float32).reshape(4, 2).to(torch.float8_e4m3fn)
    )
    weight_scale = torch.tensor([0.25], dtype=torch.bfloat16)

    loaded = module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", shard_0),
            ("ngram_embedding.shard_1.weight", shard_1),
            ("ngram_embedding.weight_scale", weight_scale),
        ]
    )

    assert loaded == {"ngram_embedding.weight", "ngram_embedding.weight_scale"}
    assert module.ngram_embedding.weight.dtype == torch.float8_e4m3fn
    assert torch.equal(
        module.ngram_embedding.weight.float(),
        torch.cat((shard_0[2:4], shard_1[0:2])).float(),
    )
    assert torch.equal(module.ngram_embedding.weight_scale, weight_scale)


def _make_fp8_embedding_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> embedding_module.VocabParallelEmbedding:
    monkeypatch.setattr(embedding_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        embedding_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    method = Qwen4ExpPLEFp8EmbeddingMethod()
    layer = embedding_module.VocabParallelEmbedding(
        3,
        2,
        params_dtype=torch.bfloat16,
        padding_size=1,
        quant_method=method,
    )
    weight = torch.tensor([[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]])
    layer.weight.data.copy_(weight.to(torch.float8_e4m3fn))
    layer.weight_scale.data.copy_(torch.tensor([0.25], dtype=torch.bfloat16))
    return layer


def test_ple_fp8_embedding_dequantizes_in_ple_layer(monkeypatch) -> None:
    layer = _make_fp8_embedding_layer(monkeypatch)
    quantized_output = layer(torch.tensor([2, 0]))
    ple_layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(ple_layer)
    ple_layer.ple_embedding = nn.Module()
    ple_layer.ple_embedding.ngram_embedding = layer

    output = ple_layer._dequantize_embeddings(
        quantized_output,
        torch.bfloat16,
    )

    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.dtype == torch.bfloat16
    assert quantized_output.dtype == torch.float8_e4m3fn
    assert output.dtype == torch.bfloat16
    weight = torch.tensor([[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]])
    torch.testing.assert_close(output, (weight[[2, 0]] * 0.25).bfloat16())


def test_ple_fp8_embedding_uses_int8_for_tp_reduce(monkeypatch) -> None:
    monkeypatch.setattr(embedding_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        embedding_module, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(
        embedding_module,
        "get_masked_input_and_mask",
        lambda *args: (
            torch.tensor([0, 0]),
            torch.tensor([False, True]),
        ),
    )
    reduced_dtypes = []

    def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        reduced_dtypes.append(tensor.dtype)
        return tensor.clone()

    monkeypatch.setattr(
        embedding_module,
        "tensor_model_parallel_all_reduce",
        all_reduce,
    )
    layer = embedding_module.VocabParallelEmbedding(
        4,
        2,
        params_dtype=torch.bfloat16,
        padding_size=1,
        quant_method=Qwen4ExpPLEFp8EmbeddingMethod(),
    )
    layer.weight.data.copy_(
        torch.tensor([[1.0, 2.0], [4.0, 8.0]]).to(torch.float8_e4m3fn)
    )

    output = layer(torch.tensor([0, 2]))

    assert reduced_dtypes == [torch.int8]
    assert output.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(output[0].float(), layer.weight[0].float())
    assert torch.count_nonzero(output[1].float()) == 0


def test_ple_fp8_embedding_respects_checkpoint_shard_exclusions() -> None:
    prefix = "model.layers.1.ple.ple_embedding.ngram_embedding"
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        ignored_layers=[],
        weight_block_size=[128, 128],
    )
    assert isinstance(
        _get_ple_embedding_quant_method(quant_config, prefix),
        Qwen4ExpPLEFp8EmbeddingMethod,
    )

    quant_config.ignored_layers = [f"{prefix}.shard_0"]
    assert _get_ple_embedding_quant_method(quant_config, prefix) is None


def test_dilated_ple_spec_state_rolls_back_before_next_forward() -> None:
    module = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(module)
    module.conv_state_len = 6
    module.short_conv_dilation = 2

    conv_weights = torch.tensor([[0.25, -0.5, 0.75, 1.0]])
    conv_state = torch.zeros(2, 1, 9)
    conv_state[1] = torch.arange(1, 10, dtype=torch.float32).reshape(1, 9)
    first_inputs = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    initial_state = conv_state[1:].clone()
    first_history = torch.cat(
        (initial_state[..., : module.conv_state_len], first_inputs.T.unsqueeze(0)),
        dim=-1,
    )

    graph_padded_inputs = F.pad(first_inputs, (0, 0, 0, 4))
    first_output = module._short_conv_dilated_spec_batched(
        graph_padded_inputs,
        conv_state,
        conv_weights,
        torch.tensor([1, 0]),
        torch.tensor([0, 4, 4]),
        torch.tensor([1, 0]),
        spec_query_len=4,
    )

    expected_first_output = F.silu(
        F.conv1d(
            first_history,
            conv_weights.unsqueeze(1),
            groups=1,
            dilation=module.short_conv_dilation,
        )
    ).transpose(1, 2)[0]
    expected_first_state = first_history[..., 1:10]
    torch.testing.assert_close(first_output[:4], expected_first_output)
    assert torch.count_nonzero(first_output[4:]) == 0
    assert torch.count_nonzero(conv_state[0]) == 0
    torch.testing.assert_close(conv_state[1:], expected_first_state)

    second_inputs = torch.tensor([[50.0], [60.0]])
    rollback_state = expected_first_state[..., 1:7]
    padded_second_inputs = F.pad(second_inputs.T.unsqueeze(0), (0, 2))
    second_history = torch.cat((rollback_state, padded_second_inputs), dim=-1)
    expected_second_state = expected_first_state.clone()
    expected_second_state[..., :7] = second_history[..., 1:8]

    second_output = module._short_conv_dilated_spec_batched(
        second_inputs,
        conv_state,
        conv_weights,
        torch.tensor([1]),
        torch.tensor([0, 2]),
        torch.tensor([2]),
        spec_query_len=4,
    )

    expected_second_output = F.silu(
        F.conv1d(
            second_history,
            conv_weights.unsqueeze(1),
            groups=1,
            dilation=module.short_conv_dilation,
        )
    ).transpose(1, 2)[0, :2]
    torch.testing.assert_close(second_output, expected_second_output)
    torch.testing.assert_close(conv_state[1:], expected_second_state)


def _reference_ngram_ids(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    layer_multipliers: torch.Tensor,
    ngram_heads_vocab_sizes: torch.Tensor,
    ngram_heads_offsets: torch.Tensor,
    eos_token_id: int,
    heads_per_ngram: int,
) -> torch.Tensor:
    tokens = input_ids.tolist()
    starts = query_start_loc.tolist()
    contexts = ngram_context.tolist()
    multipliers = layer_multipliers.tolist()
    sizes = ngram_heads_vocab_sizes.tolist()
    offsets = ngram_heads_offsets.tolist()
    rows = []
    context_len = len(contexts[0])
    for req, (start, end) in enumerate(zip(starts, starts[1:])):
        history = contexts[req] + tokens[start:end]
        for pos in range(context_len, len(history)):
            shifted = [history[pos]]
            crossed_eos = False
            for shift in range(1, context_len + 1):
                token = eos_token_id if crossed_eos else history[pos - shift]
                shifted.append(token)
                crossed_eos |= token == eos_token_id

            row = []
            mixed = shifted[0] * multipliers[0]
            for ngram_order in range(2, context_len + 2):
                shift = ngram_order - 1
                mixed ^= shifted[shift] * multipliers[shift]
                head_start = (ngram_order - 2) * heads_per_ngram
                for head in range(head_start, head_start + heads_per_ngram):
                    row.append(mixed % sizes[head] + offsets[head])
            rows.append(row)
    return torch.tensor(rows, dtype=torch.int64, device=input_ids.device)


_NGRAM_MULTIPLIERS = (
    18_014_398_509_481_983,
    17_114_398_509_481_981,
    16_214_398_509_481_979,
    15_314_398_509_481_977,
)
_NGRAM_HEADS_VOCAB_SIZES = (
    101,
    103,
    107,
    109,
    113,
    127,
    131,
    137,
    139,
    149,
    151,
    157,
    163,
    167,
    173,
    179,
    181,
    191,
    193,
    197,
    199,
    211,
    223,
    227,
)
_NGRAM_EOS_TOKEN_ID = 251
_NGRAM_HEADS_PER_NGRAM = 8


def _ngram_hash_params(device: torch.device, context_len: int) -> dict:
    num_heads = context_len * _NGRAM_HEADS_PER_NGRAM
    sizes = torch.tensor(
        _NGRAM_HEADS_VOCAB_SIZES[:num_heads], dtype=torch.int64, device=device
    )
    offsets = torch.zeros_like(sizes)
    offsets[1:] = torch.cumsum(sizes, dim=0)[:-1]
    return {
        "layer_multipliers": torch.tensor(
            _NGRAM_MULTIPLIERS[: context_len + 1],
            dtype=torch.int64,
            device=device,
        ),
        "ngram_heads_vocab_sizes": sizes,
        "ngram_heads_offsets": offsets,
        "eos_token_id": _NGRAM_EOS_TOKEN_ID,
        "heads_per_ngram": _NGRAM_HEADS_PER_NGRAM,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused PLE needs CUDA")
@pytest.mark.parametrize(
    ("query_lens", "eos_offsets", "contexts"),
    [
        ([1], [], [[11, 12]]),
        ([3], [1], [[11]]),
        ([4, 4], [0, 7], [[11, 12], [13, 14]]),
        ([4, 0, 3], [], [[11, 12], [13, 14], [15, 16]]),
        (
            [1, 33, 2],
            [5, 32],
            [[_NGRAM_EOS_TOKEN_ID, 11], [12, 13], [14, _NGRAM_EOS_TOKEN_ID]],
        ),
        (
            [5, 12, 16, 1, 16, 17],
            [10, 40],
            [[11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22]],
        ),
        ([5, 3], [3], [[11, 12, 13], [14, 15, 16]]),
    ],
    ids=[
        "single-token",
        "bigram-only",
        "power-of-two",
        "empty-request",
        "three-requests",
        "six-requests",
        "four-gram",
    ],
)
def test_fused_ngram_ids_correctness(
    query_lens: list[int],
    eos_offsets: list[int],
    contexts: list[list[int]],
) -> None:
    from vllm.models.qwen4_exp.nvidia.ops.ple_ngram import ple_ngram_ids

    device = torch.device("cuda")
    query_start_loc = torch.tensor(
        [0, *accumulate(query_lens)], dtype=torch.int32, device=device
    )
    input_ids = torch.arange(
        20,
        20 + sum(query_lens),
        dtype=torch.int32,
        device=device,
    )
    if eos_offsets:
        input_ids[eos_offsets] = _NGRAM_EOS_TOKEN_ID
    ngram_context = torch.tensor(contexts, dtype=torch.int32, device=device)
    params = _ngram_hash_params(device, ngram_context.shape[1])

    expected = _reference_ngram_ids(input_ids, query_start_loc, ngram_context, **params)
    actual = ple_ngram_ids(input_ids, query_start_loc, ngram_context, **params)

    assert torch.equal(actual, expected)


_KERNEL_STATE_BLOCKS = 64


def _make_conv_case(
    device: torch.device,
    seed: int,
    channels: int,
    kernel_size: int,
    dilation: int,
    spec_query_len: int = 1,
) -> tuple[torch.Generator, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    state_len = (kernel_size - 1) * dilation
    state_width = state_len + spec_query_len - 1
    state = torch.randn(
        _KERNEL_STATE_BLOCKS,
        state_width,
        channels,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    weights = torch.randn(
        channels,
        kernel_size,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    return generator, state, weights


def _make_conv_states(
    state_sd: torch.Tensor,
    state_layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state_reference_ds = state_sd.clone().transpose(-1, -2)
    if state_layout == "SD":
        state_kernel = state_sd.clone()
        state_kernel_ds = state_kernel.transpose(-1, -2)
    else:
        state_kernel = state_sd.clone().transpose(-1, -2).contiguous()
        state_kernel_ds = state_kernel
    return state_reference_ds, state_kernel, state_kernel_ds


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused conv needs CUDA")
@pytest.mark.parametrize("state_layout", ["SD", "DS"])
@pytest.mark.parametrize(
    ("num_tokens", "channels", "kernel_size", "dilation"),
    [(1, 512, 4, 3), (4, 640, 3, 2), (33, 2048, 5, 1)],
)
def test_fused_conv_decode_correctness(
    num_tokens: int,
    channels: int,
    kernel_size: int,
    dilation: int,
    state_layout: str,
) -> None:
    from vllm.models.qwen4_exp.nvidia.ops.ple_conv import ple_conv

    device = torch.device("cuda")
    module = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(module)
    module.conv_state_len = (kernel_size - 1) * dilation
    module.short_conv_dilation = dilation
    generator, state, weights = _make_conv_case(
        device, num_tokens, channels, kernel_size, dilation
    )
    inputs = torch.randn(
        num_tokens,
        channels,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    residual = torch.randn(
        num_tokens,
        channels,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    state_indices = torch.arange(1, num_tokens + 1, dtype=torch.int32, device=device)
    if num_tokens > 1:
        state_indices[-1] = 0
    has_initial_states = torch.arange(num_tokens, device=device) % 2 == 0

    state_reference, state_kernel, state_kernel_ds = _make_conv_states(
        state, state_layout
    )
    null_state = state_kernel_ds[0].clone()
    conv_reference = module._short_conv_dilated_decode_batched(
        inputs,
        state_reference,
        weights,
        state_indices,
        has_initial_states,
    )
    output_reference = residual + conv_reference
    residual_kernel = residual.clone()
    ple_conv(
        inputs,
        residual_kernel,
        state_kernel,
        weights,
        state_indices,
        mode="decode",
        dilation=dilation,
        has_initial_states=has_initial_states,
    )
    torch.testing.assert_close(
        residual_kernel.float(), output_reference.float(), atol=3e-2, rtol=3e-2
    )
    assert torch.equal(state_kernel_ds, state_reference)
    assert torch.equal(state_kernel_ds[0], null_state)
    assert torch.equal(
        residual_kernel[state_indices == 0], residual[state_indices == 0]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused conv needs CUDA")
@pytest.mark.parametrize("state_layout", ["SD", "DS"])
@pytest.mark.parametrize(
    (
        "query_lens",
        "num_accepted",
        "spec_query_len",
        "channels",
        "kernel_size",
        "dilation",
    ),
    [
        ([1], [2], 2, 640, 3, 2),
        ([1, 4, 4], [1, 2, 4], 4, 2048, 4, 3),
        ([5, 5, 5, 2, 1, 3], [4, 3, 1, 2, 0, 1], 5, 512, 5, 1),
    ],
)
def test_fused_conv_spec_correctness(
    query_lens: list[int],
    num_accepted: list[int],
    spec_query_len: int,
    channels: int,
    kernel_size: int,
    dilation: int,
    state_layout: str,
) -> None:
    from vllm.models.qwen4_exp.nvidia.ops.ple_conv import ple_conv

    device = torch.device("cuda")
    module = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(module)
    module.conv_state_len = (kernel_size - 1) * dilation
    module.short_conv_dilation = dilation
    num_reqs = len(query_lens)
    generator, state, weights = _make_conv_case(
        device,
        num_reqs + 100,
        channels,
        kernel_size,
        dilation,
        spec_query_len,
    )
    query_start_loc = torch.tensor(
        [0, *accumulate(query_lens)], dtype=torch.int32, device=device
    )
    num_real_tokens = sum(query_lens)
    num_tokens = num_real_tokens + spec_query_len
    inputs = torch.randn(
        num_tokens,
        channels,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    residual = torch.randn(
        num_tokens,
        channels,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    state_indices = torch.arange(1, num_reqs + 1, dtype=torch.int32, device=device)
    if num_reqs > 1:
        state_indices[1] = 0
    num_accepted_tensor = torch.tensor(num_accepted, dtype=torch.int32, device=device)

    state_reference, state_kernel, state_kernel_ds = _make_conv_states(
        state, state_layout
    )
    null_state = state_kernel_ds[0].clone()
    conv_reference = module._short_conv_dilated_spec_batched(
        inputs,
        state_reference,
        weights,
        state_indices,
        query_start_loc,
        num_accepted_tensor,
        spec_query_len,
    )
    output_reference = residual + conv_reference
    residual_kernel = residual.clone()
    ple_conv(
        inputs,
        residual_kernel,
        state_kernel,
        weights,
        state_indices,
        mode="spec",
        dilation=dilation,
        query_start_loc=query_start_loc,
        num_accepted_tokens=num_accepted_tensor,
        spec_query_len=spec_query_len,
    )
    torch.testing.assert_close(
        residual_kernel.float(), output_reference.float(), atol=3e-2, rtol=3e-2
    )
    assert torch.equal(state_kernel_ds, state_reference)
    assert torch.equal(state_kernel_ds[0], null_state)
    assert torch.equal(residual_kernel[num_real_tokens:], residual[num_real_tokens:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused conv needs CUDA")
@pytest.mark.parametrize("state_layout", ["SD", "DS"])
@pytest.mark.parametrize(
    ("query_lens", "channels", "kernel_size", "dilation"),
    [([37, 0, 5, 128], 2048, 4, 3), ([1, 8, 3], 640, 3, 2)],
)
def test_fused_conv_prefill_correctness(
    query_lens: list[int],
    channels: int,
    kernel_size: int,
    dilation: int,
    state_layout: str,
) -> None:
    from vllm.models.qwen4_exp.nvidia.ops.ple_conv import ple_conv

    device = torch.device("cuda")
    module = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(module)
    module.conv_state_len = (kernel_size - 1) * dilation
    module.short_conv_dilation = dilation
    generator, state, weights = _make_conv_case(
        device, sum(query_lens), channels, kernel_size, dilation
    )
    query_start_loc = torch.tensor(
        [0, *accumulate(query_lens)], dtype=torch.int32, device=device
    )
    inputs = torch.randn(
        sum(query_lens),
        channels,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    residual = torch.randn(
        sum(query_lens),
        channels,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    num_reqs = len(query_lens)
    state_indices = torch.arange(1, num_reqs + 1, dtype=torch.int32, device=device)
    state_indices[-2] = 0
    has_initial_states = torch.tensor(
        [i % 2 == 0 for i in range(num_reqs)], device=device
    )
    metadata = SimpleNamespace(
        non_spec_query_start_loc=query_start_loc,
        has_initial_states_p=has_initial_states,
        max_prefill_query_len=max(query_lens),
    )

    state_reference, state_kernel, state_kernel_ds = _make_conv_states(
        state, state_layout
    )
    null_state = state_kernel_ds[0].clone()
    conv_reference = module._short_conv_dilated_prefill_batched(
        inputs,
        metadata,
        state_reference,
        weights,
        state_indices,
        num_reqs,
        0,
        inputs.shape[0],
    )
    output_reference = residual + conv_reference
    residual_kernel = residual.clone()
    ple_conv(
        inputs,
        residual_kernel,
        state_kernel,
        weights,
        state_indices,
        mode="prefill",
        dilation=dilation,
        query_start_loc=query_start_loc,
        has_initial_states=has_initial_states,
    )
    torch.testing.assert_close(
        residual_kernel.float(), output_reference.float(), atol=3e-2, rtol=3e-2
    )
    assert torch.equal(state_kernel_ds, state_reference)
    assert torch.equal(state_kernel_ds[0], null_state)
    query_start = 0
    for state_index, query_len in zip(state_indices.tolist(), query_lens):
        if state_index == 0:
            output_slice = residual_kernel[query_start : query_start + query_len]
            residual_slice = residual[query_start : query_start + query_len]
            assert torch.equal(
                output_slice,
                residual_slice,
            )
        query_start += query_len


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused gate needs CUDA")
@pytest.mark.parametrize(("num_tokens", "strided_kv"), [(1, False), (64, True)])
def test_fused_gate_correctness(num_tokens: int, strided_kv: bool) -> None:
    import math

    from vllm.models.qwen4_exp.nvidia.ops.ple_gate import ple_gate

    device = torch.device("cuda")
    hc, h = 4, 2560
    generator = torch.Generator(device=device).manual_seed(num_tokens)
    kv = torch.randn(
        num_tokens,
        hc * h + h,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    key = kv[:, : hc * h]
    value = kv[:, hc * h :]
    if not strided_kv:
        key = key.contiguous()
        value = value.contiguous()
    hidden = torch.randn(
        num_tokens,
        hc * h,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    norm_key = torch.randn(
        hc * h,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.05)
    norm_query = torch.randn(
        hc * h,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.05)
    norm_conv = torch.randn(
        hc * h,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.05)

    def grouped_norm(inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        grouped = inputs.float().unflatten(-1, (hc, h))
        var = grouped.square().mean(dim=-1, keepdim=True)
        normalized = grouped * torch.rsqrt(var + 1e-6)
        return (normalized.flatten(-2) * (1.0 + weight.float())).to(inputs.dtype)

    gated, normed = ple_gate(
        key,
        value,
        hidden,
        norm_key,
        norm_query,
        norm_conv,
        1e-6,
    )
    key_normalized = grouped_norm(key, norm_key).reshape(num_tokens, hc, h)
    query_normalized = grouped_norm(hidden, norm_query).reshape(num_tokens, hc, h)
    dot = (key_normalized * query_normalized).sum(dim=-1, keepdim=True)
    dot = (dot / math.sqrt(h)).to(torch.bfloat16)
    gate = torch.sigmoid(dot.sign() * dot.abs().clamp_min(1e-6).sqrt()).to(
        torch.bfloat16
    )
    expected_gated = (gate * value.unsqueeze(-2)).flatten(-2)
    expected_normed = grouped_norm(expected_gated, norm_conv)
    torch.testing.assert_close(gated, expected_gated, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(normed, expected_normed, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused conv needs CUDA")
def test_fused_conv_mixed_batch_correctness() -> None:
    from vllm.models.qwen4_exp.nvidia.ops.ple_conv import BLOCK_C

    device = torch.device("cuda")
    channels = 4 * BLOCK_C
    module = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(module)
    module.conv_state_len = 9
    module.short_conv_dilation = 3

    generator = torch.Generator(device=device).manual_seed(17)
    inputs = torch.randn(
        13,
        channels,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    state = torch.randn(
        64,
        12,
        channels,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    weights = torch.randn(
        channels,
        4,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    residual = torch.randn(
        inputs.shape,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    metadata = SimpleNamespace(
        num_prefills=1,
        num_decodes=1,
        num_decode_tokens=1,
        num_prefill_tokens=5,
        spec_sequence_masks=torch.tensor([True, True], device=device),
        spec_token_indx=torch.arange(7, dtype=torch.int32, device=device),
        non_spec_token_indx=torch.arange(7, 13, dtype=torch.int32, device=device),
        num_actual_tokens=13,
        spec_state_indices_tensor=torch.tensor(
            [11, 12], dtype=torch.int32, device=device
        ),
        spec_query_start_loc=torch.tensor([0, 3, 7], dtype=torch.int32, device=device),
        num_accepted_tokens=torch.tensor([2, 4], dtype=torch.int32, device=device),
        spec_query_len=4,
        state_indices_tensor=torch.tensor([13, 14], dtype=torch.int32, device=device),
        has_initial_states_d=torch.tensor([True], device=device),
        non_spec_query_start_loc=torch.tensor(
            [0, 1, 6], dtype=torch.int32, device=device
        ),
        has_initial_states_p=torch.tensor([True], device=device),
        max_prefill_query_len=5,
        num_spec_decodes=2,
    )
    state_kernel = state.clone()
    state_eager = state.clone().transpose(-1, -2)
    residual_kernel = residual.clone()
    residual_eager = residual.clone()
    module._short_conv_dilated_dispatch(
        inputs=inputs,
        residual=residual_kernel,
        metadata=metadata,
        conv_state=state_kernel,
        conv_weights=weights,
        use_fused_kernels=True,
    )
    module._short_conv_dilated_dispatch(
        inputs=inputs,
        residual=residual_eager,
        metadata=metadata,
        conv_state=state_eager,
        conv_weights=weights,
        use_fused_kernels=False,
    )
    torch.testing.assert_close(
        residual_kernel.float(), residual_eager.float(), atol=3e-2, rtol=3e-2
    )
    assert torch.equal(state_kernel, state_eager.transpose(-1, -2).contiguous())
