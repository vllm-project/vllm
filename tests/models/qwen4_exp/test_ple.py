# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
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
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionMetadata,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


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
        nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False),
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
    assert module.ngram_embedding.weight_scale.dtype == torch.float32
    torch.testing.assert_close(
        module.ngram_embedding.weight_scale,
        weight_scale.float(),
    )


def _make_fp8_embedding_layer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    load_scale: bool = True,
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
    if load_scale:
        layer.weight_scale.data.copy_(torch.tensor([0.25], dtype=torch.bfloat16))
    return layer


def test_ple_fp8_embedding_dequantizes_in_ple_layer(monkeypatch) -> None:
    layer = _make_fp8_embedding_layer(monkeypatch)
    layer.quant_method.process_weights_after_loading(layer)
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
    assert layer.weight_scale.dtype == torch.float32
    assert quantized_output.dtype == torch.float8_e4m3fn
    assert output.dtype == torch.bfloat16
    weight = torch.tensor([[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]])
    torch.testing.assert_close(output, (weight[[2, 0]] * 0.25).bfloat16())


def test_ple_fp8_embedding_rejects_missing_global_scale(monkeypatch) -> None:
    layer = _make_fp8_embedding_layer(monkeypatch, load_scale=False)

    with pytest.raises(ValueError, match="missing its global scale"):
        layer.quant_method.process_weights_after_loading(layer)


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
    conv_state_len = 6
    dilation = 2

    conv_weights = torch.tensor([[0.25, -0.5, 0.75, 1.0]])
    conv_state = torch.zeros(2, 1, 9)
    conv_state[1] = torch.arange(1, 10, dtype=torch.float32).reshape(1, 9)
    first_inputs = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    initial_state = conv_state[1:].clone()
    first_history = torch.cat(
        (initial_state[..., :conv_state_len], first_inputs.T.unsqueeze(0)),
        dim=-1,
    )

    graph_padded_inputs = F.pad(first_inputs, (0, 0, 0, 4))
    first_output = _short_conv_dilated_spec_pytorch(
        graph_padded_inputs,
        conv_state,
        conv_weights,
        torch.tensor([1, 0]),
        torch.tensor([0, 4, 4]),
        torch.tensor([1, 0]),
        spec_query_len=4,
        conv_state_len=conv_state_len,
        dilation=dilation,
    )

    expected_first_output = F.silu(
        F.conv1d(
            first_history,
            conv_weights.unsqueeze(1),
            groups=1,
            dilation=dilation,
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

    second_output = _short_conv_dilated_spec_pytorch(
        second_inputs,
        conv_state,
        conv_weights,
        torch.tensor([1]),
        torch.tensor([0, 2]),
        torch.tensor([2]),
        spec_query_len=4,
        conv_state_len=conv_state_len,
        dilation=dilation,
    )

    expected_second_output = F.silu(
        F.conv1d(
            second_history,
            conv_weights.unsqueeze(1),
            groups=1,
            dilation=dilation,
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
    multipliers = layer_multipliers.cpu()
    sizes = ngram_heads_vocab_sizes.cpu()
    offsets = ngram_heads_offsets.cpu()
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
            mixed = torch.tensor(shifted[0], dtype=torch.int64) * multipliers[0]
            for ngram_order in range(2, context_len + 2):
                shift = ngram_order - 1
                mixed ^= shifted[shift] * multipliers[shift]
                head_start = (ngram_order - 2) * heads_per_ngram
                for head in range(head_start, head_start + heads_per_ngram):
                    row.append(torch.remainder(mixed, sizes[head]) + offsets[head])
            rows.append(torch.stack(row))
    return torch.stack(rows).to(input_ids.device)


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
    ("query_lens", "eos_offsets", "contexts", "first_token_id"),
    [
        ([1], [], [[11, 12]], 20),
        ([3], [1], [[11]], 20),
        ([4, 4], [0, 7], [[11, 12], [13, 14]], 20),
        ([4, 0, 3], [], [[11, 12], [13, 14], [15, 16]], 20),
        (
            [1, 33, 2],
            [5, 32],
            [[_NGRAM_EOS_TOKEN_ID, 11], [12, 13], [14, _NGRAM_EOS_TOKEN_ID]],
            20,
        ),
        (
            [5, 12, 16, 1, 16, 17],
            [10, 40],
            [[11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22]],
            20,
        ),
        ([5, 3], [3], [[11, 12, 13], [14, 15, 16]], 20),
        (
            [4, 0, 3],
            [],
            [
                [200_000, 200_001],
                [250_000, 250_001],
                [300_000, 300_001],
            ],
            350_000,
        ),
        ([3], [], [[1_000_000_000, 1_000_000_001]], 1_000_000_002),
    ],
    ids=[
        "single-token",
        "bigram-only",
        "power-of-two",
        "empty-request",
        "three-requests",
        "six-requests",
        "four-gram",
        "int64-overflow",
        "large-int32-ids",
    ],
)
def test_fused_ngram_ids_correctness(
    query_lens: list[int],
    eos_offsets: list[int],
    contexts: list[list[int]],
    first_token_id: int,
) -> None:
    from vllm.models.qwen4_exp.nvidia.ops.ple import ple_ngram_ids

    device = torch.device("cuda")
    query_start_loc = torch.tensor(
        [0, *accumulate(query_lens)], dtype=torch.int32, device=device
    )
    input_ids = torch.arange(
        first_token_id,
        first_token_id + sum(query_lens),
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
    offsets = params["ngram_heads_offsets"]
    sizes = params["ngram_heads_vocab_sizes"]
    assert torch.all((actual >= offsets) & (actual < offsets + sizes))


def _short_conv_dilated_decode_pytorch(
    x_d: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices_tensor_d: torch.Tensor,
    has_initial_states_d: torch.Tensor | None,
    *,
    conv_state_len: int,
    dilation: int,
) -> torch.Tensor:
    state_indices = state_indices_tensor_d.to(
        device=conv_state.device, dtype=torch.int64
    )
    # FULL cudagraph padded decode rows use NULL_BLOCK_ID. Remap them to
    # slot 0 for a safe gather, then zero output and skip write-back.
    valid_state = state_indices != NULL_BLOCK_ID
    state_indices = torch.where(
        valid_state, state_indices, torch.zeros_like(state_indices)
    )
    if has_initial_states_d is None:
        has_initial_state = valid_state
    else:
        if has_initial_states_d.numel() < state_indices_tensor_d.numel():
            raise ValueError(
                "has_initial_states_d size mismatch: "
                f"got {has_initial_states_d.numel()}, "
                f"need >= {state_indices_tensor_d.numel()}."
            )
        has_initial_state = has_initial_states_d[: state_indices_tensor_d.numel()].to(
            device=conv_state.device, dtype=torch.bool
        )
        has_initial_state = has_initial_state & valid_state

    cached_state = conv_state.index_select(0, state_indices)
    state = cached_state[..., :conv_state_len].to(x_d.dtype)
    if conv_state_len > 0:
        initial_state = torch.where(
            has_initial_state.view(-1, 1, 1),
            state,
            torch.zeros_like(state),
        )
        history = torch.cat((initial_state, x_d.unsqueeze(-1)), dim=-1)
    else:
        history = x_d.unsqueeze(-1)

    conv_output = F.conv1d(
        history,
        conv_weights.unsqueeze(1).contiguous(),
        groups=history.size(1),
        dilation=dilation,
    ).squeeze(-1)
    output = F.silu(conv_output)
    output = output * valid_state.view(-1, 1).to(output.dtype)

    if conv_state_len > 0:
        next_state = history[..., -conv_state_len:]
        # Padded rows are remapped to the reserved null slot. Preserve its
        # existing value while writing the new states for valid rows.
        existing_base_state = cached_state[..., :conv_state_len]
        safe_next_state = torch.where(
            valid_state.view(-1, 1, 1),
            next_state.to(conv_state.dtype),
            existing_base_state,
        )
        cached_state[..., :conv_state_len] = safe_next_state
        conv_state.index_copy_(0, state_indices, cached_state)

    return output


def _short_conv_dilated_prefill_pytorch(
    x_p: torch.Tensor,
    metadata: PleShortConvAttentionMetadata,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices_tensor_p: torch.Tensor,
    num_prefills: int,
    num_decode_tokens: int,
    num_prefill_tokens: int,
    *,
    conv_state_len: int,
    dilation: int,
) -> torch.Tensor:
    # ``non_spec_query_start_loc`` covers the non-spec (decode + prefill)
    # requests and equals ``query_start_loc`` when spec-decode is inactive.
    non_spec_query_start_loc = metadata.non_spec_query_start_loc
    if non_spec_query_start_loc is None:
        raise ValueError("query_start_loc is required for prefill short-conv")
    query_start_loc_p = (
        non_spec_query_start_loc[-num_prefills - 1 :] - num_decode_tokens
    )
    # The metadata builder guarantees that the prefill query offsets start
    # at 0 and end at num_prefill_tokens. Avoid reading those values here,
    # since doing so would force a device-to-host synchronization.
    has_initial_states_p = metadata.has_initial_states_p
    if has_initial_states_p is None:
        raise ValueError("has_initial_states_p is required for prefill short-conv")

    output = torch.empty_like(x_p)
    q_starts = query_start_loc_p.to(torch.int64)
    if state_indices_tensor_p.numel() < num_prefills:
        raise ValueError(
            "state_indices_tensor_p size mismatch: "
            f"got {state_indices_tensor_p.numel()}, "
            f"need >= {num_prefills}."
        )
    if has_initial_states_p.numel() < num_prefills:
        raise ValueError(
            "has_initial_states_p size mismatch: "
            f"got {has_initial_states_p.numel()}, "
            f"need >= {num_prefills}."
        )
    if num_prefills == 0 or x_p.numel() == 0:
        return output
    lengths = q_starts[1:] - q_starts[:-1]
    # Use the CPU-computed packing width from the metadata builder instead
    # of synchronizing on lengths.max().
    max_len = metadata.max_prefill_query_len
    if max_len <= 0:
        return output

    hidden_size = x_p.shape[1]
    positions = torch.arange(num_prefill_tokens, device=x_p.device, dtype=torch.int64)
    req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
    col_indices = positions - q_starts[req_indices]

    packed_tokens = x_p.new_zeros((num_prefills, max_len, hidden_size))
    packed_tokens[req_indices, col_indices] = x_p
    packed_tokens = packed_tokens.transpose(1, 2).contiguous()

    state_indices = state_indices_tensor_p[:num_prefills].to(
        device=conv_state.device, dtype=torch.int64
    )
    valid_state = state_indices != NULL_BLOCK_ID
    state_indices = torch.where(
        valid_state, state_indices, torch.zeros_like(state_indices)
    )
    has_initial = has_initial_states_p[:num_prefills].to(
        device=conv_state.device, dtype=torch.bool
    )
    if conv_state_len > 0:
        if conv_state.shape[0] == 0:
            state = conv_state.new_zeros(
                (num_prefills, hidden_size, conv_state_len),
                dtype=x_p.dtype,
            )
        else:
            state = conv_state.index_select(0, state_indices)[..., :conv_state_len].to(
                x_p.dtype
            )
        use_initial_mask = (valid_state & has_initial).view(num_prefills, 1, 1)
        initial_state = torch.where(
            use_initial_mask,
            state,
            torch.zeros_like(state),
        )
        history = torch.cat((initial_state, packed_tokens), dim=-1)
    else:
        history = packed_tokens

    conv_output = F.conv1d(
        history,
        conv_weights.unsqueeze(1).contiguous(),
        groups=history.size(1),
        dilation=dilation,
    )
    conv_output = F.silu(conv_output).transpose(1, 2).contiguous()

    token_positions = torch.arange(max_len, device=x_p.device, dtype=torch.int64)
    valid_tokens = token_positions.view(1, max_len) < lengths.view(num_prefills, 1)
    valid_output_mask = valid_tokens & valid_state.to(device=x_p.device).view(
        num_prefills, 1
    )
    conv_output.masked_fill_(~valid_output_mask.unsqueeze(-1), 0)
    output.copy_(conv_output[req_indices, col_indices])

    if conv_state_len > 0 and conv_state.shape[0] > 0:
        state_starts = lengths.to(device=history.device, dtype=torch.int64).view(
            num_prefills, 1, 1
        )
        state_offsets = torch.arange(
            conv_state_len, device=history.device, dtype=torch.int64
        ).view(1, 1, conv_state_len)
        next_state = history.gather(
            dim=2,
            index=(state_starts + state_offsets).expand(-1, history.size(1), -1),
        )
        # Write back without a host synchronization. Valid, non-empty rows
        # receive their new state; padding and zero-length rows keep the
        # current cache value.
        existing_state = conv_state.index_select(0, state_indices)
        existing_base_state = existing_state[..., :conv_state_len]
        update_mask = valid_state & (lengths.to(device=conv_state.device) > 0)
        safe_next_state = torch.where(
            update_mask.view(num_prefills, 1, 1),
            next_state.to(conv_state.dtype),
            existing_base_state,
        )
        existing_state[..., :conv_state_len] = safe_next_state
        conv_state.index_copy_(0, state_indices, existing_state)
    return output


def _short_conv_dilated_spec_pytorch(
    x_spec: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    spec_state_indices_tensor: torch.Tensor,
    spec_query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    spec_query_len: int,
    *,
    conv_state_len: int,
    dilation: int,
) -> torch.Tensor:
    """Dilated short-conv for speculative-decode (MTP) requests.

    Each spec request feeds multiple (draft + 1) query tokens. The conv
    outputs are computed causally after rolling back the previous draft
    state by ``num_accepted_tokens - 1``. The current candidate inputs stay
    in the extended cache for the next forward, matching
    ``causal_conv1d_update``.

    ``spec_query_len`` (== num_speculative_tokens + 1) is the maximum query
    length and is a Python int, so no host synchronization is needed; this
    keeps the path safe for full CUDA-graph capture/replay where the buffers
    are padded at the request level.
    """
    num_reqs = spec_state_indices_tensor.numel()
    hidden_size = x_spec.size(-1)
    # Use a fixed packing width instead of synchronizing on lengths.max().
    max_len = spec_query_len
    # Full CUDA graphs can pad these buffers. Only the first num_reqs
    # accepted-token counts belong to actual speculative requests.
    num_accepted_tokens = num_accepted_tokens[:num_reqs]
    q_starts = spec_query_start_loc[: num_reqs + 1].to(torch.int64)
    # Keep the number of real speculative tokens on the device.
    total_real_tokens = q_starts[num_reqs]

    state_indices = spec_state_indices_tensor.to(
        device=conv_state.device, dtype=torch.int64
    )
    valid_state = state_indices != NULL_BLOCK_ID
    state_indices = torch.where(
        valid_state, state_indices, torch.zeros_like(state_indices)
    )
    positions = torch.arange(x_spec.size(0), device=x_spec.device, dtype=torch.int64)
    # Route graph-padded token rows to the discarded dummy request so that
    # they cannot overwrite real packed data.
    req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
    valid_tokens = (positions < total_real_tokens) & (req_indices < num_reqs)
    clamped_req_indices = req_indices.clamp_max(max(num_reqs - 1, 0))
    col_indices = (positions - q_starts[clamped_req_indices]).clamp_(0, max_len - 1)
    pack_req_indices = torch.where(
        valid_tokens,
        clamped_req_indices,
        torch.full_like(req_indices, num_reqs),
    )
    pack_col_indices = torch.where(
        valid_tokens, col_indices, torch.zeros_like(col_indices)
    )

    # The last request row is the dummy sink for graph padding.
    packed = x_spec.new_zeros((num_reqs + 1, max_len, hidden_size))
    packed[pack_req_indices, pack_col_indices] = x_spec
    packed = packed.transpose(1, 2).contiguous()

    if conv_state_len > 0:
        cached_state = conv_state.index_select(0, state_indices)
        rollback_offsets = num_accepted_tokens.to(
            device=conv_state.device, dtype=torch.int64
        ).sub(1)
        rollback_offsets = torch.where(
            valid_state,
            rollback_offsets.clamp_(0, max_len - 1),
            torch.zeros_like(rollback_offsets),
        )
        state_offsets = torch.arange(
            conv_state_len, device=conv_state.device, dtype=torch.int64
        ).view(1, 1, conv_state_len)
        rollback_indices = rollback_offsets.view(-1, 1, 1) + state_offsets
        state = cached_state.gather(2, rollback_indices.expand(-1, hidden_size, -1)).to(
            x_spec.dtype
        )
        state = torch.where(
            valid_state.view(num_reqs, 1, 1),
            state,
            torch.zeros_like(state),
        )
        # Append a zeroed dummy-row state to match the [num_reqs + 1] pack.
        dummy_state = state.new_zeros((1, hidden_size, conv_state_len))
        state_full = torch.cat((state, dummy_state), dim=0)
        history = torch.cat((state_full, packed), dim=-1)
    else:
        history = packed

    conv_output = F.conv1d(
        history,
        conv_weights.unsqueeze(1).contiguous(),
        groups=history.size(1),
        dilation=dilation,
    )
    conv_output = F.silu(conv_output).transpose(1, 2).contiguous()

    output = conv_output[pack_req_indices, pack_col_indices]
    output = output * valid_tokens.view(-1, 1).to(output.dtype)

    # Keep all current candidate inputs in the extended state. On the next
    # target forward, ``num_accepted_tokens - 1`` selects the rollback
    # window before processing the newly scheduled tokens.
    if conv_state_len > 0:
        state_capacity = conv_state_len + max_len - 1
        if conv_state.size(-1) < state_capacity:
            raise RuntimeError(
                "PLE short-conv cache cannot retain speculative tokens: "
                f"got {conv_state.size(-1)}, need {state_capacity}."
            )
        candidate_state = history[:num_reqs, :, 1 : state_capacity + 1]
        query_lengths = q_starts[1:] - q_starts[:-1]
        state_positions = torch.arange(
            state_capacity, device=history.device, dtype=torch.int64
        ).view(1, 1, state_capacity)
        update_lengths = (conv_state_len + query_lengths - 1).view(num_reqs, 1, 1)
        update_mask = valid_state.view(num_reqs, 1, 1) & (
            state_positions < update_lengths
        )
        existing_state = cached_state[..., :state_capacity]
        next_state = torch.where(
            update_mask,
            candidate_state.to(conv_state.dtype),
            existing_state,
        )
        cached_state[..., :state_capacity] = next_state
        conv_state.index_copy_(0, state_indices, cached_state)

    return output


def _short_conv_dilated_dispatch_pytorch(
    inputs: torch.Tensor,
    residual: torch.Tensor,
    metadata: PleShortConvAttentionMetadata,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    *,
    conv_state_len: int,
    dilation: int,
) -> None:
    """PyTorch reference for fused short-convolution dispatch."""
    num_prefills = metadata.num_prefills
    num_decodes = metadata.num_decodes
    num_decode_tokens = metadata.num_decode_tokens
    num_prefill_tokens = metadata.num_prefill_tokens
    has_prefill = num_prefills > 0
    has_decode = num_decodes > 0
    has_spec = metadata.spec_sequence_masks is not None
    x = inputs[: metadata.num_actual_tokens]
    residual = residual[: metadata.num_actual_tokens]

    if has_spec:
        if has_prefill or has_decode:
            assert metadata.spec_token_indx is not None
            assert metadata.non_spec_token_indx is not None
            spec_token_indices = metadata.spec_token_indx.long()
            non_spec_token_indices = metadata.non_spec_token_indx.long()
            x_spec = x.index_select(0, spec_token_indices)
            x_non_spec = x.index_select(0, non_spec_token_indices)
            residual_spec = residual.index_select(0, spec_token_indices)
            residual_non_spec = residual.index_select(0, non_spec_token_indices)
        else:
            spec_token_indices = None
            non_spec_token_indices = None
            x_spec = x
            x_non_spec = None
            residual_spec = residual
            residual_non_spec = None
    else:
        spec_token_indices = None
        non_spec_token_indices = None
        x_spec = None
        x_non_spec = x
        residual_spec = None
        residual_non_spec = residual

    if has_spec:
        assert metadata.spec_state_indices_tensor is not None
        assert metadata.spec_query_start_loc is not None
        assert metadata.num_accepted_tokens is not None
        assert x_spec is not None
        assert residual_spec is not None
        spec_state_indices = metadata.spec_state_indices_tensor[
            : metadata.num_spec_decodes
        ]
        conv_output = _short_conv_dilated_spec_pytorch(
            x_spec=x_spec,
            conv_state=conv_state,
            conv_weights=conv_weights,
            spec_state_indices_tensor=spec_state_indices,
            spec_query_start_loc=metadata.spec_query_start_loc,
            num_accepted_tokens=metadata.num_accepted_tokens,
            spec_query_len=metadata.spec_query_len,
            conv_state_len=conv_state_len,
            dilation=dilation,
        )
        residual_spec.add_(conv_output)

    state_indices = metadata.state_indices_tensor
    if x_non_spec is not None:
        assert state_indices is not None
        assert residual_non_spec is not None
        if has_prefill:
            state_indices_d, state_indices_p = torch.split(
                state_indices, [num_decodes, num_prefills], dim=0
            )
            x_d, x_p = torch.split(
                x_non_spec, [num_decode_tokens, num_prefill_tokens], dim=0
            )
            residual_d, residual_p = torch.split(
                residual_non_spec,
                [num_decode_tokens, num_prefill_tokens],
                dim=0,
            )
            if has_decode:
                conv_output = _short_conv_dilated_decode_pytorch(
                    x_d=x_d,
                    conv_state=conv_state,
                    conv_weights=conv_weights,
                    state_indices_tensor_d=state_indices_d,
                    has_initial_states_d=metadata.has_initial_states_d,
                    conv_state_len=conv_state_len,
                    dilation=dilation,
                )
                residual_d.add_(conv_output)
            conv_output = _short_conv_dilated_prefill_pytorch(
                x_p=x_p,
                metadata=metadata,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices_tensor_p=state_indices_p,
                num_prefills=num_prefills,
                num_decode_tokens=num_decode_tokens,
                num_prefill_tokens=num_prefill_tokens,
                conv_state_len=conv_state_len,
                dilation=dilation,
            )
            residual_p.add_(conv_output)
        else:
            conv_output = _short_conv_dilated_decode_pytorch(
                x_d=x_non_spec,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices_tensor_d=state_indices[: x_non_spec.size(0)],
                has_initial_states_d=metadata.has_initial_states_d,
                conv_state_len=conv_state_len,
                dilation=dilation,
            )
            residual_non_spec.add_(conv_output)

    if has_spec and residual_non_spec is not None:
        assert spec_token_indices is not None
        assert non_spec_token_indices is not None
        assert residual_spec is not None
        residual.index_copy_(0, spec_token_indices, residual_spec)
        residual.index_copy_(0, non_spec_token_indices, residual_non_spec)


_KERNEL_STATE_BLOCKS = 64


def _make_conv_case(
    device: torch.device,
    seed: int,
    channels: int,
    kernel_size: int,
    dilation: int,
    state_layout: str,
    spec_query_len: int = 1,
) -> tuple[torch.Generator, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = torch.Generator(device=device).manual_seed(seed)
    state_len = (kernel_size - 1) * dilation
    state_width = state_len + spec_query_len - 1
    state_shape = (
        (_KERNEL_STATE_BLOCKS, state_width, channels)
        if state_layout == "SD"
        else (_KERNEL_STATE_BLOCKS, channels, state_width)
    )
    state_kernel = torch.randn(
        state_shape, device=device, dtype=torch.bfloat16, generator=rng
    )
    conv_state = (
        state_kernel.transpose(-1, -2) if state_layout == "SD" else state_kernel
    )
    state_reference = conv_state.clone()
    weights = torch.randn(
        channels, kernel_size, device=device, dtype=torch.bfloat16, generator=rng
    )
    return rng, state_reference, conv_state, weights


@dataclass(frozen=True)
class _ConvBatchCase:
    spec_query_lens: tuple[int, ...] = ()
    num_accepted: tuple[int, ...] = ()
    num_decodes: int = 0
    prefill_query_lens: tuple[int, ...] = ()
    channels: int = 2048
    kernel_size: int = 4
    dilation: int = 3
    spec_query_len: int = 1
    graph_padding: int = 0


def _make_conv_metadata(
    case: _ConvBatchCase,
    device: torch.device,
) -> tuple[SimpleNamespace, int]:
    request_groups: dict[str, list[int]] = {
        "spec": [],
        "decode": [],
        "prefill": [],
    }
    token_offset = 0
    max_num_reqs = max(
        len(case.spec_query_lens),
        case.num_decodes,
        len(case.prefill_query_lens),
    )
    request_kinds = []
    for req_idx in range(max_num_reqs):
        if req_idx < len(case.spec_query_lens):
            request_kinds.append(("spec", case.spec_query_lens[req_idx]))
        if req_idx < case.num_decodes:
            request_kinds.append(("decode", 1))
        if req_idx < len(case.prefill_query_lens):
            request_kinds.append(("prefill", case.prefill_query_lens[req_idx]))
    for kind, query_len in request_kinds:
        request_groups[kind].extend(range(token_offset, token_offset + query_len))
        token_offset += query_len

    num_spec_reqs = len(case.spec_query_lens)
    num_prefills = len(case.prefill_query_lens)
    has_spec = num_spec_reqs > 0
    has_non_spec = case.num_decodes > 0 or num_prefills > 0
    mixed_batch = has_spec and has_non_spec

    spec_state_indices = torch.arange(
        1, num_spec_reqs + 1, dtype=torch.int32, device=device
    )
    if num_spec_reqs >= 3:
        spec_state_indices[1] = NULL_BLOCK_ID
    non_spec_state_indices = torch.arange(
        num_spec_reqs + 1,
        num_spec_reqs + case.num_decodes + num_prefills + 1,
        dtype=torch.int32,
        device=device,
    )
    if case.num_decodes > 1:
        non_spec_state_indices[case.num_decodes - 1] = NULL_BLOCK_ID
    if num_prefills > 1:
        non_spec_state_indices[case.num_decodes + 1] = NULL_BLOCK_ID

    spec_query_start_loc = torch.tensor(
        [0, *accumulate(case.spec_query_lens)], dtype=torch.int32, device=device
    )
    non_spec_query_lens = (1,) * case.num_decodes + case.prefill_query_lens
    non_spec_query_start_loc = torch.tensor(
        [0, *accumulate(non_spec_query_lens)], dtype=torch.int32, device=device
    )
    spec_token_indices = torch.tensor(
        request_groups["spec"], dtype=torch.int64, device=device
    )
    non_spec_token_indices = torch.tensor(
        request_groups["decode"] + request_groups["prefill"],
        dtype=torch.int64,
        device=device,
    )

    metadata = SimpleNamespace(
        num_prefills=num_prefills,
        num_decodes=case.num_decodes,
        num_decode_tokens=case.num_decodes,
        num_prefill_tokens=sum(case.prefill_query_lens),
        spec_sequence_masks=(
            torch.tensor([kind == "spec" for kind, _ in request_kinds], device=device)
            if has_spec
            else None
        ),
        spec_token_indx=spec_token_indices if mixed_batch else None,
        non_spec_token_indx=non_spec_token_indices if mixed_batch else None,
        num_actual_tokens=token_offset + case.graph_padding,
        spec_state_indices_tensor=spec_state_indices if has_spec else None,
        spec_query_start_loc=spec_query_start_loc if has_spec else None,
        num_accepted_tokens=(
            torch.tensor(case.num_accepted, dtype=torch.int32, device=device)
            if has_spec
            else None
        ),
        spec_query_len=case.spec_query_len,
        state_indices_tensor=non_spec_state_indices if has_non_spec else None,
        has_initial_states_d=(
            torch.arange(case.num_decodes, device=device) % 2 == 0
            if case.num_decodes
            else None
        ),
        non_spec_query_start_loc=(non_spec_query_start_loc if has_non_spec else None),
        has_initial_states_p=(
            torch.arange(num_prefills, device=device) % 2 == 0 if num_prefills else None
        ),
        max_prefill_query_len=max(case.prefill_query_lens, default=0),
        num_spec_decodes=num_spec_reqs,
    )
    return metadata, token_offset


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused conv needs CUDA")
@pytest.mark.parametrize("state_layout", ["SD", "DS"])
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            _ConvBatchCase(num_decodes=1, channels=512),
            id="decode-single",
        ),
        pytest.param(
            _ConvBatchCase(
                num_decodes=33,
                channels=2048,
                kernel_size=5,
                dilation=1,
            ),
            id="decode-padded",
        ),
        pytest.param(
            _ConvBatchCase(prefill_query_lens=(37, 0, 5, 128)),
            id="prefill",
        ),
        pytest.param(
            _ConvBatchCase(
                spec_query_lens=(1, 4, 4, 0),
                num_accepted=(1, 2, 4, 1),
                spec_query_len=4,
                graph_padding=4,
            ),
            id="spec-graph-padded",
        ),
        pytest.param(
            _ConvBatchCase(
                spec_query_lens=(3, 4),
                num_accepted=(2, 4),
                num_decodes=1,
                prefill_query_lens=(5,),
                spec_query_len=4,
            ),
            id="mixed",
        ),
        pytest.param(
            _ConvBatchCase(
                spec_query_lens=(1, 2),
                num_accepted=(0, 1),
                num_decodes=2,
                prefill_query_lens=(0, 8, 3),
                channels=640,
                kernel_size=3,
                dilation=2,
                spec_query_len=3,
            ),
            id="mixed-varied",
        ),
    ],
)
def test_fused_conv_correctness(
    case: _ConvBatchCase,
    state_layout: str,
) -> None:
    device = torch.device("cuda")
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
        state_layout=state_layout,
        spec_query_len=case.spec_query_len,
    )
    inputs = torch.randn(
        metadata.num_actual_tokens,
        case.channels,
        device=device,
        dtype=torch.bfloat16,
        generator=rng,
    )
    residual = torch.randn(
        inputs.shape,
        device=device,
        dtype=torch.bfloat16,
        generator=rng,
    )
    null_state = conv_state[NULL_BLOCK_ID].clone()
    residual_kernel = residual.clone()
    residual_reference = residual.clone()

    module._short_conv_dilated_dispatch(
        inputs=inputs,
        residual=residual_kernel,
        metadata=metadata,
        conv_state=conv_state,
        conv_weights=weights,
    )
    _short_conv_dilated_dispatch_pytorch(
        inputs=inputs,
        residual=residual_reference,
        metadata=metadata,
        conv_state=state_reference,
        conv_weights=weights,
        conv_state_len=module.conv_state_len,
        dilation=module.short_conv_dilation,
    )

    torch.testing.assert_close(
        residual_kernel.float(), residual_reference.float(), atol=3e-2, rtol=3e-2
    )
    assert torch.equal(conv_state, state_reference)
    assert torch.equal(conv_state[NULL_BLOCK_ID], null_state)
    if case.graph_padding:
        assert torch.equal(
            residual_kernel[num_real_tokens:], residual[num_real_tokens:]
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused gate needs CUDA")
@pytest.mark.parametrize(("num_tokens", "strided_kv"), [(1, False), (64, True)])
def test_fused_gate_correctness(num_tokens: int, strided_kv: bool) -> None:
    import math

    from vllm.models.qwen4_exp.nvidia.ops.ple import ple_gate

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

    def make_norm_weight() -> torch.Tensor:
        weight = torch.empty(hc * h, device=device, dtype=torch.bfloat16)
        return weight.normal_(mean=-0.1, std=0.1, generator=generator)

    norm_key = make_norm_weight()
    norm_query = make_norm_weight()
    norm_conv = make_norm_weight()

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
    assert torch.equal(gated, expected_gated)
    torch.testing.assert_close(normed, expected_normed, atol=1e-2, rtol=1e-2)
