# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
from vllm.models.qwen4_exp.nvidia import ple_layer as ple_layer_module
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
    return module


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


def test_ngram_embedding_rejects_mismatched_checkpoint_shard() -> None:
    module = _make_ngram_embedding_for_load_test()

    with pytest.raises(
        ValueError,
        match=r"Shape mismatch for PLE embedding shard 0",
    ):
        module.load_weights([("ngram_embedding.shard_0.weight", torch.zeros(3, 2))])


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
    assert module.get_offload_output_dtype(torch.bfloat16) == torch.float8_e4m3fn


def test_ngram_gpu_offload_retains_only_fp8_global_scale(monkeypatch) -> None:
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(module)
    weight_scale = torch.tensor([0.25], dtype=torch.bfloat16)
    monkeypatch.setattr(ple_layer_module.envs, "VLLM_PLE_CPU_OFFLOAD", True)
    monkeypatch.setattr(ple_layer_module, "is_offload_process", lambda: False)
    monkeypatch.setattr(
        torch.accelerator,
        "current_accelerator",
        lambda: torch.device("cpu"),
    )

    loaded = module.load_weights(
        [
            ("ngram_embedding.shard_0.weight", torch.empty(4, 2)),
            ("ngram_embedding.weight_scale", weight_scale),
        ]
    )

    assert loaded == {"ngram_embedding.weight_scale"}
    assert torch.equal(module._offload_weight_scale, weight_scale)
    assert module.get_offload_output_dtype(torch.bfloat16) == torch.float8_e4m3fn

    ple_layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(ple_layer)
    ple_layer.ple_embedding = module
    embeddings = torch.tensor([[4.0, 8.0]]).to(torch.float8_e4m3fn)
    output = ple_layer._dequantize_embeddings(embeddings, torch.bfloat16)
    torch.testing.assert_close(
        output,
        torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
    )


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


def test_ngram_cpu_offload_padding_does_not_overwrite_real_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(module)
    module.embedding_dim = 1
    module.head_dim = 1
    module.ngram_size = 2
    module.heads_per_ngram = 1
    module.eos_token_id = 99
    module.register_buffer("positions_buffer", torch.arange(4))
    module.register_buffer("padded_buffer", torch.empty(1, 4, dtype=torch.long))
    module.register_buffer("layer_multipliers", torch.tensor([3, 5]))
    module.register_buffer("ngram_heads_vocab_sizes", torch.tensor([101]))
    module.register_buffer("ngram_heads_offsets", torch.tensor([0]))
    module.ngram_embedding = nn.Embedding(101, 1)
    module.ngram_embedding.weight.requires_grad_(False)
    with torch.no_grad():
        module.ngram_embedding.weight.copy_(torch.arange(101).reshape(-1, 1))

    monkeypatch.setattr(ple_layer_module, "is_offload_process", lambda: True)
    query_start_loc = torch.tensor([0, 2])
    ngram_context = torch.full((1, 1), 99, dtype=torch.long)
    expected = module.forward_impl(
        torch.empty(2, 0),
        torch.tensor([11, 13]),
        query_start_loc,
        ngram_context,
        output_buffer=torch.empty(2, 1),
    )
    actual = module.forward_impl(
        torch.empty(4, 0),
        torch.tensor([11, 13, 777, 888]),
        query_start_loc,
        ngram_context,
        output_buffer=torch.empty(4, 1),
    )

    torch.testing.assert_close(actual[:2], expected)


def test_ngram_fp8_cpu_offload_preserves_quantized_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(module)
    module.embedding_dim = 2
    module.head_dim = 2
    module.ngram_size = 2
    module.heads_per_ngram = 1
    module.eos_token_id = 99
    module.register_buffer("positions_buffer", torch.arange(2))
    module.register_buffer("padded_buffer", torch.empty(1, 2, dtype=torch.long))
    module.register_buffer("layer_multipliers", torch.tensor([1, 1]))
    module.register_buffer("ngram_heads_vocab_sizes", torch.tensor([3]))
    module.register_buffer("ngram_heads_offsets", torch.tensor([0]))
    module.ngram_embedding = _make_fp8_embedding_layer(monkeypatch)

    monkeypatch.setattr(ple_layer_module, "is_offload_process", lambda: True)
    hidden_states = torch.empty(2, 0)
    input_ids = torch.tensor([0, 1])
    query_start_loc = torch.tensor([0, 2])
    ngram_context = torch.tensor([[99]])
    quantized = module.forward_impl(
        hidden_states,
        input_ids,
        query_start_loc,
        ngram_context,
    )
    output_buffer = torch.empty(2, 2, dtype=torch.float8_e4m3fn)

    output = module.forward_impl(
        hidden_states,
        input_ids,
        query_start_loc,
        ngram_context,
        output_buffer=output_buffer,
    )

    assert output.data_ptr() == output_buffer.data_ptr()
    assert output.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(output.float(), quantized.float())


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


def test_ple_state_shape_reserves_speculative_tokens() -> None:
    module = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(module)
    module.hc_hidden_size = 32
    module.conv_state_len = 9
    module.num_spec_tokens = 3

    assert module.get_state_shape()[0] in ((32, 12), (12, 32))
