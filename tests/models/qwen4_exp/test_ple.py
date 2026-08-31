# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import vllm.model_executor.layers.vocab_parallel_embedding as embedding_module
import vllm.model_executor.parameter as parameter_module
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.models.qwen4_exp.common.ple import (
    PLEShardOverlap,
    compute_ple_shard_overlap,
    copy_ple_embedding_shard_,
)
from vllm.models.qwen4_exp.nvidia import ple_layer as ple_layer_module
from vllm.models.qwen4_exp.nvidia.ple_layer import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPinnedHostEmbedding,
    Qwen4ExpPLEDeviceEmbedding,
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEFp8EmbeddingMethod,
    Qwen4ExpPLELayer,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)


def _mock_np_group(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int = 1,
    all_reduce=lambda tensor: tensor,
) -> None:
    group = SimpleNamespace(
        rank_in_group=0,
        world_size=world_size,
        all_reduce=all_reduce,
    )
    monkeypatch.setattr(ple_layer_module, "get_np_group", lambda: group)


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


@pytest.mark.parametrize(("dp_rank", "local_tokens"), [(0, 2), (1, 3)])
def test_np_lookup_gathers_and_returns_dp_local_rows(
    monkeypatch: pytest.MonkeyPatch,
    dp_rank: int,
    local_tokens: int,
) -> None:
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(module)
    module.np_data_parallel_size = 2
    module.data_parallel_rank = dp_rank
    gathered_ids = torch.tensor([[10], [11], [0], [20], [21], [22]])
    group = SimpleNamespace(
        rank_in_group=dp_rank,
        all_gather=lambda input_ids, dim: gathered_ids,
    )
    forward_context = SimpleNamespace(
        dp_metadata=SimpleNamespace(
            num_tokens_across_dp_cpu=torch.tensor([2, 3]),
        )
    )
    monkeypatch.setattr(ple_layer_module, "get_np_dp_group", lambda: group)
    monkeypatch.setattr(
        ple_layer_module, "get_forward_context", lambda: forward_context
    )
    local_ids = gathered_ids[dp_rank * 3 : dp_rank * 3 + local_tokens]
    slot_size, slot_offset = module._get_np_gather_slot(local_tokens)
    lookup_ids = module._gather_np_ids(local_ids, slot_size)
    embeddings = torch.arange(12).reshape(6, 2)
    output = module._select_embeddings(embeddings, local_tokens, slot_offset)

    assert torch.equal(lookup_ids, gathered_ids)
    expected = embeddings[dp_rank * 3 : dp_rank * 3 + local_tokens]
    assert torch.equal(output, expected)


def _make_fp8_embedding_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> Qwen4ExpPLEDeviceEmbedding:
    _mock_np_group(monkeypatch)
    monkeypatch.setattr(embedding_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        embedding_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    method = Qwen4ExpPLEFp8EmbeddingMethod()
    layer = Qwen4ExpPLEDeviceEmbedding(
        3,
        2,
        params_dtype=torch.bfloat16,
        padding_size=1,
        prefix="test.ple_embedding",
        embedding_method=method,
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


def test_ple_fp8_embedding_uses_int8_for_parallel_reduce(monkeypatch) -> None:
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

    _mock_np_group(monkeypatch, world_size=2, all_reduce=all_reduce)
    layer = Qwen4ExpPLEDeviceEmbedding(
        4,
        2,
        params_dtype=torch.bfloat16,
        padding_size=1,
        prefix="test.ple_embedding",
        embedding_method=Qwen4ExpPLEFp8EmbeddingMethod(),
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
        Qwen4ExpPLEEmbeddingMethod.from_quant_config(quant_config, prefix),
        Qwen4ExpPLEFp8EmbeddingMethod,
    )

    quant_config.ignored_layers = [f"{prefix}.shard_0"]
    assert isinstance(
        Qwen4ExpPLEEmbeddingMethod.from_quant_config(quant_config, prefix),
        Qwen4ExpPLEUnquantizedEmbeddingMethod,
    )
    assert isinstance(
        Qwen4ExpPLEEmbeddingMethod.from_quant_config(None, prefix),
        Qwen4ExpPLEUnquantizedEmbeddingMethod,
    )


def test_ple_embedding_rejects_unsupported_quantization_configs() -> None:
    prefix = "model.layers.1.ple.ple_embedding.ngram_embedding"
    with pytest.raises(NotImplementedError, match="SimpleNamespace"):
        Qwen4ExpPLEEmbeddingMethod.from_quant_config(SimpleNamespace(), prefix)

    nvfp4_config = ModelOptNvFp4Config(exclude_modules=[])
    with pytest.raises(NotImplementedError, match="ModelOptNvFp4Config"):
        Qwen4ExpPLEEmbeddingMethod.from_quant_config(nvfp4_config, prefix)

    dynamic_fp8_config = Fp8Config(
        is_checkpoint_fp8_serialized=False,
        ignored_layers=[],
    )
    with pytest.raises(NotImplementedError, match="serialized FP8"):
        Qwen4ExpPLEEmbeddingMethod.from_quant_config(dynamic_fp8_config, prefix)


def test_ple_embedding_respects_modelopt_exclusion() -> None:
    prefix = "model.layers.1.ple.ple_embedding.ngram_embedding"
    quant_config = ModelOptNvFp4Config(exclude_modules=[prefix])

    assert isinstance(
        Qwen4ExpPLEEmbeddingMethod.from_quant_config(quant_config, prefix),
        Qwen4ExpPLEUnquantizedEmbeddingMethod,
    )


def test_ple_embedding_dtype_overrides_modelopt_exclusion() -> None:
    prefix = "model.layers.1.ple.ple_embedding.ngram_embedding"
    quant_config = ModelOptNvFp4Config(exclude_modules=[prefix])

    assert isinstance(
        Qwen4ExpPLEEmbeddingMethod.from_quant_config(
            quant_config,
            prefix,
            "float8_e4m3fn",
        ),
        Qwen4ExpPLEFp8EmbeddingMethod,
    )


def test_ple_prefetch_streams_are_scoped_by_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streams = Qwen4ExpNGramEmbedding._PREFETCH_STREAMS
    streams.clear()
    monkeypatch.setattr(torch.cuda, "Stream", object)

    try:
        with pytest.raises(RuntimeError, match="is not initialized"):
            Qwen4ExpNGramEmbedding._get_prefetch_stream("layers.1.ple")

        Qwen4ExpNGramEmbedding._setup_prefetch_stream("layers.1.ple")
        Qwen4ExpNGramEmbedding._setup_prefetch_stream("layers.1.ple")
        Qwen4ExpNGramEmbedding._setup_prefetch_stream("layers.2.ple")
        first = Qwen4ExpNGramEmbedding._get_prefetch_stream("layers.1.ple")
        first_again = Qwen4ExpNGramEmbedding._get_prefetch_stream("layers.1.ple")
        second = Qwen4ExpNGramEmbedding._get_prefetch_stream("layers.2.ple")

        assert first is first_again
        assert first is not second
    finally:
        streams.clear()


def test_ple_device_embedding_allocates_on_active_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_np_group(monkeypatch)
    monkeypatch.setattr(embedding_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        embedding_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    embedding_method = Qwen4ExpPLEUnquantizedEmbeddingMethod()

    with torch.device("cuda:0"):
        embedding = Qwen4ExpPLEDeviceEmbedding(
            4,
            3,
            params_dtype=torch.bfloat16,
            padding_size=1,
            prefix="test.ple_embedding",
            embedding_method=embedding_method,
        )

    assert embedding.weight.device == torch.device("cuda:0")
    assert embedding.embedding_method is embedding_method
    assert embedding.quant_method is embedding_method


@pytest.mark.parametrize("fp8_checkpoint", [False, True])
def test_ple_pinned_embedding_loads_on_cpu_and_looks_up_through_uva(
    monkeypatch: pytest.MonkeyPatch,
    fp8_checkpoint: bool,
) -> None:
    _mock_np_group(monkeypatch)
    monkeypatch.setattr(embedding_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        embedding_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

    with torch.device("cuda:0"):
        embedding_method = (
            Qwen4ExpPLEFp8EmbeddingMethod()
            if fp8_checkpoint
            else Qwen4ExpPLEUnquantizedEmbeddingMethod()
        )
        embedding = Qwen4ExpPinnedHostEmbedding(
            4,
            3,
            params_dtype=torch.bfloat16,
            padding_size=1,
            prefix="test.ple_embedding",
            embedding_method=embedding_method,
        )

    storage_dtype = torch.float8_e4m3fn if fp8_checkpoint else torch.bfloat16
    loaded_weight = (
        torch.arange(12, dtype=torch.float32).reshape(4, 3).to(storage_dtype)
    )
    copied = copy_ple_embedding_shard_(
        embedding.weight,
        loaded_weight,
        checkpoint_start=0,
        tp_start=0,
        tp_end=4,
    )
    input_ids = torch.tensor([[3, 0], [1, 2]], device="cuda:0")
    output = embedding.lookup(input_ids)
    if fp8_checkpoint:
        embedding.weight_scale.data.fill_(0.25)
    dequantized = embedding.dequantize(output, torch.bfloat16)

    assert copied == 4
    assert embedding.weight.device.type == "cpu"
    assert embedding.weight.is_pinned()
    assert embedding.embedding_method is embedding_method
    assert not hasattr(embedding, "quant_method")
    assert embedding._uva_weight.device.type == "cuda"
    assert output.dtype == torch.bfloat16
    expected = loaded_weight[input_ids.cpu()].to(device="cuda:0", dtype=torch.bfloat16)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    expected_scale = 0.25 if fp8_checkpoint else 1.0
    torch.testing.assert_close(
        dequantized,
        expected * expected_scale,
        rtol=0,
        atol=0,
    )


def test_ple_ngram_ids_custom_op_uses_current_request_layout(monkeypatch) -> None:
    class RuntimeNGramEmbedding(nn.Module):
        def _compute_ngram_ids_impl(
            self,
            input_ids: torch.Tensor,
            query_start_loc: torch.Tensor,
            ngram_context: torch.Tensor,
        ) -> torch.Tensor:
            del input_ids, ngram_context
            num_reqs = query_start_loc.numel() - 1
            return torch.full((4, 2), num_reqs, dtype=torch.long)

    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    layer.ple_embedding = RuntimeNGramEmbedding()
    monkeypatch.setattr(
        ple_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"ple": layer}),
    )
    input_ids = torch.arange(4)
    ngram_context = torch.zeros(2, 2, dtype=torch.long)
    output = torch.empty(4, 2, dtype=torch.long)

    ple_layer_module.qwen4_exp_compute_ple_ngram_ids(
        input_ids,
        torch.tensor([0, 4]),
        ngram_context,
        output,
        "ple",
    )
    assert torch.equal(output, torch.ones_like(output))

    ple_layer_module.qwen4_exp_compute_ple_ngram_ids(
        input_ids,
        torch.tensor([0, 2, 4]),
        ngram_context,
        output,
        "ple",
    )
    assert torch.equal(output, torch.full_like(output, 2))


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


def test_ple_short_conv_uses_fallback_when_profile_metadata_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(module)
    module.prefix = "model.layers.1.ple"
    inputs = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    expected = inputs + 1
    monkeypatch.setattr(module, "_short_conv_fallback", lambda _: expected)
    monkeypatch.setattr(
        ple_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"model.layers.0.self_attn": object()}),
    )

    output = module._short_conv(inputs)

    assert output is expected
