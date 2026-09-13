# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig
from vllm.models.qwen3_ksa.common.reference import (
    build_ksa_visibility_mask,
    dense_ksa_attention,
    expand_ksa_batch,
    expand_ksa_cudagraph_decode,
)
from vllm.models.qwen3_ksa.nvidia.attention import (
    KSATextCacheLayer,
    get_ksa_summary_cache_spec,
)
from vllm.models.qwen3_ksa.nvidia.model import (
    _pack_ksa_intermediate_tensors,
    _unpack_ksa_intermediate_tensors,
    parse_ksa_layer_pattern,
    validate_ksa_config,
    validate_ksa_runtime_config,
)
from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec


def _released_config(**overrides: object) -> SimpleNamespace:
    values = {
        "use_summary_attention": True,
        "summary_chunk_size": 8,
        "summary_token_begin": 151936,
        "summary_token_num": 1,
        "summary_chunk_position_ids_type": "origin",
        "summary_token_position_ids_type": "last_chunk_slice_right",
        "summary_independent_parameters": True,
        "summary_independent_attention_layernorm": False,
        "summary_sliding_chunk_num": "([128]*3+[16768]*1)*9",
        "mix_coeff": 0,
        "num_hidden_layers": 36,
        "vocab_size": 151937,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _runtime_config(
    *,
    enforce_eager: bool = True,
    enable_prefix_caching: bool = False,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    decode_context_parallel_size: int = 1,
    prefill_context_parallel_size: int = 1,
    cache_dtype: str = "bfloat16",
    speculative_config: object | None = None,
    quant_config: object | None = None,
    enable_prompt_embeds: bool = False,
    disable_hybrid_kv_cache_manager: bool = False,
    block_size: int | None = None,
) -> VllmConfig:
    return cast(
        VllmConfig,
        SimpleNamespace(
            device_config=SimpleNamespace(device_type="cuda"),
            model_config=SimpleNamespace(
                enforce_eager=enforce_eager,
                dtype=torch.bfloat16,
                enable_prompt_embeds=enable_prompt_embeds,
            ),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                decode_context_parallel_size=decode_context_parallel_size,
                prefill_context_parallel_size=prefill_context_parallel_size,
            ),
            cache_config=SimpleNamespace(
                enable_prefix_caching=enable_prefix_caching,
                block_size=(
                    block_size
                    if block_size is not None
                    else 8
                    if enable_prefix_caching
                    else 16
                ),
                cache_dtype=cache_dtype,
            ),
            scheduler_config=SimpleNamespace(
                disable_hybrid_kv_cache_manager=disable_hybrid_kv_cache_manager,
            ),
            compilation_config=SimpleNamespace(
                mode=CompilationMode.NONE,
                cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            ),
            speculative_config=speculative_config,
            quant_config=quant_config,
        ),
    )


@pytest.mark.parametrize(
    ("pattern", "layers", "expected"),
    [
        (128, 3, (128, 128, 128)),
        ([128, 128, 16384], 3, (128, 128, 16384)),
        ("([128]*3+[16384]*1)*2", 8, (128, 128, 128, 16384) * 2),
    ],
)
def test_parse_ksa_layer_pattern(
    pattern: int | list[int] | str,
    layers: int,
    expected: tuple[int, ...],
) -> None:
    assert parse_ksa_layer_pattern(pattern, num_hidden_layers=layers) == expected


@pytest.mark.parametrize(
    "pattern",
    ["__import__('os').system('true')", "[128, 0]", "[128] + 1", "[128] * 99"],
)
def test_parse_ksa_layer_pattern_rejects_unsafe_or_invalid(pattern: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_ksa_layer_pattern(pattern, num_hidden_layers=2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("summary_chunk_size", 4),
        ("summary_token_num", 2),
        ("summary_token_position_ids_type", "zeros"),
        ("mix_coeff", 0.5),
    ],
)
def test_validate_ksa_config_rejects_unsupported_values(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        validate_ksa_config(_released_config(**{field: value}))


def test_validate_public_4b_shape_config() -> None:
    settings = validate_ksa_config(_released_config())
    assert settings.summary_token_begin == 151936
    assert settings.sliding_chunk_nums == (128, 128, 128, 16768) * 9


def test_public_4b_cache_specs_reduce_active_states() -> None:
    runtime_config = _runtime_config()
    text_cache_owner = SimpleNamespace(
        cache_config=runtime_config.cache_config,
        num_kv_heads=8,
        head_dim=128,
        cache_torch_dtype=torch.bfloat16,
        cache_dtype="bfloat16",
        is_small_layer=True,
        sliding_chunk_num=128,
    )

    text_spec = KSATextCacheLayer.get_kv_cache_spec(
        cast(Any, text_cache_owner), runtime_config
    )
    summary_spec = get_ksa_summary_cache_spec(
        cache_config=runtime_config.cache_config,
        vllm_config=runtime_config,
        num_kv_heads=8,
        head_dim=128,
    )

    unified_specs = unify_kv_cache_spec_page_size(
        {"text": text_spec, "summary": summary_spec}
    )
    unified_text_spec = unified_specs["text"]
    unified_summary_spec = unified_specs["summary"]

    assert isinstance(unified_text_spec, SlidingWindowSpec)
    assert isinstance(unified_summary_spec, FullAttentionSpec)
    assert unified_text_spec.sliding_window == 1032
    assert unified_text_spec.block_size == 16
    assert unified_summary_spec.tokens_per_state == 8
    assert unified_summary_spec.block_size == 128
    assert unified_summary_spec.num_states == 16
    assert unified_text_spec.page_size_bytes == unified_summary_spec.page_size_bytes

    settings = validate_ksa_config(_released_config())
    sequence_length = 131072
    small_layers = settings.sliding_chunk_nums.count(128)
    large_layers = len(settings.sliding_chunk_nums) - small_layers
    small_text_states = (
        (unified_text_spec.sliding_window + unified_text_spec.block_size - 1)
        // unified_text_spec.block_size
        * unified_text_spec.num_states
    )
    summary_states = (
        (sequence_length + unified_summary_spec.block_size - 1)
        // unified_summary_spec.block_size
        * unified_summary_spec.num_states
    )
    large_text_states = (
        (sequence_length + unified_text_spec.block_size - 1)
        // unified_text_spec.block_size
        * unified_text_spec.num_states
    )
    active_ksa_states = (
        small_layers * (small_text_states + summary_states)
        + large_layers * large_text_states
    )
    dense_states = len(settings.sliding_chunk_nums) * sequence_length

    assert active_ksa_states < dense_states * 0.36


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"speculative_config": object()}, "speculative decoding"),
        ({"decode_context_parallel_size": 2}, "DCP"),
        ({"prefill_context_parallel_size": 2}, "PCP"),
        ({"cache_dtype": "fp8"}, "KV cache quantization"),
        ({"quant_config": object()}, "weight quantization"),
        ({"enable_prompt_embeds": True}, "token IDs"),
        ({"disable_hybrid_kv_cache_manager": True}, "hybrid KV cache manager"),
        (
            {"enable_prefix_caching": True, "block_size": 16},
            "block_size=8",
        ),
        (
            {
                "enable_prefix_caching": True,
                "tensor_parallel_size": 2,
            },
            "prefix caching",
        ),
        (
            {
                "enforce_eager": False,
                "tensor_parallel_size": 2,
            },
            "CUDA Graph",
        ),
    ],
)
def test_runtime_config_rejects_unvalidated_features_early(
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(NotImplementedError, match=message):
        validate_ksa_runtime_config(_runtime_config(**overrides))


def test_runtime_config_accepts_eager_tp2_pp2() -> None:
    validate_ksa_runtime_config(
        _runtime_config(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        )
    )


@pytest.mark.parametrize("length", [1, 7, 8, 9, 15, 16, 17])
def test_expansion_boundaries(length: int) -> None:
    input_ids = torch.arange(100, 100 + length)
    positions = torch.arange(length)
    expanded = expand_ksa_batch(
        input_ids,
        positions,
        query_start_loc=torch.tensor([0, length]),
        summary_chunk_size=8,
        summary_token_begin=151936,
    )

    expected_summary_count = length // 8
    assert expanded.summary_row_indices.numel() == expected_summary_count
    assert expanded.expanded_input_ids.numel() == length + expected_summary_count
    torch.testing.assert_close(
        expanded.expanded_input_ids[expanded.output_gather_indices], input_ids
    )
    torch.testing.assert_close(
        expanded.expanded_positions[expanded.output_gather_indices], positions
    )
    if expected_summary_count:
        torch.testing.assert_close(
            expanded.expanded_positions[expanded.summary_row_indices],
            torch.arange(7, length, 8),
        )
        assert torch.all(
            expanded.expanded_input_ids[expanded.summary_row_indices] == 151936
        )


def test_expansion_is_request_local() -> None:
    lengths = torch.tensor([8, 9])
    input_ids = torch.arange(lengths.sum())
    positions = torch.cat((torch.arange(8), torch.arange(9)))
    expanded = expand_ksa_batch(
        input_ids,
        positions,
        query_start_loc=torch.tensor([0, 8, 17]),
        summary_chunk_size=8,
        summary_token_begin=151936,
    )

    torch.testing.assert_close(
        expanded.row_to_request[expanded.summary_row_indices], torch.tensor([0, 1])
    )
    torch.testing.assert_close(
        expanded.expanded_positions[expanded.summary_row_indices], torch.tensor([7, 7])
    )
    torch.testing.assert_close(
        expanded.expanded_input_ids[expanded.output_gather_indices], input_ids
    )


def test_expansion_uses_request_local_row_count() -> None:
    input_ids = torch.arange(8)
    positions = torch.arange(100, 108)
    expanded = expand_ksa_batch(
        input_ids,
        positions,
        query_start_loc=torch.tensor([0, 8]),
        summary_chunk_size=8,
        summary_token_begin=151936,
    )

    assert expanded.summary_row_indices.numel() == 1
    assert expanded.expanded_positions[expanded.summary_row_indices].item() == 107


@pytest.mark.parametrize(
    ("num_computed", "creates_summary"),
    [(6, False), (7, True), (8, False), (15, True)],
)
def test_expansion_uses_absolute_request_token_count_during_decode(
    num_computed: int,
    creates_summary: bool,
) -> None:
    input_ids = torch.tensor([123])
    positions = torch.tensor([num_computed])
    expanded = expand_ksa_batch(
        input_ids,
        positions,
        query_start_loc=torch.tensor([0, 1]),
        summary_chunk_size=8,
        summary_token_begin=151936,
        num_computed_tokens=torch.tensor([num_computed]),
    )

    assert bool(expanded.summary_row_indices.numel()) is creates_summary
    torch.testing.assert_close(
        expanded.row_logical_positions[expanded.text_row_indices],
        torch.tensor([num_computed]),
    )


def test_cudagraph_decode_expansion_has_fixed_masked_summary_rows() -> None:
    expanded = expand_ksa_cudagraph_decode(
        torch.tensor([101, 102, 0]),
        torch.tensor([6, 7, 0]),
        text_row_is_valid=torch.tensor([True, True, False]),
        summary_chunk_size=8,
        summary_token_begin=151936,
    )

    torch.testing.assert_close(expanded.text_row_indices, torch.tensor([0, 2, 4]))
    torch.testing.assert_close(
        expanded.summary_row_indices,
        torch.tensor([1, 3, 5]),
    )
    torch.testing.assert_close(
        expanded.expanded_input_ids,
        torch.tensor([101, 151936, 102, 151936, 0, 151936]),
    )
    torch.testing.assert_close(
        expanded.summary_row_is_active,
        torch.tensor([False, True, False]),
    )
    torch.testing.assert_close(
        expanded.expanded_input_ids[expanded.output_gather_indices],
        torch.tensor([101, 102, 0]),
    )


def test_pipeline_payload_round_trips_expanded_rows() -> None:
    input_ids = torch.arange(17)
    expanded = expand_ksa_batch(
        input_ids,
        input_ids,
        query_start_loc=torch.tensor([0, 8, 17]),
        summary_chunk_size=8,
        summary_token_begin=151936,
    )
    hidden_states = torch.randn(expanded.expanded_input_ids.numel(), 4)
    residual = torch.randn_like(hidden_states)

    intermediate = _pack_ksa_intermediate_tensors(
        hidden_states,
        residual,
        expanded,
    )
    restored_hidden, restored_residual = _unpack_ksa_intermediate_tensors(
        intermediate,
        expanded,
    )

    assert all(tensor.shape == (17, 4) for _, tensor in intermediate.items())
    torch.testing.assert_close(restored_hidden, hidden_states)
    torch.testing.assert_close(restored_residual, residual)
    non_boundary = ~expanded.logical_boundary_mask
    assert not torch.any(intermediate["ksa_summary_hidden_states"][non_boundary])
    assert not torch.any(intermediate["ksa_summary_residual"][non_boundary])


def test_small_window_visibility_has_no_gap_or_duplicate() -> None:
    input_ids = torch.arange(5)
    expanded = expand_ksa_batch(
        input_ids,
        input_ids,
        query_start_loc=torch.tensor([0, 5]),
        summary_chunk_size=2,
        summary_token_begin=99,
    )
    mask = build_ksa_visibility_mask(
        expanded,
        summary_chunk_size=2,
        sliding_chunk_num=1,
    )

    final_text_row = int(expanded.text_row_indices[-1])
    visible_rows = torch.nonzero(mask[final_text_row], as_tuple=False).flatten()
    expected_rows = torch.tensor(
        [
            int(expanded.summary_row_indices[0]),
            int(expanded.text_row_indices[2]),
            int(expanded.text_row_indices[3]),
            int(expanded.text_row_indices[4]),
        ]
    )
    torch.testing.assert_close(visible_rows, expected_rows.sort().values)

    second_summary_row = int(expanded.summary_row_indices[1])
    visible_rows = torch.nonzero(mask[second_summary_row], as_tuple=False).flatten()
    torch.testing.assert_close(
        visible_rows,
        torch.tensor(
            [
                int(expanded.text_row_indices[2]),
                int(expanded.text_row_indices[3]),
                second_summary_row,
            ]
        ),
    )


def test_large_window_uses_text_and_not_historical_summaries() -> None:
    input_ids = torch.arange(5)
    expanded = expand_ksa_batch(
        input_ids,
        input_ids,
        query_start_loc=torch.tensor([0, 5]),
        summary_chunk_size=2,
        summary_token_begin=99,
    )
    mask = build_ksa_visibility_mask(
        expanded,
        summary_chunk_size=2,
        sliding_chunk_num=16,
    )

    final_text_row = int(expanded.text_row_indices[-1])
    visible_rows = torch.nonzero(mask[final_text_row], as_tuple=False).flatten()
    torch.testing.assert_close(visible_rows, expanded.text_row_indices)


def test_dense_attention_supports_gqa_and_request_isolation() -> None:
    input_ids = torch.arange(8)
    positions = torch.cat((torch.arange(4), torch.arange(4)))
    expanded = expand_ksa_batch(
        input_ids,
        positions,
        query_start_loc=torch.tensor([0, 4, 8]),
        summary_chunk_size=2,
        summary_token_begin=99,
    )
    row_count = expanded.expanded_input_ids.numel()
    query = torch.ones(row_count, 4, 2)
    key = torch.ones(row_count, 2, 2)
    value = torch.arange(row_count, dtype=torch.float32).view(-1, 1, 1)
    value = value.expand(-1, 2, 2).contiguous()

    result = dense_ksa_attention(
        query,
        key,
        value,
        expanded,
        summary_chunk_size=2,
        sliding_chunk_num=1,
        return_debug_mask=True,
    )

    assert result.output.shape == query.shape
    assert result.lse.shape == (row_count, query.shape[1])
    assert result.visibility_mask is not None
    request = expanded.row_to_request
    assert not torch.any(
        result.visibility_mask & (request[:, None] != request[None, :])
    )
