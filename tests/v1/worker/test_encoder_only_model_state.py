# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState

ENCODER_LAYER = "encoder.layer.0.attn"
BUILD_ENCODER_METADATA = "_build_encoder_attn_metadata"


def make_state(*, supports_mm_inputs: bool = False) -> EncoderOnlyModelState:
    state = object.__new__(EncoderOnlyModelState)
    state.supports_mm_inputs = supports_mm_inputs
    return state


def make_kv_cache_config(*, has_groups: bool) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[Mock()] if has_groups else [],
    )


def test_prepare_attn_text_only_empty_kv_skips_parent_and_returns_fresh_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = make_state()
    input_batch = object()
    kv_cache_config = make_kv_cache_config(has_groups=False)
    encoder_metadata = {ENCODER_LAYER: object()}
    parent_prepare_attn = Mock(return_value={"parent": object()})
    build_encoder_metadata = Mock(return_value=encoder_metadata)
    monkeypatch.setattr(DefaultModelState, "prepare_attn", parent_prepare_attn)
    monkeypatch.setattr(
        EncoderOnlyModelState, BUILD_ENCODER_METADATA, build_encoder_metadata
    )

    prepare_kwargs = dict(
        input_batch=input_batch,
        cudagraph_mode=CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=Mock(),
        attn_groups=[],
        kv_cache_config=kv_cache_config,
    )
    first_metadata = state.prepare_attn(**prepare_kwargs)
    second_metadata = state.prepare_attn(**prepare_kwargs)

    parent_prepare_attn.assert_not_called()
    assert first_metadata == encoder_metadata
    assert second_metadata == encoder_metadata
    first_metadata["caller-added"] = object()
    assert first_metadata is not encoder_metadata
    assert second_metadata is not encoder_metadata
    assert first_metadata is not second_metadata
    assert "caller-added" not in second_metadata
    assert "caller-added" not in encoder_metadata
    build_encoder_metadata.assert_called_with(input_batch, CUDAGraphMode.NONE, False)


@pytest.mark.parametrize(
    ("has_kv_groups", "supports_mm_inputs"), [(True, False), (False, True)]
)
def test_prepare_attn_fallback_calls_parent_then_merges_encoder_metadata(
    monkeypatch: pytest.MonkeyPatch,
    has_kv_groups: bool,
    supports_mm_inputs: bool,
) -> None:
    state = make_state(supports_mm_inputs=supports_mm_inputs)
    input_batch = object()
    block_tables = (Mock(),)
    slot_mappings = Mock()
    attn_groups = [[Mock()]]
    kv_cache_config = make_kv_cache_config(has_groups=has_kv_groups)
    parent_metadata = {"parent": object()}
    encoder_metadata = {ENCODER_LAYER: object()}
    parent_prepare_attn = Mock(return_value=parent_metadata)
    build_encoder_metadata = Mock(return_value=encoder_metadata)
    monkeypatch.setattr(DefaultModelState, "prepare_attn", parent_prepare_attn)
    monkeypatch.setattr(
        EncoderOnlyModelState, BUILD_ENCODER_METADATA, build_encoder_metadata
    )

    metadata = state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=CUDAGraphMode.FULL,
        block_tables=block_tables,
        slot_mappings=slot_mappings,
        attn_groups=attn_groups,
        kv_cache_config=kv_cache_config,
        for_capture=True,
    )

    parent_prepare_attn.assert_called_once_with(
        input_batch,
        CUDAGraphMode.FULL,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        True,
    )
    build_encoder_metadata.assert_called_once_with(
        input_batch, CUDAGraphMode.FULL, True
    )
    assert metadata is parent_metadata
    assert metadata[ENCODER_LAYER] is encoder_metadata[ENCODER_LAYER]


def test_prepare_attn_ubatch_asserts_before_parent_or_encoder_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = make_state()
    parent_prepare_attn = Mock()
    build_encoder_metadata = Mock()
    monkeypatch.setattr(DefaultModelState, "prepare_attn", parent_prepare_attn)
    monkeypatch.setattr(
        EncoderOnlyModelState, BUILD_ENCODER_METADATA, build_encoder_metadata
    )

    with pytest.raises(AssertionError, match="DBO is not supported"):
        state.prepare_attn(
            input_batch=object(),
            cudagraph_mode=CUDAGraphMode.NONE,
            block_tables=(),
            slot_mappings=Mock(),
            attn_groups=[],
            kv_cache_config=make_kv_cache_config(has_groups=False),
            ubatch_idx=1,
        )

    parent_prepare_attn.assert_not_called()
    build_encoder_metadata.assert_not_called()
