# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState


class FakeEncoderGroup:
    def __init__(self, layer_names: list[str], metadata: object):
        self.layer_names = layer_names
        self.builder = Mock()
        self.builder.build.return_value = metadata
        self.builder.build_for_cudagraph_capture.return_value = metadata

    def get_metadata_builder(self, kv_cache_group_id: int) -> Mock:
        assert kv_cache_group_id == 0
        return self.builder


def make_state(*, supports_mm_inputs: bool = False) -> EncoderOnlyModelState:
    state = object.__new__(EncoderOnlyModelState)
    state.supports_mm_inputs = supports_mm_inputs
    state.max_model_len = 8192
    state.encoder_attn_groups = [
        FakeEncoderGroup(["encoder.layer.0.attn", "encoder.layer.1.attn"], object()),
        FakeEncoderGroup(["encoder.layer.2.attn"], object()),
    ]
    state._dummy_block_table = torch.zeros(4, 1, dtype=torch.int32)
    state._dummy_slot_mapping = torch.zeros(16, dtype=torch.int64)
    return state


def make_input_batch() -> SimpleNamespace:
    query_start_loc_np = np.array([0, 2, 5], dtype=np.int32)
    return SimpleNamespace(
        num_reqs=2,
        num_tokens=5,
        num_reqs_after_padding=3,
        num_tokens_after_padding=8,
        query_start_loc=torch.tensor(query_start_loc_np),
        query_start_loc_np=query_start_loc_np,
        num_scheduled_tokens=torch.tensor([2, 3], dtype=torch.int32),
        seq_lens=torch.tensor([4, 6, 0], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([4, 6, 0], dtype=torch.int32),
        positions=torch.arange(8, dtype=torch.int64),
    )


def make_kv_cache_config(*, has_groups: bool) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[Mock()] if has_groups else [],
    )


def call_prepare_attn(
    state: EncoderOnlyModelState,
    input_batch: SimpleNamespace,
    kv_cache_config: KVCacheConfig,
    *,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    for_capture: bool = False,
    attn_groups: list[list[object]] | None = None,
) -> dict[str, object]:
    return state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=cudagraph_mode,
        block_tables=(),
        slot_mappings=torch.empty(0, dtype=torch.int64),
        attn_groups=[] if attn_groups is None else attn_groups,
        kv_cache_config=kv_cache_config,
        for_capture=for_capture,
    )


def test_prepare_attn_empty_text_only_skips_parent_and_builds_encoder_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = make_state()
    input_batch = make_input_batch()
    parent_prepare_attn = Mock(return_value={"parent": object()})
    monkeypatch.setattr(DefaultModelState, "prepare_attn", parent_prepare_attn)

    metadata = call_prepare_attn(
        state, input_batch, make_kv_cache_config(has_groups=False)
    )

    parent_prepare_attn.assert_not_called()
    assert set(metadata) == {
        "encoder.layer.0.attn",
        "encoder.layer.1.attn",
        "encoder.layer.2.attn",
    }
    assert metadata["encoder.layer.0.attn"] is metadata["encoder.layer.1.attn"]
    assert metadata["encoder.layer.0.attn"] is not metadata["encoder.layer.2.attn"]
    for group in state.encoder_attn_groups:
        group.builder.build.assert_called_once()
        group.builder.build_for_cudagraph_capture.assert_not_called()


def test_prepare_attn_nonempty_kv_group_uses_parent_and_forwards_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = make_state()
    input_batch = make_input_batch()
    kv_cache_config = make_kv_cache_config(has_groups=True)
    parent_metadata = {"parent": object()}
    parent_prepare_attn = Mock(return_value=parent_metadata)
    monkeypatch.setattr(DefaultModelState, "prepare_attn", parent_prepare_attn)
    block_tables = (torch.empty(0, dtype=torch.int32),)
    slot_mappings = torch.empty(0, dtype=torch.int64)
    attn_groups: list[list[object]] = [[object()]]

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
    assert metadata is parent_metadata
    assert "parent" in metadata
    assert "encoder.layer.0.attn" in metadata
    for group in state.encoder_attn_groups:
        group.builder.build.assert_not_called()
        group.builder.build_for_cudagraph_capture.assert_called_once()


def test_prepare_attn_mm_inputs_with_empty_kv_groups_uses_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = make_state(supports_mm_inputs=True)
    input_batch = make_input_batch()
    kv_cache_config = make_kv_cache_config(has_groups=False)
    parent_metadata = {"parent": object()}
    parent_prepare_attn = Mock(return_value=parent_metadata)
    monkeypatch.setattr(DefaultModelState, "prepare_attn", parent_prepare_attn)
    slot_mappings = torch.empty(0, dtype=torch.int64)
    attn_groups: list[list[object]] = [[]]

    metadata = state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=slot_mappings,
        attn_groups=attn_groups,
        kv_cache_config=kv_cache_config,
        for_capture=False,
    )

    parent_prepare_attn.assert_called_once_with(
        input_batch,
        CUDAGraphMode.NONE,
        (),
        slot_mappings,
        attn_groups,
        kv_cache_config,
        False,
    )
    assert metadata is parent_metadata
    assert "parent" in metadata
    assert "encoder.layer.0.attn" in metadata


def test_prepare_attn_keeps_ubatch_assert_before_parent_or_encoder_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = make_state()
    input_batch = make_input_batch()
    parent_prepare_attn = Mock()
    encoder_build = Mock()
    monkeypatch.setattr(DefaultModelState, "prepare_attn", parent_prepare_attn)
    monkeypatch.setattr(
        EncoderOnlyModelState,
        "_build_encoder_attn_metadata",
        encoder_build,
    )

    with pytest.raises(AssertionError, match="DBO is not supported"):
        state.prepare_attn(
            input_batch=input_batch,
            cudagraph_mode=CUDAGraphMode.NONE,
            block_tables=(),
            slot_mappings=torch.empty(0, dtype=torch.int64),
            attn_groups=[],
            kv_cache_config=make_kv_cache_config(has_groups=False),
            ubatch_idx=1,
        )

    parent_prepare_attn.assert_not_called()
    encoder_build.assert_not_called()


def test_prepare_attn_fast_path_uses_fresh_metadata_dict_and_does_not_mutate_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = make_state()
    input_batch = make_input_batch()
    kv_cache_config = make_kv_cache_config(has_groups=False)
    attn_groups: list[list[object]] = [[]]
    parent_prepare_attn = Mock()
    monkeypatch.setattr(DefaultModelState, "prepare_attn", parent_prepare_attn)
    reqs_before = input_batch.num_reqs
    query_start_loc_before = input_batch.query_start_loc_np.copy()
    kv_groups_before = list(kv_cache_config.kv_cache_groups)

    first_metadata = call_prepare_attn(
        state, input_batch, kv_cache_config, attn_groups=attn_groups
    )
    first_metadata["caller-added"] = object()
    second_metadata = call_prepare_attn(
        state, input_batch, kv_cache_config, attn_groups=attn_groups
    )

    parent_prepare_attn.assert_not_called()
    assert first_metadata is not second_metadata
    assert "caller-added" not in second_metadata
    assert attn_groups == [[]]
    assert kv_cache_config.kv_cache_groups == kv_groups_before
    assert input_batch.num_reqs == reqs_before
    np.testing.assert_array_equal(
        input_batch.query_start_loc_np, query_start_loc_before
    )
