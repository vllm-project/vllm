# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Qwen4Exp MTP proposer cache topology."""

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.spec_decode.qwen4_exp as qwen_proposer
from vllm.config.speculative import SpeculativeConfig
from vllm.models.qwen4_exp.common.qsa_cache import (
    circular_qsa_slot_mapping,
    compressed_qsa_slot_mapping,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.spec_decode.qwen4_exp import (
    Qwen4ExpMTPProposer,
)

SCHEDULER_BLOCK_SIZE = 256
KERNEL_BLOCK_SIZE = 64
RAW_CAPACITY = 128
MAIN_LAYER = "draft.mtp.layers.0.self_attn.attn"
RAW_LAYER = "draft.mtp.layers.0.self_attn.indexer.raw_key_cache"
COMPRESSED_LAYER = "draft.mtp.layers.0.self_attn.indexer.compressed_key_cache"


class _FakeBackend:
    def __init__(self, name: str) -> None:
        self.name = name

    def full_cls_name(self) -> tuple[str, str]:
        return (__name__, self.name)


class _FakeAttentionGroup:
    def __init__(self, backend, layer_names, kv_cache_spec, kv_cache_group_id):
        self.backend = backend
        self.layer_names = list(layer_names)
        self.kv_cache_spec = kv_cache_spec
        self.kv_cache_group_id = kv_cache_group_id
        self.kernel_block_size = None
        self.builder = SimpleNamespace(kv_cache_spec=kv_cache_spec)

    def create_metadata_builders(self, vllm_config, device, kernel_block_size=None):
        self.kernel_block_size = kernel_block_size
        if kernel_block_size is not None:
            self.builder.kv_cache_spec = self.kv_cache_spec.copy_with_new_block_size(
                kernel_block_size
            )

    def get_metadata_builder(self):
        return self.builder


def _make_specs():
    main_spec = FullAttentionSpec(
        block_size=SCHEDULER_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=128,
        head_size_v=128,
        dtype=torch.bfloat16,
    )
    compressed_spec = MLAAttentionSpec(
        block_size=SCHEDULER_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=64,
    )
    raw_spec = CircularBufferSpec(
        block_size=RAW_CAPACITY,
        num_kv_heads=1,
        head_size=128,
        head_size_v=0,
        dtype=torch.bfloat16,
    )
    return main_spec, compressed_spec, raw_spec


def _make_proposer_and_config(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Qwen4ExpMTPProposer, SimpleNamespace]:
    main_spec, compressed_spec, raw_spec = _make_specs()
    backends = {
        MAIN_LAYER: _FakeBackend("MainBackend"),
        RAW_LAYER: _FakeBackend("QSAStateBackend"),
        COMPRESSED_LAYER: _FakeBackend("QSAStateBackend"),
    }
    fake_layers = {
        layer_name: SimpleNamespace(
            get_attn_backend=lambda backend=backend: backend,
            num_heads=1,
        )
        for layer_name, backend in backends.items()
    }
    monkeypatch.setattr(
        qwen_proposer,
        "get_layers_from_vllm_config",
        lambda *args, **kwargs: fake_layers,
    )
    monkeypatch.setattr(qwen_proposer, "AttentionGroup", _FakeAttentionGroup)

    proposer = Qwen4ExpMTPProposer.__new__(Qwen4ExpMTPProposer)
    proposer.vllm_config = None
    proposer.draft_model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(mtp_num_hidden_layers=1)
    )
    proposer.device = torch.device("cpu")
    proposer._draft_attn_layer_names = {MAIN_LAYER, RAW_LAYER, COMPRESSED_LAYER}
    proposer.kv_cache_gid = -1
    proposer.draft_attn_groups = []
    proposer.block_size = -1
    proposer._per_group_block_tables = {}

    packed_main_group = UniformTypeKVCacheSpecs(
        block_size=SCHEDULER_BLOCK_SIZE,
        kv_cache_specs={
            MAIN_LAYER: main_spec,
            COMPRESSED_LAYER: compressed_spec,
        },
    )
    packed_raw_group = UniformTypeKVCacheSpecs(
        block_size=RAW_CAPACITY,
        kv_cache_specs={RAW_LAYER: raw_spec},
    )
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=[MAIN_LAYER, COMPRESSED_LAYER],
                kv_cache_spec=packed_main_group,
            ),
            SimpleNamespace(
                layer_names=[RAW_LAYER],
                kv_cache_spec=packed_raw_group,
            ),
        ]
    )
    return proposer, config


def test_initializes_current_qwen_cache_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposer, config = _make_proposer_and_config(monkeypatch)

    proposer.initialize_attn_backend(
        config,
        kernel_block_sizes=[KERNEL_BLOCK_SIZE, RAW_CAPACITY],
    )

    assert proposer.kv_cache_gid == 0
    assert proposer.block_size == KERNEL_BLOCK_SIZE
    assert len(proposer.draft_attn_groups) == 3
    assert [group.kv_cache_group_id for group in proposer.draft_attn_groups] == [
        0,
        0,
        1,
    ]
    assert {type(group.kv_cache_spec) for group in proposer.draft_attn_groups} == {
        FullAttentionSpec,
        MLAAttentionSpec,
        CircularBufferSpec,
    }


def test_preserves_builder_slot_mapping_in_each_cache_owner_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposer, config = _make_proposer_and_config(monkeypatch)
    proposer.initialize_attn_backend(
        config,
        kernel_block_sizes=[KERNEL_BLOCK_SIZE, RAW_CAPACITY],
    )
    logical_positions = torch.tensor([62, 63, 64, 65], dtype=torch.int64)
    token_to_req = torch.zeros(4, dtype=torch.int32)

    def build_for_group(group):
        def build_for_drafting(*, common_attn_metadata, draft_index):
            spec = group.builder.kv_cache_spec
            if isinstance(spec, CircularBufferSpec):
                slots = circular_qsa_slot_mapping(
                    common_attn_metadata.block_table_tensor,
                    token_to_req,
                    logical_positions,
                    spec.block_size,
                    query_start_loc=common_attn_metadata.query_start_loc,
                )
            elif isinstance(spec, MLAAttentionSpec):
                slots = compressed_qsa_slot_mapping(
                    common_attn_metadata.block_table_tensor,
                    token_to_req,
                    logical_positions,
                    spec.storage_block_size,
                    spec.compress_ratio,
                )
            else:
                slots = common_attn_metadata.slot_mapping
            return SimpleNamespace(
                common=common_attn_metadata,
                draft_index=draft_index,
                slot_mapping=slots,
            )

        return build_for_drafting

    for group in proposer.draft_attn_groups:
        group.builder.build_for_drafting = build_for_group(group)

    main_table = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    raw_table = torch.tensor([[20]], dtype=torch.int32)
    main_slots = torch.tensor([702, 703, 704, 705], dtype=torch.int64)
    proposer.set_per_group_block_table(1, raw_table)
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=4,
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        block_table_tensor=main_table,
        slot_mapping=main_slots,
    )

    _, per_layer = proposer.build_per_group_and_layer_attn_metadata(
        common,
        draft_index=2,
    )

    assert torch.equal(per_layer[MAIN_LAYER].slot_mapping, main_slots)
    assert torch.equal(
        per_layer[COMPRESSED_LAYER].slot_mapping,
        torch.tensor([-1, 10, -1, -1], dtype=torch.int64),
    )
    assert torch.equal(
        per_layer[RAW_LAYER].slot_mapping,
        torch.tensor([2622, 2623, 2624, 2625], dtype=torch.int64),
    )
    assert torch.equal(per_layer[RAW_LAYER].common.block_table_tensor, raw_table)


def test_rejects_packed_group_without_direct_draft_layer_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposer, config = _make_proposer_and_config(monkeypatch)
    main_spec, _, _ = _make_specs()
    config.kv_cache_groups[0].kv_cache_spec = UniformTypeKVCacheSpecs(
        block_size=SCHEDULER_BLOCK_SIZE,
        kv_cache_specs={MAIN_LAYER: main_spec},
    )

    with pytest.raises(AssertionError, match=f"no spec for {COMPRESSED_LAYER}"):
        proposer.initialize_attn_backend(
            config,
            kernel_block_sizes=[KERNEL_BLOCK_SIZE, RAW_CAPACITY],
        )


def test_rejects_multiple_mtp_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    proposer, config = _make_proposer_and_config(monkeypatch)
    proposer.draft_model_config.hf_text_config.mtp_num_hidden_layers = 2

    with pytest.raises(NotImplementedError, match="only supports one MTP layer"):
        proposer.initialize_attn_backend(
            config,
            kernel_block_sizes=[KERNEL_BLOCK_SIZE, RAW_CAPACITY],
        )


def test_qwen_proposer_owns_hidden_and_return_contract() -> None:
    proposer = Qwen4ExpMTPProposer.__new__(Qwen4ExpMTPProposer)
    proposer.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(hc_mult=4),
        get_hidden_size=lambda: 1024,
    )

    assert proposer._get_hidden_size() == 4096
    assert proposer.model_returns_tuple()


def test_speculative_config_selects_qwen_proposer() -> None:
    config = SimpleNamespace(
        method="mtp",
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen4_exp_mtp")
        ),
    )

    assert SpeculativeConfig.use_qwen4_exp_mtp(config)
