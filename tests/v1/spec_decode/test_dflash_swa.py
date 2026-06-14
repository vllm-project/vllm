# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.config import SpeculativeConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.qwen3_dflash import DFlashAttention
from vllm.transformers_utils.configs.speculators import SpeculatorsConfig
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.spec_decode.dflash import DFlashProposer


class _FakeBuilder:
    def __init__(
        self, kv_cache_spec=None, layer_names=None, vllm_config=None, device=None
    ):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names

    def build_for_drafting(self, common_attn_metadata, draft_index):
        return SimpleNamespace(
            causal=common_attn_metadata.causal,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
        )


class _FakeAttentionGroup:
    def __init__(self, layer_names, kv_cache_group_id=0):
        self.layer_names = layer_names
        self.kv_cache_group_id = kv_cache_group_id
        self._builder = _FakeBuilder()

    def get_metadata_builder(self):
        return self._builder


def _make_cad(block_table, slot_mapping) -> CommonAttentionMetadata:
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=2,
        max_query_len=2,
        max_seq_len=2,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        causal=False,
    )


def test_dflash_speculators_preserves_swa_config():
    layer_types = [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    config = {
        "speculators_model_type": "dflash",
        "transformer_layer_config": {
            "num_hidden_layers": len(layer_types),
            "sliding_window": None,
        },
        "draft_vocab_size": 100,
        "target_hidden_size": 64,
        "aux_hidden_state_layer_ids": [0, 1, 2],
        "mask_token_id": 99,
        "layer_types": layer_types,
        "use_sliding_window": True,
        "sliding_window": 2048,
        "max_window_layers": len(layer_types),
    }

    hf_config = SpeculatorsConfig.extract_transformers_pre_trained_config(config)

    assert hf_config["layer_types"] == layer_types
    assert hf_config["use_sliding_window"] is True
    assert hf_config["sliding_window"] == 2048
    assert hf_config["max_window_layers"] == len(layer_types)
    assert hf_config["eagle_aux_hidden_state_layer_ids"] == [1, 2, 3]
    assert hf_config["dflash_config"]["target_layer_ids"] == [0, 1, 2]


def _compute_dflash_hash(hf_config: SimpleNamespace) -> str:
    config = object.__new__(SpeculativeConfig)
    config.method = "dflash"
    config.draft_model_config = SimpleNamespace(hf_config=hf_config)
    return config.compute_hash()


def test_dflash_compile_hash_uses_checkpoint_layer_id_semantics():
    dflash_hash = _compute_dflash_hash(
        SimpleNamespace(dflash_config={"target_layer_ids": [0, 2]})
    )
    shifted_aux_hash = _compute_dflash_hash(
        SimpleNamespace(eagle_aux_hidden_state_layer_ids=[1, 3])
    )
    different_hash = _compute_dflash_hash(
        SimpleNamespace(dflash_config={"target_layer_ids": [0, 3]})
    )

    assert dflash_hash == shifted_aux_hash
    assert dflash_hash != different_hash


def test_dflash_swa_layers_use_full_kv_cache_spec(monkeypatch):
    sliding_spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
        sliding_window=4,
    )
    monkeypatch.setattr(
        Attention,
        "get_kv_cache_spec",
        lambda self, vllm_config: sliding_spec,
    )

    spec = DFlashAttention.get_kv_cache_spec(
        object.__new__(DFlashAttention), SimpleNamespace()
    )

    assert isinstance(spec, FullAttentionSpec)
    assert spec.block_size == sliding_spec.block_size
    assert spec.num_kv_heads == sliding_spec.num_kv_heads
    assert spec.head_size == sliding_spec.head_size
    assert spec.sliding_window is None


def test_dflash_swa_layers_use_causal_metadata():
    proposer = object.__new__(DFlashProposer)
    proposer.model = SimpleNamespace(sliding_attention_layer_names={"layer.sw"})
    proposer.draft_attn_groups = [_FakeAttentionGroup(["layer.sw", "layer.full"])]
    proposer.kv_cache_gid = 0
    proposer._draft_kv_cache_group_ids = [0]
    proposer._draft_layer_to_kv_cache_gid = {
        "layer.sw": 0,
        "layer.full": 0,
    }
    proposer._draft_block_tables = {}
    cad = _make_cad(
        torch.empty(1, 1, dtype=torch.int32),
        torch.empty(2, dtype=torch.int64),
    )
    proposer._slot_mapping_buffers_by_gid = {0: (cad.slot_mapping, cad.slot_mapping)}

    per_group, per_layer = DFlashProposer.build_per_group_and_layer_attn_metadata(
        proposer, cad
    )

    assert per_group[0].causal is False
    assert per_layer["layer.sw"].causal is True
    assert per_layer["layer.full"].causal is False


def test_dflash_metadata_uses_per_kv_group_slot_mapping():
    proposer = object.__new__(DFlashProposer)
    proposer.model = SimpleNamespace(sliding_attention_layer_names={"layer.sw"})
    proposer.draft_attn_groups = [
        _FakeAttentionGroup(["layer.full"], kv_cache_group_id=1),
        _FakeAttentionGroup(["layer.sw"], kv_cache_group_id=2),
    ]
    proposer.kv_cache_gid = 1
    proposer._draft_kv_cache_group_ids = [1, 2]
    proposer._draft_layer_to_kv_cache_gid = {
        "layer.full": 1,
        "layer.sw": 2,
    }

    full_block_table = torch.tensor([[11, 12]], dtype=torch.int32)
    sw_block_table = torch.tensor([[21, 22]], dtype=torch.int32)
    full_slots = torch.tensor([111, 112], dtype=torch.int64)
    sw_slots = torch.tensor([211, 212], dtype=torch.int64)

    base_cad = _make_cad(full_block_table, full_slots)
    proposer._draft_block_tables = {
        1: full_block_table,
        2: sw_block_table,
    }
    proposer._slot_mapping_buffers_by_gid = {
        1: (full_slots, full_slots),
        2: (sw_slots, sw_slots),
    }

    _, per_layer = DFlashProposer.build_per_group_and_layer_attn_metadata(
        proposer, base_cad
    )

    assert per_layer["layer.full"].block_table_tensor is full_block_table
    torch.testing.assert_close(per_layer["layer.full"].slot_mapping, full_slots)
    assert per_layer["layer.full"].causal is False
    assert per_layer["layer.sw"].block_table_tensor is sw_block_table
    torch.testing.assert_close(per_layer["layer.sw"].slot_mapping, sw_slots)
    assert per_layer["layer.sw"].causal is True
