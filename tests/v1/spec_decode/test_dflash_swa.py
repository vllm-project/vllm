# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from vllm.config import AttentionConfig, SpeculativeConfig
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.models.qwen3_dflash import DFlashAttention
from vllm.transformers_utils.configs.speculators import SpeculatorsConfig
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


@dataclass
class _FakeVllmConfig:
    attention_config: AttentionConfig


class _FakeBuilder:
    def __init__(
        self, kv_cache_spec=None, layer_names=None, vllm_config=None, device=None
    ):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names

    def build_for_drafting(self, common_attn_metadata, draft_index):
        return SimpleNamespace(
            block_table_tensor=common_attn_metadata.block_table_tensor,
            causal=common_attn_metadata.causal,
            slot_mapping=common_attn_metadata.slot_mapping,
        )


class _FakeBackend:
    @staticmethod
    def get_builder_cls():
        return _FakeBuilder


class _FakeAttentionGroup:
    def __init__(self, layer_names, kv_cache_group_id=0):
        self.backend = _FakeBackend
        self.layer_names = layer_names
        self.kv_cache_group_id = kv_cache_group_id
        self._builder = _FakeBuilder(layer_names=layer_names)

    def get_metadata_builder(self):
        return self._builder


def test_dflash_speculators_preserves_swa_config():
    layer_types = [
        "sliding_attention",
        "sliding_attention",
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
        "aux_hidden_state_layer_ids": [1, 6, 11, 17, 22, 27],
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
    assert hf_config["dflash_config"]["target_layer_ids"] == [1, 6, 11, 17, 22, 27]


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


def test_dflash_draft_preserves_explicit_flashinfer_backend(monkeypatch):
    proposer = object.__new__(DFlashProposer)
    proposer.speculative_config = SimpleNamespace(
        moe_backend=None,
        attention_backend=AttentionBackendEnum.FLASHINFER,
    )

    def fake_create_draft_vllm_config(self):
        return _FakeVllmConfig(
            attention_config=AttentionConfig(
                backend=self.speculative_config.attention_backend
            )
        )

    monkeypatch.setattr(
        SpecDecodeBaseProposer,
        "_create_draft_vllm_config",
        fake_create_draft_vllm_config,
    )

    draft_config = DFlashProposer._create_draft_vllm_config(proposer)

    assert draft_config.attention_config.backend == AttentionBackendEnum.FLASHINFER
    assert draft_config.attention_config.use_non_causal is True


def test_dflash_swa_layers_use_causal_metadata():
    proposer = object.__new__(DFlashProposer)
    proposer.vllm_config = SimpleNamespace()
    proposer.device = None
    proposer.model = SimpleNamespace(sliding_attention_layer_names={"layer.sw"})
    proposer.draft_attn_groups = [_FakeAttentionGroup(["layer.sw", "layer.full"])]
    proposer.kv_cache_gid = 0
    proposer._draft_kv_cache_group_ids = [0]
    proposer._draft_layer_to_kv_cache_gid = {"layer.sw": 0, "layer.full": 0}
    proposer._draft_block_tables = {}
    cad = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=2,
        max_query_len=2,
        max_seq_len=2,
        block_table_tensor=torch.empty(1, 1, dtype=torch.int32),
        slot_mapping=torch.empty(2, dtype=torch.int64),
        causal=False,
    )
    proposer._slot_mapping_buffers_by_gid = {0: (cad.slot_mapping, cad.slot_mapping)}

    per_group, per_layer = DFlashProposer.build_per_group_and_layer_attn_metadata(
        proposer, cad
    )

    assert per_group[0].causal is False
    assert per_layer["layer.sw"].causal is True
    assert per_layer["layer.full"].causal is False


def test_dflash_uses_per_kv_group_slot_mapping_and_block_table():
    proposer = object.__new__(DFlashProposer)
    proposer.vllm_config = SimpleNamespace()
    proposer.device = None
    proposer.model = SimpleNamespace(sliding_attention_layer_names=set())
    proposer.draft_attn_groups = [
        _FakeAttentionGroup(["layer.0"], kv_cache_group_id=0),
        _FakeAttentionGroup(["layer.1"], kv_cache_group_id=1),
    ]
    proposer.kv_cache_gid = 0
    proposer._draft_kv_cache_group_ids = [0, 1]
    proposer._draft_attn_layer_names = {"layer.0", "layer.1"}
    proposer._draft_layer_to_kv_cache_gid = {"layer.0": 0, "layer.1": 1}

    context_slots_0 = torch.tensor([0, 1], dtype=torch.int64)
    context_slots_1 = torch.tensor([16, 17], dtype=torch.int64)
    query_slots_0 = torch.tensor([2, 3], dtype=torch.int64)
    query_slots_1 = torch.tensor([18, 19], dtype=torch.int64)
    proposer._slot_mapping_buffers_by_gid = {
        0: (context_slots_0, query_slots_0),
        1: (context_slots_1, query_slots_1),
    }
    block_table_0 = torch.tensor([[0]], dtype=torch.int32)
    block_table_1 = torch.tensor([[1]], dtype=torch.int32)
    proposer._draft_block_tables = {1: block_table_1}

    cad = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=2,
        max_query_len=2,
        max_seq_len=2,
        block_table_tensor=block_table_0,
        slot_mapping=query_slots_0,
        causal=False,
    )

    context_mapping = proposer._get_dflash_context_slot_mapping(num_context=2)
    _, per_layer = DFlashProposer.build_per_group_and_layer_attn_metadata(
        proposer, cad
    )

    assert torch.equal(context_mapping["layer.0"], context_slots_0)
    assert torch.equal(context_mapping["layer.1"], context_slots_1)
    assert per_layer["layer.0"].block_table_tensor is block_table_0
    assert torch.equal(per_layer["layer.0"].slot_mapping, query_slots_0)
    assert per_layer["layer.1"].block_table_tensor is block_table_1
    assert torch.equal(per_layer["layer.1"].slot_mapping, query_slots_1)
