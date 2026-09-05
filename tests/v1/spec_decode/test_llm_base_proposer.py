# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SpecDecodeBaseProposer attention and slot-mapping setup.

Block tables are stored at kernel-block granularity, so the proposer's
``block_size`` (used for slot-mapping math) must be the kernel block size,
not the KV cache manager's block size — the two differ when manager blocks
are split for the attention kernel. The value must also be deterministic:
``_draft_attn_layer_names`` is a set, whose iteration order varies across
processes, so anything derived from iteration order must not leak into
``block_size``.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.spec_decode.llm_base_proposer as llm_base_proposer
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.step3p5 import Step3p5MTPProposer
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID

SCHEDULER_BLOCK_SIZE = 256
KERNEL_BLOCK_SIZE = 64


def _cpu_step_update(
    positions_1d,
    block_table_tensor,
    seq_lens,
    block_size,
    max_model_len,
    out_clamped_positions,
    out_slot_mapping,
    input_batch_size=None,
):
    new_positions = positions_1d + 1
    exceeds_max = new_positions >= max_model_len
    clamped_positions = torch.where(
        exceeds_max, torch.zeros_like(new_positions), new_positions
    )
    block_numbers = (clamped_positions // block_size).clamp(
        max=block_table_tensor.shape[1] - 1
    )
    block_ids = block_table_tensor.gather(1, block_numbers[:, None]).squeeze(1)
    slot_mapping = block_ids * block_size + clamped_positions.remainder(block_size)
    slot_mapping.masked_fill_(exceeds_max, PADDING_SLOT_ID)
    out_slot_mapping[: positions_1d.shape[0]].copy_(slot_mapping)
    out_clamped_positions.copy_(clamped_positions)
    seq_lens.copy_(torch.where(exceeds_max, torch.ones_like(seq_lens), seq_lens + 1))


class _FakeAttentionGroup:
    def __init__(self, backend, layer_names, kv_cache_spec, kv_cache_group_id):
        self.backend = backend
        self.layer_names = list(layer_names)
        self.kv_cache_spec = kv_cache_spec
        self.kv_cache_group_id = kv_cache_group_id
        self.kernel_block_size = None

    def create_metadata_builders(self, vllm_config, device, kernel_block_size=None):
        self.kernel_block_size = kernel_block_size

    def get_metadata_builder(self):
        return SimpleNamespace(kv_cache_spec=self.kv_cache_spec)


def _make_proposer(
    monkeypatch: pytest.MonkeyPatch, layer_names: set[str]
) -> EagleProposer:
    fake_layers = {}
    for name in layer_names:
        backend = SimpleNamespace(full_cls_name=lambda: "FakeBackend")
        fake_layers[name] = SimpleNamespace(
            get_attn_backend=lambda backend=backend: backend
        )
    monkeypatch.setattr(
        llm_base_proposer, "get_layers_from_vllm_config", lambda *a, **k: fake_layers
    )
    monkeypatch.setattr(llm_base_proposer, "AttentionGroup", _FakeAttentionGroup)

    proposer = EagleProposer.__new__(EagleProposer)
    proposer.vllm_config = None
    proposer.device = None
    proposer._draft_attn_layer_names = set(layer_names)
    proposer.kv_cache_gid = -1
    proposer.draft_attn_groups = []
    proposer.block_size = -1
    return proposer


def _make_kv_cache_config(layer_names: set[str]) -> SimpleNamespace:
    spec = SimpleNamespace(block_size=SCHEDULER_BLOCK_SIZE)
    group = SimpleNamespace(layer_names=list(layer_names), kv_cache_spec=spec)
    return SimpleNamespace(kv_cache_groups=[group])


def test_block_size_uses_kernel_block_size(monkeypatch: pytest.MonkeyPatch):
    """The proposer's slot-mapping math runs against the kernel-granularity
    block table, so block_size must come from kernel_block_sizes."""
    layer_names = {"draft.0.self_attn.attn"}
    proposer = _make_proposer(monkeypatch, layer_names)

    proposer.initialize_attn_backend(
        _make_kv_cache_config(layer_names),
        kernel_block_sizes=[KERNEL_BLOCK_SIZE],
    )

    assert proposer.block_size == KERNEL_BLOCK_SIZE
    assert proposer.block_size != SCHEDULER_BLOCK_SIZE
    # The metadata builder keeps receiving the kernel block size as well.
    assert proposer.draft_attn_groups[0].kernel_block_size == KERNEL_BLOCK_SIZE


def test_block_size_falls_back_to_kv_cache_spec(monkeypatch: pytest.MonkeyPatch):
    layer_names = {"draft.0.self_attn.attn"}
    proposer = _make_proposer(monkeypatch, layer_names)

    proposer.initialize_attn_backend(
        _make_kv_cache_config(layer_names), kernel_block_sizes=None
    )

    assert proposer.block_size == SCHEDULER_BLOCK_SIZE


@pytest.mark.parametrize(
    (
        "mrope_position",
        "seq_len",
        "expected_slot",
        "expected_seq_len",
        "expected_mrope_position",
    ),
    [
        (19, 110, 1710, 111, 20),
        (19, 1024, PADDING_SLOT_ID, 1, 20),
        (1023, 1024, PADDING_SLOT_ID, 1, 0),
    ],
)
def test_mrope_slot_mapping_uses_absolute_sequence_position(
    monkeypatch: pytest.MonkeyPatch,
    mrope_position: int,
    seq_len: int,
    expected_slot: int,
    expected_seq_len: int,
    expected_mrope_position: int,
):
    proposer = EagleProposer.__new__(EagleProposer)
    proposer.uses_mrope = True
    proposer.uses_xdrope_dim = 0
    proposer.draft_uses_xdrope_dim = 0
    proposer.max_model_len = 1024
    proposer._slot_positions = torch.empty(1, dtype=torch.int64)
    proposer._slot_mapping_buffer = torch.empty(1, dtype=torch.int64)
    proposer.mrope_positions = torch.empty((3, 2), dtype=torch.int64)

    positions = torch.full((3, 1), mrope_position, dtype=torch.int64)
    block_size = 16
    block_table = torch.arange(100, 164, dtype=torch.int64).unsqueeze(0)
    metadata = SimpleNamespace(
        block_table_tensor=block_table,
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        slot_mapping=None,
        max_seq_len=seq_len,
        _seq_lens_cpu=None,
        _num_computed_tokens_cpu=None,
        seq_lens_cpu_upper_bound=None,
    )

    monkeypatch.setattr(
        llm_base_proposer,
        "eagle_step_update_slot_mapping_and_metadata",
        _cpu_step_update,
    )

    updated_positions = proposer._update_positions_dependent_metadata(
        positions,
        metadata,
        batch_size=1,
        input_batch_size=1,
        block_size=block_size,
    )

    assert metadata.slot_mapping.tolist() == [expected_slot]
    assert metadata.seq_lens.tolist() == [expected_seq_len]
    assert updated_positions.tolist() == [
        [expected_mrope_position],
        [expected_mrope_position],
        [expected_mrope_position],
    ]


@pytest.mark.parametrize(
    (
        "uses_mrope",
        "seq_len",
        "expected_primary",
        "expected_secondary",
        "expected_seq_len",
        "expected_positions",
    ),
    [
        (True, 110, 1710, 3310, 111, [[20], [20], [20]]),
        (True, 1024, PADDING_SLOT_ID, PADDING_SLOT_ID, 1, [[20], [20], [20]]),
        (False, 110, 1710, 3310, 111, [110]),
    ],
)
def test_step3p5_slots_use_absolute_position_for_all_groups(
    monkeypatch: pytest.MonkeyPatch,
    uses_mrope: bool,
    seq_len: int,
    expected_primary: int,
    expected_secondary: int,
    expected_seq_len: int,
    expected_positions: list,
):
    proposer = Step3p5MTPProposer.__new__(Step3p5MTPProposer)
    proposer.uses_mrope = uses_mrope
    proposer.uses_xdrope_dim = 0
    proposer.draft_uses_xdrope_dim = 0
    proposer.max_model_len = 1024
    proposer.max_positions = 1
    proposer.device = torch.device("cpu")
    proposer._slot_mapping_buffer = torch.empty(1, dtype=torch.int64)
    if uses_mrope:
        proposer._slot_positions = torch.empty(1, dtype=torch.int64)
        proposer.mrope_positions = torch.empty((3, 2), dtype=torch.int64)
        positions = torch.full((3, 1), 19, dtype=torch.int64)
    else:
        proposer.positions = torch.empty(1, dtype=torch.int64)
        positions = torch.tensor([109], dtype=torch.int64)
    proposer.kv_cache_gid = 0
    proposer.draft_attn_groups = [
        SimpleNamespace(kv_cache_group_id=0),
        SimpleNamespace(kv_cache_group_id=1),
    ]
    proposer._per_group_block_tables = {
        1: torch.arange(200, 264, dtype=torch.int64).unsqueeze(0)
    }
    proposer._per_group_slot_mappings = {}
    proposer._per_group_slot_mapping_buffers = {}

    block_size = 16
    metadata = SimpleNamespace(
        block_table_tensor=torch.arange(100, 164, dtype=torch.int64).unsqueeze(0),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        slot_mapping=None,
        max_seq_len=seq_len,
        _seq_lens_cpu=None,
        _num_computed_tokens_cpu=None,
        seq_lens_cpu_upper_bound=None,
    )

    monkeypatch.setattr(
        llm_base_proposer,
        "eagle_step_update_slot_mapping_and_metadata",
        _cpu_step_update,
    )

    updated_positions = proposer._update_positions_dependent_metadata(
        positions,
        metadata,
        batch_size=1,
        input_batch_size=1,
        block_size=block_size,
    )

    assert proposer._per_group_slot_mappings[0].tolist() == [expected_primary]
    assert proposer._per_group_slot_mappings[1].tolist() == [expected_secondary]
    assert metadata.seq_lens.tolist() == [expected_seq_len]
    assert updated_positions.tolist() == expected_positions


def test_draft_layer_iteration_is_deterministic(monkeypatch: pytest.MonkeyPatch):
    """_draft_attn_layer_names is a set; the attention groups built from it
    must not depend on its (process-random) iteration order."""
    layer_names = {"draft.c.attn", "draft.a.attn", "draft.b.attn"}
    expected_order = sorted(layer_names)

    for insertion_order in (expected_order, expected_order[::-1]):
        proposer = _make_proposer(monkeypatch, set(insertion_order))
        proposer.initialize_attn_backend(
            _make_kv_cache_config(set(insertion_order)),
            kernel_block_sizes=[KERNEL_BLOCK_SIZE],
        )
        assert len(proposer.draft_attn_groups) == 1
        assert proposer.draft_attn_groups[0].layer_names == expected_order
        assert proposer.block_size == KERNEL_BLOCK_SIZE
