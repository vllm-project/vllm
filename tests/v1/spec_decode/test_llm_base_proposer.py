# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SpecDecodeBaseProposer.initialize_attn_backend.

Block tables are stored at kernel-block granularity, so the proposer's
``block_size`` (used for slot-mapping math) must be the kernel block size,
not the KV cache manager's block size — the two differ when manager blocks
are split for the attention kernel. The value must also be deterministic:
``_draft_attn_layer_names`` is a set, whose iteration order varies across
processes, so anything derived from iteration order must not leak into
``block_size``.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.spec_decode.llm_base_proposer as llm_base_proposer
from vllm.config import CUDAGraphMode
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.worker.utils import AttentionGroup

SCHEDULER_BLOCK_SIZE = 256
KERNEL_BLOCK_SIZE = 64


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


@pytest.fixture
def draft_metadata():
    """Real CPU metadata builder, without loading draft weights or running kernels."""
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.PIECEWISE, static_forward_context={}
        ),
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda _: 8,
            get_num_kv_heads=lambda _: 8,
            get_head_size=lambda: 64,
            rswa_window=None,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1, decode_context_parallel_size=1
        ),
    )
    spec = FullAttentionSpec(
        block_size=128, num_kv_heads=8, head_size=64, dtype=torch.float32
    )
    names = ["draft.attn"]
    builder = TritonAttentionMetadataBuilder(spec, names, config, torch.device("cpu"))
    proposer = EagleProposer.__new__(EagleProposer)
    proposer.method = "eagle3"
    proposer.device = torch.device("cpu")
    proposer.vllm_config = config
    proposer.speculative_config = SimpleNamespace(disable_padded_drafter_batch=False)
    proposer.parallel_drafting = False
    proposer.constant_draft_positions = False
    proposer.needs_extra_input_slots = False
    proposer.supports_mm_inputs = False
    proposer.uses_mrope = False
    proposer.draft_model_config = SimpleNamespace(uses_mrope=False)
    proposer.num_speculative_tokens = 3
    proposer._draft_attn_layer_names = set(names)
    proposer.draft_attn_groups = [
        AttentionGroup(TritonAttentionBackend, names, spec, 0, [builder])
    ]
    proposer._draft_query_start_loc_cpu_cache = {}
    proposer.token_arange_np = np.arange(16, dtype=np.int32)
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 5], dtype=torch.int32),
        seq_lens=torch.tensor([10, 20], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=5,
        max_query_len=3,
        max_seq_len=20,
        block_table_tensor=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        slot_mapping=torch.tensor([9, 10, 275, 276, 277], dtype=torch.int64),
    )
    return proposer, builder, common


def _propose_until_decode(monkeypatch, proposer, builder, common, k=3):
    """Run propose through real metadata construction; omit GPU/model execution."""

    class DraftDecodeReady(Exception):
        pass

    class TestEagle(Eagle3LlamaForCausalLM):
        def __init__(self):
            torch.nn.Module.__init__(self)

        def combine_hidden_states(self, hidden_states):
            return hidden_states

        def forward(self, **kwargs):
            hidden = torch.zeros(common.num_actual_tokens, 2)
            return hidden, hidden

    proposer.model = TestEagle()
    proposer.hidden_size = 2
    proposer._share_mtp_indices = False
    proposer.eplb_state = None
    proposer.positions = torch.arange(16)
    proposer.arange = torch.arange(16, dtype=torch.int32)
    proposer.allowed_attn_types = None
    proposer.block_size = 128
    proposer.use_heterogeneous_vocab = False
    monkeypatch.setattr(
        llm_base_proposer, "set_forward_context", lambda *a, **kw: nullcontext()
    )
    monkeypatch.setattr(proposer, "_get_slot_mapping", lambda *a: None)
    monkeypatch.setattr(
        proposer,
        "_determine_batch_execution_and_padding",
        lambda n: (CUDAGraphMode.PIECEWISE, n, None),
    )
    monkeypatch.setattr(
        proposer,
        "set_inputs_first_pass",
        lambda **kw: (common.num_actual_tokens, torch.arange(common.num_reqs), common),
    )
    monkeypatch.setattr(
        proposer,
        "build_model_inputs_first_pass",
        lambda *a: ({}, common.num_actual_tokens),
    )
    monkeypatch.setattr(
        proposer,
        "_sample_draft_tokens",
        lambda *a: (torch.zeros(common.num_reqs, dtype=torch.int64), None),
    )

    def update_positions(positions, metadata, *args):
        metadata.seq_lens.add_(1)
        metadata.max_seq_len += 1
        return positions + 1

    monkeypatch.setattr(
        proposer, "_update_positions_dependent_metadata", update_positions
    )

    build = builder.build_for_drafting
    captured = []

    def capture(common_attn_metadata, draft_index):
        result = build(common_attn_metadata, draft_index)
        if draft_index == 1:
            captured.append(result)
            raise DraftDecodeReady
        return result

    with monkeypatch.context() as patch:
        patch.setattr(builder, "build_for_drafting", capture)
        with pytest.raises(DraftDecodeReady):
            proposer.propose(
                k,
                torch.zeros(common.num_reqs, dtype=torch.int32),
                torch.arange(common.num_reqs),
                torch.zeros(common.num_reqs, 2),
                torch.zeros(common.num_reqs, dtype=torch.int32),
                None,
                common,
                None,
            )
    metadata = captured[0]
    assert metadata.max_query_len == 1
    assert metadata.num_actual_tokens == common.num_reqs
    assert metadata.seq_lens is common.seq_lens
    assert metadata.max_seq_len == common.max_seq_len
    assert metadata.block_table is common.block_table_tensor
    assert metadata.slot_mapping is common.slot_mapping
    assert common.query_start_loc_cpu.tolist() == list(range(common.num_reqs + 1))
    return common.query_start_loc_cpu


def test_propose_reuses_cpu_query_offsets_across_batch_and_draft_length_changes(
    monkeypatch, draft_metadata
):
    proposer, builder, common = draft_metadata
    first = _propose_until_decode(monkeypatch, proposer, builder, common)
    smaller = common.replace(
        num_reqs=1, num_actual_tokens=1, seq_lens=common.seq_lens[:1]
    )
    _propose_until_decode(monkeypatch, proposer, builder, smaller, k=2)
    common.seq_lens = torch.tensor([40, 50], dtype=torch.int32)
    common.max_seq_len = 50
    common.slot_mapping = common.slot_mapping + 2
    second = _propose_until_decode(monkeypatch, proposer, builder, common, k=4)
    assert second is first
    assert common.seq_lens.tolist() == [41, 51]
    assert common.max_seq_len == 51
    proposer.token_arange_np[:] = -1
    assert first.tolist() == [0, 1, 2]


def test_propose_cpu_offsets_fall_back_for_unrecognized_builder(
    monkeypatch, draft_metadata
):
    proposer, builder, common = draft_metadata
    cached = _propose_until_decode(monkeypatch, proposer, builder, common)

    class OtherBuilder(TritonAttentionMetadataBuilder):
        pass

    builder.__class__ = OtherBuilder
    first = _propose_until_decode(monkeypatch, proposer, builder, common)
    first.fill_(-1)
    second = _propose_until_decode(monkeypatch, proposer, builder, common)
    assert second is not first
    assert second.tolist() == [0, 1, 2]
    assert cached.tolist() == [0, 1, 2]


@pytest.mark.parametrize("num_dims", [3, 4])
def test_cpu_query_offsets_mrope_fallback_has_private_storage(draft_metadata, num_dims):
    proposer, _, _ = draft_metadata
    proposer.draft_model_config = SimpleNamespace(
        uses_mrope=True, mrope_num_dims=num_dims
    )
    proposer.uses_mrope = proposer.draft_model_config.uses_mrope
    first = proposer._get_draft_query_start_loc_cpu(2)
    first.fill_(-1)
    assert proposer._get_draft_query_start_loc_cpu(2).tolist() == [0, 1, 2]


def test_cpu_query_offsets_released_on_backend_reinitialization(
    monkeypatch, draft_metadata
):
    proposer, builder, _ = draft_metadata
    first = proposer._get_draft_query_start_loc_cpu(2)
    monkeypatch.setattr(
        llm_base_proposer,
        "get_layers_from_vllm_config",
        lambda *a, **kw: {
            "draft.attn": SimpleNamespace(
                get_attn_backend=lambda: TritonAttentionBackend
            )
        },
    )
    kv_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["draft.attn"], kv_cache_spec=builder.kv_cache_spec
            )
        ]
    )
    proposer.initialize_attn_backend(kv_config)
    second = proposer._get_draft_query_start_loc_cpu(2)
    assert second is not first
    assert second.tolist() == [0, 1, 2]
