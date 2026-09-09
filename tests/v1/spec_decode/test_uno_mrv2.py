# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for Uno's MRV2 inputs and adapter ownership.

The proposer must form the same seed/noise suffix after rejection and request
reordering without advancing request state. Unit tests directly exercise its
preparation and adapter scope; model/sampler/graph numerics require GPU tests.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.uno import (
    UNO_LORA_ID,
    UnoSpeculator,
    prepare_uno_inputs_reference,
)
from vllm.v1.worker.gpu.spec_decode.uno_lora import draft_lora_mapping


@pytest.mark.parametrize("k", [1, 4, 8])
def test_noise_mapping_excludes_seed_and_padding(k):
    mapping = draft_lora_mapping(3 * k, 2, k, UNO_LORA_ID)
    assert mapping[:k] == (0,) + (UNO_LORA_ID,) * (k - 1)
    assert mapping[k : 2 * k] == mapping[:k]
    assert mapping[2 * k :] == (0,) * k


@pytest.mark.parametrize("args", [(-1, 4, 4), (1, 0, 4), (2, 4, 7)])
def test_query_capacity_rejected(args):
    with pytest.raises(ValueError):
        draft_lora_mapping(args[2], args[0], args[1], UNO_LORA_ID)


def test_rejection_and_prefill_use_correct_seed_and_persistent_slot():
    buffers = InputBuffers(4, 16, torch.device("cpu"))
    slots = torch.full((16,), 99, dtype=torch.int64)
    sample_slots = torch.full((16,), 99, dtype=torch.int32)
    batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping=torch.tensor([3, 1]),
        query_start_loc=torch.tensor([0, 3, 6]),
        positions=torch.tensor([2, 3, 4, 9, 10, 11]),
    )
    last_sampled = torch.tensor([[100], [101], [102], [103]])
    prefill_tokens = torch.tensor([[200, 201, 202, 203]])
    seeds = torch.tensor([10, 11, 12, 13])
    block_table = torch.tensor([[2, 3, 4, 5], [8, 9, 10, 11]])
    prepare_uno_inputs_reference(
        buffers,
        slots,
        sample_slots,
        batch,
        torch.tensor([1, 0]),
        torch.tensor([0, 2]),
        last_sampled,
        prefill_tokens,
        seeds,
        block_table,
        4,
        4,
        16,
        42,
        1000,
        1,
    )
    assert buffers.input_ids[[0, 4]].tolist() == [103, 201]
    assert buffers.positions[:8].tolist() == [5, 6, 7, 8, 10, 11, 12, 13]
    assert slots.tolist() == [13, 14, 15, 16, 42, 43, 44, 45] + [-1] * 8
    assert buffers.seq_lens.tolist() == [9, 14, 0, 0]
    assert buffers.query_start_loc.tolist() == [0, 4, 8, 8, 8]
    assert sample_slots.tolist() == [3] * 4 + [1] * 4 + [-1] * 8
    noise = buffers.input_ids[[1, 2, 3, 5, 6, 7]]
    assert ((noise >= 1) & (noise < 1000)).all()
    assert last_sampled.tolist() == [[100], [101], [102], [103]]
    assert batch.positions.tolist() == [2, 3, 4, 9, 10, 11]


@pytest.mark.parametrize(
    "block_ids,max_len,expected",
    [
        ([2, 3, 4], 10, [15, 16, 17, -1]),
        ([2, 3, 0], 12, [15, -1, -1, -1]),
        ([2, 3], 12, [15, -1, -1, -1]),
    ],
)
def test_draft_does_not_write_null_or_unallocated_or_out_of_context_slots(
    block_ids, max_len, expected
):
    buffers = InputBuffers(2, 8, torch.device("cpu"))
    slots = torch.empty(8, dtype=torch.int64)
    sample_slots = torch.empty(8, dtype=torch.int32)
    batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping=torch.tensor([1]),
        query_start_loc=torch.tensor([0, 1]),
        positions=torch.tensor([6]),
    )
    prepare_uno_inputs_reference(
        buffers,
        slots,
        sample_slots,
        batch,
        torch.tensor([1]),
        torch.tensor([0]),
        torch.tensor([[100], [101]]),
        torch.tensor([[200, 201]]),
        torch.tensor([10, 11]),
        torch.tensor([block_ids]),
        4,
        4,
        max_len,
        42,
        1000,
        1,
    )
    assert slots.tolist() == expected + [-1] * 4
    assert buffers.positions[:4].max() < max_len
    assert buffers.seq_lens[0] <= max_len


@pytest.mark.parametrize("fail_at", [None, "routing", "forward"])
def test_adapter_scope_restores_base_on_success_and_failure(fail_at):
    proposer = object.__new__(UnoSpeculator)
    proposer.k = 4
    history = []

    def hook(mapping):
        history.append(mapping)
        if mapping is not None and fail_at == "routing":
            raise RuntimeError("routing failed")

    proposer.set_lora_hook(hook)
    if fail_at:
        with pytest.raises(RuntimeError), proposer._draft_lora(1, 8):
            raise RuntimeError("forward failed")
    else:
        with proposer._draft_lora(1, 8):
            pass
    assert history == [(1, 8), None]


def test_native_sampling_handoff_uses_persistent_slots_and_columns(monkeypatch):
    proposer = object.__new__(UnoSpeculator)
    proposer.k = 2
    proposer.input_buffers = InputBuffers(2, 4, torch.device("cpu"))
    proposer.input_buffers.positions[:] = torch.tensor([4, 5, 8, 9])
    proposer.sample_idx_mapping = torch.tensor([3, 3, 1, 1])
    proposer.sample_col = torch.tensor([0, 1, 0, 1])
    proposer.temperature = torch.ones(4)
    proposer.seeds = torch.arange(4)
    proposer.draft_logits = torch.empty(4, 2, 7)
    proposer.draft_tokens = torch.zeros(2, 2, dtype=torch.int64)
    proposer.model = Mock(return_value=torch.randn(4, 8))
    proposer.sample_draft = Mock(return_value=torch.tensor([11, 12, 13, 14]))
    proposer.vllm_config = Mock()
    from contextlib import nullcontext

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.uno.set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )
    proposer._generate_draft(2, 4, None, {})
    args = proposer.sample_draft.call_args.args
    assert args[1].tolist() == [4, 5, 8, 9]
    assert args[2].tolist() == [3, 3, 1, 1]
    assert args[5].tolist() == [0, 1, 0, 1]
    assert args[6] is proposer.draft_logits
    assert proposer.draft_tokens.tolist() == [[11, 12], [13, 14]]


@pytest.mark.parametrize("full_graph", [False, True])
def test_graph_replay_refreshes_native_backend_without_rebuilding_metadata(
    monkeypatch, full_graph
):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    module = "vllm.v1.worker.gpu.spec_decode.uno"
    proposer = object.__new__(UnoSpeculator)
    proposer.k = 2
    proposer._step = 0
    proposer.num_graph_replays = 0
    proposer.num_eager_proposals = 0
    proposer.max_model_len = 32
    proposer.speculative_config = SimpleNamespace(
        uno_mask_token_id=1000, uno_noise_seed=42
    )
    proposer.input_buffers = InputBuffers(4, 8, torch.device("cpu"))
    proposer.sample_idx_mapping = torch.empty(8, dtype=torch.int32)
    proposer.draft_tokens = torch.empty((4, 2), dtype=torch.int64)
    proposer.block_tables = SimpleNamespace(
        slot_mappings=torch.empty((1, 8), dtype=torch.int64),
        input_block_tables=[torch.ones((4, 8), dtype=torch.int32)],
        kernel_block_sizes=[4],
    )
    proposer.kv_cache_config = Mock()
    desc = BatchExecutionDescriptor(
        CUDAGraphMode.FULL if full_graph else CUDAGraphMode.NONE, 8, 4
    )
    events = []
    captured_attn = {"layer": object()}
    proposer._graph_attn_metadata = {desc: captured_attn}
    group = Mock()
    group.update_draft_decode_metadata.side_effect = lambda metadata: events.append(
        ("refresh", metadata)
    )
    proposer.attn_groups = [[group]]
    proposer.cudagraph_manager = Mock()
    proposer.cudagraph_manager.dispatch.return_value = desc
    proposer.cudagraph_manager.run_fullgraph.side_effect = lambda _: events.append(
        ("replay", None)
    )
    proposer._copy_request_inputs = Mock()
    proposer._build_draft_attn_metadata = Mock(return_value={"eager": object()})
    proposer._generate_draft = Mock()
    proposer.set_lora_hook(lambda mapping: events.append(("lora", mapping)))
    fused_prepare = Mock()
    slot_builder = Mock(return_value={})
    monkeypatch.setattr(f"{module}.prepare_uno_inputs_fused", fused_prepare)
    monkeypatch.setattr(f"{module}.build_slot_mappings_by_layer", slot_builder)
    batch = SimpleNamespace(num_reqs=3, idx_mapping=torch.tensor([2, 0, 1]))
    # A full replay must not read or construct eager CPU length metadata.
    if not full_graph:
        batch.seq_lens_cpu_upper_bound = torch.tensor([10, 20, 30])
    tensor = torch.empty(4)
    proposer.propose(
        batch, {}, {}, tensor, None, tensor, tensor, tensor, tensor, tensor, tensor
    )
    fused_prepare.assert_called_once()
    assert events[0] == ("lora", (3, 8))
    assert events[-1] == ("lora", None)
    if full_graph:
        assert events[1:3] == [("refresh", captured_attn), ("replay", None)]
        proposer._build_draft_attn_metadata.assert_not_called()
        slot_builder.assert_not_called()
        proposer._generate_draft.assert_not_called()
        assert proposer.num_graph_replays == 1
    else:
        group.update_draft_decode_metadata.assert_not_called()
        proposer.cudagraph_manager.run_fullgraph.assert_not_called()
        proposer._build_draft_attn_metadata.assert_called_once()
        slot_builder.assert_called_once()
        proposer._generate_draft.assert_called_once()
        assert proposer.draft_max_seq_len == 32
        assert proposer.num_eager_proposals == 1


@pytest.mark.parametrize("supports_update", [False, True])
def test_uno_rejects_builder_without_native_decode_update_at_initialization(
    monkeypatch, supports_update
):
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    proposer = object.__new__(UnoSpeculator)
    group = SimpleNamespace(supports_draft_decode_metadata_update=supports_update)

    def install_groups(self, *args):
        self.attn_groups = [[group]]

    monkeypatch.setattr(DraftModelSpeculator, "set_attn", install_groups)
    cache = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=FullAttentionSpec(
                    block_size=4,
                    num_kv_heads=1,
                    head_size=16,
                    dtype=torch.bfloat16,
                )
            )
        ]
    )
    if supports_update:
        proposer.set_attn(None, cache, None, None, [])
        assert proposer.attn_groups == [[group]]
    else:
        with pytest.raises(ValueError, match="native draft decode updates"):
            proposer.set_attn(None, cache, None, None, [])
