# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for Uno's MRV2 inputs and adapter ownership.

The proposer must form the same seed/noise suffix after rejection and request
reordering without advancing request state. Unit tests directly exercise its
preparation and adapter scope; model/sampler/graph numerics require GPU tests.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

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


def test_eager_draft_attn_metadata_keeps_k_row_physical_capacity(monkeypatch):
    """The eager draft must size attention metadata to the K query rows.

    Upstream sizes autoregressive draft metadata one query per request
    (`num_tokens_padded = num_reqs`). Uno prepares K queries per request, so the
    eager path must pass its physical K-row capacity and never inherit that
    one-query value.
    """
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    module = "vllm.v1.worker.gpu.spec_decode.uno"
    k = 4
    n = 3
    count = n * k
    proposer = object.__new__(UnoSpeculator)
    proposer.k = k
    proposer._step = 0
    proposer.num_graph_replays = 0
    proposer.num_eager_proposals = 0
    proposer.max_model_len = 64
    proposer.speculative_config = SimpleNamespace(
        uno_mask_token_id=1000, uno_noise_seed=42
    )
    proposer.input_buffers = InputBuffers(4, 32, torch.device("cpu"))
    proposer.sample_idx_mapping = torch.empty(32, dtype=torch.int32)
    proposer.draft_tokens = torch.empty((4, k), dtype=torch.int64)
    proposer.block_tables = SimpleNamespace(
        slot_mappings=torch.empty((1, 32), dtype=torch.int64),
        input_block_tables=[torch.ones((4, 8), dtype=torch.int32)],
        kernel_block_sizes=[4],
    )
    proposer.kv_cache_config = Mock()
    desc = BatchExecutionDescriptor(CUDAGraphMode.NONE, count, n)
    proposer.cudagraph_manager = Mock()
    proposer.cudagraph_manager.dispatch.return_value = desc
    proposer._copy_request_inputs = Mock()
    proposer._generate_draft = Mock()
    captured: dict = {}

    def fake_build(
        num_reqs,
        num_reqs_padded,
        num_tokens_padded,
        seq_lens_cpu_upper_bound,
        step,
        **kwargs,
    ):
        captured.update(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            num_tokens_padded=num_tokens_padded,
            step=step,
            **kwargs,
        )
        return {"eager": object()}

    proposer._build_draft_attn_metadata = fake_build
    proposer.set_lora_hook(lambda mapping: None)
    monkeypatch.setattr(f"{module}.prepare_uno_inputs_fused", Mock())
    monkeypatch.setattr(f"{module}.build_slot_mappings_by_layer", Mock(return_value={}))
    batch = SimpleNamespace(
        num_reqs=n,
        idx_mapping=torch.tensor([2, 0, 1]),
        seq_lens_cpu_upper_bound=torch.tensor([10, 20, 30]),
    )
    tensor = torch.empty(4)
    proposer.propose(
        batch, {}, {}, tensor, None, tensor, tensor, tensor, tensor, tensor, tensor
    )
    assert captured["num_tokens_padded"] == count
    assert captured["num_reqs_padded"] == n
    assert captured["step"] == k
    assert captured["num_query_per_req"] == k


def _cpu_uno_proposer(k: int) -> UnoSpeculator:
    """A real UnoSpeculator wired to a CPU CudaGraphManager (no CUDA/build)."""
    from vllm.config import (
        CompilationConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.v1.attention.backend import AttentionCGSupport

    compilation_config = CompilationConfig(
        cudagraph_mode="FULL_DECODE_ONLY",
        cudagraph_capture_sizes=[8, 16, 32, 64],
    )
    compilation_config.max_cudagraph_capture_size = 64
    compilation_config.post_init_cudagraph_sizes()

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.compilation_config = compilation_config
    vllm_config.scheduler_config = SchedulerConfig.default_factory(max_num_seqs=4)
    vllm_config.parallel_config = ParallelConfig()
    vllm_config.speculative_config = None
    vllm_config.num_speculative_tokens = 0

    proposer = object.__new__(UnoSpeculator)
    proposer.k = k
    proposer.vllm_config = vllm_config
    proposer.device = torch.device("cpu")
    proposer.attn_cg_support = SimpleNamespace(
        min_cg_support=AttentionCGSupport.UNIFORM_BATCH
    )
    proposer._graph_attn_metadata = {}
    proposer._step = 0
    proposer.num_graph_replays = 0
    proposer.num_eager_proposals = 0
    proposer.max_model_len = 64
    proposer.speculative_config = SimpleNamespace(
        uno_mask_token_id=1000, uno_noise_seed=42
    )
    proposer.input_buffers = InputBuffers(4, 32, torch.device("cpu"))
    proposer.sample_idx_mapping = torch.empty(32, dtype=torch.int32)
    proposer.draft_tokens = torch.empty((4, k), dtype=torch.int64)
    proposer.block_tables = SimpleNamespace(
        slot_mappings=torch.empty((1, 32), dtype=torch.int64),
        input_block_tables=[torch.ones((4, 8), dtype=torch.int32)],
        kernel_block_sizes=[4],
    )
    proposer.kv_cache_config = Mock()
    proposer._copy_request_inputs = Mock()
    proposer._build_draft_attn_metadata = Mock(return_value={"eager": object()})
    proposer._generate_draft = Mock()
    proposer.set_lora_hook(lambda mapping: None)
    proposer.attn_groups = [[Mock()]]
    return proposer


@pytest.mark.parametrize(
    ("k", "expected_active_loras"),
    [(1, 0), (8, 2)],
)
def test_draft_graph_engagement_follows_the_k_rule(
    k, expected_active_loras, monkeypatch
):
    """Draft graph engagement is derived, not assumed from K.

    The capture case follows ``2 if k > 1 else 0`` (adapter noise rows exist
    only for k > 1), but whether a captured decode graph can serve the K-row
    batch is the separate ``CudaGraphManager._init_candidates`` rule: a
    candidate is skipped when ``round_up(num_tokens, decode_query_len)``
    exceeds ``max_num_reqs * decode_query_len``, so a graph exists only when
    some ``cudagraph_capture_sizes`` entry is at most ``max_num_seqs * k``. At
    K=1 here that is ``min([8, 16, 32, 64]) > 4``, so drafting is eager. The
    expectation is computed from that arithmetic so a capture-size or
    max-num-seqs change moves the assertion instead of breaking it, while a
    future change to the ``2 if k > 1 else 0`` rule still fails on CPU.
    """
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu import cudagraph_utils as gpu_cudagraph_utils

    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    monkeypatch.setattr(gpu_cudagraph_utils, "get_offloader", lambda: Mock())

    module = "vllm.v1.worker.gpu.spec_decode.uno"
    proposer = _cpu_uno_proposer(k)
    proposer.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    manager = proposer.cudagraph_manager
    assert manager is not None
    assert manager.lora_capture_cases == [expected_active_loras], (
        f"K={k} must capture draft graphs for num_active_loras="
        f"{expected_active_loras}, got {manager.lora_capture_cases}"
    )

    capture_sizes = proposer.vllm_config.compilation_config.cudagraph_capture_sizes
    max_num_seqs = proposer.vllm_config.scheduler_config.max_num_seqs
    expect_graph = min(capture_sizes) <= max_num_seqs * k

    if expect_graph:
        manager._graphs_captured = True
        desc = manager.dispatch(4, 4 * k, k, expected_active_loras)
        assert desc.cg_mode == CUDAGraphMode.FULL, desc
        proposer._graph_attn_metadata[desc] = {"layer": object()}
        manager.graphs[desc] = Mock()
    else:
        assert not manager.needs_capture(), manager._capture_descs

    dispatched_loras: list[int] = []
    real_dispatch = manager.dispatch

    def recording_dispatch(*args, **kwargs):
        dispatched_loras.append(args[3])
        return real_dispatch(*args, **kwargs)

    manager.dispatch = recording_dispatch
    monkeypatch.setattr(f"{module}.prepare_uno_inputs_fused", Mock())
    monkeypatch.setattr(f"{module}.build_slot_mappings_by_layer", Mock(return_value={}))

    batch = SimpleNamespace(
        num_reqs=4,
        idx_mapping=torch.arange(4),
        seq_lens_cpu_upper_bound=torch.tensor([10, 20, 30, 40]),
    )
    tensor = torch.empty(4)
    proposer.propose(
        batch, {}, {}, tensor, None, tensor, tensor, tensor, tensor, tensor, tensor
    )

    assert dispatched_loras == [expected_active_loras], (
        f"K={k} must dispatch num_active_loras={expected_active_loras}, "
        f"got {dispatched_loras}"
    )
    if expect_graph:
        assert proposer.num_graph_replays == 1
        assert proposer.num_eager_proposals == 0
    else:
        assert proposer.num_eager_proposals == 1
        assert proposer.num_graph_replays == 0


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


def test_survivor_kv_budget_clears_the_engine_admission_floor():
    """The e2e survivor budget must clear vLLM's single-request floor.

    A `kv_cache_memory_bytes` below the floor makes `get_kv_cache_configs`
    raise before the engine starts, which is exactly how the survivor e2e test
    failed on the GB10. The floor here is the engine's own admission rule fed a
    full-attention spec built from the pinned Qwen3-8B geometry, so a change to
    the model pin, block size or context fails on CPU instead of only on a GPU.
    """
    from tests.v1.e2e.spec_decode import uno_kv_budget
    from vllm.v1.core.kv_cache_utils import check_enough_kv_cache_memory
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    block = uno_kv_budget.kv_bytes_per_block()

    def max_len_config(max_model_len: int):
        # The admission path under test reads only these fields; a real
        # ModelConfig here would resolve a default model over the hub.
        return SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=max_model_len),
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=None),
        )

    def qwen3_specs() -> dict:
        num_layers, num_kv_heads, head_dim = uno_kv_budget.qwen3_geometry()
        spec = FullAttentionSpec(
            block_size=uno_kv_budget.BLOCK_SIZE,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            dtype=torch.bfloat16,
        )
        return {f"model.layers.{index}.self_attn": spec for index in range(num_layers)}

    # Both e2e budgets sit at or above the floor and the engine admits them.
    for max_model_len, budget in (
        (uno_kv_budget.SURVIVOR_MAX_MODEL_LEN, uno_kv_budget.survivor_kv_budget()),
        (uno_kv_budget.MATRIX_MAX_MODEL_LEN, uno_kv_budget.MATRIX_KV_BUDGET_BYTES),
    ):
        floor = uno_kv_budget.engine_minimum_kv_bytes(max_model_len)
        assert budget >= floor, (
            f"KV budget {budget} B ({budget // block} blocks) is below the engine "
            f"floor {floor} B ({floor // block} blocks) at "
            f"max_model_len={max_model_len}"
        )
        check_enough_kv_cache_memory(
            max_len_config(max_model_len), qwen3_specs(), budget
        )

    # fix1 shipped 64 blocks at max_model_len=1024, one below that floor; the
    # engine refuses it. The inverted run mutates the budget helper to this
    # value and records exit 1.
    fix1_budget = 64 * block
    survivor_floor = uno_kv_budget.engine_minimum_kv_bytes(
        uno_kv_budget.SURVIVOR_MAX_MODEL_LEN
    )
    assert fix1_budget < survivor_floor, (
        "the fix1 64-block budget unexpectedly clears the engine floor; the "
        "inverted run no longer proves the guard fires"
    )
    with pytest.raises(ValueError, match="To serve at least one request"):
        check_enough_kv_cache_memory(
            max_len_config(uno_kv_budget.SURVIVOR_MAX_MODEL_LEN),
            qwen3_specs(),
            fix1_budget,
        )


def test_survivor_preemption_arithmetic_fits_then_overflows_the_budget():
    """The survivor window admits four prompts but their growth exceeds it.

    The e2e test derives these token counts from the tokenizer at runtime; here
    they are pinned to the same prompt shapes at the pinned revision, so the
    preemption geometry is provable on CPU. The finish-peer cap is read from
    ``uno_kv_budget.SURVIVOR_FINISH_MAX_TOKENS`` so the CPU pin and the e2e
    cannot drift. ``mixed_growth_blocks`` counts the warmed shared prefix once;
    an earlier per-request sum over-estimated the resident set, so the gate
    passed its inequality while the engine never preempted. The inverted run
    swaps max_tokens back to [96, 4, 4, 2] and must fail the growth assertion.

    The compared request is one of the two long peers, so the crossing point
    (when the pair alone outgrows the budget) must sit below the cap: above it a
    peer could finish naturally before preemption and the resume path would
    never run. An inverted run with the cap at 448 fails this assertion.
    """
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    shared_prefix_tokens = 161
    prompt_tokens = [171, 265, 265, 249]  # seed, finish-0, finish-1, abort
    max_tokens = [
        96,
        budget.SURVIVOR_FINISH_MAX_TOKENS,
        budget.SURVIVOR_FINISH_MAX_TOKENS,
        64,
    ]
    budget_blocks = budget.survivor_kv_budget() // budget.kv_bytes_per_block()

    admission = budget.mixed_admission_blocks(prompt_tokens, shared_prefix_tokens)
    growth = budget.mixed_growth_blocks(prompt_tokens, max_tokens, shared_prefix_tokens)
    crossing = budget.mixed_crossing_tokens(
        prompt_tokens[1], shared_prefix_tokens, budget_blocks
    )

    assert admission < budget_blocks, (
        f"all four prompts must be admitted together: {admission} blocks of "
        f"unique prompt footprint (incl. one decode block each) vs budget "
        f"{budget_blocks}"
    )
    assert growth > budget_blocks, (
        f"the mixed phase must exhaust the budget by growth: {growth} blocks if "
        f"every request reached its cap vs budget {budget_blocks}"
    )
    assert crossing < budget.SURVIVOR_FINISH_MAX_TOKENS, (
        f"the two long peers alone cross the budget only at {crossing} generated "
        f"tokens, at or past the {budget.SURVIVOR_FINISH_MAX_TOKENS} cap, so a "
        "peer could finish before the scheduler preempts it"
    )
    assert all(
        tokens + cap < budget.SURVIVOR_MAX_MODEL_LEN
        for tokens, cap in zip(prompt_tokens, max_tokens)
    ), (prompt_tokens, max_tokens)

    # The old peer cap finished before growth could empty the budget; the same
    # arithmetic must not clear the gate, which is why the gate was vacuous.
    old_growth = budget.mixed_growth_blocks(
        prompt_tokens, [96, 4, 4, 2], shared_prefix_tokens
    )
    assert old_growth <= budget_blocks, (
        f"the old max_tokens=4 peers ({old_growth} blocks) unexpectedly clear "
        f"the growth gate ({budget_blocks} blocks); the inverted run would not "
        "fire"
    )
