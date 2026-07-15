# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

from vllm.config.compilation import CUDAGraphMode  # noqa: E402
from vllm.v1.attention.backend import (  # noqa: E402
    AttentionCGSupport,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mla.indexer import (  # noqa: E402
    DeepseekV32IndexerMetadataBuilder,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # noqa: E402
from vllm.v1.worker.gpu.cudagraph_utils import (  # noqa: E402
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.input_batch import (  # noqa: E402
    InputBatch,
    prepare_pos_seq_lens,
)
from vllm.v1.worker.gpu.spec_decode.confidence import (  # noqa: E402
    MaskedConfidenceManager,
    VarlenConfidenceManager,
    build_cost_tables_from_curves,
    compute_prefix_survival,
    make_confidence_manager,
)
from vllm.v1.worker.gpu.spec_decode.dspark.online_sts import (  # noqa: E402
    DSparkOnlineSTS,
)
from vllm.v1.worker.gpu.states import RequestState  # noqa: E402


def _spec_config(**overrides: Any) -> SimpleNamespace:
    config = SimpleNamespace(
        dspark_budget_frac=1.0,
        dspark_sps_curve=None,
        dspark_online_sts=False,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _logit(probs: np.ndarray) -> np.ndarray:
    return np.log(probs / (1.0 - probs))


def _plan_budget(
    survival: np.ndarray,
    num_sampling_requests: int,
    num_required_target_tokens: int,
    budget_frac: float = 1.0,
    cost_tables: tuple[np.ndarray, np.ndarray] | None = None,
) -> int:
    num_reqs, num_steps = survival.shape
    req_ids = [f"req{i}" for i in range(num_reqs)]
    required = np.full(num_reqs, num_required_target_tokens // num_reqs, dtype=np.int32)
    required[: num_required_target_tokens % num_reqs] += 1
    prefill_lens = required.copy()
    prefill_lens[num_sampling_requests:] += 1
    manager = object.__new__(VarlenConfidenceManager)
    manager.num_speculative_steps = num_steps
    manager.req_states = SimpleNamespace(
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        num_computed_tokens_np=np.zeros(num_reqs, dtype=np.int32),
        prefill_len=SimpleNamespace(np=prefill_lens),
    )
    manager._stale_confidences = (req_ids, survival)
    manager.budget_frac = budget_frac
    manager.cost_tables = cost_tables
    return manager._plan_draft_token_budget(
        req_ids,
        required,
        np.full(num_reqs, num_steps, dtype=np.int32),
    )


@pytest.fixture(autouse=True)
def _single_rank_tp_group(monkeypatch):
    group = SimpleNamespace(
        broadcast_object=lambda value, src=0: value,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.confidence.get_tp_group", lambda: group
    )


@pytest.mark.parametrize(
    ("cg_support", "expected_type"),
    [
        (AttentionCGSupport.ALWAYS, VarlenConfidenceManager),
        (AttentionCGSupport.UNIFORM_BATCH, MaskedConfidenceManager),
    ],
)
def test_auto_verification_selects_supported_manager(cg_support, expected_type):
    device = torch.device("cuda")
    req_states = RequestState(
        max_num_reqs=2,
        max_model_len=4,
        max_num_batched_tokens=8,
        num_speculative_steps=2,
        vocab_size=16,
        device=device,
    )
    attn_cg_support = SimpleNamespace(
        min_cg_support=cg_support,
        min_cg_attn_backend="test",
    )

    manager = make_confidence_manager(
        "auto",
        attn_cg_support,
        req_states=req_states,
        speculative_config=_spec_config(),
    )

    assert isinstance(manager, expected_type)


def test_masked_verification_does_not_time_cudagraphs():
    device = torch.device("cuda")
    manager = MaskedConfidenceManager(
        req_states=RequestState(
            max_num_reqs=2,
            max_model_len=4,
            max_num_batched_tokens=8,
            num_speculative_steps=2,
            vocab_size=16,
            device=device,
        ),
        speculative_config=_spec_config(dspark_sps_curve="auto"),
    )

    assert not manager.time_graphs
    assert manager.cost_tables is None


def test_prepare_pos_seq_lens_clears_active_padding():
    device = torch.device("cuda")
    is_padding = torch.ones(7, dtype=torch.bool, device=device)
    prepare_pos_seq_lens(
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.tensor([0, 2, 5], dtype=torch.int32, device=device),
        torch.zeros(2, dtype=torch.int32, device=device),
        torch.empty(7, dtype=torch.int64, device=device),
        torch.empty(4, dtype=torch.int32, device=device),
        is_padding=is_padding,
    )

    assert is_padding.cpu().tolist() == [False] * 5 + [True] * 2


def test_token_to_request_mapping_uses_device_offsets():
    device = torch.device("cuda")
    metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2, 5, -1], dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.tensor([0, 3, 5], dtype=torch.int32),
        seq_lens=torch.tensor([2, 3], dtype=torch.int32, device=device),
        num_reqs=2,
        num_actual_tokens=5,
        max_query_len=3,
        max_seq_len=3,
        block_table_tensor=torch.empty((2, 0), dtype=torch.int32, device=device),
        slot_mapping=torch.empty(5, dtype=torch.int64, device=device),
    )

    mapping = metadata.token_to_req_indices(
        torch.empty(5, dtype=torch.int32, device=device)
    )

    assert mapping.cpu().tolist() == [0, 0, 1, 1, 1]


def test_token_to_request_mapping_ignores_cudagraph_token_padding():
    device = torch.device("cuda")
    metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2, 8, 5], dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.tensor([0, 3, 5, 5], dtype=torch.int32),
        seq_lens=torch.tensor([2, 3, 0], dtype=torch.int32, device=device),
        num_reqs=3,
        num_actual_tokens=8,
        max_query_len=3,
        max_seq_len=3,
        block_table_tensor=torch.empty((3, 0), dtype=torch.int32, device=device),
        slot_mapping=torch.empty(8, dtype=torch.int64, device=device),
    )

    mapping = metadata.token_to_req_indices(
        torch.empty(8, dtype=torch.int32, device=device)
    )

    assert mapping.cpu().tolist() == [0, 0, 1, 1, 1, 0, 0, 0]


def test_varlen_indexer_indices_use_device_lengths():
    device = torch.device("cuda")
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.decode_indices_buffer = torch.empty(5, dtype=torch.int32, device=device)
    builder.arange_buffer = torch.arange(5, dtype=torch.int32, device=device)

    indices = builder._build_varlen_decode_indices(
        decode_lens=torch.tensor([1, 2], dtype=torch.int32, device=device),
        decode_lens_cpu=torch.tensor([2, 1], dtype=torch.int32),
        num_decode_tokens=5,
    )

    assert indices.cpu().tolist() == [0, 1, 1, 2, 3]


def test_select_budget_applies_fraction_globally():
    survival = compute_prefix_survival(_logit(np.array([[0.90, 0.80], [0.80, 0.80]])))
    budget = _plan_budget(survival, 2, 2, budget_frac=0.5)
    assert budget == 2


def test_select_budget_is_hard_cap_under_ties():
    survival = compute_prefix_survival(_logit(np.array([[0.90, 0.80], [0.90, 0.80]])))
    budget = _plan_budget(survival, 2, 2, budget_frac=0.25)
    assert budget == 1


def test_select_budget_is_hard_cap_for_saturated_scores():
    """Saturated (tied) survival scores must not escape the budget.

    With every confidence saturated to 1.0, a kth-score-threshold recount
    would admit all tokens; the budget must remain a hard cap on the total.
    """
    survival = compute_prefix_survival(np.full((4, 7), 40.0))
    budget = _plan_budget(survival, 4, 4, budget_frac=0.5)
    assert budget == int(4 * 7 * 0.5)


def test_select_budget_never_admits_zero_survival():
    """Zero-survival tokens are not candidates (DSpark Alg. 1: a_{r,j} > 0),
    so leftover budget must not be spent past the first dead position."""
    # Positions 0-1 confident, position 2 dead (sigmoid underflows to an
    # exact zero) -> survival is exactly 0 from there on.
    logits = np.full((2, 5), 40.0)
    logits[:, 2] = -800.0
    survival = compute_prefix_survival(logits)
    budget = _plan_budget(survival, 2, 2, budget_frac=0.9)
    assert budget == 4


def test_select_budget_uses_sps_argmax():
    """With an SPS curve, verification lengths maximize tau * SPS(B)
    (DSpark Alg. 1) instead of spending the whole budget."""
    survival = compute_prefix_survival(_logit(np.array([[0.90, 0.80], [0.60, 0.50]])))
    # Survival: r0 [0.9, 0.72], r1 [0.6, 0.3]; admission order
    # 0.9, 0.72, 0.6, 0.3 with B = 2 + k.
    # SPS drops sharply after B=4 so theta peaks at k=2:
    #   k=0: 2.00*1.00, k=1: 2.90*0.95=2.755, k=2: 3.62*0.90=3.258,
    #   k=3: 4.22*0.20=0.844, k=4: 4.52*0.10=0.452.
    rates = np.array([1.0, 1.0, 1.0, 0.95, 0.90, 0.20, 0.10])
    budget = _plan_budget(
        survival,
        num_sampling_requests=2,
        num_required_target_tokens=2,
        cost_tables=(np.zeros(3), 1000.0 / rates),
    )
    assert budget == 2


def test_select_budget_counts_only_requests_that_sample():
    survival = np.array([[0.9, 0.8], [0.7, 0.6]])
    rates = np.ones(9)
    rates[5:] = 0.1

    budget = _plan_budget(
        survival,
        num_sampling_requests=1,
        num_required_target_tokens=4,
        cost_tables=(np.zeros(3), 1000.0 / rates),
    )

    assert budget == 0


def test_cost_tables_keep_draft_and_verification_costs_separate():
    draft_curve = [(1, 0.2), (4, 0.8), (8, 1.6)]
    verify_curve = [(1, 2.0), (8, 3.0), (64, 5.0)]

    draft_cost_ms, verify_cost_ms = build_cost_tables_from_curves(
        draft_curve, verify_curve, max_num_reqs=10, max_batch_tokens=66
    )

    assert draft_cost_ms[2] == pytest.approx(0.8)
    assert verify_cost_ms[2] == pytest.approx(3.0)
    assert draft_cost_ms[9] == pytest.approx(1.8)
    assert verify_cost_ms[65] == pytest.approx(5.0 + 2.0 / 56.0)


def test_cudagraph_manager_optionally_times_replays(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.graph_capture",
        lambda device: nullcontext(),
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.is_global_first_rank", lambda: False
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.cudagraph_utils.get_offloader",
        lambda: SimpleNamespace(
            sync_prev_onload=lambda: None,
            join_after_forward=lambda: None,
        ),
    )

    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 1, 1)
    manager = object.__new__(CudaGraphManager)
    manager.device = torch.device("cuda")
    manager._capture_descs = {CUDAGraphMode.FULL: [desc]}
    manager.graphs = {}
    manager.pool = None
    manager.time_graphs = True
    manager.graph_timings = {}
    manager._graphs_captured = False
    value = torch.zeros((), device="cuda")

    def create_forward_fn(desc, warmup):
        return lambda mode: value.add_(1)

    def prepare_timing(desc):
        value.zero_()

    manager.capture(create_forward_fn, prepare_timing=prepare_timing)
    torch.accelerator.synchronize()

    assert value.item() == 3
    assert manager.graph_timings[desc] >= 0.0


def test_graph_costs_use_rank_zero_curves_on_every_tp_rank(monkeypatch):
    canonical_curves = ([(1, 2.0)], [(1, 3.0), (2, 4.0)])
    broadcasts = []

    def broadcast_object(value, src=0):
        broadcasts.append((value, src))
        return canonical_curves

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.confidence.get_tp_group",
        lambda: SimpleNamespace(broadcast_object=broadcast_object),
    )

    manager = object.__new__(VarlenConfidenceManager)
    manager.req_states = SimpleNamespace(
        max_num_reqs=1,
        max_num_batched_tokens=2,
    )
    manager.online_sts = None
    speculator = SimpleNamespace(
        query_cudagraph_manager=SimpleNamespace(graph_timings={})
    )
    model_graphs = SimpleNamespace(graph_timings={})

    manager.set_graph_costs(model_graphs, speculator)

    expected = build_cost_tables_from_curves(*canonical_curves, 1, 2)
    assert broadcasts == [(([], []), 0)]
    assert manager.cost_tables is not None
    assert np.array_equal(manager.cost_tables[0], expected[0])
    assert np.array_equal(manager.cost_tables[1], expected[1])


def test_select_budget_temperature_desaturates_zeros():
    """A confidence temperature > 1 keeps saturated-negative positions in the
    candidate set (no exact-zero survival), so the budget is spent instead of
    being truncated by miscalibrated zeros."""
    logits = np.full((2, 5), 40.0)
    logits[:, 2] = -800.0
    budget_t1 = _plan_budget(compute_prefix_survival(logits), 2, 2, budget_frac=0.9)
    budget_t80 = _plan_budget(
        compute_prefix_survival(logits, 80.0), 2, 2, budget_frac=0.9
    )
    # T=1: exact-zero survival past position 2 truncates both requests.
    assert budget_t1 == 4
    # T=80: sigmoid(-10) > 0, so 90% of the candidates are admitted.
    assert budget_t80 == 9


def test_online_sts_fits_order_preserving_temperatures():
    """Online STS fits per-position temperatures from rejection-sampler
    outcomes: identity before data, softens over-confident positions,
    sharpens under-confident ones, and never reorders candidates."""
    sts = DSparkOnlineSTS(num_steps=3, device=torch.device("cuda"))

    # Cold start: identity calibration.
    assert np.array_equal(sts.temperatures, np.ones(3))

    # Head claims p~0.88 everywhere (logit 2.0).
    logits = np.full((2, 3), 2.0)
    # Alternate outcomes so pos0 accepts 50% (head over-confident there)
    # while pos1/pos2 always accept once reached (head under-confident).
    acc_hi = np.array([3, 3])
    acc_lo = np.array([0, 0])
    ver = np.array([3, 3])
    for _ in range(1000):
        sts.update(logits, acc_hi, ver)
        sts.update(logits, acc_lo, ver)

    temps = sts.temperatures
    calibrated = 1.0 / (1.0 + np.exp(-logits[0] / temps))
    # pos0 empirical 0.5 vs raw 0.88: temperature must soften (T >> 1,
    # pushed toward the grid edge since sigmoid(2/T) -> 0.5+).
    assert temps[0] > 2.0
    assert calibrated[0] < 0.65
    # pos1/pos2 empirical 1.0 (conditioned on the prefix surviving):
    # temperature sharpens (T < 1).
    assert temps[1] < 1.0 and temps[2] < 1.0
    assert calibrated[1] > 0.9

    # Order preservation within every position, regardless of fit.
    lo = 0.5 / temps
    hi = 3.0 / temps
    assert (hi > lo).all()

    sts.reset()
    assert np.array_equal(sts.temperatures, np.ones(3))
    assert sts.copy_temperatures_to_gpu().cpu().tolist() == [1.0, 1.0, 1.0]


def test_capacity_manager_assigns_budget_on_gpu():
    device = torch.device("cuda")
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=4,
        max_num_batched_tokens=16,
        num_speculative_steps=3,
        vocab_size=32,
        device=device,
    )
    req_states.req_id_to_index = {"req0": 0, "req1": 1}
    handler = VarlenConfidenceManager(
        req_states=req_states,
        speculative_config=_spec_config(),
    )
    input_batch: Any = SimpleNamespace(
        req_ids=["req0", "req1"],
        num_reqs=2,
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32, device=device),
        cu_num_logits_np=np.array([0, 4, 8], dtype=np.int32),
        num_draft_tokens=6,
    )
    current_logits = torch.from_numpy(
        _logit(np.array([[0.90, 0.80, 0.70], [0.95, 0.40, 0.30]]))
    ).to(
        device=device,
        dtype=torch.float32,
    )
    handler._confidence_logits[:2].copy_(current_logits)

    torch.cuda.set_sync_debug_mode(2)
    try:
        capacities = handler._assign_draft_token_budget(
            input_batch,
            valid_draft_tokens_per_req=np.array([3, 3], dtype=np.int32),
            draft_token_budget=3,
        )
    finally:
        torch.cuda.set_sync_debug_mode(0)

    assert capacities.cpu().tolist() == [2, 1]

    handler._confidence_logits[:2].zero_()
    assert handler._assign_draft_token_budget(
        input_batch,
        valid_draft_tokens_per_req=np.array([3, 3], dtype=np.int32),
        draft_token_budget=1,
    ).cpu().tolist() == [1, 0]
    assert handler._assign_draft_token_budget(
        input_batch,
        valid_draft_tokens_per_req=np.array([0, 3], dtype=np.int32),
        draft_token_budget=1,
    ).cpu().tolist() == [0, 1]
    handler._confidence_logits[:2].copy_(current_logits)

    for _ in range(3):
        handler.stage_confidences(
            current_logits,
            torch.tensor([2, 1], dtype=torch.int32, device=device),
            input_batch,
        )
    assert handler._stale_confidences is not None
    assert handler._assign_draft_token_budget(
        input_batch,
        valid_draft_tokens_per_req=np.array([3, 3], dtype=np.int32),
        draft_token_budget=6,
    ).cpu().tolist() == [3, 3]
    assert handler._assign_draft_token_budget(
        input_batch,
        valid_draft_tokens_per_req=np.array([3, 3], dtype=np.int32),
        draft_token_budget=0,
    ).cpu().tolist() == [0, 0]
    handler.warmup()
    assert handler._stale_confidences is None


def test_varlen_capacity_manager_compacts_verifier_batch():
    device = torch.device("cuda")
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=8,
        max_num_batched_tokens=16,
        num_speculative_steps=3,
        vocab_size=32,
        device=device,
    )
    req_states.last_sampled_tokens[:2] = torch.tensor(
        [[101], [201]], dtype=torch.int64, device=device
    )
    req_states.draft_tokens[:2] = torch.tensor(
        [[11, 12, 13], [21, 22, 23]], dtype=torch.int64, device=device
    )
    req_states.req_id_to_index = {"req0": 0, "req1": 1}
    handler = VarlenConfidenceManager(
        req_states=req_states,
        speculative_config=_spec_config(dspark_budget_frac=0.34),
    )
    handler._confidence_logits[:2].copy_(
        torch.from_numpy(_logit(np.array([[0.90, 0.10, 0.10], [0.95, 0.95, 0.10]]))).to(
            device=device, dtype=torch.float32
        )
    )
    draft_tokens = {"req0": [-1, -1, -1], "req1": [-1, -1, -1]}
    assert handler.get_num_tokens({"req0": 4, "req1": 4}, draft_tokens) == 5

    input_ids = torch.tensor(
        [101, 11, 12, 13, 201, 21, 22, 23, 0, 0],
        dtype=torch.int32,
        device=device,
    )
    positions = torch.tensor(
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 0],
        dtype=torch.int64,
        device=device,
    )
    is_padding = torch.zeros(10, dtype=torch.bool, device=device)
    num_scheduled_tokens = np.array([4, 4], dtype=np.int32)
    query_start_loc_np = np.array([0, 4, 8, 8, 8], dtype=np.int32)
    cu_num_logits_np = np.array([0, 4, 8], dtype=np.int32)
    input_batch = InputBatch(
        req_ids=["req0", "req1"],
        num_reqs=2,
        num_reqs_after_padding=4,
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32, device=device),
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        expanded_idx_mapping=torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=device
        ),
        expanded_local_pos=torch.tensor(
            [0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32, device=device
        ),
        num_scheduled_tokens=num_scheduled_tokens,
        num_tokens=8,
        num_tokens_after_padding=5,
        num_draft_tokens=6,
        num_draft_tokens_per_req=np.array([3, 3], dtype=np.int32),
        query_start_loc=torch.tensor([0, 4, 8, 8, 8], dtype=torch.int32, device=device),
        query_start_loc_np=query_start_loc_np,
        seq_lens=torch.zeros(4, dtype=torch.int32, device=device),
        seq_lens_cpu_upper_bound=torch.tensor([4, 4, 0, 0], dtype=torch.int32),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=np.array([0, 0], dtype=np.int32),
        prefill_len_np=np.array([0, 0], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0, 0], dtype=np.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
        max_seq_len_np=None,
        input_ids=input_ids[:5],
        positions=positions[:5],
        is_padding=is_padding[:5],
        logits_indices=torch.arange(8, dtype=torch.int64, device=device),
        cu_num_logits=torch.tensor([0, 4, 8], dtype=torch.int32, device=device),
        cu_num_logits_np=cu_num_logits_np,
        has_structured_output_reqs=False,
        prompt_lens=None,
    )

    handler.trim_batch(input_batch, draft_tokens)

    torch.accelerator.synchronize()
    # CPU metadata carries only the scalar budget; GPU offsets follow current
    # confidence ordering and may distribute that budget differently.
    assert input_batch.num_scheduled_tokens.tolist() == [3, 2]
    assert input_batch.num_draft_tokens_per_req.tolist() == [2, 1]
    assert input_batch.num_tokens == 5
    assert input_batch.num_draft_tokens == 3
    assert input_batch.num_scheduled_tokens is num_scheduled_tokens
    assert input_batch.cu_num_logits_np is cu_num_logits_np
    assert input_batch.query_start_loc_np is query_start_loc_np
    assert input_batch.cu_num_logits_np.tolist() == [0, 3, 5]
    assert input_batch.query_start_loc_np.tolist() == [0, 3, 5, 5, 5]
    assert input_batch.cu_num_logits.cpu().tolist() == [0, 2, 5]
    assert input_batch.query_start_loc.cpu().tolist() == [0, 2, 5, 5, 5]
    assert input_batch.input_ids.shape[0] == input_batch.num_tokens_after_padding
    assert input_batch.input_ids[: input_batch.num_tokens].cpu().tolist() == [
        101,
        11,
        201,
        21,
        22,
    ]
    assert input_batch.positions[: input_batch.num_tokens].cpu().tolist() == [
        0,
        1,
        0,
        1,
        2,
    ]
    assert input_batch.seq_lens.cpu().tolist() == [2, 3, 0, 0]
    assert input_batch.logits_indices.cpu().tolist() == [0, 1, 2, 3, 4]
    assert (
        input_batch.is_padding[: input_batch.num_tokens].cpu().tolist() == [False] * 5
    )


def test_masked_capacity_manager_marks_pruned_tokens_for_forward_and_sampler():
    device = torch.device("cuda")
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=8,
        max_num_batched_tokens=16,
        num_speculative_steps=3,
        vocab_size=32,
        device=device,
    )
    req_states.req_id_to_index = {"req0": 0, "req1": 1}
    handler = MaskedConfidenceManager(
        req_states=req_states,
        speculative_config=_spec_config(dspark_budget_frac=0.34),
    )
    handler._confidence_logits[:2].copy_(
        torch.from_numpy(_logit(np.array([[0.90, 0.10, 0.10], [0.95, 0.95, 0.10]]))).to(
            device=device, dtype=torch.float32
        )
    )

    input_ids = torch.arange(16, dtype=torch.int32, device=device)
    input_batch = InputBatch(
        req_ids=["req0", "req1"],
        num_reqs=2,
        num_reqs_after_padding=2,
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32, device=device),
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        expanded_idx_mapping=torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=device
        ),
        expanded_local_pos=torch.tensor(
            [0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32, device=device
        ),
        num_scheduled_tokens=np.array([4, 4], dtype=np.int32),
        num_tokens=8,
        num_tokens_after_padding=10,
        num_draft_tokens=6,
        num_draft_tokens_per_req=np.array([3, 3], dtype=np.int32),
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32, device=device),
        query_start_loc_np=np.array([0, 4, 8], dtype=np.int32),
        seq_lens=torch.tensor([4, 4], dtype=torch.int32, device=device),
        seq_lens_cpu_upper_bound=torch.tensor([4, 4], dtype=torch.int32),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=np.array([0, 0], dtype=np.int32),
        prefill_len_np=np.array([0, 0], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0, 0], dtype=np.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
        max_seq_len_np=None,
        input_ids=input_ids,
        positions=torch.arange(16, dtype=torch.int64, device=device),
        is_padding=torch.zeros(10, dtype=torch.bool, device=device),
        logits_indices=torch.arange(8, dtype=torch.int64, device=device),
        cu_num_logits=torch.tensor([0, 4, 8], dtype=torch.int32, device=device),
        cu_num_logits_np=np.array([0, 4, 8], dtype=np.int32),
        has_structured_output_reqs=False,
        prompt_lens=None,
    )

    handler.trim_batch(input_batch, {})
    slot_mappings = torch.arange(20, dtype=torch.int64, device=device).view(2, 10)
    slot_mappings.masked_fill_(
        input_batch.is_padding[: slot_mappings.shape[1]].unsqueeze(0),
        PAD_SLOT_ID,
    )
    draft_sampled = input_batch.input_ids[input_batch.logits_indices]
    draft_sampled.masked_fill_(input_batch.is_padding[input_batch.logits_indices], -1)

    torch.accelerator.synchronize()
    assert input_batch.num_scheduled_tokens.tolist() == [4, 4]
    assert input_batch.num_draft_tokens_per_req.tolist() == [3, 3]
    assert input_batch.is_padding.cpu().tolist() == [
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert slot_mappings.cpu().tolist() == [
        [0, 1, -1, -1, 4, 5, 6, -1, 8, 9],
        [10, 11, -1, -1, 14, 15, 16, -1, 18, 19],
    ]
    assert draft_sampled.cpu().tolist() == [0, 1, -1, -1, 4, 5, 6, -1]


def test_capacity_cudagraph_dispatch_filters_by_max_query_len():
    manager = object.__new__(CudaGraphManager)
    manager._graphs_captured = True
    manager._resolve_effective_loras = lambda num_loras: num_loras
    regular_desc = BatchExecutionDescriptor(
        CUDAGraphMode.FULL,
        num_tokens=12,
        num_reqs=12,
        uniform_token_count=6,
    )
    full_capacity_desc = BatchExecutionDescriptor(
        CUDAGraphMode.FULL,
        num_tokens=15,
        num_reqs=4,
        max_req_tokens=6,
    )
    piecewise_desc = BatchExecutionDescriptor(
        CUDAGraphMode.PIECEWISE,
        num_tokens=16,
        num_reqs=None,
    )
    manager._candidates = {(11, 0): [regular_desc, full_capacity_desc, piecewise_desc]}

    desc = manager.dispatch(4, 11, None, 0, max_req_tokens=6)
    assert desc is full_capacity_desc

    desc = manager.dispatch(4, 11, None, 0, max_req_tokens=7)
    assert desc is piecewise_desc

    manager._candidates[(15, 0)] = [
        regular_desc,
        full_capacity_desc,
        piecewise_desc,
    ]
    desc = manager.dispatch(4, 15, None, 0, max_req_tokens=6)
    assert desc is full_capacity_desc

    desc = manager.dispatch(4, 11, 6, 0)
    assert desc is regular_desc


def test_varlen_cudagraph_capture_adds_full_desc():
    manager = object.__new__(CudaGraphManager)
    manager.vllm_config = SimpleNamespace(speculative_config=None)
    manager.compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=[5],
        max_cudagraph_capture_size=16,
    )
    manager.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    manager.decode_query_len = 4
    manager.varlen_spec_decode = True
    manager.max_num_reqs = 16
    manager.lora_capture_cases = [0]
    manager._candidates = {}
    manager._capture_descs = {}

    manager._init_candidates()

    descs = manager._capture_descs[CUDAGraphMode.FULL]
    assert [desc.num_reqs for desc in descs] == [3, 5]
    assert all(desc.max_req_tokens == manager.decode_query_len for desc in descs)
