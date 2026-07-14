# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

from vllm.config.compilation import CUDAGraphMode  # noqa: E402
from vllm.v1.attention.backend import AttentionCGSupport  # noqa: E402
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
    allocate_draft_token_capacity,
    assign_draft_token_budget,
    build_sps_table_from_costs,
    compute_prefix_survival,
    make_confidence_manager,
)
from vllm.v1.worker.gpu.spec_decode.dspark.online_sts import (  # noqa: E402
    DSparkOnlineSTS,
)
from vllm.v1.worker.gpu.states import RequestState  # noqa: E402


def _spec_config(**overrides: Any) -> SimpleNamespace:
    config = SimpleNamespace(
        dspark_confidence_threshold=0.0,
        dspark_budget_frac=1.0,
        dspark_sps_curve=None,
        dspark_online_sts=False,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _logit(probs: np.ndarray) -> np.ndarray:
    return np.log(probs / (1.0 - probs))


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
        max_num_tokens=8,
        req_states=req_states,
        device=device,
        speculative_config=_spec_config(),
    )

    assert isinstance(manager, expected_type)


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


def test_allocate_capacity_uses_global_prefix_order():
    survival = compute_prefix_survival(
        _logit(
            np.array(
                [
                    [0.90, 0.90, 0.90],
                    [0.95, 0.10, 0.99],
                    [0.70, 0.70, 0.70],
                ]
            )
        )
    )
    capacities = allocate_draft_token_capacity(survival, min_survival=0.75)
    assert capacities.tolist() == [2, 1, 0]


def test_allocate_capacity_uses_budgeted_global_prefix_order():
    survival = compute_prefix_survival(_logit(np.array([[0.90, 0.80], [0.80, 0.80]])))
    capacities = allocate_draft_token_capacity(survival, budget_frac=0.5)
    assert capacities.tolist() == [2, 1]


def test_allocate_capacity_keeps_threshold_ties():
    survival = compute_prefix_survival(_logit(np.array([[0.90, 0.80], [0.90, 0.80]])))
    capacities = allocate_draft_token_capacity(survival, budget_frac=0.25)
    assert capacities.tolist() == [1, 1]


def test_allocate_capacity_budget_is_hard_cap_under_ties():
    """Saturated (tied) survival scores must not escape the budget.

    With every confidence saturated to 1.0, a kth-score-threshold recount
    would admit all tokens; the budget must remain a hard cap on the total.
    """
    survival = compute_prefix_survival(np.full((4, 7), 40.0))
    capacities = allocate_draft_token_capacity(survival, budget_frac=0.5)
    assert int(capacities.sum()) == int(4 * 7 * 0.5) + 1
    assert int(capacities.max()) <= 7


def test_allocate_capacity_never_admits_zero_survival():
    """Zero-survival tokens are not candidates (DSpark Alg. 1: a_{r,j} > 0),
    so leftover budget must not be spent past the first dead position."""
    # Positions 0-1 confident, position 2 dead (sigmoid underflows to an
    # exact zero) -> survival is exactly 0 from there on.
    logits = np.full((2, 5), 40.0)
    logits[:, 2] = -800.0
    survival = compute_prefix_survival(logits)
    capacities = allocate_draft_token_capacity(survival, budget_frac=0.9)
    assert capacities.tolist() == [2, 2]


def test_allocate_capacity_sps_curve_argmax():
    """With an SPS curve, verification lengths maximize tau * SPS(B)
    (DSpark Alg. 1) instead of spending the whole budget."""
    survival = compute_prefix_survival(_logit(np.array([[0.90, 0.80], [0.60, 0.50]])))
    # Survival: r0 [0.9, 0.72], r1 [0.6, 0.3]; admission order
    # 0.9, 0.72, 0.6, 0.3 with B = 2 + k.
    # SPS drops sharply after B=4 so theta peaks at k=2:
    #   k=0: 2.00*1.00, k=1: 2.90*0.95=2.755, k=2: 3.62*0.90=3.258,
    #   k=3: 4.22*0.20=0.844, k=4: 4.52*0.10=0.452.
    sps_row = np.array([1.0, 1.0, 1.0, 0.95, 0.90, 0.20, 0.10])
    capacities = allocate_draft_token_capacity(survival, sps_row=sps_row)
    assert capacities.tolist() == [2, 0]


def test_assign_draft_token_budget_uses_current_scores():
    current_survival = np.array([[0.9, 0.8, 0.7], [0.95, 0.4, 0.3]])

    capacities = assign_draft_token_budget(current_survival, num_admitted=3)

    assert capacities.tolist() == [2, 1]


def test_sps_table_combines_direct_draft_and_verification_costs():
    draft_curve = [(1, 0.2), (4, 0.8), (8, 1.6)]
    verify_curve = [(1, 2.0), (8, 3.0), (64, 5.0)]

    table = build_sps_table_from_costs(
        draft_curve, verify_curve, max_num_reqs=8, max_batch_tokens=64
    )

    assert 1000.0 / table[1, 8] == pytest.approx(3.2)
    assert 1000.0 / table[8, 8] == pytest.approx(4.6)


def test_allocate_capacity_temperature_desaturates_zeros():
    """A confidence temperature > 1 keeps saturated-negative positions in the
    candidate set (no exact-zero survival), so the budget is spent instead of
    being truncated by miscalibrated zeros."""
    logits = np.full((2, 5), 40.0)
    logits[:, 2] = -800.0
    caps_t1 = allocate_draft_token_capacity(
        compute_prefix_survival(logits), budget_frac=0.9
    )
    caps_t80 = allocate_draft_token_capacity(
        compute_prefix_survival(logits, 80.0), budget_frac=0.9
    )
    # T=1: exact-zero survival past position 2 truncates both requests.
    assert caps_t1.tolist() == [2, 2]
    # T=80: sigmoid(-10) > 0, so the budget (int(10*0.9)+1 = 10 admissions,
    # capped at 5 per request) is fully spent.
    assert caps_t80.tolist() == [5, 5]


def test_online_sts_fits_order_preserving_temperatures():
    """Online STS fits per-position temperatures from rejection-sampler
    outcomes: identity before data, softens over-confident positions,
    sharpens under-confident ones, and never reorders candidates."""
    sts = DSparkOnlineSTS(num_steps=3)

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


def test_capacity_manager_uses_stale_budget_with_current_assignment():
    """The prior host snapshot sets the budget; current GPU scores assign it."""
    device = torch.device("cuda")
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=4,
        max_num_batched_tokens=16,
        num_speculative_steps=3,
        vocab_size=32,
        device=device,
    )
    req_states.req_id_to_index = {"req0": 2, "req1": 0}
    handler = VarlenConfidenceManager(
        max_num_tokens=16,
        req_states=req_states,
        device=device,
        speculative_config=_spec_config(dspark_budget_frac=0.51),
    )
    handler.add_request(2)
    handler.add_request(0)
    input_batch: Any = SimpleNamespace(
        req_ids=["req0", "req1"],
        num_reqs=2,
        idx_mapping_np=np.array([2, 0], dtype=np.int32),
        num_tokens=0,
        num_tokens_after_padding=0,
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
        cu_num_logits_np=np.array([0, 1, 2], dtype=np.int32),
        input_ids=torch.empty(0, dtype=torch.int32, device=device),
        positions=torch.empty(0, dtype=torch.int64, device=device),
        is_padding=torch.empty(0, dtype=torch.bool, device=device),
    )

    def stage(logits: torch.Tensor) -> None:
        handler.stage_confidences(
            logits,
            torch.ones(2, dtype=torch.int32, device=device),
            input_batch,
        )

    # req0 confident on 2 positions, req1 on all 3; budget
    # int(6*0.51)+1 = 4 admissions.
    confident = torch.tensor(
        [[8.0, 8.0, -800.0], [8.0, 8.0, 8.0]], dtype=torch.float32, device=device
    )
    current = torch.tensor(
        [[20.0, 20.0, 20.0], [8.0, 8.0, 8.0]],
        dtype=torch.float32,
        device=device,
    )

    stage(confident)
    handler._apply_current_assignment()
    # Only one copy staged: defaults (full k) must remain untouched.
    assert handler.draft_token_capacity_np.tolist() == [3, 3, 3, 3]

    stage(current)
    handler._apply_current_assignment()
    # The first snapshot chooses four admissions, while stable current-score
    # ordering assigns three to req0 and one to req1.
    caps = handler.draft_token_capacity_np
    assert caps[2] == 3 and caps[0] == 1
    # Batch trimming does not advance the staging pipeline.
    handler._apply_current_assignment()
    assert handler.draft_token_capacity_np.tolist() == caps.tolist()

    # Finished requests are skipped when scattering capacities.
    del req_states.req_id_to_index["req0"]
    handler.draft_token_capacity_np.fill(3)
    stage(current)
    handler._apply_current_assignment()
    assert handler.draft_token_capacity_np[0] != 3  # req1 updated
    assert handler.draft_token_capacity_np[2] == 3  # req0 untouched


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
    handler = VarlenConfidenceManager(
        max_num_tokens=16,
        req_states=req_states,
        device=device,
        speculative_config=_spec_config(),
    )
    handler.add_request(0)
    handler.add_request(1)
    handler.draft_token_capacity_np[:2] = np.array([1, 2], dtype=np.int32)

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
        num_tokens_after_padding=5,
        num_draft_tokens=6,
        num_draft_tokens_per_req=np.array([3, 3], dtype=np.int32),
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32, device=device),
        query_start_loc_np=np.array([0, 4, 8], dtype=np.int32),
        seq_lens=torch.zeros(2, dtype=torch.int32, device=device),
        seq_lens_cpu_upper_bound=torch.tensor([4, 4], dtype=torch.int32),
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
        cu_num_logits_np=np.array([0, 4, 8], dtype=np.int32),
        has_structured_output_reqs=False,
        prompt_lens=None,
    )

    handler.trim_batch(input_batch, {})

    torch.accelerator.synchronize()
    assert input_batch.num_scheduled_tokens.tolist() == [2, 3]
    assert input_batch.num_draft_tokens_per_req.tolist() == [1, 2]
    assert input_batch.num_tokens == 5
    assert input_batch.num_draft_tokens == 3
    assert input_batch.cu_num_logits_np.tolist() == [0, 2, 5]
    assert input_batch.query_start_loc_np.tolist() == [0, 2, 5]
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
    assert input_batch.seq_lens.cpu().tolist() == [2, 3]
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
    handler = MaskedConfidenceManager(
        max_num_tokens=16,
        req_states=req_states,
        device=device,
        speculative_config=_spec_config(),
    )
    handler.add_request(0)
    handler.add_request(1)
    handler.draft_token_capacity_np[:2] = np.array([1, 2], dtype=np.int32)

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

    assert any(
        desc.max_req_tokens == manager.decode_query_len
        for desc in manager._capture_descs[CUDAGraphMode.FULL]
    )
