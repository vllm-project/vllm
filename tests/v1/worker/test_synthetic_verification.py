# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.synthetic_verification import (
    can_compact_synthetic_verification,
    compact_synthetic_verification_counts,
    resolve_synthetic_verify_max_drafts,
)


def _config(**overrides) -> VllmConfig:
    values = {
        "rejection_sample_method": "synthetic",
        "draft_sample_method": "greedy",
        "enable_adaptive_verification": False,
        "synthetic_acceptance_rates": [1.0, 1.0, 0.75, 0.0, 0.0, 0.0],
        "num_speculative_tokens": 6,
    }
    values.update(overrides)
    spec_config = SimpleNamespace(
        **values,
        use_dspark=lambda: True,
    )
    return cast(
        VllmConfig,
        SimpleNamespace(
            speculative_config=spec_config,
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(model_type="kimi_linear")
            ),
        ),
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("draft_sample_method", "rates", "num_speculative_tokens", "expected"),
    [
        ("greedy", [1.0, 1.0, 0.75, 0.0, 0.0, 0.0], 6, 3),
        ("probabilistic", [1.0, 1.0, 0.0], 3, 2),
    ],
)
def test_resolve_synthetic_verify_max_drafts(
    monkeypatch: pytest.MonkeyPatch,
    draft_sample_method: str,
    rates: list[float],
    num_speculative_tokens: int,
    expected: int,
):
    monkeypatch.setattr(
        envs,
        "VLLM_KIMI_K3_SYNTHETIC_VERIFY_COMPACTION",
        True,
    )
    assert (
        resolve_synthetic_verify_max_drafts(
            _config(
                draft_sample_method=draft_sample_method,
                synthetic_acceptance_rates=rates,
                num_speculative_tokens=num_speculative_tokens,
            )
        )
        == expected
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "rates",
    [
        [0.0] * 6,
        [1.0] * 6,
    ],
)
def test_synthetic_verify_compaction_requires_strict_nonempty_suffix(
    monkeypatch: pytest.MonkeyPatch,
    rates: list[float],
):
    monkeypatch.setattr(
        envs,
        "VLLM_KIMI_K3_SYNTHETIC_VERIFY_COMPACTION",
        True,
    )
    assert (
        resolve_synthetic_verify_max_drafts(_config(synthetic_acceptance_rates=rates))
        is None
    )


@pytest.mark.cpu_test
def test_synthetic_verify_compaction_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        envs,
        "VLLM_KIMI_K3_SYNTHETIC_VERIFY_COMPACTION",
        False,
    )
    assert resolve_synthetic_verify_max_drafts(_config()) is None


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "overrides",
    [
        {"rejection_sample_method": "standard"},
        {"enable_adaptive_verification": True},
    ],
)
def test_synthetic_verify_compaction_rejects_incompatible_modes(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict,
):
    monkeypatch.setattr(
        envs,
        "VLLM_KIMI_K3_SYNTHETIC_VERIFY_COMPACTION",
        True,
    )
    with pytest.raises(ValueError):
        resolve_synthetic_verify_max_drafts(_config(**overrides))


@pytest.mark.cpu_test
def test_compact_synthetic_verification_counts_preserves_logical_input():
    scheduled = np.array([7, 4, 7, 1], dtype=np.int32)
    logical_drafts = np.array([6, 6, 6, 0], dtype=np.int32)

    physical_tokens, physical_drafts = compact_synthetic_verification_counts(
        scheduled,
        logical_drafts,
        max_drafts=3,
    )

    np.testing.assert_array_equal(physical_tokens, [4, 4, 4, 1])
    np.testing.assert_array_equal(physical_drafts, [3, 3, 3, 0])
    np.testing.assert_array_equal(scheduled, [7, 4, 7, 1])
    np.testing.assert_array_equal(logical_drafts, [6, 6, 6, 0])


@pytest.mark.cpu_test
def test_compact_counts_preserves_already_compacted_k3_warmup():
    scheduled = np.array([4, 3], dtype=np.int32)
    logical_drafts = np.array([3, 3], dtype=np.int32)

    physical_tokens, physical_drafts = compact_synthetic_verification_counts(
        scheduled,
        logical_drafts,
        max_drafts=2,
    )

    np.testing.assert_array_equal(physical_tokens, [3, 3])
    np.testing.assert_array_equal(physical_drafts, [2, 2])


@pytest.mark.cpu_test
def test_compaction_uses_lengths_not_placeholder_values():
    assert can_compact_synthetic_verification(
        np.array([6, 6], dtype=np.int32),
        max_drafts=3,
    )
    assert can_compact_synthetic_verification(
        np.array([6, 0], dtype=np.int32), max_drafts=3
    )
    assert not can_compact_synthetic_verification(
        np.array([3], dtype=np.int32), max_drafts=3
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("third_draw", [0.25, 0.90])
def test_compacted_schedule_rejects_at_same_target_row(third_draw: float):
    rates = [1.0, 1.0, 0.75, 0.0, 0.0, 0.0]
    draws = [0.5, 0.5, third_draw, 0.5, 0.5, 0.5]

    def accepted_length(candidate_rates: list[float]) -> int:
        accepted = 0
        for draw, rate in zip(
            draws[: len(candidate_rates)], candidate_rates, strict=True
        ):
            if draw >= rate:
                break
            accepted += 1
        return accepted

    full_accepted = accepted_length(rates)
    compact_accepted = accepted_length(rates[:4])
    assert compact_accepted == full_accepted
    # The resampling row is indexed by accepted length in both layouts.
    assert compact_accepted in (2, 3)


@pytest.mark.cpu_test
def test_model_runner_gathers_compacted_uniform_verification_batch():
    owner = SimpleNamespace(
        decode_query_len=4,
        synthetic_verify_max_drafts=3,
        adaptive_verification=None,
        req_states=SimpleNamespace(
            req_id_to_index={"req": 0},
            num_computed_tokens_np=np.array([100], dtype=np.int64),
            prefill_len=SimpleNamespace(np=np.array([10], dtype=np.int64)),
            num_computed_prefill_tokens=np.array([10], dtype=np.int64),
        ),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req": 7},
        total_num_scheduled_tokens=7,
        scheduled_spec_decode_tokens={"req": [11, 12, 13, 14, 15, 16]},
    )

    batch_state, uniform_token_count = GPUModelRunner.gather_batch_req_state(
        owner,
        scheduler_output,
        dummy_run=False,
    )

    assert batch_state is not None
    assert batch_state.num_tokens == 4
    np.testing.assert_array_equal(batch_state.num_scheduled_tokens, [4])
    assert uniform_token_count == 4


@pytest.mark.cpu_test
def test_model_runner_preserves_already_compacted_k3_warmup_batch():
    owner = SimpleNamespace(
        decode_query_len=3,
        synthetic_verify_max_drafts=2,
        adaptive_verification=None,
        req_states=SimpleNamespace(
            req_id_to_index={"req": 0},
            num_computed_tokens_np=np.array([100], dtype=np.int64),
            prefill_len=SimpleNamespace(np=np.array([10], dtype=np.int64)),
            num_computed_prefill_tokens=np.array([10], dtype=np.int64),
        ),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req": 3},
        total_num_scheduled_tokens=3,
        scheduled_spec_decode_tokens={"req": [0, 0, 0]},
    )

    batch_state, uniform_token_count = GPUModelRunner.gather_batch_req_state(
        owner,
        scheduler_output,
        dummy_run=False,
    )

    assert batch_state is not None
    assert batch_state.num_tokens == 3
    np.testing.assert_array_equal(batch_state.num_scheduled_tokens, [3])
    assert uniform_token_count == 3


@pytest.mark.cpu_test
def test_model_runner_compacts_async_placeholder_drafts():
    owner = SimpleNamespace(
        decode_query_len=4,
        synthetic_verify_max_drafts=3,
        adaptive_verification=None,
        req_states=SimpleNamespace(
            req_id_to_index={"req": 0},
            num_computed_tokens_np=np.array([100], dtype=np.int64),
            prefill_len=SimpleNamespace(np=np.array([10], dtype=np.int64)),
            num_computed_prefill_tokens=np.array([10], dtype=np.int64),
        ),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req": 7},
        total_num_scheduled_tokens=7,
        scheduled_spec_decode_tokens={"req": [-1, -1, -1, -1, -1, -1]},
    )

    batch_state, uniform_token_count = GPUModelRunner.gather_batch_req_state(
        owner,
        scheduler_output,
        dummy_run=False,
    )

    assert batch_state is not None
    assert batch_state.num_tokens == 4
    np.testing.assert_array_equal(batch_state.num_scheduled_tokens, [4])
    assert uniform_token_count == 4


@pytest.mark.cpu_test
def test_model_runner_compacts_only_spec_request_in_mixed_batch():
    owner = SimpleNamespace(
        decode_query_len=4,
        synthetic_verify_max_drafts=3,
        adaptive_verification=None,
        req_states=SimpleNamespace(
            req_id_to_index={"spec": 0, "plain": 1},
            num_computed_tokens_np=np.array([100, 100], dtype=np.int64),
            prefill_len=SimpleNamespace(np=np.array([10, 10], dtype=np.int64)),
            num_computed_prefill_tokens=np.array([10, 10], dtype=np.int64),
        ),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"spec": 7, "plain": 1},
        total_num_scheduled_tokens=8,
        scheduled_spec_decode_tokens={"spec": [11, 12, 13, 14, 15, 16]},
    )

    batch_state, uniform_token_count = GPUModelRunner.gather_batch_req_state(
        owner,
        scheduler_output,
        dummy_run=False,
    )

    assert batch_state is not None
    assert batch_state.num_tokens == 5
    np.testing.assert_array_equal(batch_state.num_scheduled_tokens, [4, 1])
    assert uniform_token_count is None
