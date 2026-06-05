# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for decoupling the draft model's pipeline-parallel size from the
target model's, so that PP>1 + speculative decoding (MTP) can place the draft
on a single stage (draft_pp=1) without tripping the SupportsPP guard.

Spike E1 for docs/superpowers/specs/2026-06-04-pp-mtp-spike-plan.md.
CPU-only: exercises config plumbing, no model download, no distributed init.
"""

import pytest

from vllm.config import ParallelConfig
from vllm.config.speculative import SpeculativeConfig


def _target_pp2() -> ParallelConfig:
    """A target ParallelConfig with pp=2 that constructs on a single-GPU host
    (mp backend + nnodes=2 bypasses the world-size>=num-GPUs validation)."""
    return ParallelConfig(
        pipeline_parallel_size=2,
        tensor_parallel_size=1,
        distributed_executor_backend="mp",
        nnodes=2,
    )


def test_create_draft_parallel_config_decouples_pp():
    """With draft_pp=1 the draft ParallelConfig must report pp=1 even when the
    target runs pp=2. This is the mechanism that bypasses the
    `pipeline_parallel_size > 1` SupportsPP guard for the draft model."""
    draft = SpeculativeConfig.create_draft_parallel_config(
        target_parallel_config=_target_pp2(),
        speculative_draft_tensor_parallel_size=1,
        speculative_draft_pipeline_parallel_size=1,
    )

    assert draft.pipeline_parallel_size == 1
    assert draft.tensor_parallel_size == 1


def test_verify_and_get_draft_pp_defaults_to_one():
    """Unset draft_pp defaults to 1 (draft on a single stage)."""
    assert SpeculativeConfig._verify_and_get_draft_pp(_target_pp2(), None) == 1


def test_verify_and_get_draft_pp_allows_one_or_target():
    target = _target_pp2()
    assert SpeculativeConfig._verify_and_get_draft_pp(target, 1) == 1
    assert SpeculativeConfig._verify_and_get_draft_pp(target, 2) == 2


def test_verify_and_get_draft_pp_rejects_other_values():
    """Only 1 or the target pp size are valid, mirroring draft_tp."""
    with pytest.raises(ValueError):
        SpeculativeConfig._verify_and_get_draft_pp(_target_pp2(), 3)


# --- Spike E3 step 1: the Qwen3.5 MTP draft must declare SupportsPP so that,
# under Design B (draft_pp == target_pp), it passes the model.py guard, which
# reads `model_cls.supports_pp` via registry.is_pp_supported_model.


def test_qwen3_5_mtp_declares_supports_pp():
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MoeMTP, Qwen3_5MTP

    assert Qwen3_5MTP.supports_pp is True
    assert Qwen3_5MoeMTP.supports_pp is True
