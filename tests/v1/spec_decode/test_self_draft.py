# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ``method='self_draft'`` configuration path.

These tests are intentionally offline: they exercise
``SpeculativeConfig.__post_init__`` and the small set of method-predicate
helpers added alongside ``SelfDraftProposer``. End-to-end accept-rate and
CUDA-graph correctness are covered by the integration tests (which require
a GPU + a model checkpoint) and are run separately by the submitter.
"""

from unittest import mock

import pytest

from vllm.config import ParallelConfig, SpeculativeConfig


def _make_target_model_config(enforce_eager: bool = False):
    """Minimal ``ModelConfig`` stand-in for the ``self_draft`` branch.

    ``SpeculativeConfig.__post_init__`` only reads ``enforce_eager`` and
    assigns the whole object to ``draft_model_config`` / inspects
    ``hf_config`` later via the proposer (not exercised here), so a small
    ``MagicMock`` is sufficient and avoids any HF download.
    """
    target = mock.MagicMock(name="target_model_config")
    target.enforce_eager = enforce_eager
    return target


def test_predicates_are_mutually_exclusive_for_self_draft():
    """``use_self_draft()`` must not overlap with the other method gates."""
    sc = SpeculativeConfig.__new__(SpeculativeConfig)
    sc.method = "self_draft"
    assert sc.use_self_draft() is True
    assert sc.use_eagle() is False
    assert sc.use_dflash() is False
    assert sc.uses_draft_model() is False
    assert sc.uses_extract_hidden_states() is False

    sc.method = "eagle"
    assert sc.use_self_draft() is False
    sc.method = "draft_model"
    assert sc.use_self_draft() is False
    sc.method = "ngram"
    assert sc.use_self_draft() is False


def test_self_draft_shares_target_configs_and_enables_parallel_drafting():
    target = _make_target_model_config()
    spec = SpeculativeConfig(
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
        method="self_draft",
        num_speculative_tokens=3,
    )
    # Drafter shares the target's weights & parallelism by construction.
    assert spec.draft_model_config is spec.target_model_config
    assert spec.draft_parallel_config is spec.target_parallel_config
    # Parallel drafting is the whole point of this method.
    assert spec.parallel_drafting is True
    # ``self.model`` is set to the method name purely to satisfy the
    # "model is not None" guard.
    assert spec.model == "self_draft"


def test_self_draft_inherits_enforce_eager_from_target():
    """Drafter and target share the same module instance, so their
    cudagraph modes must agree when the user did not pin ``enforce_eager``
    on the speculative config explicitly."""
    target = _make_target_model_config(enforce_eager=True)
    spec = SpeculativeConfig(
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
        method="self_draft",
        num_speculative_tokens=2,
    )
    assert spec.enforce_eager is True


def test_self_draft_respects_explicit_enforce_eager():
    """An explicit ``enforce_eager=False`` on the speculative config must
    win over the target's default."""
    target = _make_target_model_config(enforce_eager=True)
    spec = SpeculativeConfig(
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
        method="self_draft",
        num_speculative_tokens=2,
        enforce_eager=False,
    )
    assert spec.enforce_eager is False


def test_self_draft_auto_generates_chain_token_tree():
    target = _make_target_model_config()
    spec = SpeculativeConfig(
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
        method="self_draft",
        num_speculative_tokens=4,
    )
    # Chain shape: K nodes, each at depth i+1.
    # SpecDecodeBaseProposer parses ``speculative_token_tree``; parallel
    # drafting ignores its shape but the field must be non-None.
    assert spec.speculative_token_tree is not None
    tree = eval(spec.speculative_token_tree)  # noqa: S307 - test-only
    assert tree == [(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0)]


def test_self_draft_requires_num_speculative_tokens():
    target = _make_target_model_config()
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        SpeculativeConfig(
            target_model_config=target,
            target_parallel_config=ParallelConfig(),
            method="self_draft",
            # num_speculative_tokens intentionally omitted
        )


def test_self_draft_disables_prompt_lookup():
    """``prompt_lookup_{min,max}`` are an ngram-only artifact; the
    self_draft branch must zero them out so downstream code paths that
    branch on these values do not accidentally trigger."""
    target = _make_target_model_config()
    spec = SpeculativeConfig(
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
        method="self_draft",
        num_speculative_tokens=2,
    )
    assert spec.prompt_lookup_min == 0
    assert spec.prompt_lookup_max == 0


def test_parallel_drafting_mask_token_id_override_is_stored_verbatim():
    """The override is consumed lazily by ``SelfDraftProposer`` (not
    exercised here without a model). What we verify is that the
    ``SpeculativeConfig`` field round-trips through ``__post_init__``
    without being clobbered by the self_draft branch."""
    target = _make_target_model_config()
    spec = SpeculativeConfig(
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
        method="self_draft",
        num_speculative_tokens=1,
        parallel_drafting_mask_token_id=123456,
    )
    assert spec.parallel_drafting_mask_token_id == 123456
