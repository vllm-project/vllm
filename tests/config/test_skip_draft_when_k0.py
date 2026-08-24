# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the opt-in ``skip_draft_when_k0`` speculative config option.

The option skips the draft-model forward on engine steps whose
dynamically resolved ``num_speculative_tokens`` is 0 (the ``DSD K=0``
prefill normally kept to sync draft KV state for a possible K>0
resume). It is refused for EAGLE-family methods: a full-attention
drafter's acceptance collapses non-recovering after unsynced K=0 steps
(issue #53420).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
    DFlash2Speculator,
)

MODEL = "facebook/opt-125m"
DYNAMIC_SCHEDULE = [(1, 1, 0), (2, 16, 2)]


def _spec_config(**kwargs) -> SpeculativeConfig:
    target = ModelConfig(model=MODEL)
    parallel = ParallelConfig(pipeline_parallel_size=1, tensor_parallel_size=1)
    defaults = dict(
        method="draft_model",
        model=MODEL,
        target_model_config=target,
        target_parallel_config=parallel,
        num_speculative_tokens=2,
    )
    defaults.update(kwargs)
    return SpeculativeConfig(**defaults)


def _validator_target(method: str, dynamic: bool = True):
    config = SpeculativeConfig.__new__(SpeculativeConfig)
    config.method = method
    config.skip_draft_when_k0 = True
    config.num_speculative_tokens_per_batch_size = DYNAMIC_SCHEDULE if dynamic else None
    return config


@pytest.mark.cpu_test
def test_validator_accepts_mtp():
    _validator_target("mtp")._validate_skip_draft_when_k0(
        warn_without_dynamic_schedule=True
    )  # must not raise


@pytest.mark.cpu_test
def test_rejected_for_unimplemented_speculators():
    """Methods whose speculators do not implement the skip (or are
    unmeasured) are rejected explicitly, not silently ignored."""
    for method in ("dspark",):
        with pytest.raises(ValueError, match="not implemented"):
            _validator_target(method)._validate_skip_draft_when_k0()


@pytest.mark.cpu_test
def test_validator_accepts_dflash():
    _validator_target("dflash")._validate_skip_draft_when_k0(
        warn_without_dynamic_schedule=True
    )  # must not raise


@pytest.mark.cpu_test
def test_dflash_tier_rungs_must_be_max_or_skipped_zero():
    """DFlash proposes a fixed block width; tier rungs must resolve to
    that width (the scheduler clamps rungs via min(k, max)) or skip
    drafting entirely (the skip requires the flag)."""
    _dflash_tiers(
        [(1, 2, 7), (3, 16, 0)], skip=True
    )._validate_dflash_dynamic_schedule()  # noqa: E501
    # Over-max rungs clamp to the full width — scheduler-valid, accepted.
    _dflash_tiers(
        [(1, 2, 8), (3, 16, 0)], skip=True
    )._validate_dflash_dynamic_schedule()  # noqa: E501
    with pytest.raises(ValueError, match="rungs resolving to 7"):
        _dflash_tiers(
            [(1, 2, 7), (3, 16, 3)], skip=True
        )._validate_dflash_dynamic_schedule()  # noqa: E501
    with pytest.raises(ValueError, match="skip_draft_when_k0=true"):
        _dflash_tiers(
            [(1, 2, 7), (3, 16, 0)], skip=False
        )._validate_dflash_dynamic_schedule()  # noqa: E501


def _dflash_tiers(schedule, skip: bool):
    config = SpeculativeConfig.__new__(SpeculativeConfig)
    config.method = "dflash"
    config.num_speculative_tokens = 7
    config.skip_draft_when_k0 = skip
    config.num_speculative_tokens_per_batch_size = schedule
    return config


@pytest.mark.cpu_test
def test_refused_for_explicit_eagle_methods_without_model_access():
    """The rejection must fire before any draft-config work: an
    explicit eagle-family method is refused even with a bogus model
    path (no download / filesystem access happens)."""
    with pytest.raises(ValueError, match="skip_draft_when_k0"):
        SpeculativeConfig(
            method="eagle3",
            model="/nonexistent/path",
            num_speculative_tokens=2,
            skip_draft_when_k0=True,
        )
    with pytest.raises(ValueError, match="skip_draft_when_k0"):
        SpeculativeConfig(
            method="eagle",
            model="/nonexistent/path",
            num_speculative_tokens=2,
            skip_draft_when_k0=True,
        )


@pytest.mark.cpu_test
def test_no_effect_warning_without_dynamic_schedule(caplog):
    with caplog.at_level("WARNING", logger="vllm.config.speculative"):
        _validator_target("mtp", dynamic=False)._validate_skip_draft_when_k0(
            warn_without_dynamic_schedule=True
        )
    assert any(
        "skip_draft_when_k0 has no effect" in rec.message for rec in caplog.records
    )


@pytest.mark.cpu_test
def test_skip_does_not_fire_when_k_unresolved():
    """num_speculative_tokens=None (no dynamic resolution information):
    the skip must not fire — propose proceeds (sentinel)."""
    spec = _bare_speculator(skip=True)
    with pytest.raises(SentinelError):
        _call_propose(spec, _k0_batch(num_reqs=2), num_speculative_tokens=None)


class _ConcreteSpeculator(AutoRegressiveSpeculator):
    """Stub making the abstract base instantiable via __new__."""

    def load_draft_model(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


def _bare_speculator(skip: bool) -> AutoRegressiveSpeculator:
    """Only the state propose() touches before the K=0 branch."""
    spec = _ConcreteSpeculator.__new__(_ConcreteSpeculator)
    spec.skip_draft_when_k0 = skip
    spec.draft_tokens = torch.zeros(4, 2, dtype=torch.int64)
    spec.num_speculative_steps = 2
    spec.max_model_len = 128
    spec.hidden_states = torch.zeros(8, 8)
    spec.dtype = torch.float32
    # sentinel on the first method call after the hidden-state copy:
    # if propose() runs past the K=0 skip branch it reaches this and
    # raises, proving the prefill path was entered.
    spec._copy_request_inputs = _sentinel
    return spec


def _sentinel(*args, **kwargs):
    raise SentinelError()


def _k0_batch(num_reqs: int = 2):
    return SimpleNamespace(
        num_tokens=num_reqs,
        num_tokens_after_padding=num_reqs,
        num_reqs=num_reqs,
        num_scheduled_tokens=torch.ones(num_reqs, dtype=torch.int64),
        seq_lens_cpu_upper_bound=torch.full((num_reqs,), 8, dtype=torch.int64),
        idx_mapping=torch.arange(num_reqs, dtype=torch.int64),
    )


def _call_propose(spec, batch, num_speculative_tokens=0):
    return spec.propose(
        batch,
        {},
        {},
        torch.zeros(batch.num_tokens, 8),
        None,
        torch.zeros(batch.num_reqs, dtype=torch.int64),  # num_sampled
        torch.zeros(batch.num_reqs, dtype=torch.int64),  # num_rejected
        torch.zeros(4, dtype=torch.int64),  # last_sampled
        torch.zeros(4, dtype=torch.int64),  # next_prefill_tokens
        torch.zeros(4, dtype=torch.float32),  # temperature
        torch.zeros(4, dtype=torch.int64),  # seeds
        num_speculative_tokens=num_speculative_tokens,
    )


@pytest.mark.cpu_test
def test_skip_returns_zero_width_without_running_prefill():
    """K=0 + flag on: propose returns [num_reqs, 0] immediately; the
    draft prefill path is never entered (proved by a sentinel raised
    from the first prefill-path helper propose would otherwise reach)."""
    spec = _bare_speculator(skip=True)
    with patch(
        "vllm.v1.worker.gpu.spec_decode.autoregressive.speculator."
        "prepare_prefill_inputs",
        side_effect=AssertionError("prefill path must not run"),
    ):
        out = _call_propose(spec, _k0_batch(num_reqs=2))  # must NOT raise
    assert out.shape == (2, 0)
    assert out.dtype == torch.int64


@pytest.mark.cpu_test
def test_flag_off_still_enters_prefill_path_at_k0():
    """K=0 + flag off (default): behavior unchanged — propose continues
    into the prefill path (sentinel fires), preserving the draft KV
    sync forward."""
    spec = _bare_speculator(skip=False)
    with pytest.raises(SentinelError):
        _call_propose(spec, _k0_batch(num_reqs=2))


@pytest.mark.cpu_test
def test_skip_does_not_fire_when_k_is_positive():
    """K=2 + flag on: no early return — propose proceeds (sentinel)."""
    spec = _bare_speculator(skip=True)
    with pytest.raises(SentinelError):
        _call_propose(spec, _k0_batch(num_reqs=2), num_speculative_tokens=2)


class SentinelError(Exception):
    pass


# --- DFlash speculator K=0 skip (mirrors the AutoRegressive tests above) ---


def _bare_dflash(skip: bool) -> "DFlash2Speculator":
    """Only the state propose() touches before the K=0 branch. The
    hidden_states buffer is intentionally absent: propose() reaching the
    forward path raises on it, proving the skip did (or did not) fire."""
    spec = DFlash2Speculator.__new__(DFlash2Speculator)
    spec.skip_draft_when_k0 = skip
    spec.draft_tokens = torch.zeros(4, 7, dtype=torch.int64)
    spec.num_query_per_req = 8
    spec.max_model_len = 128
    return spec


@pytest.mark.cpu_test
def test_dflash_skip_returns_zero_width_at_k0():
    spec = _bare_dflash(skip=True)
    out = _call_propose(spec, _k0_batch(num_reqs=2))  # must NOT raise
    assert out.shape == (2, 0)
    assert out.dtype == torch.int64


@pytest.mark.cpu_test
def test_dflash_flag_off_runs_forward_at_k0():
    spec = _bare_dflash(skip=False)
    with pytest.raises(AttributeError, match="hidden_states"):
        _call_propose(spec, _k0_batch(num_reqs=2))


@pytest.mark.cpu_test
def test_dflash_skip_does_not_fire_when_k_unresolved_or_positive():
    spec = _bare_dflash(skip=True)
    with pytest.raises(AttributeError, match="hidden_states"):
        _call_propose(spec, _k0_batch(num_reqs=2), num_speculative_tokens=None)
    with pytest.raises(AttributeError, match="hidden_states"):
        _call_propose(spec, _k0_batch(num_reqs=2), num_speculative_tokens=7)


@pytest.mark.cpu_test
def test_support_matrix_validator_level():
    """Every method boundary resolves to accept or a specific rejection
    — none fall through to silently ignoring the option."""
    # accepted (measured)
    _validator_target("mtp")._validate_skip_draft_when_k0(
        warn_without_dynamic_schedule=True
    )  # must not raise
    # measured-unsafe: EAGLE family (one EAGLE3 checkpoint collapsed;
    # eagle excluded conservatively — shared full-attention risk)
    for method in ("eagle", "eagle3"):
        with pytest.raises(ValueError, match="not supported"):
            _validator_target(method)._validate_skip_draft_when_k0()
    # method strings whose speculators override propose() without the
    # skip -> would be silently ignored -> rejected explicitly. (Real
    # multi-module/gemma4 drafts route through method='mtp' + draft
    # architecture — covered by the test_real_routing_* tests below.)
    for method in ("dspark",):
        with pytest.raises(ValueError, match="silently ignored"):
            _validator_target(method)._validate_skip_draft_when_k0()
    # accepted since the DFlash propose() gained the K=0 skip
    _validator_target("dflash")._validate_skip_draft_when_k0(
        warn_without_dynamic_schedule=True
    )  # must not raise
    # any other non-mtp method string (including impossible states)
    # -> rejected as out of scope
    for method in ("gemma4", "multi_module_mtp", "medusa", "ngram"):
        with pytest.raises(ValueError, match="validated only for method"):
            _validator_target(method)._validate_skip_draft_when_k0()


@pytest.mark.cpu_test
def test_early_validation_defers_method_inference_placeholder():
    """method='draft_model' is the placeholder for method=None; the
    early (pre-model-access) validation must NOT reject it — the real
    method is inferred later in __post_init__ and checked at the final
    call. Early rejection would block automatically detected MTP."""
    cfg = _validator_target("draft_model")
    cfg._validate_skip_draft_when_k0(early=True)  # deferred: no raise
    cfg.method = "mtp"  # after successful inference: accepts
    cfg._validate_skip_draft_when_k0(warn_without_dynamic_schedule=True)
    cfg.method = "eagle"  # after inference to an unsafe method: rejects
    with pytest.raises(ValueError, match="not supported"):
        cfg._validate_skip_draft_when_k0()


@pytest.mark.cpu_test
def test_real_construction_unresolved_method_defers_then_rejects():
    """End-to-end construction with the flag on and method left to
    inference (placeholder 'draft_model', non-MTP draft): the early
    call defers, construction proceeds, and the FINAL validation
    rejects with a clear message — no silent ignore, no early block of
    legitimate inference."""
    with pytest.raises(ValueError, match="validated only for methods"):
        _spec_config(skip_draft_when_k0=True)


@pytest.mark.cpu_test
def test_cli_json_plumbing_via_engine_args():
    """The real CLI path: --speculative-config JSON -> dict (hyphens
    normalized) -> SpeculativeConfig(**kwargs). The flag must survive
    with the correct type and reach validation."""
    import json

    from vllm.engine.arg_utils import EngineArgs

    ea = EngineArgs(
        model=MODEL,
        speculative_config=json.loads(
            '{"method": "draft_model", "model": "' + MODEL + '", '
            '"num_speculative_tokens": 2, "skip-draft-when-k0": true}'
        ),
    )
    # hyphenated CLI key normalized + flag typed bool + final validation
    with pytest.raises(ValueError, match="validated only for methods"):
        ea.create_speculative_config(
            ModelConfig(model=MODEL),
            ParallelConfig(pipeline_parallel_size=1, tensor_parallel_size=1),
        )

    ea_false = EngineArgs(
        model=MODEL,
        speculative_config=json.loads(
            '{"method": "draft_model", "model": "' + MODEL + '", '
            '"num_speculative_tokens": 2, "skip_draft_when_k0": false}'
        ),
    )
    spec = ea_false.create_speculative_config(
        ModelConfig(model=MODEL),
        ParallelConfig(pipeline_parallel_size=1, tensor_parallel_size=1),
    )
    assert spec.skip_draft_when_k0 is False


def _mtp_target(model_type: str, num_nextn: int = 1):
    """Target ModelConfig whose (callable) hf_overrides flow to the
    draft config as well, routing method='mtp' to the requested draft
    architecture — the same resolution init_speculator performs."""

    def _override(cfg):
        cfg.model_type = model_type
        cfg.num_nextn_predict_layers = num_nextn
        return cfg

    return ModelConfig(model=MODEL, hf_overrides=_override)


@pytest.mark.cpu_test
def test_real_routing_accepts_single_module_mtp():
    """Construction-level accept: method='mtp' resolving to the
    standard single-module MTPSpeculator (the measured configuration)."""
    spec = SpeculativeConfig(
        method="mtp",
        model=MODEL,
        num_speculative_tokens=2,
        num_speculative_tokens_per_batch_size=[[1, 1, 0], [2, 16, 2]],
        skip_draft_when_k0=True,
        target_model_config=_mtp_target("bailing_hybrid_mtp", num_nextn=1),
        target_parallel_config=ParallelConfig(
            pipeline_parallel_size=1, tensor_parallel_size=1
        ),
    )
    assert spec.skip_draft_when_k0 is True
    assert not spec.use_multi_module_mtp()
    assert not spec.use_gemma4_mtp()


@pytest.mark.cpu_test
def test_real_routing_rejects_multi_module_mtp():
    """method='mtp' + multi-module draft (num_nextn>1 with K>1):
    MultiModuleMTPSpeculator overrides propose() without the skip —
    the flag would be silently ignored, so construction rejects."""
    with pytest.raises(ValueError, match="multi-module"):
        SpeculativeConfig(
            method="mtp",
            model=MODEL,
            num_speculative_tokens=2,
            num_speculative_tokens_per_batch_size=[[1, 1, 0], [2, 16, 2]],
            skip_draft_when_k0=True,
            target_model_config=_mtp_target("bailing_hybrid_mtp", num_nextn=4),
            target_parallel_config=ParallelConfig(
                pipeline_parallel_size=1, tensor_parallel_size=1
            ),
        )


@pytest.mark.cpu_test
def test_real_routing_rejects_gemma4_mtp():
    """method='mtp' + gemma4_mtp draft: Gemma4Speculator inherits the
    skip but is unmeasured — rejected rather than applied."""
    with pytest.raises(ValueError, match="gemma4"):
        SpeculativeConfig(
            method="mtp",
            model=MODEL,
            num_speculative_tokens=2,
            num_speculative_tokens_per_batch_size=[[1, 1, 0], [2, 16, 2]],
            skip_draft_when_k0=True,
            target_model_config=_mtp_target("gemma4_mtp"),
            target_parallel_config=ParallelConfig(
                pipeline_parallel_size=1, tensor_parallel_size=1
            ),
        )
