# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for Kimi-K3's DCP combine default.

On ROCm, Kimi-K3 prefers the ``a2a`` DCP combine over the default ``ag_rs``: it
is one ``all_to_all_single`` instead of ``allgather(lse)`` +
``reduce_scatter(out)``, and combine runs per MLA layer per decode step.

The preference is measured on MI355X and the tradeoff is fabric-dependent, so
the hook returns early off ROCm. Both branches are exercised here by patching
the platform, so these tests are meaningful on any CI runner rather than
silently vacuous on half of them.

``set_dcp_defaults`` only fills options the user left unset, so the other
property that matters is that an explicit user choice still wins.
"""

from types import SimpleNamespace

import pytest

from vllm.config import VllmConfig
from vllm.config.parallel import ParallelConfig
from vllm.model_executor.models import config as models_config
from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    KimiK3ForConditionalGenerationConfig,
)


@pytest.fixture
def on_rocm(monkeypatch):
    """Run the hook as if on ROCm, whatever the CI runner actually is."""
    monkeypatch.setattr(
        models_config, "current_platform", SimpleNamespace(is_rocm=lambda: True)
    )


@pytest.fixture
def off_rocm(monkeypatch):
    monkeypatch.setattr(
        models_config, "current_platform", SimpleNamespace(is_rocm=lambda: False)
    )


def _apply(parallel_config: ParallelConfig) -> ParallelConfig:
    KimiK3ForConditionalGenerationConfig.verify_and_update_config(
        SimpleNamespace(parallel_config=parallel_config)
    )
    return parallel_config


def test_k3_defaults_to_the_a2a_combine_on_rocm(on_rocm):
    assert _apply(ParallelConfig()).dcp_comm_backend == "a2a"


def test_the_default_is_untouched_off_rocm(off_rocm):
    """The measurement is MI355X-only and the tradeoff is fabric-dependent, so
    no other platform's default may move."""
    assert _apply(ParallelConfig()).dcp_comm_backend is None


def test_an_explicit_backend_is_not_overridden(on_rocm):
    """set_dcp_defaults fills unset options only; --dcp-comm-backend ag_rs must
    survive, or the flag would be silently inert for this model."""
    cfg = _apply(ParallelConfig(dcp_comm_backend="ag_rs"))
    assert cfg.dcp_comm_backend == "ag_rs"


def test_q_replicate_resolves_to_false_not_none(on_rocm):
    """set_dcp_defaults fills BOTH fields, so q_replicate is decided here.

    It is resolved None -> False, not left as None. That is the value
    MLAAttention already defaults to, so behaviour is unchanged -- but the hook
    does make the decision, and saying it is "left alone" would be wrong.

    What matters is that it is not True: GlmMoeDsa pairs a2a with
    q_replicate=True, which changes weight loading and is an independent,
    unmeasured question.
    """
    assert _apply(ParallelConfig()).dcp_q_replicate is False


def test_an_explicit_q_replicate_survives(on_rocm):
    """A user who asked for q_replicate=True keeps it."""
    cfg = ParallelConfig()
    cfg.dcp_q_replicate = True
    assert _apply(cfg).dcp_q_replicate is True


def test_pcp_keeps_the_upstream_default(on_rocm):
    """a2a under PCP is unmeasured and #56677 pins ag_rs for a GLM PCP4+DCP4
    config, so leave the default untouched rather than opt K3 users in."""
    cfg = ParallelConfig()
    cfg.prefill_context_parallel_size = 4
    out = _apply(cfg)
    assert out.dcp_comm_backend is None, (
        "PCP is enabled, so the a2a default must not be applied -- it is "
        "unmeasured there and upstream pinned ag_rs for that combination"
    )
    assert out.dcp_q_replicate is None


def test_pcp_still_honours_an_explicit_a2a(on_rocm):
    """Opting in under PCP is the user's call and must still work."""
    cfg = ParallelConfig()
    cfg.prefill_context_parallel_size = 4
    cfg.dcp_comm_backend = "a2a"
    assert _apply(cfg).dcp_comm_backend == "a2a"


def test_hook_is_registered_for_both_k3_architectures():
    """The MTP draft shares the target's config class; if only the main model
    were mapped, the draft would combine with ag_rs and the two halves of the
    same run would disagree."""
    for arch in ("KimiK3ForConditionalGeneration", "KimiK3MTPModel"):
        assert MODELS_CONFIG_MAP[arch] is KimiK3ForConditionalGenerationConfig


@pytest.mark.parametrize(
    "architecture", ["KimiK3ForConditionalGeneration", "KimiK3MTPModel"]
)
def test_the_dispatcher_actually_reaches_the_hook(on_rocm, architecture):
    """End-to-end through VllmConfig.try_verify_and_update_config.

    The registration test above proves the MODELS_CONFIG_MAP entry exists, and
    the default tests prove the hook does the right thing when called -- but
    neither proves the dispatcher connects the two. This closes that gap by
    going through the real lookup path.

    Uses the same object.__new__ + SimpleNamespace model_config shape as
    tests/model_executor/model_loader/test_modelexpress_loader.py, so nothing
    is downloaded and no model config has to resolve.
    """
    vllm_config = object.__new__(VllmConfig)
    vllm_config.model_config = SimpleNamespace(
        architecture=architecture,
        config_updated=False,
        convert_type=None,
        is_hybrid=False,
    )
    vllm_config.parallel_config = ParallelConfig()

    vllm_config.try_verify_and_update_config()

    assert vllm_config.parallel_config.dcp_comm_backend == "a2a"
