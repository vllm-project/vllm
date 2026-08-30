# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPress method registration.

These are config-only and run in milliseconds. They exist because the failures
they catch are silent: an unregistered method name surfaces as a confusing
"unsupported" error much later, and a missing entry in the async-scheduling
whitelist does not raise at all -- vLLM simply logs a warning and runs the
engine ~17% slower, which is easy to miss and hard to attribute.
"""

import inspect
from typing import get_args

from vllm.config.speculative import SpeculativeConfig, SpeculativeMethod
from vllm.model_executor.models.registry import _SPECULATIVE_DECODING_MODELS


def test_xpress_is_a_known_speculative_method():
    assert "xpress" in get_args(SpeculativeMethod)


def test_draft_architecture_is_registered():
    """The checkpoint's config.json says Qwen3XPressModel, so the registry must
    map exactly that string, or from_pretrained cannot find the draft model."""
    assert _SPECULATIVE_DECODING_MODELS["Qwen3XPressModel"] == (
        "qwen3_xpress",
        "Qwen3XPressForCausalLM",
    )


def test_async_scheduling_is_allowed_in_both_paths():
    """vllm.config.vllm gates async scheduling in two independent places -- one
    that raises when it was requested explicitly, one that warns and disables it
    when it was inferred. Passing only the first still costs the overlap, with
    nothing but a warning to show for it, so pin that xpress clears both."""
    from vllm.config import vllm as vllm_config_mod

    source = inspect.getsource(vllm_config_mod)
    assert source.count('method != "xpress"') == 2, (
        "xpress must appear in BOTH async-scheduling checks; a single one leaves "
        "the other path silently disabling host/GPU overlap"
    )


def test_parallel_drafting_and_v2_runner():
    """XPress drafts a whole block per step and is implemented only by the V2 GPU
    runner, mirroring DSpark. Both flags are read from plain method-name checks."""
    source = inspect.getsource(SpeculativeConfig)
    assert '"xpress"' in source
    from vllm.config import vllm as vllm_config_mod

    v2 = inspect.getsource(vllm_config_mod)
    assert '("dspark", "xpress")' in v2, (
        "xpress must force the V2 GPU model runner, as DSpark does"
    )
