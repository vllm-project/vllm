# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from typing import get_args

from vllm.config.speculative import SpeculativeConfig, SpeculativeMethod
from vllm.model_executor.models.registry import _SPECULATIVE_DECODING_MODELS


def test_xpress_is_a_known_speculative_method():
    assert "xpress" in get_args(SpeculativeMethod)


# The checkpoint's config.json says Qwen3XPressModel, so the registry must map exactly
# that string or from_pretrained cannot find the draft model.
def test_draft_architecture_is_registered():
    assert _SPECULATIVE_DECODING_MODELS["Qwen3XPressModel"] == (
        "qwen3_xpress",
        "Qwen3XPressForCausalLM",
    )


# vllm/config/vllm.py gates async scheduling in two independent places. Passing only one
# still costs the host/GPU overlap -- roughly 17% -- with nothing but a warning to show.
def test_async_scheduling_is_allowed_in_both_paths():
    from vllm.config import vllm as vllm_config_mod

    source = inspect.getsource(vllm_config_mod)
    assert source.count('method != "xpress"') == 2, (
        "xpress must appear in BOTH async-scheduling checks; a single one leaves "
        "the other path silently disabling host/GPU overlap"
    )


# XPress drafts a whole block per step and is implemented only by the V2 GPU runner,
# mirroring DSpark.
def test_parallel_drafting_and_v2_runner():
    source = inspect.getsource(SpeculativeConfig)
    assert '"xpress"' in source
    from vllm.config import vllm as vllm_config_mod

    v2 = inspect.getsource(vllm_config_mod)
    assert '("dspark", "xpress")' in v2, (
        "xpress must force the V2 GPU model runner, as DSpark does"
    )
