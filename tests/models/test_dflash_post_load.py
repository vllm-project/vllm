# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash / DSpark drafters build their fused context-KV buffers from the
loader's post-load hook, not from inside ``load_weights`` or lazily on the
first forward."""

from unittest.mock import Mock

import pytest
from torch import nn

from vllm.model_executor.models.gemma4_dspark import Gemma4DSparkForCausalLM
from vllm.model_executor.models.laguna_dflash import DFlashLagunaForCausalLM
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM


@pytest.mark.parametrize(
    "root_cls",
    [
        DFlashQwen3ForCausalLM,
        Qwen3DSparkForCausalLM,
        Gemma4DSparkForCausalLM,
        DFlashLagunaForCausalLM,
    ],
)
def test_root_hook_builds_fused_kv_buffers(root_cls) -> None:
    model = object.__new__(root_cls)
    nn.Module.__init__(model)
    model.model = Mock()

    model.process_weights_after_loading()

    model.model._build_fused_kv_buffers.assert_called_once_with()


def test_precompute_does_not_build_buffers_lazily() -> None:
    """A model whose hook never ran must fail, not silently build on first use."""
    model = object.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model._build_fused_kv_buffers = Mock()

    with pytest.raises(AttributeError):
        model.precompute_and_store_context_kv(Mock(shape=(1,)), Mock(), None)

    model._build_fused_kv_buffers.assert_not_called()
