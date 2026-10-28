# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RL weight updates call ``load_weights`` once per bucket: the model-level
finish step runs once after the last call."""

from torch import nn

from vllm.model_executor.model_loader.reload import finalize_layerwise_reload


class _HookCounter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def process_weights_after_loading(self) -> None:
        self.calls += 1


def test_reload_finish_runs_model_hook_once():
    model = _HookCounter()
    finalize_layerwise_reload(model, None)
    assert model.calls == 1
