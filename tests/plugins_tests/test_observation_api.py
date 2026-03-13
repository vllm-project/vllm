# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.plugins.observation import (
    ObservationAction,
    ObservationPlugin,
    ObservationResult,
    RequestContext,
)


class MockObservationPlugin(ObservationPlugin):
    def __init__(self, action, prefill_chunks=None, vllm_config=None):
        super().__init__(vllm_config=vllm_config)
        self.action_to_return = action
        self.observed_contexts = []
        self.prefill_chunks = prefill_chunks or []

    def get_observation_layers(self):
        return [0, -1]

    def on_step_batch(self, batch_hidden_states, request_contexts):
        self.observed_contexts.extend(request_contexts)
        return [ObservationResult(action=self.action_to_return)] * len(request_contexts)


def test_basic_plugin_action():
    plugin = MockObservationPlugin(ObservationAction.CONTINUE)

    contexts = [
        RequestContext(
            request_id="req-1",
            is_prefill=True,
            chunk_idx=0,
            num_cached_tokens=0,
            batch_offset=0,
            num_tokens=10,
        ),
    ]
    hidden_states = {0: torch.randn(10, 128), -1: torch.randn(10, 128)}

    results = plugin.on_step_batch(hidden_states, contexts)

    assert len(results) == 1
    assert results[0].action == ObservationAction.CONTINUE
    assert len(plugin.observed_contexts) == 1
