# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dspark_target_layer_ids name the layers whose OUTPUT the drafter consumes,
but the target's capture hook fires on `idx + 1 in aux_hidden_state_layers`
(the input of layer L). The V1 runner must apply the same +1 conversion as the
DFlash branch and the V2 runner (eagle3_utils.py); passing the ids raw hands
the drafter hidden states shifted down one layer (jasl/vllm#29)."""

from types import SimpleNamespace

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def _stub_runner(hf_config) -> SimpleNamespace:
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf_config)
        )
    )

def test_dspark_target_layer_ids_are_shifted_to_capture_semantics():
    hf_config = SimpleNamespace(dspark_target_layer_ids=(40, 41, 42))
    runner = _stub_runner(hf_config)
    assert GPUModelRunner._get_eagle3_aux_layers_from_config(runner) == (41, 42, 43)

def test_eagle_aux_ids_pass_through_unshifted():
    hf_config = SimpleNamespace(eagle_aux_hidden_state_layer_ids=(10, 20))
    runner = _stub_runner(hf_config)
    assert GPUModelRunner._get_eagle3_aux_layers_from_config(runner) == (10, 20)
