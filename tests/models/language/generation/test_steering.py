# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for activation steering on Gemma 3."""

import pytest

from vllm import SamplingParams

MODEL = "google/gemma-3-4b-it"


@pytest.mark.parametrize("model", [MODEL])
def test_steering_changes_output(vllm_runner, monkeypatch, model: str) -> None:
    """Verify that non-zero steering vectors change model output
    and that clearing them restores the original behaviour."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0)

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
        ) as llm:
            # 1. Baseline (zero steering buffers)
            baseline = llm.generate([prompt], sampling)
            baseline_tokens = baseline[0][0][0]

            # Clear the clean prompt from APC so the steered run has to prefill
            # and write its own KV entries before the unsteered replay.
            assert llm.llm.reset_prefix_cache()

            # 2. Set steering on a middle layer
            def _set(worker):
                import torch

                mr = worker.model_runner
                nn_model = mr.get_model()
                # Find a decoder layer to read hidden_size and dtype
                for mod in nn_model.modules():
                    if hasattr(mod, "steering_vector"):
                        hs = mod.steering_vector.shape[1]
                        dt = mod.steering_vector.dtype
                        dv = mod.steering_vector.device
                        break
                # Steer a single middle layer with a large constant vector
                vec = torch.ones(hs, device=dv, dtype=dt) * 10.0
                target_layer = max(
                    mod.layer_idx
                    for mod in nn_model.modules()
                    if hasattr(mod, "layer_idx")
                ) // 2
                mr.set_steering_vectors({target_layer: vec})

            llm.llm.collective_rpc(_set)

            steered = llm.generate([prompt], sampling)
            steered_tokens = steered[0][0][0]

            assert steered_tokens != baseline_tokens, (
                "Non-zero steering should change model output"
            )

            # 3. Clear steering and verify output matches baseline
            llm.llm.collective_rpc(
                lambda worker: worker.model_runner.clear_steering_vectors()
            )

            restored = llm.generate([prompt], sampling)
            restored_tokens = restored[0][0][0]

            assert restored_tokens == baseline_tokens, (
                "Clearing steering should restore original output"
            )
