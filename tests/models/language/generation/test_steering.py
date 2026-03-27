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
            # Use llm.llm.generate() to get RequestOutput objects so we
            # can compare only the generated completion tokens, not the
            # prompt+completion concatenation that VllmRunner.generate()
            # returns.
            baseline = llm.llm.generate([prompt], sampling)
            baseline_tokens = list(baseline[0].outputs[0].token_ids)

            # Clear the clean prompt from APC so the steered run has to
            # prefill and write its own KV entries before the unsteered
            # replay.
            assert llm.llm.reset_prefix_cache()

            # 2. Discover hidden_size and pick a middle layer.
            def _discover(worker):
                layers = {}
                model = worker.model_runner.get_model()
                for mod in model.modules():
                    if hasattr(mod, "steering_vector") and hasattr(mod, "layer_idx"):
                        layers[mod.layer_idx] = mod.steering_vector.shape[1]
                return layers

            layer_info = llm.llm.collective_rpc(_discover)[0]
            target_layer = max(layer_info.keys()) // 2
            hidden_size = layer_info[target_layer]

            # 3. Set steering via WorkerBase (same path as HTTP API).
            #    With dummy (random) weights the magnitude must be large
            #    enough to overcome noise in the logit space.
            vec = [500.0] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                args=({target_layer: vec}, False),
            )

            steered = llm.llm.generate([prompt], sampling)
            steered_tokens = list(steered[0].outputs[0].token_ids)

            assert steered_tokens != baseline_tokens, (
                "Non-zero steering should change model output"
            )

            # 4. Clear steering and verify output matches baseline
            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored = llm.llm.generate([prompt], sampling)
            restored_tokens = list(restored[0].outputs[0].token_ids)

            assert restored_tokens == baseline_tokens, (
                "Clearing steering should restore original output"
            )
