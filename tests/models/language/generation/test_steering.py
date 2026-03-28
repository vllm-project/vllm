# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for activation steering on Gemma 3."""

import pytest

from vllm import SamplingParams
from vllm.model_executor.layers.steering import (
    DEFAULT_HOOK_POINT,
    HOOK_POINT_VECTOR_ATTR,
)

MODEL = "google/gemma-3-4b-it"

# Shorthand
_HP = DEFAULT_HOOK_POINT.value
_VEC_ATTR = HOOK_POINT_VECTOR_ATTR[DEFAULT_HOOK_POINT]


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
                    if hasattr(mod, _VEC_ATTR) and hasattr(mod, "layer_idx"):
                        layers[mod.layer_idx] = getattr(
                            mod, _VEC_ATTR
                        ).shape[1]
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
                args=({_HP: {target_layer: vec}}, False),
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


@pytest.mark.parametrize("model", [MODEL])
def test_per_request_steering_via_sampling_params(
    vllm_runner, monkeypatch, model: str
) -> None:
    """Verify that per-request steering_vectors in SamplingParams
    changes output and that different steering produces different results."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0)

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            # 1. Baseline (no steering)
            baseline = llm.llm.generate([prompt], sampling)
            baseline_tokens = list(baseline[0].outputs[0].token_ids)

            assert llm.llm.reset_prefix_cache()

            # 2. Discover steerable layers
            def _discover(worker):
                layers = {}
                model_inst = worker.model_runner.get_model()
                for mod in model_inst.modules():
                    if hasattr(mod, _VEC_ATTR) and hasattr(mod, "layer_idx"):
                        layers[mod.layer_idx] = getattr(
                            mod, _VEC_ATTR
                        ).shape[1]
                return layers

            layer_info = llm.llm.collective_rpc(_discover)[0]
            target_layer = max(layer_info.keys()) // 2
            hidden_size = layer_info[target_layer]

            # 3. Generate with per-request steering via SamplingParams
            steered_sampling = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={target_layer: [500.0] * hidden_size},
            )

            steered = llm.llm.generate([prompt], steered_sampling)
            steered_tokens = list(steered[0].outputs[0].token_ids)

            assert steered_tokens != baseline_tokens, (
                "Per-request steering should change model output"
            )

            # 4. Verify baseline is unchanged (no contamination)
            assert llm.llm.reset_prefix_cache()
            restored = llm.llm.generate([prompt], sampling)
            restored_tokens = list(restored[0].outputs[0].token_ids)

            assert restored_tokens == baseline_tokens, (
                "Per-request steering should not contaminate other requests"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_per_request_steering_concurrent_with_cuda_graphs(
    vllm_runner, monkeypatch, model: str
) -> None:
    """Test that different per-request steering configs in the same batch
    produce different outputs, and that CUDA graph replays correctly pick
    up updated steering buffers between steps.

    This sends multiple requests simultaneously so they land in the same
    batch during decode, exercising the request-indexed gather with
    CUDA graphs active.
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        base_sampling = SamplingParams(max_tokens=10, temperature=0.0)

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=False,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            # 1. Discover steerable layers
            def _discover(worker):
                layers = {}
                model_inst = worker.model_runner.get_model()
                for mod in model_inst.modules():
                    if hasattr(mod, _VEC_ATTR) and hasattr(
                        mod, "layer_idx"
                    ):
                        layers[mod.layer_idx] = getattr(
                            mod, _VEC_ATTR
                        ).shape[1]
                return layers

            layer_info = llm.llm.collective_rpc(_discover)[0]
            target_layer = max(layer_info.keys()) // 2
            hidden_size = layer_info[target_layer]

            # 2. Create three requests: no steering, positive, negative
            no_steer = SamplingParams(max_tokens=10, temperature=0.0)
            steer_pos = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={target_layer: [500.0] * hidden_size},
            )
            steer_neg = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={target_layer: [-500.0] * hidden_size},
            )

            # 3. Send all three simultaneously so they batch together
            outputs = llm.llm.generate(
                [prompt, prompt, prompt],
                [no_steer, steer_pos, steer_neg],
            )

            tokens_none = list(outputs[0].outputs[0].token_ids)
            tokens_pos = list(outputs[1].outputs[0].token_ids)
            tokens_neg = list(outputs[2].outputs[0].token_ids)

            # Positive and negative steering should produce different output
            assert tokens_pos != tokens_neg, (
                "Opposite steering vectors should produce different outputs"
            )

            # At least one steered output should differ from unsteered
            assert tokens_pos != tokens_none or tokens_neg != tokens_none, (
                "At least one steered request should differ from unsteered"
            )

            # 4. Run again without steering to verify CUDA graph replays
            #    pick up updated (cleared) buffer contents
            outputs2 = llm.llm.generate(
                [prompt, prompt],
                [no_steer, no_steer],
            )

            tokens_none2 = list(outputs2[0].outputs[0].token_ids)
            tokens_none3 = list(outputs2[1].outputs[0].token_ids)

            # Unsteered should be consistent across runs
            assert tokens_none2 == tokens_none, (
                "Unsteered output should be deterministic across runs"
            )
            assert tokens_none3 == tokens_none, (
                "Both unsteered requests should match baseline"
            )
