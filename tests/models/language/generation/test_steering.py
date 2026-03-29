# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for activation steering on Gemma 3.

Covers global steering via the worker API, per-request steering via
SamplingParams, concurrent batching with CUDA graphs, and three-tier
prefill/decode phase-specific steering.
"""

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discover_layers(llm):
    """Return (target_layer, hidden_size) for the default hook point."""

    def _discover(worker):
        layers = {}
        model_inst = worker.model_runner.get_model()
        for mod in model_inst.modules():
            if hasattr(mod, _VEC_ATTR) and hasattr(mod, "layer_idx"):
                layers[mod.layer_idx] = getattr(mod, _VEC_ATTR).shape[1]
        return layers

    layer_info = llm.llm.collective_rpc(_discover)[0]
    target_layer = max(layer_info.keys()) // 2
    hidden_size = layer_info[target_layer]
    return target_layer, hidden_size


def _gen_tokens(llm, prompt, sampling):
    """Generate and return token ids list."""
    result = llm.llm.generate([prompt], sampling)
    return list(result[0].outputs[0].token_ids)


# ---------------------------------------------------------------------------
# Existing tests (updated for kwargs-based collective_rpc API)
# ---------------------------------------------------------------------------


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
            baseline_tokens = _gen_tokens(llm, prompt, sampling)

            # Clear the clean prompt from APC so the steered run has to
            # prefill and write its own KV entries before the unsteered
            # replay.
            assert llm.llm.reset_prefix_cache()

            # 2. Discover hidden_size and pick a middle layer.
            target_layer, hidden_size = _discover_layers(llm)

            # 3. Set steering via WorkerBase (same path as HTTP API).
            #    With dummy (random) weights the magnitude must be large
            #    enough to overcome noise in the logit space.
            vec = [500.0] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={"vectors": {_HP: {target_layer: vec}}},
            )

            steered_tokens = _gen_tokens(llm, prompt, sampling)

            assert steered_tokens != baseline_tokens, (
                "Non-zero steering should change model output"
            )

            # 4. Clear steering and verify output matches baseline
            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored_tokens = _gen_tokens(llm, prompt, sampling)

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
            baseline_tokens = _gen_tokens(llm, prompt, sampling)

            assert llm.llm.reset_prefix_cache()

            # 2. Discover steerable layers
            target_layer, hidden_size = _discover_layers(llm)

            # 3. Generate with per-request steering via SamplingParams
            steered_sampling = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={
                    _HP: {target_layer: [500.0] * hidden_size},
                },
            )

            steered_tokens = _gen_tokens(llm, prompt, steered_sampling)

            assert steered_tokens != baseline_tokens, (
                "Per-request steering should change model output"
            )

            # 4. Verify baseline is unchanged (no contamination)
            assert llm.llm.reset_prefix_cache()
            restored_tokens = _gen_tokens(llm, prompt, sampling)

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

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=False,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            # 1. Discover steerable layers
            target_layer, hidden_size = _discover_layers(llm)

            # 2. Create three requests: no steering, positive, negative
            no_steer = SamplingParams(max_tokens=10, temperature=0.0)
            steer_pos = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={
                    _HP: {target_layer: [500.0] * hidden_size},
                },
            )
            steer_neg = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={
                    _HP: {target_layer: [-500.0] * hidden_size},
                },
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


# ---------------------------------------------------------------------------
# Prefill steering tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", [MODEL])
def test_prefill_steering_changes_output(vllm_runner, monkeypatch, model: str) -> None:
    """Verify that prefill-specific steering via SamplingParams changes
    output and does not contaminate subsequent unsteered requests."""
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
            baseline_tokens = _gen_tokens(llm, prompt, sampling)

            assert llm.llm.reset_prefix_cache()

            # 2. Discover steerable layers
            target_layer, hidden_size = _discover_layers(llm)

            # 3. Generate with prefill-specific steering
            prefill_steered = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                prefill_steering_vectors={
                    _HP: {target_layer: [500.0] * hidden_size},
                },
            )

            steered_tokens = _gen_tokens(llm, prompt, prefill_steered)

            assert steered_tokens != baseline_tokens, (
                "Prefill steering should change model output"
            )

            # 4. Reset and verify no contamination
            assert llm.llm.reset_prefix_cache()
            restored_tokens = _gen_tokens(llm, prompt, sampling)

            assert restored_tokens == baseline_tokens, (
                "Prefill steering should not contaminate subsequent requests"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_decode_only_steering_via_new_field(
    vllm_runner, monkeypatch, model: str
) -> None:
    """Verify that decode_steering_vectors (the new field) changes output
    compared to an unsteered baseline."""
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
            baseline_tokens = _gen_tokens(llm, prompt, sampling)

            assert llm.llm.reset_prefix_cache()

            # 2. Discover steerable layers
            target_layer, hidden_size = _discover_layers(llm)

            # 3. Generate with decode-specific steering
            decode_steered = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                decode_steering_vectors={
                    _HP: {target_layer: [500.0] * hidden_size},
                },
            )

            steered_tokens = _gen_tokens(llm, prompt, decode_steered)

            assert steered_tokens != baseline_tokens, (
                "Decode-only steering should change model output"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_prefill_and_decode_different_steering(
    vllm_runner, monkeypatch, model: str
) -> None:
    """Verify that using different vectors for prefill vs decode produces
    different output than using the same vector for both phases."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            target_layer, hidden_size = _discover_layers(llm)

            # 1. Same vector for both phases via base steering_vectors
            both_same = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={
                    _HP: {target_layer: [500.0] * hidden_size},
                },
            )

            result_both = _gen_tokens(llm, prompt, both_same)

            assert llm.llm.reset_prefix_cache()

            # 2. Different vectors for prefill vs decode
            split = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                prefill_steering_vectors={
                    _HP: {target_layer: [500.0] * hidden_size},
                },
                decode_steering_vectors={
                    _HP: {target_layer: [-500.0] * hidden_size},
                },
            )

            result_split = _gen_tokens(llm, prompt, split)

            assert result_both != result_split, (
                "Different prefill vs decode steering should produce "
                "different output than uniform steering"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_additive_composition(vllm_runner, monkeypatch, model: str) -> None:
    """Verify the three-tier additive model: base + decode-specific should
    equal a single vector with the summed magnitude.

    base=250 + decode=250 should produce the same decode-phase output as
    base=500 alone (since base applies to decode too).
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            target_layer, hidden_size = _discover_layers(llm)

            H = hidden_size

            # 1. Base only at 250
            base_only = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={
                    _HP: {target_layer: [250.0] * H},
                },
            )

            result_base = _gen_tokens(llm, prompt, base_only)

            assert llm.llm.reset_prefix_cache()

            # 2. Base 250 + decode 250 (additive = 500 for decode)
            additive = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={
                    _HP: {target_layer: [250.0] * H},
                },
                decode_steering_vectors={
                    _HP: {target_layer: [250.0] * H},
                },
            )

            result_additive = _gen_tokens(llm, prompt, additive)

            assert llm.llm.reset_prefix_cache()

            # 3. Base only at 500
            full = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={
                    _HP: {target_layer: [500.0] * H},
                },
            )

            result_full = _gen_tokens(llm, prompt, full)

            # Sanity: base-only at 250 should differ from base at 500
            # (confirms the magnitudes are meaningfully different).
            assert result_base != result_full, "base=250 should differ from base=500"

            # The additive composition (250+250=500 for decode) should match
            # the single 500 vector (since most generated tokens are decode).
            assert result_additive == result_full, (
                "Additive composition (base=250 + decode=250) should equal "
                "a single base=500 vector for decode-phase tokens"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_prefix_cache_respects_prefill_steering(
    vllm_runner, monkeypatch, model: str
) -> None:
    """Verify that prefix cache correctly separates different prefill
    steering: same prompt with different prefill steering should produce
    different outputs, but same prompt with same prefill steering should
    hit the cache and produce identical output."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling_no_steer = SamplingParams(max_tokens=10, temperature=0.0)

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            target_layer, hidden_size = _discover_layers(llm)

            # 1. Request A: with prefill steering
            steered_sampling = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                prefill_steering_vectors={
                    _HP: {target_layer: [500.0] * hidden_size},
                },
            )

            tokens_a = _gen_tokens(llm, prompt, steered_sampling)

            # 2. Request B: same prompt, NO prefill steering
            tokens_b = _gen_tokens(llm, prompt, sampling_no_steer)

            # Different prefill steering means different KV cache
            assert tokens_a != tokens_b, (
                "Different prefill steering should produce different output "
                "even with prefix caching enabled"
            )

            # 3. Request C: same prompt, same prefill steering as A
            #    Should hit prefix cache and produce identical output
            tokens_c = _gen_tokens(llm, prompt, steered_sampling)

            assert tokens_c == tokens_a, (
                "Same prefill steering should hit prefix cache and produce "
                "identical output"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_co_located_scale(vllm_runner, monkeypatch, model: str) -> None:
    """Verify that the co-located scale format produces the same result
    as a pre-scaled bare vector: [500]*H should equal {"vector": [250]*H,
    "scale": 2.0}."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            target_layer, hidden_size = _discover_layers(llm)

            H = hidden_size

            # 1. Bare vector at magnitude 500
            bare = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={
                    _HP: {target_layer: [500.0] * H},
                },
            )

            result_bare = _gen_tokens(llm, prompt, bare)

            assert llm.llm.reset_prefix_cache()

            # 2. Co-located scale: vector=250, scale=2.0 => effective 500
            scaled = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={
                    _HP: {
                        target_layer: {
                            "vector": [250.0] * H,
                            "scale": 2.0,
                        },
                    },
                },
            )

            result_scaled = _gen_tokens(llm, prompt, scaled)

            assert result_bare == result_scaled, (
                "Co-located scale (250 * 2.0) should produce same output "
                "as bare vector (500)"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_global_prefill_steering_via_worker_api(
    vllm_runner, monkeypatch, model: str
) -> None:
    """Verify global three-tier steering via the worker API: setting
    prefill-specific global vectors changes output, and clearing them
    restores the baseline."""
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
            # 1. Baseline (no global steering)
            baseline_tokens = _gen_tokens(llm, prompt, sampling)

            assert llm.llm.reset_prefix_cache()

            # 2. Discover steerable layers
            target_layer, hidden_size = _discover_layers(llm)

            # 3. Set global prefill-specific vectors via worker API
            vec = [500.0] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={
                    "prefill_vectors": {_HP: {target_layer: vec}},
                },
            )

            steered_tokens = _gen_tokens(llm, prompt, sampling)

            assert steered_tokens != baseline_tokens, (
                "Global prefill steering should change model output"
            )

            # 4. Clear and verify restoration
            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored_tokens = _gen_tokens(llm, prompt, sampling)

            assert restored_tokens == baseline_tokens, (
                "Clearing global prefill steering should restore baseline"
            )
