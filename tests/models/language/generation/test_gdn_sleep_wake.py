# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for hybrid GDN/Mamba models under sleep -> wake.

Hybrid Mamba / gated-delta-net (GDN) models (e.g. Qwen3-Next) keep a
persisted conv + recurrent state cache. With sleep mode (the RLHF reuse
pattern: ``sleep()`` -> weight update -> ``wake_up()``) the state-cache tag is
discarded on sleep and its device memory is re-created on wake. If a *new*
sequence's state slot is consumed before being reset, the gated-delta-rule
kernel faithfully propagates whatever is in that (now non-zeroed) memory; when
it contains NaN/inf the output becomes NaN and ``argmax`` collapses every token
to id 0 (which decodes to ``"!"``), giving ``reward=0`` / NaN log-probs in RL
training.

This test repeatedly exercises both public wake paths on a real hybrid GDN
model. It verifies the bound recurrent state is zero immediately after every
wake, then checks that deterministic generation matches the pre-sleep result.
"""

import math

import pytest

from vllm import LLM, SamplingParams

from ....utils import create_new_process_for_each_test
from ...utils import check_outputs_equal

# Small Qwen3-Next (GDN) model already used by the hybrid model test-suite.
MODEL = "tiny-random/qwen3-next-moe"

PROMPTS = [
    "The capital of France is",
    "Once upon a time,",
    "1, 2, 3, 4,",
    "Water is made of",
]


def _qwen_gdn_state_summary(model):
    """Return primitive state metadata suitable for ``LLM.apply_model``."""
    import torch

    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    layers = [
        module
        for module in model.modules()
        if isinstance(module, QwenGatedDeltaNetAttention)
    ]
    states = [state for layer in layers for state in layer.kv_cache]
    unique_states: dict[
        tuple[int, torch.dtype, tuple[int, ...], tuple[int, ...]],
        torch.Tensor,
    ] = {}
    for state in states:
        key = (
            state.data_ptr(),
            state.dtype,
            tuple(state.shape),
            tuple(state.stride()),
        )
        unique_states.setdefault(key, state)
    # These views can span most of the reserved KV allocation and are strided
    # by the packed page size. Reducing the full view may materialize hundreds
    # of GiB on ROCm. The next requests consume the first ``max_num_seqs``
    # slots, so inspect exactly those slots without a giant temporary.
    has_nonzero = any(
        bool(torch.count_nonzero(state[: len(PROMPTS)]).item())
        for state in unique_states.values()
    )
    return len(layers), len(states), len(unique_states), has_nonzero


def _normalize_and_check(outputs):
    normalized = []
    for output in outputs:
        completion = output.outputs[0]
        token_ids = list(completion.token_ids)
        assert token_ids, "empty generation after wake_up"
        assert completion.logprobs is not None
        assert len(completion.logprobs) == len(token_ids)
        for token_id, step_logprobs in zip(token_ids, completion.logprobs):
            assert step_logprobs
            assert token_id in step_logprobs
            assert all(
                math.isfinite(logprob.logprob) for logprob in step_logprobs.values()
            ), "non-finite log-prob after wake_up (stale GDN state)"
        normalized.append((token_ids, completion.text))
    return normalized


@pytest.mark.hybrid_model
@create_new_process_for_each_test()
def test_gdn_sleep_wake_no_stale_state(enable_pickle):
    sampling_params = SamplingParams(
        temperature=0.0,
        seed=0,
        ignore_eos=True,
        max_tokens=16,
        logprobs=1,
    )

    # Keep the reserved fraction low. On some (notably ROCm/amdgpu) drivers the
    # VRAM discarded by ``sleep()`` is not returned to the free pool before
    # ``wake_up()`` re-creates it, so the woken allocation must coexist with the
    # not-yet-reclaimed one (~2x peak). A high ``gpu_memory_utilization`` then
    # OOMs in ``cuMemCreate`` on wake. The model is tiny, so a small fraction
    # still leaves ample KV/state cache while keeping the sleep/wake cycle well
    # within device memory.
    llm = LLM(
        model=MODEL,
        enable_sleep_mode=True,
        enforce_eager=True,
        max_model_len=1024,
        max_num_seqs=len(PROMPTS),
        gpu_memory_utilization=0.4,
        trust_remote_code=True,
    )

    baseline = _normalize_and_check(
        llm.generate(PROMPTS, sampling_params, use_tqdm=False)
    )
    baseline_state = llm.apply_model(_qwen_gdn_state_summary)
    for num_layers, num_states, num_unique_states, has_nonzero in baseline_state:
        assert num_layers > 0, "test model has no bound Qwen GDN layers"
        assert num_states == 2 * num_layers
        assert num_unique_states > 0
        assert has_nonzero, "warm generation did not populate Qwen GDN state"

    # Exercise each public path twice without recreating the engine. Default
    # sleep offloads weights and discards the KV/GDN state allocation.
    for cycle, wake_mode in enumerate(("full", "split", "full", "split"), 1):
        llm.sleep(level=1)
        if wake_mode == "full":
            llm.wake_up()
        else:
            llm.wake_up(tags=["weights"])
            llm.wake_up(tags=["kv_cache"])

        for num_layers, num_states, num_unique_states, has_nonzero in llm.apply_model(
            _qwen_gdn_state_summary
        ):
            assert num_layers > 0
            assert num_states == 2 * num_layers
            assert num_unique_states > 0
            assert not has_nonzero, (
                f"Qwen GDN state was not zero after {wake_mode} wake, cycle {cycle}"
            )

        after = _normalize_and_check(
            llm.generate(PROMPTS, sampling_params, use_tqdm=False)
        )
        check_outputs_equal(
            outputs_0_lst=baseline,
            outputs_1_lst=after,
            name_0="before sleep",
            name_1=f"after {wake_mode} wake cycle {cycle}",
        )

        for _, _, _, has_nonzero in llm.apply_model(_qwen_gdn_state_summary):
            assert has_nonzero, "post-wake generation did not populate Qwen GDN state"
