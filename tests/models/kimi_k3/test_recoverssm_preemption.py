# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 KDA RecoverSSM under decode preemption and aborts.

A KV pool sized for the running requests' first block forces preemption once
their decode crosses into a second attention block. Preempted requests are
recomputed (prefill again, from their prompt and generated tokens) while others
keep verifying and committing; aborted requests leave mid-decode. The weights
are random (uniform logits), so this checks that every surviving request
completes, not token values; the CuTe kernels are compared against the Triton
ones and against speculative decoding without RecoverSSM in
test_recoverssm_kernels.py.
"""

import random

import pytest

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform

# Official Kimi-K3 config with 8 layers and no weights, and its DSpark draft.
MODEL = "riverclouds/Kimi-K3-8L-dummy"
DRAFT = "riverclouds/Kimi-K3-DSpark-for-8L-dummy"
NUM_REQUESTS = 6
MAX_TOKENS = 48

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="Kimi-K3 NVIDIA kernels require the SM100 family",
)


def _llm(backend: str, num_gpu_blocks: int | None = None) -> LLM:
    recoverssm = backend != "off"
    return LLM(
        MODEL,
        trust_remote_code=True,
        language_model_only=True,
        load_format="dummy",
        enable_flashinfer_autotune=False,
        enable_prefix_caching=False,
        max_model_len=16384,
        max_num_seqs=NUM_REQUESTS,
        gpu_memory_utilization=0.6,
        disable_log_stats=False,
        num_gpu_blocks_override=num_gpu_blocks,
        mamba_ssm_cache_dtype="bfloat16",
        use_replayssm=recoverssm,
        additional_config={"kda_recoverssm_backend": backend} if recoverssm else {},
        speculative_config={
            "method": "dspark",
            "model": DRAFT,
            "num_speculative_tokens": 4,
            "draft_sample_method": "probabilistic",
            "rejection_sample_method": "block",
        },
    )


def _shutdown(llm: LLM) -> None:
    llm.llm_engine.engine_core.shutdown()
    cleanup_dist_env_and_memory()


@pytest.mark.parametrize("backend", ["off", "triton", "cute"])
def test_recoverssm_decode_preemption_and_abort(backend: str) -> None:
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0, ignore_eos=True)

    # The attention block size, and the KV pool blocks one decoding request
    # holds (its Mamba state blocks and one attention block).
    llm = _llm(backend)
    try:
        cache_config = llm.llm_engine.vllm_config.cache_config
        block_size = cache_config.block_size
        assert cache_config.num_gpu_blocks is not None
        rng = random.Random(0)
        prompts = [
            {
                "prompt_token_ids": [
                    rng.randrange(1000, 60000) for _ in range(block_size - 24)
                ]
            }
            for _ in range(NUM_REQUESTS)
        ]
        # After its prefill, before decoding reserves lookahead slots.
        llm.llm_engine.add_request("probe", prompts[0], params)
        llm.llm_engine.step()
        usage = next(
            m.value for m in llm.get_metrics() if m.name == "vllm:kv_cache_usage_perc"
        )
        blocks_per_request = round(usage * cache_config.num_gpu_blocks)
        assert blocks_per_request > 0
    finally:
        _shutdown(llm)

    # Room for every request's first blocks, and 2 spare: decoding into a
    # second attention block preempts all but two of them.
    llm = _llm(backend, NUM_REQUESTS * blocks_per_request + 2)
    try:
        outputs = llm.generate(prompts, params)
        assert [len(o.outputs[0].token_ids) for o in outputs] == [MAX_TOKENS] * len(
            prompts
        )
        preemptions = sum(
            m.value for m in llm.get_metrics() if m.name == "vllm:num_preemptions"
        )
        assert preemptions > 0

        # Abort half of the requests mid-decode; the rest finish.
        engine = llm.llm_engine
        for i, prompt in enumerate(prompts):
            engine.add_request(f"r{i}", prompt, params)
        for _ in range(4):
            engine.step()
        engine.abort_request([f"r{i}" for i in range(0, NUM_REQUESTS, 2)])
        finished = {}
        while engine.has_unfinished_requests():
            for output in engine.step():
                if output.finished:
                    finished[output.request_id] = len(output.outputs[0].token_ids)
        assert finished == {f"r{i}": MAX_TOKENS for i in range(1, NUM_REQUESTS, 2)}
    finally:
        _shutdown(llm)
