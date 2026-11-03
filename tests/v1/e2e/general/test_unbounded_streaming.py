# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""--enable-unbounded-streaming-input on a generic sliding-window model.

One streaming-input session runs twice on Llama-3.2-1B, every layer overridden to a
sliding window: with a small max_model_len and the flag on, then with a max_model_len
that needs no shift. The greedy logprobs must match up to rounding.
"""

import random

import pytest

import vllm.envs as envs
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import StreamingInput
from vllm.platforms import current_platform
from vllm.sampling_params import RequestOutputKind
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.engine.async_llm import AsyncLLM

if not current_platform.is_cuda():
    pytest.skip(reason="Re-anchoring is CUDA only.", allow_module_level=True)

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
SLIDING_WINDOW = 512
CHUNK_TOKENS = 120
NUM_CHUNKS = 40  # 8 to 11 shifts at max_model_len 1536
# Measured in float32: up to 0.05 with the shifts, up to 2.9 without the rotation.
LOGPROB_TOLERANCE = 0.2

HF_OVERRIDES = {
    "layer_types": ["sliding_attention"] * 16,
    "sliding_window": SLIDING_WINDOW,
}


async def _run_session(
    gen_tokens: int, **engine_kwargs
) -> list[tuple[int, dict[int, float]]]:
    """Returns each generated token with the logprobs of the top tokens."""
    engine_args = AsyncEngineArgs(
        model=MODEL,
        hf_overrides=HF_OVERRIDES,
        dtype="float32",
        seed=0,
        enforce_eager=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.5,
        **engine_kwargs,
    )
    with set_default_torch_num_threads(1):
        engine = AsyncLLM.from_engine_args(engine_args)

    rng = random.Random(0)
    chunks = [
        [rng.randrange(3, 100000) for _ in range(CHUNK_TOKENS)]
        for _ in range(NUM_CHUNKS)
    ]

    async def inputs():
        for ids in chunks:
            yield StreamingInput(prompt={"prompt_token_ids": ids})

    params = SamplingParams(
        max_tokens=gen_tokens,
        ignore_eos=True,
        temperature=0.0,
        logprobs=5,
        output_kind=RequestOutputKind.DELTA,
    )
    outputs: list[tuple[int, dict[int, float]]] = []
    try:
        async for out in engine.generate(inputs(), params, "session"):
            for completion in out.outputs:
                for token, top in zip(completion.token_ids, completion.logprobs or ()):
                    assert top is not None
                    outputs.append(
                        (token, {tok: lp.logprob for tok, lp in top.items()})
                    )
    finally:
        engine.shutdown()
    return outputs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "use_v2_model_runner, gen_tokens",
    [(False, 1), (True, 1), (True, 8)],
    ids=["v1", "v2", "v2-eight-tokens-per-chunk"],
)
async def test_reanchoring_matches_reference(
    monkeypatch, use_v2_model_runner, gen_tokens
):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(use_v2_model_runner)))
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    envs.disable_envs_cache()

    reanchored = await _run_session(
        gen_tokens, max_model_len=1536, enable_unbounded_streaming_input=True
    )
    reference = await _run_session(gen_tokens, max_model_len=8192)

    assert len(reanchored) == len(reference) == NUM_CHUNKS * gen_tokens
    for (token, top), (ref_token, ref_top) in zip(reanchored, reference):
        assert abs(top[token] - ref_top[ref_token]) < LOGPROB_TOLERANCE
        if token != ref_token:
            # Rounding can only flip a near tie.
            assert ref_top[ref_token] - ref_top[token] < LOGPROB_TOLERANCE
            # With one token per chunk the sampled token is dropped, so the later
            # inputs stay the same. Otherwise they differ from here on.
            if gen_tokens > 1:
                break
