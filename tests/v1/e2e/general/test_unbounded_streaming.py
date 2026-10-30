# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""--enable-unbounded-streaming-input on a generic sliding-window model.

One streaming-input session runs twice on a tiny model with random weights: with a
small max_model_len and the flag on, then with a max_model_len that needs no shift.
The greedy outputs must match. Rounding from the rotation stays far below the gaps
between logits in fp32.
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

MODEL = "hmellor/tiny-random-LlamaForCausalLM"
SLIDING_WINDOW = 512
CHUNK_TOKENS = 120
NUM_CHUNKS = 40  # 8 to 11 shifts at max_model_len 1536

HF_OVERRIDES = {
    "layer_types": ["sliding_attention"] * 2,
    "sliding_window": SLIDING_WINDOW,
}


async def _run_session(gen_tokens: int, **engine_kwargs) -> list[int]:
    engine_args = AsyncEngineArgs(
        model=MODEL,
        hf_overrides=HF_OVERRIDES,
        dtype="float32",
        seed=0,
        enforce_eager=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.3,
        **engine_kwargs,
    )
    with set_default_torch_num_threads(1):
        engine = AsyncLLM.from_engine_args(engine_args)

    rng = random.Random(0)
    chunks = [
        [rng.randrange(3, 32000) for _ in range(CHUNK_TOKENS)]
        for _ in range(NUM_CHUNKS)
    ]

    async def inputs():
        for ids in chunks:
            yield StreamingInput(prompt={"prompt_token_ids": ids})

    params = SamplingParams(
        max_tokens=gen_tokens,
        ignore_eos=True,
        temperature=0.0,
        output_kind=RequestOutputKind.DELTA,
    )
    tokens: list[int] = []
    try:
        async for out in engine.generate(inputs(), params, "session"):
            for completion in out.outputs:
                tokens.extend(completion.token_ids)
    finally:
        engine.shutdown()
    return tokens


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "use_v2_model_runner, gen_tokens",
    [(False, 8), (True, 8), (True, 1)],
    ids=["v1", "v2", "v2-one-token-per-chunk"],
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
    assert reanchored == reference
