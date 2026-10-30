# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end check that RoPE re-anchoring (--enable-unbounded-streaming) is
transparent on a generic sliding-window model.

One long streaming-input session runs twice on a small Mistral-architecture
model (dummy weights, float32): once with a small max_model_len and
re-anchoring on, once with a max_model_len large enough to never need it. The
greedy outputs must be identical token for token.
"""

import asyncio
import random

import pytest

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import StreamingInput
from vllm.platforms import current_platform
from vllm.sampling_params import RequestOutputKind
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.engine.async_llm import AsyncLLM

if not current_platform.is_cuda():
    pytest.skip(reason="Re-anchoring is CUDA only.", allow_module_level=True)

MODEL = "mistralai/Mistral-7B-v0.1"
SLIDING_WINDOW = 512
CHUNK_TOKENS = 120
GEN_TOKENS = 8
NUM_CHUNKS = 40  # 5120 session tokens: ~9 re-anchors at max_model_len 1536

HF_OVERRIDES = {
    "num_hidden_layers": 2,
    "hidden_size": 1024,
    "intermediate_size": 2048,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "sliding_window": SLIDING_WINDOW,
}


async def _run_session(**engine_kwargs) -> list[int]:
    engine_args = AsyncEngineArgs(
        model=MODEL,
        load_format="dummy",
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
        max_tokens=GEN_TOKENS,
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
        await asyncio.sleep(0.1)
    return tokens


@pytest.mark.asyncio
async def test_reanchoring_is_transparent():
    reanchored = await _run_session(
        max_model_len=1536,
        enable_unbounded_streaming=True,
        reanchor_margin_tokens=512,
    )
    reference = await _run_session(max_model_len=8192)

    assert len(reanchored) == len(reference) == NUM_CHUNKS * GEN_TOKENS
    assert reanchored == reference
