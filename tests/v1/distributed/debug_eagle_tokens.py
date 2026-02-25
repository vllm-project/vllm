# SPDX-License-Identifier: Apache-2.0
# Debug script to track token-by-token outputs
import asyncio
import os
from contextlib import AsyncExitStack
from dataclasses import replace

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

DP_SIZE = int(os.getenv("DP_SIZE", 1))


async def test_debug_eagle():
    target_model = "meta-llama/Llama-3.1-8B-Instruct"
    draft_model = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"

    engine_args = AsyncEngineArgs(
        model=target_model,
        tokenizer_mode="auto",
        enforce_eager=True,
        tensor_parallel_size=int(os.getenv("TP_SIZE", 1)),
        data_parallel_size=DP_SIZE,
        data_parallel_backend="mp",
        trust_remote_code=True,
        max_model_len=16384,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )

    eagle_engine_args = replace(
        engine_args,
        speculative_config={
            "model": draft_model,
            "method": "eagle",
            "num_speculative_tokens": 1,
        },
    )

    prompt = "This is a test of data parallel with eagle"
    num_expected_tokens = 100
    sampling_params = SamplingParams(
        min_tokens=num_expected_tokens,
        max_tokens=num_expected_tokens,
        ignore_eos=True,
        output_kind=RequestOutputKind.DELTA,  # Get incremental updates
        temperature=0,
    )

    async def collect_tokens(given_engine: AsyncLLM, label: str):
        tokens = []
        async for out in given_engine.generate(
            request_id=f"test-{label}", prompt=prompt, sampling_params=sampling_params
        ):
            new_tokens = out.outputs[0].token_ids
            tokens.extend(new_tokens)
            print(f"[{label}] Step {len(tokens)}: tokens={new_tokens}, total={len(tokens)}")

            if len(tokens) >= num_expected_tokens:
                break
        return tokens[:num_expected_tokens]

    async def engine_create_and_collect(engine_args: AsyncEngineArgs, label: str):
        async with AsyncExitStack() as after:
            engine = AsyncLLM.from_engine_args(engine_args)
            after.callback(engine.shutdown)
            tokens = await asyncio.wait_for(collect_tokens(engine, label), timeout=60)
        return tokens

    print("=" * 80)
    print("Running EAGLE...")
    print("=" * 80)
    token_ids_with_eagle = await engine_create_and_collect(eagle_engine_args, "EAGLE")

    print("\n" + "=" * 80)
    print("Running Baseline...")
    print("=" * 80)
    token_ids_no_eagle = await engine_create_and_collect(engine_args, "BASELINE")

    print("\n" + "=" * 80)
    print("Comparison:")
    print("=" * 80)

    divergence_point = None
    for i, (t1, t2) in enumerate(zip(token_ids_with_eagle, token_ids_no_eagle)):
        if t1 != t2:
            if divergence_point is None:
                divergence_point = i
            print(f"Token {i}: EAGLE={t1}, BASELINE={t2} *** DIVERGED ***")
        else:
            if divergence_point is not None and i < divergence_point + 5:
                print(f"Token {i}: {t1} (match)")

    if divergence_point is not None:
        print(f"\n!!! Divergence detected at token index {divergence_point} !!!")
        print(f"EAGLE tokens around divergence: {token_ids_with_eagle[max(0, divergence_point-5):divergence_point+5]}")
        print(f"BASELINE tokens around divergence: {token_ids_no_eagle[max(0, divergence_point-5):divergence_point+5]}")
    else:
        print("\nNo divergence - outputs match!")


if __name__ == "__main__":
    asyncio.run(test_debug_eagle())
