# ruff: noqa: E501
# SPDX-License-Identifier: Apache-2.0
from vllm import LLM, SamplingParams


def main():
    block_size = 16

    llm = LLM(
        model="facebook/opt-125m",
        enforce_eager=True,
        block_size=block_size,
        # two slots for ongoing compute and two slots for free queue.
        num_gpu_blocks_override=5,
    )

    x_tokens = {"prompt_token_ids": [101] * (block_size + 1)}
    y_tokens = {"prompt_token_ids": [102] * (block_size + 1)}
    a_tokens = {"prompt_token_ids": [103] * (block_size + 1)}
    b_tokens = {"prompt_token_ids": [104] * (block_size + 1)}

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)

    print("Sending P1 requests...")
    for tokens in [x_tokens, y_tokens]:
        output = llm.generate([tokens],
                              sampling_params=sampling_params,
                              priority=[1])
        assert output[0].num_cached_tokens == 0

    # The KV cache should be [x_tokens: cached, y_tokens: cached]

    print("Verifying cache hit...")
    for tokens in [x_tokens, y_tokens]:
        outputs = llm.generate([tokens],
                               sampling_params=sampling_params,
                               priority=[1])
        assert (
            outputs[0].num_cached_tokens == block_size
        ), f"P1 requests should cache {block_size} tokens, but got {outputs[0].num_cached_tokens}"

    print("Cache hit verified.")

    print("Sending P0 requests...")
    for tokens in [a_tokens, b_tokens]:
        outputs = llm.generate([tokens],
                               sampling_params=sampling_params,
                               priority=[0])
        assert outputs[0].num_cached_tokens == 0

    # The KV cache should be [x_tokens: evicted, y_tokens: cached, a_tokens: evicted, b_tokens: cached]

    print("Now send request A and B again...")
    for tokens in [a_tokens, b_tokens]:
        outputs = llm.generate([tokens],
                               sampling_params=sampling_params,
                               priority=[0])
        # A and B should trash each other's cache.
        assert outputs[0].num_cached_tokens == 0

    # The KV cache should be [x_tokens: evicted, y_tokens: cached, a_tokens: evicted, b_tokens: cached]

    print("P1's cache should be [x_tokens: evicted, y_tokens: cached]")
    outputs = llm.generate([x_tokens],
                           sampling_params=sampling_params,
                           priority=[1])
    assert outputs[0].num_cached_tokens == 0

    outputs = llm.generate([y_tokens],
                           sampling_params=sampling_params,
                           priority=[1])
    assert outputs[0].num_cached_tokens == block_size


if __name__ == "__main__":
    main()
