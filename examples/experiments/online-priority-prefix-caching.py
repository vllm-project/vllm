# ruff: noqa: E501
# SPDX-License-Identifier: Apache-2.0
from openai import OpenAI

# Start a vllm server with the following flags:
# vllm serve \
#   facebook/opt-125m \
#   --port 8001 \
#   --enable-prompt-tokens-details \
#   --block-size 16 \
#   --num-gpu-blocks-override 5

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)
models = client.models.list()
model = models.data[0].id


def main():
    block_size = 16  # Should match the block_size in server config

    # Define prompts with exact token length
    # Using distinct integer tokens for easier tracking
    # (convert to strings since the API expects string prompts)
    x_prompt = " ".join([str(101)] * block_size)
    y_prompt = " ".join([str(102)] * block_size)
    a_prompt = " ".join([str(103)] * block_size)
    b_prompt = " ".join([str(104)] * block_size)

    print("Sending P1 requests...")
    for prompt in [x_prompt, y_prompt]:
        response = client.completions.create(model=model,
                                             prompt=prompt,
                                             max_tokens=1,
                                             temperature=0.0,
                                             extra_body={"priority": 1})
        cached = 0
        if hasattr(response.usage, 'prompt_tokens_details'
                   ) and response.usage.prompt_tokens_details:
            cached = response.usage.prompt_tokens_details.cached_tokens or 0

        print(f"Cached tokens: {cached}")
        assert cached == 0, "First request should have no cached tokens"

    # The KV cache should be [x_prompt: cached, y_prompt: cached]

    print("Verifying cache hit...")
    for prompt in [x_prompt, y_prompt]:
        response = client.completions.create(model=model,
                                             prompt=prompt,
                                             max_tokens=1,
                                             temperature=0.0,
                                             extra_body={"priority": 1})
        cached = 0
        if hasattr(response.usage, 'prompt_tokens_details'
                   ) and response.usage.prompt_tokens_details:
            cached = response.usage.prompt_tokens_details.cached_tokens or 0

        print(f"Cached tokens: {cached}")
        assert cached == block_size, f"P1 requests should cache {block_size} tokens, but got {cached}"

    print("Cache hit verified.")

    print("Sending P0 requests...")
    for prompt in [a_prompt, b_prompt]:
        response = client.completions.create(model=model,
                                             prompt=prompt,
                                             max_tokens=1,
                                             temperature=0.0,
                                             extra_body={"priority": 0})
        cached = 0
        if hasattr(response.usage, 'prompt_tokens_details'
                   ) and response.usage.prompt_tokens_details:
            cached = response.usage.prompt_tokens_details.cached_tokens or 0

        print(f"Cached tokens: {cached}")
        assert cached == 0, "First P0 request should have no cached tokens"

    # The KV cache should be [x_prompt: evicted, y_prompt: cached, a_prompt: evicted, b_prompt: cached]

    print("Now send request A and B again...")
    for prompt in [a_prompt, b_prompt]:
        response = client.completions.create(model=model,
                                             prompt=prompt,
                                             max_tokens=1,
                                             temperature=0.0,
                                             extra_body={"priority": 0})
        cached = 0
        if hasattr(response.usage, 'prompt_tokens_details'
                   ) and response.usage.prompt_tokens_details:
            cached = response.usage.prompt_tokens_details.cached_tokens or 0

        print(f"Cached tokens: {cached}")
        # A and B should trash each other's cache.
        assert cached == 0, f"P0 requests should trash each other's cache, but got {cached} cached tokens"

    # The KV cache should be [x_prompt: evicted, y_prompt: cached, a_prompt: evicted, b_prompt: cached]

    print("P1's cache should be [x_prompt: evicted, y_prompt: cached]")
    response = client.completions.create(model=model,
                                         prompt=x_prompt,
                                         max_tokens=1,
                                         temperature=0.0,
                                         extra_body={"priority": 1})
    cached = 0
    if hasattr(
            response.usage,
            'prompt_tokens_details') and response.usage.prompt_tokens_details:
        cached = response.usage.prompt_tokens_details.cached_tokens or 0

    print(f"X cached tokens: {cached}")
    assert cached == 0, f"x_prompt should be evicted, but got {cached} cached tokens"

    response = client.completions.create(model=model,
                                         prompt=y_prompt,
                                         max_tokens=1,
                                         temperature=0.0,
                                         extra_body={"priority": 1})
    cached = 0
    if hasattr(
            response.usage,
            'prompt_tokens_details') and response.usage.prompt_tokens_details:
        cached = response.usage.prompt_tokens_details.cached_tokens or 0

    print(f"Y cached tokens: {cached}")
    assert cached == block_size, f"y_prompt should cache {block_size} tokens, but got {cached} cached tokens"

    print("Test completed successfully!")


if __name__ == "__main__":
    main()
