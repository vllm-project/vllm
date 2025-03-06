# V1 LoRA

This page lists some caveats in using LoRA with V1. For general information about using LoRA adapters, please refer to [LoRA Adapters](../../features/lora.md)

## Caveats

### Using Long Context LoRA Adapters with V1

Long Context LoRA adapters are LoRA adapters that are fine-tuned to increase the context sizes of pre-trained large language models.  

Support for Long Context LoRA adapters is essentially a case for supporting request-level model-length setting. However, vLLM’s V1 architecture assumes a static `max_model_len` and is less amenable to this use case.

#### Workaround

When using Long Context LoRA adapters with V1, we recommend that you,

- Explicitly set the engine's `max_model_len` argument to the largest context length you expect the engine to process.
- Explicitly set each request's `max_num_tokens` appropriately. i.e.,

  - For requests targeting the base model, set `max_num_tokens` to less than or equal to the base model's model length,
  - For requests targeting LoRA adapters, set `max_num_tokens` to less than or equal to the context length of the LoRA adapter.

#### Expected Behaviour

The following is the expected behaviour for different scenarios when Long Context LoRA models are in play.

- The engine is constructed with `max_model_len` set to None or set to the base model's model length. When the engine receives Long Context LoRA requests,
  - If prompt length <= `max_model_len` : The request will be accepted, but the output will be truncated at max_model_len.
  - If prompt length  > `max_model_len` : The request will be rejected citing that the prompt is too long.

- The engine is constructed with `max_model_len` set to the largest context length of any adapter that it might receive requests for. As an example, let the base model’s model length be 4K and let there be 2 Long Context LoRA adapters,
  - 16K adapter : Can support a context length upto 16K
  - 32K adapter : Can support a context length upto 32K
  - In this case, the `max_model_len` will be set to 32K

  - User Request Scenarios:
    - Request for the 16K adapter, and `max_num_tokens` is None
      - The request will produce incorrect results when the total number of tokens (prompt + generated) goes beyond 16K. (Tokens generated beyond the 16K context length will be incorrect)
    - Request for the 32K adapter, and `max_num_tokens` is None
      - The request will produce correct outputs.
    - Request for the base model, and `max_num_tokens` is None
      - The request will produce incorrect requests when the total number tokens (prompt + generated) goes beyond 4K. (Tokens generated beyond the 4K context length will be incorrect)
    - Request for the 16K adapter, and `max_num_tokens` <= 16K
      - The request will produce correct results
    - Request for the 32K adapter, and `max_num_token` <= 32K
      - The request will produce correct results
    - Request for the base model, and `max_num_tokens` <= 4K
      - The request will produce correct results.

Reference: Please look at  `tests/lora/test_long_context.py`
