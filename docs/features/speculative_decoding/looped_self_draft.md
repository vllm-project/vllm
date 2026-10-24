# Looped Self-Drafting

Looped self-drafting generates draft tokens with fewer recurrent loops and verifies them with the full target model. It shares the target model's weights, so no separate draft checkpoint is needed. Dory is currently supported.

## Online Example

The following command enables looped self-drafting for Dory:

```bash
vllm serve nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 \
    --hf-overrides '{"n_recurrent_loops": 4}' \
    --speculative-config '{"method": "looped_self_draft", "num_speculative_tokens": 4, "num_draft_loops": 1, "draft_sample_method": "probabilistic"}'
```

## Configuration

- `num_speculative_tokens` sets the number of tokens to draft per step.
- `num_draft_loops` sets the draft model's recurrent depth. It defaults to `1` and must be at least `1`. For Dory, it must be less than `n_recurrent_loops`.
- `num_shared_loops` sets how many early loops share KV caches and activations with the target. It defaults to `num_draft_loops` and accepts values from `0` through `num_draft_loops`. The input block is always shared. Set this to `0` to share only the input block.
- `reuse_draft_activations` defaults to `true`. Set it to `false` to disable activation reuse while keeping KV sharing.
- `draft_sample_method` supports `greedy` and `probabilistic`. It defaults to `greedy`.

For example, `num_draft_loops=2` and `num_shared_loops=1` run two draft loops but share only the input block and first loop. When activation reuse applies, target verification resumes after the first loop. Unshared draft loops and output attention layers use separate KV caches. Sharing fewer loops uses more KV memory and requires more computation.

## Notes

- This method requires Model Runner V2. Dory requires `recurrent_kv_cache_mode="per_loop"` and tensor, pipeline, and prefill context parallel sizes of `1`.
- `torch.compile` is on by default. vLLM selects `FULL_DECODE_ONLY` CUDA graphs automatically; an explicit graph mode must be `FULL_DECODE_ONLY` or `NONE` when compiling or reusing activations. Batches without reusable activations run without graph replay.
- Set `--compilation-config '{"mode": 0}'` to disable compilation while retaining decode CUDA graphs with activation reuse, or use `--enforce-eager` to disable both. Optimization level `-O0` also leaves both disabled by default.
- Prompt embeddings, microbatching, parallel drafting, adaptive verification, and dynamic speculation schedules are not supported.
- Compiled and uncompiled execution need not be bitwise identical; floating-point differences can change outputs. See [Lossless guarantees of Speculative Decoding](README.md#lossless-guarantees-of-speculative-decoding).
