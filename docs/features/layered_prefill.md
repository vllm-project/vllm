# Layered Prefill

Layered prefill splits the decoder layers on each pipeline rank into groups and
advances a prefill batch through one group per engine iteration, instead of
splitting prompts into token chunks. Every iteration also runs a full-depth
decode step for the running requests, so a long prompt delays each decode step
by one group instead of the whole model.

Compared with chunked prefill, each prompt token passes through every layer
once with a large batch. For MoE models this avoids reloading expert weights
for every chunk, and it keeps inter-token latency bounded while prompts are
being prefilled.

See: [Layered Prefill paper](https://arxiv.org/abs/2510.08055)

## Usage

```bash
vllm serve Qwen/Qwen3-30B-A3B \
    --num-layer-groups 16 \
    --max-num-batched-tokens 8192 \
    --no-async-scheduling
```

`--num-layer-groups` is the number of groups per pipeline rank and must not
exceed the number of layers on any rank. A prompt completes prefill after
`num_layer_groups × pipeline_parallel_size` iterations. With PP, each rank runs
its own prefill batch every iteration, so a new batch can start while earlier
batches finish on downstream ranks.

## Requirements and limitations

- Model Runner V2 and synchronous scheduling (`--no-async-scheduling`).
  `torch.compile` is disabled; CUDA graphs use breakable prefill graphs for
  each layer group and full graphs for decode.
- Text-only generation models whose decoder iterates
  `layers[start_layer:end_layer]` with the standard pipeline-parallel interface.
  This covers most models that support pipeline parallelism.
- TP, PP and EP are supported. Speculative decoding, data and context
  parallelism, DBO, EPLB, KV/encoder transfer connectors, and hybrid or
  attention-free models are not yet supported.
- The `mp` or `uni` executor backend.

## Benchmark

`benchmarks/benchmark_layered_prefill.py` compares chunked and layered prefill
serving SLO attainment for the configurations in
`benchmarks/configs/layered_prefill/`.
