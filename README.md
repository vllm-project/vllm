<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
vllm download-kernels  # precompiled kernels for faster startup
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## LongSpark

This branch supports native LongSpark speculative decoding with Qwen3 targets.
Set `num_speculative_tokens` to any integer satisfying
`1 <= num_speculative_tokens <= block_size`, where `block_size` is defined by
the draft checkpoint. Omitting it uses the full checkpoint block. For a
seven-token checkpoint, lengths 1 through 7 are supported without modifying
the checkpoint configuration or weights. See the
[LongSpark documentation](docs/features/speculative_decoding/longspark.md) for
the supported model and runtime configurations.

### Start a LongSpark server

The following uses the local model paths and settings from the measurements
below. Adjust paths to match your installation. The target and draft weights
must already be downloaded.

```bash
source /home/ubuntu/workspace/.venv/bin/activate
export VLLM_USE_V2_MODEL_RUNNER=1
export NO_PROXY=127.0.0.1,localhost
unset VLLM_BATCH_INVARIANT VLLM_GPU_SYNC_CHECK VLLM_USE_BREAKABLE_CUDAGRAPH

vllm serve /home/ubuntu/workspace/models/Qwen3-4B \
    --host 127.0.0.1 \
    --port 18998 \
    --dtype bfloat16 \
    --kv-cache-dtype bfloat16 \
    --attention-backend FLASH_ATTN \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.9 \
    --max-model-len 45056 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --hf-overrides '{"max_position_embeddings": 45056}' \
    --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128]}' \
    --speculative-config '{
      "method": "longspark",
      "model": "/home/ubuntu/workspace/models/Qwen3-4B-LongSpark",
      "num_speculative_tokens": 7
    }'
```

To try a shorter draft, change only `num_speculative_tokens`, for example to 3.
The draft backbone retains the trained block shape; the sampler and target
verifier use only the requested number of proposals.

For the DFlash comparison, stop the LongSpark server and run the same command
with this speculative configuration:

```json
{
  "method": "dflash",
  "model": "/home/ubuntu/workspace/models/Qwen3-4B-DFlash",
  "num_speculative_tokens": 7
}
```

The 45056-position override accommodates the supplied 40k dataset, whose inputs
plus output exceed the target's native 40960-position limit. No input was
truncated, and weights and RoPE theta were unchanged. These measurements do not
evaluate output quality beyond the native context limit.

### Benchmark commands

After the server is ready, run these commands in a second terminal. The local
datasets must already exist. Run both loops once for each server, setting
`BENCH_METHOD` to the active method so the result files remain separate.

```bash
source /home/ubuntu/workspace/.venv/bin/activate
export NO_PROXY=127.0.0.1,localhost
BENCH_MODEL=/home/ubuntu/workspace/models/Qwen3-4B
BENCH_DATA=/home/ubuntu/workspace/data
BENCH_METHOD=longspark  # Set to dflash after restarting the server with DFlash.
BENCH_RESULTS=./benchmark-results/$BENCH_METHOD
mkdir -p "$BENCH_RESULTS"

# MT-Bench: 32 measured requests at each concurrency.
for concurrency in 1 4 8 16; do
    vllm bench serve \
        --base-url http://127.0.0.1:18998 \
        --model "$BENCH_MODEL" \
        --tokenizer "$BENCH_MODEL" \
        --dataset-name hf \
        --dataset-path "$BENCH_DATA/philschmid/mt-bench" \
        --hf-name philschmid/mt-bench \
        --hf-output-len 256 \
        --num-prompts 32 \
        --max-concurrency "$concurrency" \
        --temperature 0 --seed 42 --ignore-eos \
        --num-warmups 1 --no-oversample --disable-tqdm \
        --save-result --save-detailed \
        --result-dir "$BENCH_RESULTS" \
        --result-filename "mt-bench_c${concurrency}.json" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,95,99
done

# Long context: 10 measured requests per file, concurrency fixed at 4.
for length in 28k 32k 40k; do
    vllm bench serve \
        --base-url http://127.0.0.1:18998 \
        --model "$BENCH_MODEL" \
        --tokenizer "$BENCH_MODEL" \
        --dataset-name custom \
        --dataset-path "$BENCH_DATA/longcontext_en_prompt_${length}.jsonl" \
        --custom-output-len 256 \
        --num-prompts 10 \
        --max-concurrency 4 \
        --temperature 0 --seed 42 --ignore-eos \
        --num-warmups 1 --no-oversample --disable-tqdm \
        --save-result --save-detailed \
        --result-dir "$BENCH_RESULTS" \
        --result-filename "longcontext_${length}_c4.json" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,95,99
done
```

### Serving benchmark results

Measured on 2026-10-27 at commit `3f569234d9c9c8b35f79e4d5aa61380edfd24d53`
using one NVIDIA RTX PRO 6000 Blackwell Server Edition (97887 MiB, driver
580.178.04), Qwen3-4B, BF16 weights/KV, Model Runner V2, FlashAttention, and
TP/PP/DCP/PCP = 1. Both methods used seven speculative tokens, prefix caching
disabled, and the server settings above. These are measurements of the
seven-token configuration, not new measurements of shorter draft lengths.

All end-to-end measurements used `vllm bench serve`, temperature 0, seed 42,
and exactly 256 output tokens per request. All 316 measured requests across
14 conditions succeeded, with no KV-cache preemptions. Startup and one warmup
request per condition are excluded. The input sequence is identical between
methods. Each condition was run once; the samples do not establish statistical
significance. The local MT-Bench file contains repeated prompts; the selected
32 requests contain 28 distinct first-turn prompts.

- Throughput is output tokens divided by measured wall time.
- TTFT and TPOT are request means in milliseconds; TPOT is per output token.
  Client-side waiting for the concurrency semaphore is excluded.
- Draft throughput is proposed draft tokens divided by the same measured wall
  time, including target/prefill work and rejected proposals. It is not isolated
  drafter-kernel throughput.
- Acceptance length is `1 + accepted draft tokens / verification steps`,
  including the bonus token.

Long-context datasets: concurrency 4, 10 measured requests per condition.

| Dataset | Method | Output tok/s | TTFT (ms) | TPOT (ms/token) | Draft tok/s | Acceptance length |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 28k | LongSpark | 99.44 | 2744.77 | 27.44 | 320.31 | 2.18 |
| 28k | DFlash | 76.41 | 2847.19 | 37.14 | 391.75 | 1.36 |
| 32k | LongSpark | 83.77 | 3525.70 | 33.33 | 298.22 | 1.96 |
| 32k | DFlash | 63.51 | 3157.76 | 45.15 | 332.91 | 1.33 |
| 40k | LongSpark | 60.19 | 4959.80 | 44.52 | 219.05 | 1.92 |
| 40k | DFlash | 46.98 | 5038.99 | 58.58 | 263.07 | 1.25 |

MT-Bench: 32 measured requests per condition.

| Concurrency | Method | Output tok/s | TTFT (ms) | TPOT (ms/token) | Draft tok/s | Acceptance length |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | LongSpark | 301.54 | 36.83 | 3.18 | 772.72 | 2.74 |
| 1 | DFlash | 248.21 | 35.18 | 3.91 | 789.62 | 2.21 |
| 4 | LongSpark | 1036.78 | 53.24 | 3.51 | 2663.08 | 2.73 |
| 4 | DFlash | 930.18 | 35.87 | 4.13 | 2941.68 | 2.22 |
| 8 | LongSpark | 1816.35 | 61.16 | 3.88 | 4597.18 | 2.77 |
| 8 | DFlash | 1607.87 | 44.99 | 4.50 | 5075.22 | 2.22 |
| 16 | LongSpark | 2755.57 | 86.42 | 4.66 | 7233.38 | 2.67 |
| 16 | DFlash | 2546.62 | 82.09 | 5.19 | 8142.83 | 2.19 |

In this run, LongSpark's output throughput was 28.1–31.9% higher on the
long-context datasets and 8.2–21.5% higher on MT-Bench. DFlash had lower mean
TTFT on MT-Bench. DFlash's checkpoint has `block_size=16`; this comparison uses
seven speculative tokens for both methods and does not separately tune DFlash
at width 15.

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
