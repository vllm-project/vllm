<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
vLLM — feat/steering branch
</h3>

<p align="center">
Activation steering support for vLLM inference
</p>

---

This branch adds **activation steering** to vLLM: injecting precomputed
vectors into the residual stream of decoder layers during inference, enabling
tone/style changes, behavioral interventions, and SAE-derived steering vectors
without fine-tuning.

Full docs live in the repo:

- [User guide](docs/features/steering.md) — setup, API reference, examples
- [Runtime design](docs/design/steering_runtime.md) — internals for contributors

---

## Steering Model

Steering uses a three-tier additive composition:

```text
effective_prefill = global_base + global_prefill + request_base + request_prefill
effective_decode  = global_base + global_decode  + request_base + request_decode
```

Three hook points are available per decoder layer:

| Hook | Where |
| --- | --- |
| `pre_attn` | Residual stream before attention |
| `post_attn` | Residual stream after attention |
| `post_mlp` | Residual stream after MLP |

---

## Quickstart

### Serving

```bash
# Global steering only (always available for steerable models)
vllm serve google/gemma-3-4b-it

# Enable per-request steering
vllm serve google/gemma-3-4b-it \
  --enable-steering \
  --max-steering-configs 4
```

### Global steering via HTTP

Global endpoints require `VLLM_SERVER_DEV_MODE=1`.

```bash
# Set steering vectors
curl -X POST http://localhost:8000/v1/steering/set \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "post_mlp": {
        "15": {"vector": [0.1, 0.2], "scale": 2.0}
      }
    },
    "prefill_vectors": {"pre_attn": {"15": [0.3, 0.4]}},
    "decode_vectors":  {"pre_attn": {"15": [0.5, 0.6]}},
    "replace": false
  }'

# Clear all global steering
curl -X POST http://localhost:8000/v1/steering/clear

# Inspect active steering
curl http://localhost:8000/v1/steering
```

### Per-request steering (Python)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="google/gemma-3-4b-it", enable_steering=True, max_steering_configs=4)

params = SamplingParams(
    max_tokens=64,
    temperature=0.0,
    steering_vectors={"post_mlp": {15: {"vector": [0.1, 0.2], "scale": 2.0}}},
    prefill_steering_vectors={"pre_attn": {15: [0.3, 0.4]}},
    decode_steering_vectors={"pre_attn": {15: [0.5, 0.6]}},
)
outputs = llm.generate(["Hello"], params)
```

### Per-request steering (OpenAI-compatible server)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="google/gemma-3-4b-it",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "steering_vectors":         {"post_mlp": {15: [0.1, 0.2]}},
        "prefill_steering_vectors": {"pre_attn": {15: [0.3, 0.4]}},
        "decode_steering_vectors":  {"pre_attn": {15: [0.5, 0.6]}},
    },
)
```

### Named steering modules

Pre-register a config once; reference it by name in requests.

```bash
# Register
curl -X POST http://localhost:8000/v1/steering/modules/register \
  -H "Content-Type: application/json" \
  -d '{"name": "creativity", "vectors": {"post_mlp": {"15": [0.1, 0.2, 0.3]}}}'

# Use in a request
client.chat.completions.create(
    model="google/gemma-3-4b-it",
    messages=[{"role": "user", "content": "Write a poem"}],
    extra_body={"steering_name": "creativity"},
)
```

Named modules and inline vectors compose additively per tier.

---

## Runtime Design Summary

The runtime is built around these invariants:

- **Persistent GPU buffers** — steering data lives in pre-allocated tables
  updated between steps; CUDA graph replay reads live values, no recompilation.
- **Phase-aware admission** — prefill and decode consume separate steering-table
  capacity; the scheduler tracks per-request phase transitions.
- **Prefix-cache correctness** — prefill steering is part of the cache key;
  decode-only steering is not; global base/prefill changes invalidate cache reuse.
- **Two-queue deferral** — decode transitions are retried before new-request
  registrations so in-flight requests are never starved of table rows.

See [docs/design/steering_runtime.md](docs/design/steering_runtime.md) for full details.

---

## Supported Architectures

Steering is wired into all major decoder families:

- Llama, Qwen, Gemma, Mixtral/MoE, GLM4, InternLM, Olmo, Exaone, Phi, Plamo,
  Step, Molmo, Falcon, Baichuan, CommandR, StableLM, and more.

End-to-end tested with real weights: Gemma 3, StableLM, step3p5, Mixtral,
DeepSeek V2, PhiMoE, GLM4 MoE, Exaone MoE.

See the [full list](docs/features/steering.md#supported-scope).

---

## About vLLM

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, and x86/ARM/PowerPC CPUs

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
