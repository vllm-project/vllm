# Committers

This document lists the current committers of the vLLM project and the core areas they maintain.
Committers have write access to the vLLM repository and are responsible for reviewing and merging PRs.
You can also refer to the [CODEOWNERS](https://github.com/vllm-project/vllm/blob/main/.github/CODEOWNERS) file for concrete file-level ownership and reviewers.

## Core Committers

These committers are responsible for the overall governance of the vLLM project with administrative permissions.

- [@WoosukKwon](https://github.com/WoosukKwon)
- [@zhuohan123](https://github.com/zhuohan123)
- [@simon-mo](https://github.com/simon-mo)
- [@youkaichao](https://github.com/youkaichao)

## Active Committers

We try to summarize each committer's role in vLLM in a few words. In general, vLLM committers cover a wide range of areas and help each other in the maintenance process.
Please refer to the later section about Area Owners for exact component ownership details.
Sorted alphabetically by GitHub handle:

- [@22quinn](https://github.com/22quinn): RL API
- [@aarnphm](https://github.com/aarnphm): Structured output
- [@alexm-redhat](https://github.com/alexm-redhat): Performance
- [@ApostaC](https://github.com/ApostaC): Connectors, offloading
- [@benchislett](https://github.com/benchislett): Engine core and spec decode
- [@bigPYJ1151](https://github.com/bigPYJ1151): Intel CPU/XPU integration
- [@chaunceyjiang](https://github.com/chaunceyjiang): Tool use and reasoning parser
- [@DarkLight1337](https://github.com/DarkLight1337): Multimodality, API server
- [@gshtras](https://github.com/gshtras): AMD integration
- [@heheda12345](https://github.com/heheda12345): Hybrid memory allocator
- [@hmellor](https://github.com/hmellor): Hugging Face integration, documentation
- [@houseroad](https://github.com/houseroad): Engine core and Llama models
- [@Isotr0py](https://github.com/Isotr0py): Multimodality, new model support
- [@jeejeelee](https://github.com/jeejeelee): LoRA, new model support
- [@jikunshang](https://github.com/jikunshang): Intel CPU/XPU integration
- [@khluu](https://github.com/khluu): CI infrastructure
- [@KuntaiDu](https://github.com/KuntaiDu): KV Connector
- [@LucasWilkinson](https://github.com/LucasWilkinson): Kernels and performance
- [@luccafong](https://github.com/luccafong): Llama models, speculative decoding, distributed
- [@mgoin](https://github.com/mgoin): Quantization and performance
- [@NickLucche](https://github.com/NickLucche): KV connector
- [@njhill](https://github.com/njhill): Distributed, API server, engine core
- [@patrickvonplaten](https://github.com/patrickvonplaten): Mistral models, new model support
- [@ProExpertProg](https://github.com/ProExpertProg): Compilation, startup UX
- [@robertgshaw2-redhat](https://github.com/robertgshaw2-redhat): Core, distributed, disagg
- [@ruisearch42](https://github.com/ruisearch42): Pipeline parallelism, Ray Support
- [@russellb](https://github.com/russellb): Structured output, engine core, security
- [@sighingnow](https://github.com/sighingnow): Qwen models, new model support
- [@tdoublep](https://github.com/tdoublep): State space models
- [@tlrmchlsmth](https://github.com/tlrmchlsmth): Kernels and performance
- [@yaochengji](https://github.com/yaochengji): TPU integration
- [@yewentao256](https://github.com/yewentao256): Kernels and performance
- [@yeqcharlotte](https://github.com/yeqcharlotte): Benchmark, Llama models
- [@Yikun](https://github.com/Yikun): Pluggable hardware interface
- [@ywang96](https://github.com/ywang96): Multimodality, benchmarks
- [@zou3519](https://github.com/zou3519): Compilation

### Emeritus Committers

Committers who have contributed to vLLM significantly in the past (thank you!) but no longer active:

- [@andoorve](https://github.com/andoorve): Pipeline parallelism
- [@cadedaniel](https://github.com/cadedaniel): Speculative decoding
- [@comaniac](https://github.com/comaniac): KV cache management, pipeline parallelism
- [@esmeetu](https://github.com/esmeetu)
- [@LiuXiaoxuanPKU](https://github.com/LiuXiaoxuanPKU): Speculative decoding
- [@pcmoritz](https://github.com/pcmoritz)
- [@rkooo567](https://github.com/rkooo567): Chunked prefill
- [@sroy745](https://github.com/sroy745): Speculative decoding
- [@Yard1](https://github.com/Yard1): kernels and performance
- [@zhisbug](https://github.com/zhisbug): Arctic models, distributed

## Area Owners

This section breaks down the active committers by vLLM components and lists the area owners.
If you have PRs touching the area, please ping the area owner for review.

### Engine Core

- Scheduler: the core vLLM engine loop scheduling requests to next batch
    - @WoosukKwon, @robertgshaw2-redhat, @njhill, @heheda12345
- KV Cache Manager: memory management layer within scheduler maintaining KV cache logical block data
    - @heheda12345, @WoosukKwon
- AsyncLLM: the zmq based protocol hosting engine core and making it accessible for entrypoints
    - @robertgshaw2-redhat, @njhill, @russellb
- ModelRunner, Executor, Worker: the abstractions for engine wrapping model implementation
    - @WoosukKwon, @tlrmchlsmth, @heheda12345, @LucasWilkinson
- KV Connector: Connector interface and implementation for KV cache offload and transfer
    - @robertgshaw2-redhat, @njhill, @KuntaiDu, @NickLucche, @ApostaC
- Distributed, Parallelism, Process Management: Process launchers managing each worker, and assign them to the right DP/TP/PP/EP ranks
    - @youkaichao, @njhill, @WoosukKwon, @ruisearch42
- Collectives: the usage of nccl and other communication libraries/kernels
    - @tlrmchlsmth, @youkaichao
- Multimodality engine and memory management: core scheduling and memory management concerning vision, audio, and video inputs.
    - @ywang96, @DarkLight1337

### Model Implementations

- Model Interface: The `nn.Module` interface and implementation for various models
    - @zhuohan123, @mgoin, @simon-mo, @houseroad, @ywang96 (multimodality), @jeejeelee (lora)
- Logits Processors / Sampler: The provided sampler class and pluggable logits processors
    - @njhill, @houseroad, @22quinn
- Custom Layers: Utility layers in vLLM such as rotary embedding and rms norms
    - @ProExpertProg
- Attention: Attention interface for paged attention
    - @WoosukKwon, @LucasWilkinson, @heheda12345
- FusedMoE: FusedMoE kernel, Modular kernel framework, EPLB
    - @tlrmchlsmth
- Quantization: Various quantization config, weight loading, and kernel.
    - @mgoin, @Isotr0py, @yewentao256 
- Multi-modal Input Processing: Components that load and process image/video/audio data into feature tensors
    - @DarkLight1337, @ywang96, @Isotr0py
- torch compile: The torch compile integration for vLLM
    - @ProExpertProg, @zou3519, @youkaichao
- State space models: The state space models implementation in vLLM
    - @tdoublep, @tlrmchlsmth
- Reasoning and tool calling parsers
    - @chaunceyjiang, @aarnphm

### Entrypoints

- LLM Class: The LLM class for offline inference
    - @DarkLight1337
- API Server: The OpenAI-compatible API server
    - @DarkLight1337, @njhill, @aarnphm, @simon-mo, @heheda12345 (Responses API)
- Batch Runner: The OpenAI-compatible batch runner
    - @simon-mo

### Features

- Spec Decode: Covers model definition, attention, sampler, and scheduler related to n-grams, EAGLE, and MTP.
    - @WoosukKwon, @benchislett, @luccafong
- Structured Output: The structured output implementation
    - @russellb, @aarnphm
- RL: The RL related features such as collective rpc, sleep mode, etc.
    - @youkaichao, @zhuohan123, @22quinn
- LoRA: @jeejeelee
- Observability: Metrics and Logging
    - @robertgshaw2-redhat, @simon-mo

### Code Base

- Config: Configuration registration and parsing
    - @hmellor
- Documentation: @hmellor, @DarkLight1337, @simon-mo
- Benchmarks: @ywang96, @simon-mo
- CI, Build, Release Process: @khluu, @njhill, @simon-mo
- Security: @russellb

### Kernels

- FlashAttention: @LucasWilkinson
- FlashInfer: @LucasWilkinson, @mgoin, @WoosukKwon
- Blackwell Kernels: @mgoin
- DeepEP/DeepGEMM/pplx: @yewentao256

### Integrations

- Hugging Face: @hmellor, @Isotr0py
- Ray: @ruisearch42
- NIXL: @robertgshaw2-redhat, @NickLucche

### Collaboration with Model Vendors

- gpt-oss: @heheda12345, @simon-mo, @zhuohan123
- Llama: @luccafong
- Qwen: @sighingnow
- Mistral: @patrickvonplaten

### Hardware

- Plugin Interface: @youkaichao, @Yikun
- AMD GPU: @gshtras
- Intel CPU/GPU: @jikunshang, @bigPYJ1151
- Google TPU: @yaochengji

### Ecosystem Projects

- Ascend NPU: [@wangxiyuan](https://github.com/wangxiyuan) and [see more details](https://vllm-ascend.readthedocs.io/en/latest/community/contributors.html#maintainers)
