# Qwen3.6 TP3/DCP3 Research Notes

This document records an experimental serving effort for Qwen3.6-class
NVFP4 models on three independent PCIe Blackwell GPUs without NVLink. The
goal is not only to fit a model, but to use all three GPUs productively for
long-context agent workloads.

## Problem Statement

The test machine has three 16 GiB GPUs. In aggregate that is 48 GiB of VRAM,
which is enough to make Qwen3.6 27B/35B-A3B long-context serving interesting.
However, the machine is not a datacenter NVLink box:

- no NVLink or shared VRAM;
- asymmetric PCIe topology, effectively around `x8/x4/x4`;
- stable communication currently uses conservative PyNCCL paths;
- P2P, custom all-reduce, and symmetric-memory paths are limited or unstable on
  this local Blackwell PCIe topology.

The budget angle is part of the motivation. As of July 2026 spot pricing,
RTX 5060 Ti 16GB cards are commonly listed around the mid-`$500` range in US
retail, so a `3 x 5060 Ti 16GB` setup is roughly a `$1.6K-$1.8K` GPU budget for
48 GiB aggregate VRAM. By contrast, RTX 5090 32GB cards are difficult to buy at
the `$1,999` MSRP and current market trackers commonly show new cards around
`$4K+`. Prices are volatile, but the shape of the tradeoff is stable: the
three-card system is much weaker as an interconnect topology, yet it can offer
more aggregate VRAM at a materially lower GPU cost than a market-priced 5090.

Out of the box, this hardware class is awkward for vLLM. `TP=2` leaves one GPU
underused, while `TP=3` is often rejected or degraded because Qwen3.6 model
layouts contain dimensions that are not divisible by 3:

- GQA/KV layouts;
- full-attention heads in the MoE model;
- MTP hidden projection size (`5120 % 3 != 0`);
- DFlash draft attention heads (`32 % 3 != 0`);
- DFlash draft MLP intermediate size (`17408 % 3 != 0`).

The naive fallback is replication or a parallel group that technically starts
but exposes too few useful KV tokens for 128K-150K agent workloads. The research
question is therefore: can vLLM make `3 x 16 GiB` PCIe GPUs useful as one
long-context serving pool, despite weak interconnect and non-divisible model
geometry?

## Goals

- Serve dense Qwen3.6 27B NVFP4 on `TP=3` despite uneven GQA/KV geometry.
- Preserve long context: target range is 128K-150K today, with 256K as the
  stretch goal.
- Keep `max_num_seqs=8` usable for agent workloads that mix one large context
  with many small concurrent prompts.
- Support vision and eventually speculative decoding without losing the
  long-context memory budget.
- Understand and reduce communication overhead on PCIe-only multi-GPU systems.
- Identify where vLLM memory estimation and placement are too conservative or
  too unaware of real transient workspace needs.

The correct strategy is not to emulate a single large GPU blindly; it is to make
the TP/DCP, KV-cache, draft-model, and communication layouts serve the topology.

The high-level outcome is already positive: this setup makes dense 27B usable
with 150K context and `2x+` long-context concurrency on a 48 GiB aggregate
consumer/prosumer PCIe system. That is not comparable to a single 32 GiB RTX
5090 memory budget: the single card can be faster, but it cannot provide the
same long-context capacity. The current decode speed is roughly within a 2x
factor of the single-card target while enabling workloads that did not fit
there. This is a meaningful result precisely because the interconnect is only
`x8/x4/x4`-class PCIe and the stable communication path is not a specialized
NVLink all-reduce.

## Current Working Results

### Dense 27B, text only

The dense 27B NVFP4 target (`mconcat/Qwopus3.6-27B-v2-NVFP4`) runs with:

- `--tensor-parallel-size 3`
- `--decode-context-parallel-size 3`
- `--max-model-len 150K`
- `--max-num-seqs 8`
- `--max-num-batched-tokens 8192`
- `--gpu-memory-utilization 0.85`
- `--kv-cache-dtype fp8`
- Model Runner V2
- FlashInfer attention
- PyNCCL all-reduce

Observed:

- model weights: about `7.98 GiB` per GPU;
- GPU KV cache size: `386,477` logical tokens;
- maximum concurrency for `153,600` tokens per request: `2.52x`;
- warmed no-MTP sequential decode tests without the vision tower reached the
  `52-54 tok/s` class;
- a real `150K` prompt smoke returned successfully;
- no replicated full-attention fallback is required for the dense 27B target.

The important result is that TP3/DCP3 is viable for the dense 27B target even
though the standard vLLM assumptions usually expect tensor-parallel-friendly
head layouts. This is achieved by using sequence-dimension DCP for the KV
budget instead of requiring a clean KV-head split.

### Dense 27B with vision

The same dense 27B profile also runs with vision enabled:

- `--mm-encoder-tp-mode data`
- no `--language-model-only`
- same `TP=3`, `DCP=3`, `150K`, `max_num_seqs=8`, `gpu_memory_utilization=0.85`

Observed:

- model weights: about `8.84 GiB` per GPU;
- vision tower cost: about `+0.86 GiB/GPU`;
- available KV cache memory: `3.59 GiB`;
- GPU KV cache size: `327,019` logical tokens;
- maximum concurrency for `153,600` tokens per request: `2.13x`;
- multimodal warmup completed and the server reached healthy state;
- no-MTP single-request decode in this vision-enabled 150K profile measured
  about `34 tok/s`, while aggregate throughput at `8` concurrent short
  requests measured about `177 tok/s` wall-clock.

This means vision is no longer a blocker for the 150K dense profile, but it
does compete directly with speculative decoding and FlashInfer autotune
headroom.

### Agent-stable no-MTP profile

For the agent-oriented profile on the Qwen3.6-style MoE/NVFP4 model family
(`protoLabsAI/Agents-A1-NVFP4` / 35B-A3B class), a tighter no-MTP configuration
reached:

- `--max-model-len 128K`
- `--max-num-seqs 8`
- `--gpu-memory-utilization 0.92`

Observed:

- KV cache size: `481,689` logical tokens;
- across three GPUs this corresponds to roughly `1,445,067` distributed
  token-slots before accounting for the logical request view;
- maximum concurrency for `131,072` tokens per request: `3.67x`;
- direct agent smoke across `qwen`, `codex`, `claude`, and `hermes` completed
  with no CUDA OOM, HTTP 500, or EngineDeadError.

Viewed physically, this is roughly in the "1.5M token-slot" class across three
16 GiB GPUs, even though vLLM reports the logical request-level KV budget. The
more exact stable observation was `481,689 * 3 = 1,445,067` distributed
token-slots. This is the practical success: the three independent GPUs can be
made to behave like a useful long-context serving pool for agent workloads.

Rejected tighter profiles:

- `0.94` and `0.93` no-MTP started and exposed more KV tokens, but failed under
  agent prefill workspace pressure in GDN/FLA kernels.
- `0.92` with some lower-thinking request defaults increased planned KV
  reservation, but failed on the first agent request with a small CUDA
  allocation OOM.

The memory target cannot be chosen only from the final reserved KV blocks.
Transient prefill workspaces need explicit headroom.

### MoE / Agents-A1 status

The MoE branch is an important success case, not just a side experiment. It
proved that TP3 plus expert parallelism can make the three-GPU machine useful
for a 35B-A3B-class agent model.

Working profile highlights:

- `TP=3` with expert parallelism enabled;
- Model Runner V2;
- `max_num_seqs=8`;
- long-context agent serving in the 128K class;
- vision profile also reached stable startup and request handling;
- direct agent test traffic completed without engine death in the stable memory
  profile.

The practical compromise is the MoE backend. On this SM120/NVFP4 stack, Marlin
is currently the only backend that is both correct enough and stable enough for
agent serving. Native FlashInfer/CuTeDSL MoE paths are still research debt:
they are the route to higher performance, but local and upstream signals show
correctness/performance instability on desktop/prosumer Blackwell for FP4 MoE
grouped GEMM. In practice, FlashInfer/CuTeDSL MoE is not yet the production
choice here; Marlin is.

DFlash on MoE is still worth keeping in scope. Experimental runs showed
approximately the `100 tok/s` class, which is materially higher than the dense
27B no-MTP `52-54 tok/s` baseline. This needs a proper controlled benchmark
because the first results mix several moving parts:

- MoE backend choice: Marlin vs FlashInfer/CuTeDSL;
- speculative method: built-in MTP vs DFlash/external drafter;
- context length and prefill workspace pressure;
- concurrency level;
- acceptance rate by prompt type.

The next MoE decision should not be based on a single throughput number. The
correct promotion rule is: keep Marlin as the stable baseline, then test DFlash
or other draft paths only if they preserve agent stability at `max_num_seqs=8`
and improve either aggregate throughput or per-request latency without reducing
the usable long-context KV budget too much.

### MTP status

Built-in Qwen3.5/Qwen3.6 MTP runs on the dense 27B target with:

- `{"method":"mtp","num_speculative_tokens":1}`
- `TP=3`, `DCP=3`, `150K`, vision enabled

Observed:

- model weights increased from about `8.84 GiB/GPU` in the vision/no-MTP
  profile to about `9.19 GiB/GPU`;
- MTP overhead in the vision profile: about `+0.35 GiB/GPU` over the
  vision/no-MTP baseline, because the target embeddings and `lm_head` are
  shared with the drafter;
- GPU KV cache size in the `0.87` profile: about `168K` logical tokens in the
  stable run, enough for about `1.09x` concurrency at `153,600` tokens;
- single-request decode improved from the vision/no-MTP `~34 tok/s` class to
  the `~50 tok/s` class;
- aggregate throughput on `8` concurrent short requests improved from about
  `177 tok/s` wall-clock without MTP to about `226 tok/s` wall-clock with
  `num_speculative_tokens=1`;
- spec decode metrics were healthy:
  - draft acceptance rate typically in the `75-86%` range;
  - mean accepted length around `1.8`.

The key warning is:

```text
Replicating Qwen3.5 MTP fc because hidden_size=5120 is not divisible by tensor_parallel_size=3.
```

MTP is therefore functional and beneficial for the dense 27B vision profile at
`num_speculative_tokens=1`, but the current TP3 implementation still pays a
replicated-memory cost in the MTP `fc` projection. The important distinction for
reporting is:

- no-MTP, vision enabled, 150K: about `34 tok/s` single request and about
  `177 tok/s` aggregate at `8` concurrent short requests;
- MTP K=1, vision enabled, 150K: about `50 tok/s` single request and about
  `226 tok/s` aggregate at `8` concurrent short requests;
- no-vision/no-MTP warmed sequential tests are a different baseline and should
  not be quoted as the vision+MTP comparison point.

`num_speculative_tokens=2` is not currently usable in this TP3/DCP3 profile.
It starts at `gpu_memory_utilization=0.87` with about `156,767` KV tokens and
`1.02x` max concurrency for a `153,600` token request, but the first real agent
request crashes the engine with a CUDA illegal memory access followed by an
NCCL watchdog cascade. Treat K=2 as debug debt, not as a serving candidate.

Remaining MTP work items include:

- measure acceptance by prompt class and generated-token length;
- debug `num_speculative_tokens > 1` with a minimal repro and
  `CUDA_LAUNCH_BLOCKING=1`;
- reduce or shard the replicated MTP `fc` memory cost for TP3;
- tune CUDA graph and FlashInfer warmup coverage for MTP shapes.

The target is to turn MTP from "works on TP3" into an additional speedup layer
on top of the already-working TP3/DCP3 long-context baseline.

## DFlash Findings

DFlash is attractive because it drafts a whole token block in one pass instead
of autoregressively drafting token by token. The public Qwen3.6 draft model is
`z-lab/Qwen3.6-27B-DFlash`.

The local draft config is:

- `num_hidden_layers=5`
- `hidden_size=5120`
- `intermediate_size=17408`
- `num_attention_heads=32`
- `num_key_value_heads=8`
- `sliding_window=2048`
- `layer_types=[sliding_attention, sliding_attention, sliding_attention,
  sliding_attention, full_attention]`
- `dflash_config.target_layer_ids=[1,16,31,46,61]`

The first compatibility blockers on `TP=3` are:

- attention heads: `32 % 3 != 0`;
- MLP intermediate size: `17408 % 3 != 0`;
- mixed sliding/full attention support is still experimental for this model
  family.

The current experimental patches add two opt-in escape hatches:

- allow mixed SWA/full DFlash attention for proof-of-concept runs;
- allow replicated DFlash attention projections when head count is not divisible
  by TP size.

That got past the first attention assertion but immediately exposed the MLP
assertion. This is expected: DFlash is not a small head-only module. A naive
"replicate DFlash on all three target TP ranks" approach would replicate both
attention and MLP state.

Approximate BF16 DFlash memory:

- attention per layer: about 52M parameters;
- MLP per layer: about 267M parameters;
- five draft layers plus FC/context projection: roughly 1.6B-2B parameters;
- BF16 storage: roughly 3.5-4 GiB before runtime buffers.

If DFlash is truly TP-sharded over three GPUs, that is roughly `1.2-1.4 GiB/GPU`.
If the incompatible pieces are replicated on every GPU, it can become
`3.5-4 GiB/GPU`, likely worse than built-in MTP on this memory budget.

Conclusion: replicated DFlash is useful only as a diagnostic PoC. The production
path should be either true uneven/padded sharding or isolated draft placement.

## Alternatives to Built-in MTP

### DFlash

Pros:

- block diffusion drafts multiple tokens in one pass;
- vLLM has first-class `method="dflash"` plumbing;
- can outperform autoregressive draft methods if acceptance remains high.

Cons:

- Qwen3.6 DFlash support is still moving due to causal/mixed SWA layers;
- non-causal DFlash attention interacts poorly with DCP and KV-cache dtype
  choices in some vLLM/FlashInfer versions;
- the Qwen3.6 draft geometry is not TP3-friendly;
- replicated fallback is too memory-expensive for `150K + vision`.

### EAGLE/EAGLE3-style draft models

Pros:

- mature speculative-decoding path in vLLM;
- can use hidden states rather than duplicating a full target-size model.

Cons:

- needs a compatible trained draft model for the exact target;
- draft TP and placement still need attention on PCIe-only hosts;
- acceptance depends strongly on workload and prompt style.

### N-gram / prompt lookup / suffix decoding

Pros:

- very low memory overhead;
- can help coding and agent workloads with repeated prefixes, boilerplate, and
  tool-call patterns;
- does not require a large draft model.

Cons:

- not a general decode accelerator;
- gains depend on repetition and prompt structure;
- cannot replace MTP/DFlash for arbitrary natural language.

### External draft service

Pros:

- best fit for three independent GPUs if one GPU has available headroom;
- avoids replicating draft weights on all target TP ranks;
- can run a draft model with different precision, backend, or batch policy.

Cons:

- vLLM's current in-process speculative path assumes a tighter coupling between
  target runner, draft model, hidden-state transfer, and verification;
- requires new scheduling/IPC APIs or a custom proposer path;
- hidden-state transfer latency must be measured carefully.

### Custom-trained draft model

Training a custom draft model is technically possible and could be attractive if
off-the-shelf MTP/DFlash placement remains inefficient for TP3. A custom drafter
could be shaped around this exact deployment:

- TP3-friendly dimensions;
- small enough to fit on one GPU or into planned residual memory;
- trained on agent/coding/tool-call distributions rather than generic chat;
- designed for `150K` target-context behavior and Qwen3.6 hidden states.

This is a last-resort route, not the first optimization. It likely costs much
more calendar time than fixing placement/sharding and benchmarking existing
MTP/DFlash/EAGLE-style paths.

## FlashInfer Research Direction

FlashInfer is involved in several distinct paths:

- attention backend for the target model;
- FP4/NVFP4 GEMM kernels;
- autotune cache generation/loading;
- optional all-reduce backend;
- DFlash non-causal draft attention path.

Current local findings:

- FlashInfer attention works for dense 27B TP3/DCP3.
- TRTLLM FlashInfer path cannot return LSE for DCP, so vLLM falls back to
  FlashInfer native attention.
- Missing autotune cache falls back to default tactics; this is stable but can
  leave performance on the table.
- Running autotune at high `gpu_memory_utilization` can OOM even when runtime
  serving would fit, because autotune temporarily allocates profiling workspaces.
- For MoE/NVFP4 on SM120, native FlashInfer/CuTeDSL/CUTLASS paths are still
  riskier than Marlin in this environment; Marlin is slower than the theoretical
  native FP4 path, but currently more reliable.

Research items:

1. Build FlashInfer autotune cache at a lower utilization profile, then run
   production with `VLLM_FLASHINFER_AUTOTUNE_MODE=load`.
2. Verify which cache keys change across:
   - MTP on/off;
   - vision on/off;
   - `max_num_batched_tokens`;
   - FlashInfer backend variant;
   - model architecture: dense 27B vs 35B-A3B MoE.
3. Test target attention backends separately from MoE GEMM backends.
4. Revisit native FP4 MoE only when SM120/SM12x correctness and performance
   issues are resolved upstream.

## Testing Methodology

The local validation is not just "the server started." It combines vLLM-style
benchmarking ideas with end-to-end agent workload checks.

Relevant vLLM methodology:

- unit tests for sharding/math invariants where the behavior is deterministic;
- `vllm bench serve` for online server throughput/latency, including custom
  datasets and OpenAI-compatible request paths;
- `vllm bench throughput` for offline throughput isolation;
- `vllm bench latency` for lower-level latency measurements;
- benchmark sweeps to compare serve parameters while keeping server settings
  controlled and resetting caches between runs;
- SPEED-Bench-style speculative decoding measurement for acceptance rate,
  acceptance length, and throughput across prompt-length buckets;
- production-oriented benchmarking with realistic request rate and concurrency,
  not only a single synthetic prompt.

For this hardware research, the most important follow-up benchmark shape is:

- no-MTP vs MTP vs DFlash/external drafter;
- 10-20 warmed sequential requests at fixed 256/512 generated tokens;
- concurrency 4 and 8 for aggregate throughput;
- long-prefill smoke at 128K-150K;
- separate text, vision, and agent-tool workloads;
- record TTFT, TPOT/decode throughput, output throughput, acceptance rate, and
  peak/free GPU memory.

The local agent harness adds an end-to-end workload layer that plain vLLM
benchmarks do not cover. It runs real agent CLIs (`qwen`, `codex`, `claude`,
`hermes`, and optionally `opencode`) inside containers configured to use only
the local vLLM endpoint (`host.docker.internal:8902`) with dummy API keys. The
summary files record:

- whether container internet was available;
- whether the local model endpoint was available;
- whether each agent passed its task;
- whether internet was used;
- whether the local model was used.

The agent task is intentionally closer to a real coding-agent loop than a raw
completion benchmark: the agent must interact with a workspace and complete a
small edit/test-style task. This catches failures that a one-shot chat request
does not reveal, including server disconnects, request-path incompatibilities,
tool-call/rendering issues, long prompt handling, and memory instability under
multiple independent clients.

These agent tests are not a replacement for upstream unit tests or vLLM
benchmarks. They are a hardware/workload acceptance gate: a serving profile is
not considered stable for this project unless it can survive both synthetic
long-context tests and agent-style traffic.

## Intercommunication Research Direction

Current stable communication mode is PyNCCL:

- `NCCL_P2P_DISABLE=1`
- `--disable-custom-all-reduce`
- FlashInfer all-reduce disabled
- symmetric memory disabled

This is conservative but stable on a PCIe-only, no-NVLink topology.

Observed/known issues:

- vLLM custom all-reduce can hang on Blackwell PCIe systems when topology
  assumptions do not hold.
- PyTorch `SymmMemCommunicator` did not support device capability `12.0` in this
  stack.
- vLLM's `NCCL_SYMM_MEM` path was originally gated for larger world sizes and is
  not a free win for `world_size=3`.
- Local experiments that enabled symmetric-memory paths did not improve short
  decode and hit long-prefill instability.

Next technical directions:

1. Keep PyNCCL as the production baseline.
2. Add reproducible microbenchmarks for TP3 all-reduce sizes that actually occur
   in Qwen3.6 dense and MoE decode.
3. Re-test P2P after BIOS/IOMMU/ACS changes, but do not assume it solves the
   GPU2 bottleneck if the PCIe topology is inherently asymmetric.
4. Prototype a custom IPC all-reduce only for the narrow tensor sizes where
   PyNCCL is measurably dominant in latency.
5. Track whether vLLM/FlashInfer symmetric-memory support catches up for
   Blackwell consumer/prosumer SM120 systems.

## Memory Placement and Estimation Debt

vLLM's current memory planning is functional but not topology-aware enough for
this target.

Problems seen locally:

- `gpu_memory_utilization` reserves KV blocks based on planned steady-state
  memory, but GDN/FLA prefill kernels need transient workspace headroom.
- FlashInfer autotune can OOM during profiling even when the final selected
  runtime tactic would fit.
- Vision, MTP, DFlash, and CUDA graph pools all compete for the same residual
  memory, but the placement policy treats many of these costs as uniform across
  ranks.
- Replicated fallback for incompatible TP geometry can make a model "work" while
  silently destroying the KV budget.

Desired improvements:

- expose planned vs measured memory buckets per rank:
  weights, KV, CUDA graphs, compile cache, vision tower, draft model, transient
  workspace estimate;
- model transient workspace reserves explicitly instead of relying on coarse
  utilization headroom;
- allow asymmetric placement for draft modules and possibly vision encoders;
- support uneven/padded sharding for small incompatible dimensions instead of
  full replication;
- make speculative draft placement independent enough that draft weights are not
  replicated across all target TP ranks unless requested.

## Dense 27B vs 35B-A3B

Dense 27B:

- easier TP3 target geometry in the current patches;
- no replicated full-attention fallback in the dense target;
- good long-context behavior with DCP3;
- vision works at 150K/0.85.

35B-A3B / Agents-A1 MoE:

- MoE/EP makes full utilization attractive, but several dense/full-attention
  fallback paths still appear because head counts are not cleanly divisible by
  TP3;
- Marlin is currently the reliable FP4/MoE backend on SM120, while native
  FlashInfer/CuTeDSL paths need more validation;
- agent stability was achieved at `128K`, `max_num_seqs=8`, `gpu_util=0.87` for
  the MoE vision profile and `0.92` for the no-MTP text-oriented profile, but
  higher utilization failed on transient workspaces.
- DFlash-style speculation is especially interesting for MoE because early
  experiments reached the `~100 tok/s` class, but this must be remeasured against
  a stable Marlin baseline with acceptance metrics and agent success criteria.

## Upstream Research References

- vLLM Qwen3.6 27B recipe: https://recipes.vllm.ai/Qwen/Qwen3.6-27B
- vLLM speculative config docs: https://docs.vllm.ai/en/stable/api/vllm/config/speculative/
- vLLM engine args / gpu memory utilization docs: https://docs.vllm.ai/en/v0.20.1/configuration/engine_args/
- DFlash model card: https://huggingface.co/z-lab/Qwen3.6-27B-DFlash
- Qwen3.6 DFlash low-acceptance discussion: https://huggingface.co/z-lab/Qwen3.6-27B-DFlash/discussions/2
- vLLM DFlash KV dtype issue: https://github.com/vllm-project/vllm/issues/41559
- DFlash paper: https://arxiv.org/html/2602.06036v2
- NVIDIA DFlash Blackwell blog: https://developer.nvidia.com/blog/boost-inference-performance-up-to-15x-on-nvidia-blackwell-using-dflash-speculative-decoding/
- FlashInfer SM120 NVFP4 MoE issue: https://github.com/flashinfer-ai/flashinfer/issues/2723
- FlashInfer FP4 GEMM performance issue: https://github.com/flashinfer-ai/flashinfer/issues/1732
- FlashInfer SM12x vLLM CUTLASS backend issue: https://github.com/flashinfer-ai/flashinfer/issues/3013
- vLLM Blackwell PCIe custom all-reduce discussion: https://discuss.vllm.ai/t/vllm-hangs-during-worker-initialization-on-blackwell-pcie-gpus-unless-disable-custom-all-reduce-is-used/2540
- FlashInfer autotune OOM discussion: https://discuss.vllm.ai/t/getting-flashinfer-jit-autotuner-oom-detected/2565
- RTX 5060 Ti 16GB retail examples: https://www.bestbuy.com/site/searchpage.jsp?id=pcat17071&st=rtx+5060+ti
- RTX 5090 market-price tracker: https://bestvaluegpu.com/history/new-and-used-rtx-5090-price-history-and-specs/

## Recommended Next Work

1. Commit the current DFlash patch only as an experimental proof-of-concept
   helper, not as the final production design.
2. Add a DFlash MLP fallback only if the goal is to measure replicated-DFlash
   memory cost; otherwise skip it and go directly to uneven/padded sharding.
3. Design true TP3 uneven sharding for DFlash attention and MLP:
   - attention heads split as `[11, 11, 10]` or padded to 33;
   - MLP intermediate split as `[5803, 5803, 5802]` or padded to 17409;
   - trim padded outputs before residual paths.
4. Investigate external/isolated draft placement:
   - target remains TP3/DCP3;
   - draft model uses TP1 or its own placement;
   - hidden-state transfer and verification latency are measured explicitly.
5. Run matched MTP A/B benchmarks:
   - no-MTP vs MTP `num_speculative_tokens=1,2,3`;
   - 10-20 warmed sequential requests;
   - 256 and 512 generated tokens;
   - then concurrency 4 and 8.
6. Add vLLM benchmark artifacts for the promoted profiles:
   - `vllm bench serve` or GuideLLM for online throughput/latency;
   - SPEED-Bench-style speculative decoding metrics for MTP/DFlash;
   - agent harness summaries for real coding-agent traffic.
7. Build and pin FlashInfer autotune caches separately from production startup.
8. Keep PyNCCL as the stable interconnect baseline until a measured custom path
   beats it without long-context instability.
