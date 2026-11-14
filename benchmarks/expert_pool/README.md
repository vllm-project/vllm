# Expert pool: reproducing the generation-speed measurement

Sends a warmup request and then a measured request to Qwen3.8-Flash-Next
NVFP4. Each request is a software bug report; the model writes a fix
proposal and a verification plan.

## Files and requests

| file | content |
| --- | --- |
| [benchmark.py](benchmark.py) | HTTP requests, stream capture, decode-speed computation |
| [pair.json](pair.json) | the two prompts as measured: full text, frozen token ids, provenance |
| [prompts.md](prompts.md) | the prompt texts, readable |
| [provenance.json](provenance.json) | code versions measured, SHA256 of the original files, changes from the historical runner |
| [results/20261112.json](results/20261112.json) | November measurements, output hashes, memory observations and excluded runs |
| [results/20261112-latest.json](results/20261112-latest.json) | updated main comparison, rows/concurrency sweep and actual long-input validation |
| [results/20261113-fi074.json](results/20261113-fi074.json) | refreshed main, matching native wheel and FlashInfer 0.7.4 validation |
| [results/20261112-input-manifest.json](results/20261112-input-manifest.json) | frozen long-input source/tokenizer hashes, seeds and token positions |
| [LICENSE.prompts](LICENSE.prompts) | MIT license of the prompt source |

| order | role | bug report | input tokens |
| --- | --- | --- | --- |
| 1st | warmup (`role` = `warmup`) | rename the "Choose File" button to "Choose file" (task 28096_836) | 1070 |
| 2nd | measured (`role` = `measure`) | update the "Link sent!" message when the language setting changes (task 18827_741) | 753 |

The prompts come from SWE-Lancer in
[OpenAI frontier-evals](https://github.com/openai/frontier-evals/tree/51052cede8cc608f95bb00346635e03759013e5a),
two of its existing sanity tasks. This measures fix-proposal text
generation only; the code-editing and official-grading quality checks are a
separate procedure.

## Main refresh with FlashInfer 0.7.4

The final serving runtime is `87535c55bf698b71f9b6d4e6b38b024e958da4e5`,
incorporating main `c07d58dc82d75f7d1a947b3264b460384ec3e18d` (latest
snapshot checked at 2026-11-13 01:56 UTC). Its separate GPU suite and
fresh main/PR comparison are recorded below.

The broader row/concurrency, MTP and long-input campaign runs on
`40f9048f45cf1d39a66a4d067a5dd45e97a97c21`, incorporating main
`3bc1d31306e9a10fce59846f888fe106f274bc8c`. That snapshot adds
30 upstream commits after the earlier final snapshot below, including the
FlashInfer 0.7.4 requirement, NVFP4 activation-fusion contracts and MTP
changes. This campaign uses the public nightly wheel from
`b70414673330f969f07bd6d0f214fd536e6a7dd1`; its C++/CUDA build inputs
and dependency declarations match the tested main. Both PR and actual
main use that same wheel and FlashInfer **0.7.4** in a separate uv
environment. The other 406 installed dependency versions are unchanged.
Torch remains 2.14.0+cu132, Triton 3.8.0, Transformers 5.22.0, xgrammar
0.2.9 and NVIDIA Cutlass DSL 4.8.0.

The broader campaign's pytest results are **208 passed, 12 skipped,
1 failed**; the two benchmark-client tests also pass. The 12 skips are
the existing SM10x-only CuTeDSL/Torch NVFP4 cases on SM120. The failed
configuration test tries to
construct four-GPU parallelism before its intended assertion on this
one-GPU machine. The same unmodified test fails at the same constructor
on actual upstream main. No test, tolerance, GPU count or skip marker is
changed. The result artifact retains the failure and main control.
Coverage includes the original 164 pool/configuration/loader/worker/
tokenizer cases, activation contracts, Qwen MTP configuration/remapping,
and real NVFP4 tensor-versus-prequantized-input kernel checks.

Whole-repository manual pre-commit, including mypy and CI test-tethering,
passes on Linux. Selective pip-compile has no changed PR inputs (six hooks
skip). Strict MkDocs passes in **196.52 seconds**, using the RTD Ubuntu
22.04 / Python 3.12 setup, hash-locked docs requirements, no Torch, and a
clean Git-tracked checkout. The existing decord wheel's platform metadata
warning occurs in both runtime environments; its runtime import succeeds,
and its metadata is preserved.

This campaign repeats the frozen request pair, four API checks, row and
concurrency measurements, MTP checks and official-YaRN long retrieval.
The source/tokenizer hashes and seeds remain those in the existing input
manifest. The earlier campaigns below keep their original revisions and
dependencies; their results are historical rather than reattributed to
this runtime. Rust frontend, ROCm, SM90-specific execution, TP=2 and GLM
serving are not covered by this single-GPU Python-server campaign.

### Final main snapshot check

Main advanced by three more commits during the campaign: a cuBLAS
workspace cleanup at MRV2 shutdown, a ROCm AITER revert and an MRV2
structured-output draft handoff improvement. They are
included in final PR runtime `87535c55bf698b71f9b6d4e6b38b024e958da4e5`,
whose main parent is `c07d58dc82d75f7d1a947b3264b460384ec3e18d`.
The expanded GPU suite was rerun on this runtime with
209 passes, 12 architecture skips and one four-GPU configuration
failure; both benchmark-client tests pass. The extra passing CUDA test
exercises the upstream draft-handoff change. The source and build-input
checks are recorded separately from the original campaign.

Fresh servers then repeat the same 48 GiB pair and four API checks:

| Source/configuration | Decode tok/s | TTFT | Output tokens |
| --- | ---: | ---: | ---: |
| Main c07d58dc82, B12X A16 / UVA34 | 13.1477 | 9.282 s | 1838 |
| PR 87535c55bf, Marlin pool N=258 / staging OFF | 64.3164 | 7.113 s | 1519 |

This final snapshot gives **4.89x** the main decode throughput at the constrained budget.
All four API checks pass in each arm. The long, row-sweep, MTP and
full-card results below retain their actual 40f9048f45 / 3bc1d31306
runtime versions; they are not relabeled as final-head measurements.

Each final arm additionally runs the existing fixed-output protocol:
one 128-token warmup, then three 256-token c1 samples, ignoring EOS
for timing. Prompt order is identical; continuations can still differ.
Main median [range]: **13.15 [13.15–13.17] tok/s**;
pool: **71.83 [70.53–76.05] tok/s**.
These timing-only requests are separate from the normal-stop pair
and four API correctness checks.

### Refreshed serving measurements

The short comparison uses a 48 GiB available-memory budget, 768 MiB KV,
4096 context, one slot, 512-token chunks, MTP off and FULL_DECODE_ONLY
graphs in every arm. Each row is one fresh server and the original
1070-token warmup / 753-token measured request, capped at 2048 output
tokens with normal stopping. All four API checks pass in every arm.
These compare serving configurations, including their backend/offload
choices, with differing generated continuations; they do not isolate a
single implementation toggle or establish variation across restarts.

| Source/configuration | Decode tok/s | TTFT | Output tokens |
| --- | ---: | ---: | ---: |
| Main, Marlin / UVA32 | 9.2721 | 10.560 s | 1589 |
| Main, B12X A16 / UVA34 | 13.1490 | 9.349 s | 1349 |
| PR, Marlin pool N=258 / staging OFF | 65.2440 | 7.065 s | 1141 |

The pool is **4.96x** the fastest main configuration tested at this budget.

A separate full-card main run uses B12X A16 with all expert weights on
the GPU, no expert CPU offload, 8 GiB KV, maximum context 128,512, two
slots and 4608-token chunks. Engram disk offloading stays enabled.
The original measured request gives **107.85 tok/s** (TTFT 0.141 s).
Three fixed-output samples give c1 **107.59 [106.44–108.03]**
and aggregate c2 **187.59 [186.59–188.34]** tok/s.
All nine retrieval requests at 12.5K / 81.6K / 128K pass (45/45
bindings), as do all four API checks. This is a different GPU-memory,
KV and batching condition from the 48 GiB comparison.

The row sweep uses a 64 GiB budget, automatic KV sizing, maximum
context 128,512, two slots, 4608-token chunks, no prefix caching,
staging off and MTP off. Each concurrency has three 256-output-token
samples after a 128-token warmup. c2 is aggregate decode-window
throughput; end-to-end throughput is recorded separately.

| Initial rows/layer | Planner KV tokens | KV GiB | c1 median [range] tok/s | c2 aggregate median [range] tok/s | Retrieval bindings |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 1,676,633 | 43.62 | 23.53 [22.84–26.87] | 25.11 [24.12–25.89] | 15/15 |
| 128 | 1,362,824 | 35.71 | 41.23 [40.27–44.81] | 44.56 [43.00–46.63] | 15/15 |
| 192 | 1,050,510 | 27.80 | 62.32 [57.85–66.18] | 73.76 [72.07–73.82] | 15/15 |

Planner capacity is not a test at that input length. The separate
refreshed long run uses N=192, staging on, 64 GiB, fixed 24 GiB KV,
two slots, batch 4608, no prefix caching and the official YaRN4 settings
shown below. YaRN applies at every tested length in this run.

| Actual input tokens | Requests | Correct bindings | TTFT median [range] |
| --- | ---: | ---: | ---: |
| 12,500 | 3 | 15/15 | 17.06 [17.00–17.26] s |
| 81,600 | 3 | 15/15 | 101.86 [101.58–102.39] s |
| 128,000 | 3 | 15/15 | 158.18 [157.87–158.71] s |
| 1,000,000 | 1 | 5/5 | 1232.90 [1232.90–1232.90] s |

Two concurrent 12,500-token requests also pass (10/10 bindings). The
million-token result is one frozen-input retrieval sample, with actual
server usage, all five exact bindings, normal stop and complete stream
required. It is not a general accuracy or 1M decode-speed benchmark.
The earlier controlled staging ON/OFF comparison retains its original
source revisions and dependencies below.

The MTP functional run uses N=192, staging off, three Triton draft
tokens, 1.5 GiB KV and INFO logging. All four API checks pass;
counters record 448 drafts, 1344 draft tokens and
1105 accepted tokens. These establish active speculation on the smoke set,
rather than a representative acceptance-rate or speed result.

## Earlier main integration and long-input validation

The final integration merges main
`3f1ebac297fb8791a8fa8ee55e635e7fa93ddaec` in runtime commit
`893345409dcea5ebcb3a82250cea88350110b331` (main snapshot checked at
2026-11-12 16:08 UTC). This includes fifteen upstream commits after the
frozen long-input campaign below. The pool, QSA, rotary embedding,
offloader and loader sources are unchanged between those runtime commits.
The earlier final spot checks use PR `d7ec7e12f8` / main `3f96f6b923`.
The last five upstream commits add KV-connector, opt-in Mamba2 ReplaySSM
and API priority handling changes. Final serving checks use separate
immutable source trees; running measurements are not modified.

For this integration, the only C++/CUDA compilation input differing from
artifact `c9c36486c` is the FlashMLA dependency revision for SM90 sparse prefill.
That component was not rebuilt or exercised on this SM120 Marlin setup.
The Marlin native inputs still match the artifact. Rust frontend changes
were not built or exercised. This validation uses the Python frontend.
The environment uses the updated xgrammar 0.2.9 dependency; the removal of
the optional `b12x` package extra is metadata-only. This is not a complete
wheel rebuild from the final source revision.

The frozen long-input and row-sweep campaign uses main
`3b43d2b01c2fb667101746e1acd33b6a3c9d0863` and PR runtime sources
`a6248e2f6f067a93aea1c28cbb014fb31dd461c1`. The worker fixture and
generated-diagram update in `ab7c1217d23db1fb5bd58578eaebac7b851fce37`
does not change the serving implementation. Both sources use the same
precompiled CUDA/C++ artifacts from `c9c36486c`; their native build inputs
are unchanged. The newer Rust frontend/parser was not rebuilt or exercised.

The final short-request recheck uses main `3f1ebac297` and PR runtime
`893345409d`, with the same 48 GiB available-memory budget, 768 MiB KV,
4096 context, one slot, 512-token chunks, MTP off, INFO logging and
FULL_DECODE_ONLY graphs. Each uses one fresh server and the frozen
1070-token warmup / 753-token measured request, stopping naturally within
2048 output tokens. Both pass all four API quality checks. B12X34 is the
fastest tested constrained-memory main configuration, not an exhaustive
optimum.

| Configuration | Decode tok/s | TTFT | Output tokens |
| --- | ---: | ---: | ---: |
| Main `3f1ebac297`, B12X A16, UVA 34 GiB | 13.1561 | 9.251 s | 1784 |
| PR `893345409d`, Marlin pool N=258, staging OFF | 64.5497 | 7.194 s | 1094 |

This is about 4.91x decode throughput for the tested serving configurations.
It compares different MoE backends and offload strategies; generated
continuations also differ. It does not isolate one implementation toggle
or establish restart variability. For the main arm, set
`VLLM_B12X_MOE_FP4_FORCE_A16=1` and replace the pool/backend arguments in
the short-request command below with `--moe-backend b12x
--moe-expert-pool-rows 0 --offload-backend uva --cpu-offload-gb 34
--cpu-offload-params experts`. Keep all other memory, graph and request
settings identical.

At the earlier frozen campaign revisions, with the same 48 GiB
available-memory budget, 768 MiB KV cache, frozen
request pair, INFO logging and FULL_DECODE_ONLY graphs in **both** arms,
one fresh main server with 40 GiB UVA expert offload measured
7.6034 decode tok/s and 13.060 s TTFT. One fresh pool server with N=258
and prefill staging disabled measured 65.7333 tok/s and 7.198 s TTFT.
Generated continuations differ; these are single observations, and do not
establish the best available main configuration.

Main's other backends were also exercised. On main `3b43d2b01c` with the
same 48 GiB / 768 MiB KV / 4096-context short-request settings, explicit
B12X A16 (`--moe-backend b12x`, `VLLM_B12X_MOE_FP4_FORCE_A16=1`) with
40 GiB UVA expert offload measured **11.3590 tok/s**, TTFT **11.004 s**.
All four API quality checks passed. Thus 7.60 tok/s is the Marlin/UVA
reference, not the fastest tested main configuration. Marlin/UVA attempts
at 28 and 30 GiB CPU offload failed during NVFP4 scale conversion with
CUDA OOM; steady weight sizes alone do not establish startup feasibility.

A separate main `3b43d2b01c` run used the full physical card, without the
external memory allocation or expert CPU offload: about 95 GiB physical
memory, 72.86 GiB loaded weights, 8 GiB KV, 128,512 context, two slots and
4608-token prefill chunks. Three samples gave median **101.84 tok/s c1**
and **178.28 tok/s aggregate c2**. The original request pair subsequently
measured **101.00 tok/s**, TTFT **0.112 s**. All nine retrieval requests at
12,500 / 81,600 / 128,000 input tokens passed (45/45 bindings), with
median TTFT **1.298 / 8.485 / 13.523 s**. This is a different memory
condition from the 48 GiB comparison. It also does not demonstrate a
fully resident million-token deployment: the measured weights plus the
long-test 24 GiB KV already exceed this card's physical memory.

Main's prefetch offloader was measured separately at 48 GiB with group=3,
num-in-group=2, step=1. Two bounded 32-output-token timing requests gave
**0.2614 / 0.2630 tok/s** after an 8-token warmup. This timing-only arm
uses a different output protocol and does not have the four-check quality
suite; it is not included in the matched pair comparison.

The separate 64 GiB campaign uses 4608-token prefill chunks, two request
slots, no prefix caching and MTP disabled. This is TP=1 on one RTX PRO
6000 Blackwell Max-Q, unlike the community report's two RTX 5090s with
TP=2. It also uses current Engram disk offloading and a 100 GiB host-memory
limit. The server, checkpoint, offload method and hardware differences
prevent interpreting the community's numbers as a regression baseline.

The completed sweep uses automatic KV sizing, maximum context 128,512,
staging off and three decode samples per concurrency. The c2 column is
aggregate throughput over the two requests' decode window.

| Initial rows/layer | Planner KV tokens | KV GiB | c1 decode tok/s, median [range] | c2 decode-window tok/s, median [range] |
| --- | ---: | ---: | ---: | ---: |
| 64 | 1,675,138 | 43.62 | 24.32 [23.69–27.05] | 25.28 [24.61–25.53] |
| 128 | 1,361,330 | 35.71 | 41.87 [38.81–44.86] | 43.38 [43.11–44.54] |
| 192 | 1,049,016 | 27.80 | 60.81 [58.09–63.86] | 71.55 [71.24–76.08] |

Each configuration also passed three 12,500-token retrieval requests
(45/45 bindings across the sweep), with identical output hashes for each
input across all three row settings. Planner capacity is not a test at
that maximum input length. The separate long-input campaign checks actual
input-token counts and correctness.

[validate_serving.py](validate_serving.py) generates exact-length token
arrays from a shuffled archive of real vLLM Python sources, with five
random codes placed near the beginning, 25%, 50%, 75% and end. It checks
the server's actual input-token count, complete stream, normal stop, and
exact recovery of all five codes. This is a retrieval correctness check,
not a general model evaluation or a quality guarantee for other tasks.
The [input manifest](results/20261112-input-manifest.json) records source
file hashes, tokenizer hashes, seeds, token positions and prompt hashes.
Large prompt arrays are regenerated rather than checked into Git.
Set `CHECKPOINT` to the local NVIDIA checkpoint directory before preparing
the prompts. The frozen archive uses the exact source revision below;
generating from a newer tree changes its token hashes.

```bash
git worktree add --detach /tmp/pool-input-source a6248e2f6f067a93aea1c28cbb014fb31dd461c1
.venv/bin/python benchmarks/expert_pool/validate_serving.py prepare \
  --tokenizer "$CHECKPOINT" --corpus /tmp/pool-input-source/vllm \
  --output /tmp/pool-prompts \
  --lengths 12500 81600 128000 262000 524000 1000000 --repeats 3
.venv/bin/python benchmarks/expert_pool/validate_serving.py run \
  --kind plan --base-url http://127.0.0.1:18098 \
  --pair benchmarks/expert_pool/pair.json --prompts /tmp/pool-prompts \
  --lengths 12500 81600 128000 --repeats 3 --output /tmp/pool-validation
```

Each decode condition warms up with 128 output tokens, then measures three
256-output-token requests at concurrency 1 and three batches at
concurrency 2. Fixed output length uses `min_tokens` and `ignore_eos` only
for timing. Retrieval requests stop normally and require correct output.
Results retain per-request decode speed, aggregate decode-window speed
and end-to-end throughput separately; end-to-end throughput includes
prefill and must not be compared with a reported decode-only number.
For a concurrent correctness check, use `--kind retrieval-concurrent
--lengths 12500 --repeats 2 --concurrency 2` in a fresh output directory.

The checkpoint declares 262,144 positions. Setting only
`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` allowed startup at 1,001,000 but the first
actual million-token input failed with CUDA illegal memory access before
producing output. A small test on unmodified main also failed when a
position of 999,999 indexed the original RoPE cache; the same test passed
with a 1,048,576-position cache. The extended-context retry explicitly uses
`--hf-overrides '{"text_config":{"max_position_embeddings":1048576}}'`.
This enlarges the runtime cache; it does not establish that the model was
trained for that length. Extended-context results are reported separately
from tests within the checkpoint's declared limit.

The million-token retry completed without a CUDA error in 1238.1 seconds
(TTFT 1236.6 seconds), with server usage confirming 1,000,000 input tokens.
It returned all five code values but permuted four key/value bindings;
only 1/5 bindings were correct. This is a **failed correctness check**,
even though serving completed. This cache-only trial omitted YaRN and
cannot establish a pool accuracy regression. The
[official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices)
and [repository recipe](../../recipes/models/Qwen/Qwen3.8-Flash-Next.yaml)
specify YaRN factor=4 for a million-token context. The new trial uses:

```bash
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CHECKPOINT=/data/models/Qwen3.8-Flash-Next-NVFP4-nvidia
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VLLM_LOGGING_LEVEL=INFO
export PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1,pinned_max_cached_size_mb:1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset VLLM_USE_BREAKABLE_CUDAGRAPH VLLM_USE_V2_MODEL_RUNNER
unset VLLM_PLE_MMAP VLLM_PLE_MMAP_DEFERRED VLLM_PLE_MMAP_PREWARM VLLM_PLE_MMAP_PINNED
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$CHECKPOINT" --served-model-name flashnext \
  --host 127.0.0.1 --port 18098 --tensor-parallel-size 1 \
  --quantization modelopt --dtype bfloat16 --moe-backend marlin \
  --moe-expert-pool-rows 192 --language-model-only \
  --max-model-len 1001000 --max-num-seqs 2 --max-num-batched-tokens 4608 \
  --engram-config '{"disk_offload":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kv-cache-memory-bytes 25769803776 --gpu-memory-utilization 0.66 \
  --no-enable-prefix-caching --no-enable-flashinfer-autotune \
  --safetensors-load-strategy lazy --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --hf-overrides '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}'
```

The extra 1000 positions reserve room for output after an actual
1,000,000-token input; the YaRN cache covers 1,048,576 positions. Static
YaRN is evaluated separately because it can affect shorter inputs too.

With this official YaRN factor=4 configuration, the same frozen inputs
passed at 12,500, 81,600 and **1,000,000 actual input tokens**, each with
**5/5 exact bindings**. The million-token request reports 71 output tokens,
a normal stop and complete stream; TTFT was **1234.767 s**, elapsed time
**1235.624 s**. The server exited cleanly without CUDA error or OOM. This
is one million-token retrieval sample, not a general long-context accuracy
evaluation. The two earlier incorrectly configured trials remain failed
diagnostics in the result file.

Within the checkpoint's native context range, the first staging-on
campaign passed 60/60 bindings: 12,500 / 81,600 / 128,000 tokens with three
seeds each, 262,000 with one seed, and two concurrent 12,500-token
requests. The million-token test is separate from this native-range set.

### Controlled long-prefill staging comparison

Both arms below use N=192, the same 64 GiB budget, 24 GiB KV, two slots,
4608-token chunks, no prefix caching or MTP, and identical enlarged RoPE
cache settings. All inputs are within the native 262,144-token context;
neither arm uses YaRN. There are three frozen seeds per input length.

| Actual input tokens | Staging OFF median TTFT | Staging ON median TTFT | Exact bindings, each arm |
| --- | ---: | ---: | ---: |
| 81,600 | 199.487 s | 101.917 s | 15/15 |
| 128,000 | 305.321 s | 158.277 s | 15/15 |

Staging improves these long-prefill times by about 1.96x / 1.93x. It does
not reproduce the external 13-second report on a different two-GPU setup.
Actual main `3b43d2b01c`, UVA40, with the same 64 GiB long-test KV/chunk
settings passed 12,500 / 81,600 tokens with three seeds each (30/30
bindings), at median TTFT **35.139 / 221.633 s**. Its c1 and aggregate c2
decode-window medians were **7.587 / 7.766 tok/s**.

A separate uncaptured pinned-host DMA probe copied representative full
layers using CUDA-event elapsed time: **11.84 GB/s** with a 5.27 GiB source
working set and **11.79 GB/s** with 63.28 GiB. Each uses nine timed samples;
all destination bytes were checked before timing, with allocation, fill
and verification excluded. Full-layer staging transfers approximately
63.28 GiB per prefill chunk in this model. At 18 chunks for 81,600 tokens,
that bandwidth corresponds to roughly 104 seconds of transfer. This is
an explanatory estimate, not a measured per-request byte trace or a
universal PCIe limit. The implementation copies resident rows too to avoid
a placement-table readback on the host.

### Main feature coverage

At main `3f1ebac297`, no equivalent persistent shared GPU expert bank was
found. Main does contain UVA, prefetch offloading and B12X, as measured
above. [#62054](https://github.com/vllm-project/vllm/pull/62054) improves
sorted Triton dispatch for host-resident experts; Marlin already uses
aligned routing. The cache work in
[#37190](https://github.com/vllm-project/vllm/pull/37190) and the bank
primitives in [#57943](https://github.com/vllm-project/vllm/pull/57943)
remain open. UVA prefill staging in
[#62060](https://github.com/vllm-project/vllm/pull/62060), its side-stream
prefetch follow-up [#66374](https://github.com/vllm-project/vllm/pull/66374),
and the extended prefetch offloader
[#51710](https://github.com/vllm-project/vllm/pull/51710) are also open.
These do not establish that main is generally slow: the
fully resident result above is faster than the constrained pool run.

### Local CI implementation checks

Runtime `893345409d` passes **164 pytest cases** and **two benchmark-client
tests** on the RTX host. The pytest run takes 71.44 seconds; its only test
warning is Torch's synchronization-debug prototype warning.

Linux whole-repository manual pre-commit passes, including mypy. The six
selective pip-compile hooks have no changed inputs relative to the pinned
main. Strict MkDocs also passes on Linux in a fresh hash-locked Python
3.12 environment without Torch, using only Git-tracked source files and
Read the Docs external-preview variables. Runtime-only ignored vendor
files caused ten API namespace warnings in an initial docs attempt; a
clean checkout fixes that environment difference without suppressing
warnings or changing the docs configuration.

```bash
.venv/bin/python -m pytest -v -rs \
  tests/config/test_moe_expert_pool_rows.py tests/kernels/expert_pool \
  tests/model_executor/model_loader/test_reload_lifecycle.py \
  tests/model_executor/model_loader/test_load_plan.py \
  tests/v1/worker/test_gpu_worker.py tests/renderers/test_tokenizer_pool.py
.venv/bin/python -m unittest discover -s benchmarks/expert_pool -p 'test_*.py' -v
SKIP=pip-compile uvx pre-commit run --show-diff-on-failure --all-files --hook-stage manual
uvx pre-commit run pip-compile --hook-stage manual \
  --from-ref 3f1ebac297fb8791a8fa8ee55e635e7fa93ddaec --to-ref HEAD
```

These checks do not run every upstream job. The GitHub pre-run author/label
gate is a separate policy check; running lint/docs locally does not bypass
it or establish that the full Buildkite CI passed. The strict docs check
uses a fresh Python 3.12 environment, hash-locked `requirements/docs.txt`,
no Torch installed, and Read the Docs external-preview variables.

Integration regression evidence is retained in `integration_regressions`
in the latest result artifact. Restoring the original cold-row staging
implementation on the adapted tree makes the existing Marlin test fail
under `VLLM_GPU_SYNC_CHECK=error` at the placement-table `.cpu()` readback:
`RuntimeError: GPU<->CPU sync detected` (1 failed). After asynchronous
full-layer staging the path passes (1 passed). Before the unpooled-loader
guard, 43 reload/load-plan CPU cases fail at `get_current_vllm_config()`;
afterwards the same CPU subset passes (two GPU/device-dependent cases are
deselected). All 45 loader cases pass in the final GPU run above.

## Earlier validation configuration on main dac51063cc

The November performance measurements use production code at
`27e2470a154b790f67ad184d63a6796ba8371007`, based on main
`dac51063cc1295a2a8426998c4c372b19c42a0f9`. It uses current Engram disk
offloading instead of the historical deferred-PLE patch below.

The later commit `3abbddc9dfaf31149f505fce4d2d4edeeb5ee9cb` skips pool setup
when the loader finds no pooled layers. It changes only
`vllm/model_executor/model_loader/utils.py`; the pooled installation body
and inference kernels are unchanged. Its separate validation passed 89
pool, configuration and loader tests and two benchmark-client tests.
Performance was not remeasured on that commit.

The host is an RTX PRO 6000 Blackwell Max-Q with an external CUDA allocation
leaving approximately 48 GiB free before launching the server, and a
100 GiB container memory limit. This is not a measurement on RTX 6000 Ada.
The external allocation is fixed; desktop and driver allocations can change
the available memory slightly between launches.

KV cache is fixed at 768 MiB in both staging conditions. This disables the
extensible KV cache and takes precedence over `gpu-memory-utilization` for
cache sizing. An earlier automatic-sizing pilot completed four runs, but
its fifth launch failed after warmup: the extensible cache's safety reserve
left 0.53 GiB for KV, below the approximately 0.60 GiB needed for one
4096-token request. Pilot results are not mixed with fixed-KV measurements.

The utilization fraction below is specific to the physical 96 GiB-class
GPU used here; it is not a portable setting for other GPUs. Check available
memory before running these commands.

```bash
export CHECKPOINT=/data/models/Qwen3.8-Flash-Next-NVFP4-nvidia
export VLLM_DEBUG_WORKSPACE=1 VLLM_LOGGING_LEVEL=DEBUG
export PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1,pinned_max_cached_size_mb:1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FLASHINFER_WORKSPACE_BASE=/path/to/persistent/flashinfer-workspace
unset VLLM_USE_BREAKABLE_CUDAGRAPH VLLM_USE_V2_MODEL_RUNNER
unset VLLM_PLE_MMAP VLLM_PLE_MMAP_DEFERRED VLLM_PLE_MMAP_PREWARM VLLM_PLE_MMAP_PINNED
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$CHECKPOINT" --served-model-name flashnext \
  --host 127.0.0.1 --port 18098 --tensor-parallel-size 1 \
  --quantization modelopt --dtype bfloat16 --moe-backend marlin \
  --moe-expert-pool-rows 258 --language-model-only \
  --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 512 \
  --engram-config '{"disk_offload":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kv-cache-memory-bytes 805306368 --gpu-memory-utilization 0.49 \
  --no-enable-flashinfer-autotune --safetensors-load-strategy lazy \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --reasoning-parser qwen3 --generation-config vllm \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
```

Append `--no-moe-expert-pool-prefill-staging` for the comparison condition.
Run the [client](#client) against `http://127.0.0.1:18098` once per fresh
server. The two conditions use the same frozen requests and sampling
settings. Generated text may differ across fresh servers, including within
one condition; these are response-throughput measurements, not timings of
an identical generated continuation.

## November results

The following values are for the measured request (753 input tokens).
Staging ON and OFF each use three fresh servers, fixed 768 MiB KV capacity,
and the configuration above. The brackets show the observed range.

| Configuration | Fresh servers | Decode tok/s, median [range] | TTFT, median (s) | End-to-end tok/s, median |
| --- | --- | --- | --- | --- |
| Pool, staging ON | 3 | 64.9848 [64.1537–65.8998] | 9.3034 | 45.4134 |
| Pool, staging OFF | 3 | 64.8578 [62.1142–65.5667] | 6.9398 | 48.5435 |
| UVA expert offload, eager | 1 | 7.5227 | 13.0673 | 7.0312 |

For the UVA row, the pool is disabled with `--moe-expert-pool-rows 0`,
and `--offload-backend uva --cpu-offload-gb 40 --cpu-offload-params experts
--enforce-eager` is added. The source, checkpoint, fixed KV size, prompts
and sampling settings match the pool measurements. This single run changes
both offloading and graph execution; it does not isolate staging or measure
variation across restarts.

The pool decode ranges overlap. Staging ON had higher TTFT for these short
prompts and 512-token chunks; this experiment does not establish a benefit
for long-input prefill. Staging used another 2.64 GiB of GPU memory, while
both arms retained the same 4,864-token KV capacity. Generated text and
completion lengths differed even between launches of the same condition;
the JSON records both warmup and measured output hashes and token counts.

The original fixed-KV ON-2 run measured 62.1077 tok/s. CPU loader tests
overlapped its measured request from 06:03:05.516 to 06:03:36.221 UTC, with
the CPU test log finishing at 06:03:27.889 UTC. That run is retained but
excluded from the primary statistics. One fresh ON replacement was declared
before rerunning, after the six-run campaign completed; its 64.9848 tok/s
result is the third ON sample above. The exclusion is for concurrent work,
not a speed threshold. The earlier automatic-KV pilot is also retained
separately.

All included requests finished with `stop`, and all included containers
exited 0 with no OOM kill or restart. The four API checks run after the
first benchmark in each pool condition passed: exact JSON, a `read_file`
tool call, five-fact retrieval with a prompt over 512 tokens, and a bounded
JSON product list. These are lightweight checks, not official SWE-Lancer
grades or an accuracy benchmark. MTP is disabled for every speed result
above.

The maximum saved server NVML readings were 47,406 MiB with staging ON,
44,772 MiB with staging OFF, and 35,576 MiB for UVA. These are snapshots,
not continuous peak measurements. Cgroup-accounted memory, including file
cache, reached the 100 GiB limit in the pool runs without an OOM event.
Before launch, GPU free memory was
49,152–49,276 MiB under the nominal 48 GiB balloon budget. Shared model
files and compilation caches, including FlashInfer's persistent workspace,
were retained between fresh server processes.

The Python sources ran over the precompiled wheel at
`c9c36486c3f7c767c25bb37f087e38d101edf1ef`; native/build inputs were checked,
with two unrelated Rust MiniMax parser files differing from the wheel.
Exact source and wheel hashes, package versions, per-run values and
measurement limitations are in [the result file](results/20261112.json).
The historical measurements below used another vLLM and deferred-PLE
implementation and are not a controlled baseline for these results.

### MTP functional validation

The same four API checks also passed on
`3abbddc9dfaf31149f505fce4d2d4edeeb5ee9cb` with the checkpoint's FP8 draft.
This separate run used `VLLM_LOGGING_LEVEL=INFO`,
`--moe-expert-pool-rows 240`, `--no-moe-expert-pool-prefill-staging`,
`--kv-cache-memory-bytes 1610612736` (1.5 GiB), and
`--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'`.
The remaining model, context, batch and sampling settings matched those above.

Startup logs confirmed 40 decode staging rows (four verification tokens
times top-k 10), the FP8 draft experts remaining on the GPU with Triton,
and successful graph capture. These decode staging rows are separate from
the disabled prefill staging buffers. The server counters increased by
456 drafting steps, 1,368 draft tokens and 1,119 accepted tokens: 81.8%
acceptance across these four checks. All checks passed and the container
exited 0 without an OOM kill. This small set, dominated by the structured
output request, does not establish a representative acceptance rate, an
MTP speedup or an accuracy measurement.

Two earlier attempts are retained in the result JSON. With 768 MiB KV,
startup planning rejected the approximately 0.71 GiB available because
4096 context needed about 1.15 GiB. With 1.5 GiB KV and DEBUG logging,
upstream IR tensor formatting attempted a CUDA operation during draft
graph capture and invalidated capture. INFO logging passed on the same
production code; neither failed attempt was OOM-killed.

## Historical code and environment measured

| purpose | version |
| --- | --- |
| vLLM main used as the base | `a97dacb7106ee49f39f3d1fc6ae1800ff724e01d` |
| expert pool implementation (this PR, at the time of measurement) | `5fbc240ba5ddec82a10362340ac77339a1c24017` |
| deferred PLE rows ([01554/vllm#46](https://github.com/01554/vllm/pull/46), a diff on top of upstream PR [#54129](https://github.com/vllm-project/vllm/pull/54129)) | `4f859de9d0f55760b50358aee4834e6966e13bc8` |
| the combination of the above that produced the numbers below | `7dedc6d8d9b178b60f6a5b32f03d677145982441` |

The server ran the Python sources of the measured combination over a wheel
built from the base commit above, plus a separately built `_ple_memops`
extension. This directory holds client-side files added after the
measurement; `pair.json` is a byte copy of the file used.

The goal was a configuration that fits an RTX 6000 Ada (48 GB). Measurements
were taken on an RTX PRO 6000 Blackwell Max-Q (96 GiB) with a separate
process holding GPU memory so that the server had 48 GiB available; the
container's host memory limit was 100 GiB. The numbers below are from that GPU.

The expert pool holds 258 expert rows per layer for 48 layers, about
32 GiB. The memory limit is set by the external process; the
`gpu-memory-utilization` below is vLLM's budget as a fraction of the
physical GPU. Check the available memory and that the server is up before
running the client.

## Historical server launch

These commands describe the September measurement and its PLE patch;
they are not launch instructions for current vLLM. Current vLLM uses
`--engram-config '{"disk_offload":true}'` for this model's PLE table, with
different staging behavior from the deferred PLE patch measured here.
Use the checkpoint path appropriate to your host.
`VLLM_USE_BREAKABLE_CUDAGRAPH` was unset (automatic selection).
`VLLM_DEBUG_WORKSPACE`, `VLLM_LOGGING_LEVEL`, `PYTORCH_ALLOC_CONF` and the
thread counts are the values used during measurement, not requirements.

```bash
export CHECKPOINT=/data/models/Qwen3.8-Flash-Next-NVFP4-nvidia
unset VLLM_USE_BREAKABLE_CUDAGRAPH
export VLLM_DEBUG_WORKSPACE=1 VLLM_LOGGING_LEVEL=DEBUG
export PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1,pinned_max_cached_size_mb:1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_PLE_MMAP=1 VLLM_PLE_MMAP_DEFERRED=1
export VLLM_PLE_MMAP_PREWARM=0 VLLM_PLE_MMAP_PINNED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$CHECKPOINT" --served-model-name flashnext \
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 \
  --quantization modelopt --dtype bfloat16 --moe-backend marlin \
  --moe-expert-pool-rows 258 --language-model-only \
  --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 512 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --no-enable-flashinfer-autotune --gpu-memory-utilization 0.4548806288994517 \
  --safetensors-load-strategy lazy \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --reasoning-parser qwen3 --generation-config vllm
```

The speed measurement uses a context limit of 4096. The separate quality
checks used 32768.

## Client

Run once after the server is ready. The client needs only the Python
standard library.

```bash
.venv/bin/python benchmarks/expert_pool/benchmark.py \
  --base-url http://127.0.0.1:8000 --model flashnext \
  --label fresh-0 --output results/fresh-0.jsonl
```

Requests go to `/v1/completions` with the frozen token ids from
`pair.json`, so confirm that the checkpoint's tokenizer matches the SHA256
recorded under `tokenization` in `pair.json`. Each request uses
`temperature=0`, `top_p=1`, `seed=0`, `max_tokens=2048` and the
chat-template token sequence with thinking disabled.
`tokenization.pair_sha256` is the hash of the material before the token ids
were added; the hash of the whole file is in `provenance.json`.

For the three-run measurement, **stop the server and start a new process
before each run, then send warmup -> measure once**, writing to
`fresh-0.jsonl`, `fresh-1.jsonl`, `fresh-2.jsonl`. The reported value is the
median of the three measured-request speeds. The client refuses to
overwrite an existing output file, and saves HTTP errors and incomplete
streams before stopping; keep failed runs as results too.

## Metric and measured values

Decode speed is computed from the usage completion-token count and the
client-side receive times:

```text
decode_tok_s = (completion_tokens - 1) / (last text event time - first text event time)
```

`first_token_s` is the time from request start to the first text event;
`e2e_tok_s` is completion tokens over the whole request time. A stream
event may carry more than one token, so both are client-observed values.
The raw SSE events, request body, usage, finish reason, text and its SHA256
are stored in the same JSONL.

| fresh server run | measured-request decode speed |
| --- | --- |
| 1 | 63.1882 tok/s |
| 2 | 62.7087 tok/s |
| 3 | 63.6519 tok/s |
| median | **63.1882 tok/s** |

Measured on the combination listed above. Every request finished with
`stop`. Memory was sampled during startup and generation; the process
exited 0 with OOMKilled=false.

## Client self-check

```bash
.venv/bin/python -m unittest discover -s benchmarks/expert_pool -p 'test_*.py'
```
