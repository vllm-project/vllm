# Qwen3.5 XPU fused MoE decode

This opt-in stage specialization uses the complete XPU eager model and its
canonical routed/shared weights. It does not duplicate the model, weight
loader, KV cache or recurrent state manager.

## Selection and qualification

`additional_config={"decode_backend":"auto"}` is the default. It runs stock
MoE without constructing a specialized executor, allocating its workspaces or
importing the optional shared-expert provider. Dense and MTP models retain their
default implementation.

Explicit `decode_backend=latency_specialized` validates the deployment before
model construction. The initial qualified envelope is deliberately narrow:

| Dimension | Qualified combination |
| --- | --- |
| Hardware | Two Arc Pro B70 devices, ID `0xE223`, FP16 and subgroup block-2D capability |
| Model | Qwen3.5-family MoE, H=2048, routed/shared I=512, E=256, top-k=8, 40 layers |
| Parallelism | TP2, PP1, DP1, EP1, context parallel sizes 1; single-host mp, standard TP reduction |
| Numerics | FP16 activations, online `fp8_per_tensor`, FP8-e4m3 weights, FP32 scales; KV/conv state inferred, recurrent state FP32 |
| Execution | Model Runner V2 through XPUModelRunner; compilation mode 0; graph NONE or FULL_DECODE_ONLY |
| Workload | Ordinary decode, q_i=1, 1 <= B=T_actual <= T_padded <= 8; configured context limit <=40960 |

Routed experts must use `XPUExpertsFp8`; shared projections must use
`XPUW8A16FP8LinearKernel`. Static guards also check routing semantics, sharding,
scale shapes, native dimensions, device placement, contiguity and alignment.
All detectable deployment and per-layer failures are collected across the TP
CPU group and reported at startup with instructions to select `auto`.

Speculative decoding, multimodal inputs, LoRA, prefix caching, sleep/wake,
external weight/KV transfer, prompt embeddings, alternate KV/SSM paths, offload, DBO, EPLB, expert/sequence/context
parallelism and W8A8 shared projections are initially unqualified for explicit
specialization. They remain available through the default implementation where
vLLM supports them. This restriction is a validation boundary, not a claim
that the native operator cannot support other configurations.

Software validation uses driver `1.15.39122+11`, Level-Zero V2 device runtime
`20.2.0`, PyTorch `2.14.0+xpu`, and a companion `vllm-xpu-kernels` build
providing `XpuFusedMoeWithSharedExpert`. The tested grouped-GEMM binary records
oneAPI DPC++/C++ `2026.1.1` and GCC `13.3.1` in its ELF compiler metadata.
Native availability is checked against
the actual device. Other SKUs and software combinations need independent
correctness and performance qualification.

## Planning, graphs and fallback

The model selects a `MambaHybridModelState` subclass only when specialization
is requested. It keeps the existing KV/SSM handling and uses the runner's CPU
`InputBatch` metadata to plan the optional MoE stage once per model step.

B is the current request count, q_i is the scheduled query length,
T_actual=sum(q_i), and T_padded is the executed row count after graph padding.
The provider uses T_padded, including routed rows T_padded*top-k <=512.
L_i is the existing context/KV length of request i before the step, bounded by
the configured model limit. Context length does not change this MoE stage's
scratch requirement; the stock runner retains context and cache handling.

Prefill, chunked prefill, mixed batches, drafts, zero rows and batches outside
the qualified row range select stock MoE before native execution.
`XPUModelRunner.gather_batch_req_state` consults the optional XPU ModelState's
decode-graph policy before graph selection. It keeps even a one-token prompt
tail out of a graph captured for specialized ordinary decode without changing
the shared runner or request metadata used for KV/SSM handling. B3 may replay a
qualified four-row graph; B9 uses stock. No device-to-host read is added for planning.

Both paths use the stock routing/top-k selection and one TP all-reduce for the
combined routed/shared output. Initialization decisions agree across TP ranks;
runtime planning sees the same replicated batch metadata. Native or collective
errors propagate through the worker; execution is never retried through stock
after a kernel has started.

## Weights, persistent state and memory

Normal routed FP8 post-processing prepares the common provider's canonical
`[E,H,2I]` / `[E,I,H]` layout, whether specialization is enabled or disabled.
The optional executor directly references those parameters and the shared
projections, FP32 scales and gate. It creates no persistent derived weights.
Post-load preparation precedes any graph address binding; the default provider
consumes the same representation.

The executor reserves bounded scratch for each row count 1 through 8 at
post-load, before memory profiling and KV-cache sizing. There are no lazy
specialization workspace allocations in decode. Output/activation buffers are
transient; graph pools and their peak costs must be reported separately.

`DecodeRuntime.snapshot()` reports planned steps by eligibility/skip reason,
native Python launches including capture, scratch bytes and binding checks.
The zero additional weight-storage result is reported only after tensor
identity, pointer, shape, stride, dtype and device checks. Python launch counts
do not count native kernels executed by graph replay. Validation observes actual
`run_fullgraph` calls and their batch plans, checks that capture launched all
MoE layers, and observes the bound native provider through CPU operator events.
A separate lightweight process provides eager GPU kernel traces. Enabling PTI
GPU tracing in the model worker can stall later whole-model graph replays on
the tested driver, including with stock MoE; stopping the GPU profiler before
replay does not avoid this. CPU-only observation keeps that collector disabled.
This validation does not provide a device trace of whole-model graph replay.

Supported in-place kernel-format reloads preserve parameter objects, addresses,
executor identity and workspaces. Post-load reinitialization reuses existing
executors and rejects changed storage/layout or eligibility. Live requests'
KV/SSM state remains owned by stock ModelState, so path changes do not migrate
or reconstruct persistent state. Broader rebinding/update mechanisms are
explicitly unqualified.

## Validation and benefit

Correctness and measured benefit must be recorded separately for each
checkpoint/configuration. Layer tests exercise real native dispatch, numerical
agreement, default-off independence, prompt/padding boundaries, TP reduction,
workspace reuse and reload graphs. Real-model validation must exercise a single
request across default prefill -> specialized decode -> default decode ->
specialized decode, including mixed and >8-row batches.

Compare opt-in and auto on the same source revision, hardware, numerical
formats and graph mode. Report decode latency/TPOT, throughput, TTFT, warmup,
repetitions and variability, plus persistent scratch, peak allocations and
KV-cache capacity. Greedy/logprob and model evaluation results must disclose
rounding differences; native executability alone does not establish benefit.

The initial measurements and remaining coverage gaps are retained in the
experiment report. No benefit claim extends automatically to another GPU,
checkpoint, speculative mode or numerical format.

## Files and tests

- `__init__.py`: opt-in selection and row limits.
- `config.py`: deployment qualification and TP agreement.
- `hooks.py`: optional, idempotent post-load initialization.
- `weights.py`: static guards and canonical parameter preparation.
- `planning.py`: host-only batch plan and bounded diagnostics.
- `model_state.py`: stock hybrid state integration and graph eligibility.
- `moe.py`: bound provider, runtime tensor guards, native call and TP reduction.

Layer tests: `tests/kernels/moe/test_moe.py -k xpu_qwen3_5`.
Model tests: `tests/models/qwen3_5/xpu_specialized_decode/test_integration.py`.
The Intel job requests the existing B70 resource template and selects these
suites and their source dependencies. Local validation does not establish that
the new CI job has run; its execution remains to be confirmed.
