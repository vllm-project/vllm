# vLLM on SiFive P550 Implementation Plan

<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the vLLM project
SPDX-FileCopyrightText: Copyright (c) 2026 zyz
-->

This document describes the implementation strategy used to enable and validate
the vLLM CPU backend on a SiFive P550 development board. It focuses on the
minimum useful path: build vLLM from source, load a small trained chat model,
serve a minimal OpenAI-compatible chat endpoint, and validate generation from a
Windows-side browser.

The P550 enablement notes, validation workflow, and this implementation guide
include additional copyright attribution to zyz. The underlying vLLM source code
and existing CPU backend implementation remain under the project's Apache-2.0
license and existing contributor copyright notices.

## Goals

- Run vLLM on the P550 through the CPU backend.
- Avoid CUDA, ROCm, Triton, and GPU-specific execution paths.
- Validate the existing vLLM CPU attention native op on RISC-V.
- Use a small real instruction model instead of a random checkpoint for final
  chat validation.
- Provide a manual Windows browser workflow for testing the P550 service.
- Keep board addresses, SSH details, credentials, model weights, and local cache
  paths out of the repository.

## Scope

The current implementation targets correctness and bring-up, not performance.
The accepted baseline is a scalar RISC-V CPU path with short prompts and short
outputs. RVV can be enabled later only when the board exposes a supported vector
length such as `zvl128b` or `zvl256b`.

The validated runtime path is:

```text
browser or CLI client
  -> minimal HTTP chat service
  -> vLLM LLM.generate()
  -> vLLM V1 CPU backend
  -> CPU_ATTN native op
  -> precompiled C++ CPU attention implementation
  -> PyTorch CPU tensor operations for the rest of the model
```

## Repository Changes

The P550 enablement uses a small set of targeted repository changes:

- `tools/p550_probe.sh`: board environment and CPU feature probing.
- `tools/p550_make_tiny_llama.py`: local tiny random model generation for
  smoke tests.
- `tools/p550_smoke_test.sh`: minimal offline vLLM import and generation test.
- `tools/p550_real_chat_smoke_test.sh`: trained-model offline validation.
- `tools/p550_start_vllm_service.sh`: minimal HTTP chat service backed by
  `vllm.LLM`.
- `tools/p550_chat_test.sh`: simple CLI chat client.
- `tools/p550_web_chat.html`: Windows-side browser test page.
- `docs/getting_started/installation/cpu.p550.inc.md`: concise P550 CPU build
  instructions.
- `docs/getting_started/installation/p550_component_plan.md`: notes for
  additional component work if a missing runtime dependency blocks progress.
- `docs/getting_started/installation/p550_real_chat_model_plan.md`: real chat
  model validation plan and results.

The implementation also includes a small RISC-V compatibility fix in the CPU
utility layer so unsupported `py-cpuinfo` architecture probing does not block
P550 initialization.

## Runtime Configuration

The P550 service should be started from the repository root on the board:

```bash
VLLM_TARGET_DEVICE=cpu \
VLLM_RVV_VLEN=0 \
OMP_NUM_THREADS=4 \
VLLM_CPU_OMP_THREADS_BIND=nobind \
VLLM_WORKER_MULTIPROC_METHOD=fork \
VLLM_P550_MODEL=$PWD/.p550_models/qwen2.5-0.5b-instruct \
VLLM_P550_SERVED_MODEL_NAME=qwen2.5-0.5b-instruct \
VLLM_P550_MAX_MODEL_LEN=128 \
VLLM_P550_KV_CACHE_BYTES=536870912 \
    tools/p550_start_vllm_service.sh
```

Important settings:

- `VLLM_TARGET_DEVICE=cpu` forces the CPU backend.
- `VLLM_RVV_VLEN=0` keeps the first validated baseline on the scalar RISC-V
  path when RVV is not exposed by the board.
- `VLLM_WORKER_MULTIPROC_METHOD=fork` avoids Python multiprocessing issues
  seen when vLLM worker startup is launched from stdin-style scripts.
- `VLLM_P550_MAX_MODEL_LEN=128` gives Qwen 0.5B enough context for short chat
  prompts while keeping memory pressure low.
- `VLLM_P550_KV_CACHE_BYTES=536870912` explicitly reserves a small KV cache.

The model directory must exist locally on the board. Model weights are not
committed to the repository.

## Attention Native Op

The attention operator used on P550 was not newly developed for this enablement.
It is the existing vLLM CPU backend native attention implementation.

The relevant source path is:

```text
vllm/v1/attention/backends/cpu_attn.py
  -> vllm/_custom_ops.py
  -> csrc/cpu/torch_bindings.cpp
  -> csrc/cpu/cpu_attn.cpp
  -> csrc/cpu/cpu_attn_impl.hpp
  -> csrc/cpu/cpu_attn_vec.hpp
```

### Python Backend Selection

`vllm/v1/attention/backends/cpu_attn.py` defines `CPUAttentionBackend`, whose
backend name is `CPU_ATTN`.

The metadata builder determines the preferred CPU attention ISA. On the current
P550 validation target, the runtime selected the `vec` path. The board did not
expose the RVV feature set required to select the `rvv` specialization.

During execution, `CPUAttentionBackendImpl.forward()` performs these steps:

1. Receive query, key, value, KV cache, and attention metadata from the model
   runner.
2. Split the KV cache into key and value cache tensors.
3. Call `ops.cpu_attn_reshape_and_cache(...)` to write the current key/value
   tensors into vLLM's block KV cache layout.
4. Call `ops.cpu_attention_with_kv_cache(...)` to compute attention using the
   cached K/V tensors.

For the P550 decoder path, attention is not routed through ordinary PyTorch
`scaled_dot_product_attention`. The active path is the vLLM CPU native op.

### PyTorch Custom Op Binding

`vllm/_custom_ops.py` exposes Python wrappers around registered C++ ops:

```text
ops.cpu_attn_get_scheduler_metadata(...)
  -> torch.ops._C.get_scheduler_metadata(...)

ops.cpu_attn_reshape_and_cache(...)
  -> torch.ops._C.cpu_attn_reshape_and_cache(...)

ops.cpu_attention_with_kv_cache(...)
  -> torch.ops._C.cpu_attention_with_kv_cache(...)
```

`csrc/cpu/torch_bindings.cpp` registers these functions into the `_C` extension.
This makes the CPU attention code callable from Python through `torch.ops._C`.

### C++ Scheduling and Dispatch

`csrc/cpu/cpu_attn.cpp` implements the top-level native op entry points.

`get_scheduler_metadata(...)` builds an `AttentionScheduler::ScheduleInput`,
maps the requested ISA string to `cpu_attention::ISA`, dispatches by dtype,
head dimension, ISA, and KV cache dtype, and returns scheduler metadata to
Python.

`cpu_attn_reshape_and_cache(...)` validates tensor shapes and strides, maps the
KV cache dtype, dispatches to the selected implementation, and writes K/V data
into the paged KV cache.

`cpu_attention_with_kv_cache(...)` builds a `cpu_attention::AttentionInput`
structure containing raw pointers to:

- query
- key cache
- value cache
- output
- query start locations
- sequence lengths
- block tables
- scheduler metadata

It then instantiates `cpu_attention::AttentionMainLoop<attn_impl>` and executes
the selected attention implementation.

`csrc/cpu/generate_cpu_attn_dispatch.py` generates the dispatch table used by
`CPU_ATTN_DISPATCH`. The dispatch key includes:

- head dimension
- CPU ISA specialization
- KV cache dtype

For RISC-V, the generated dispatch logic supports RVV specializations only when
the build target exposes a supported vector length. Otherwise, the RISC-V path
falls back to `VEC` and `VEC16`.

### Attention Main Loop

`csrc/cpu/cpu_attn_impl.hpp` contains the scheduler, scratchpad layout, and
main attention loop. The scheduler partitions work across OpenMP threads and
KV tiles while accounting for sequence lengths, head dimensions, causal masks,
sliding-window settings, and cache size.

The main loop performs:

1. Load query tiles.
2. Load key tiles from the block KV cache.
3. Compute `Q * K^T` logits.
4. Apply causal or sliding-window masking when required.
5. Compute softmax normalization.
6. Load value tiles.
7. Compute `softmax(QK) * V`.
8. Reduce partial outputs when a query/head is split across multiple work
   items.
9. Store the final attention output tensor.

`csrc/cpu/cpu_attn_vec.hpp` provides the general vector-based implementation
used by the validated P550 path. This is a precompiled C++ implementation, not
a runtime JIT kernel.

## JIT, Triton, and PyTorch Operator Usage

The validated P550 service uses:

- vLLM CPU backend: yes.
- Triton kernels: no. Triton is not installed or used on this path.
- vLLM CPU attention native op: yes.
- PyTorch tensor operators for non-attention model layers: yes.
- PyTorch `scaled_dot_product_attention` for the active decoder path: no.
- Runtime JIT for attention: no.
- Runtime JIT for the whole model: no.

The service uses `enforce_eager=True`, and the logs show compilation mode set to
`NONE`. Torch Inductor and CUDA graph capture are disabled. The CPU attention
operator is compiled into the vLLM native extension during build time.

## Validated Models

### Tiny Random Model

The tiny random local model is useful only for build and plumbing checks. It is
not a correctness or model-quality target.

### SmolLM2 135M Instruct

`HuggingFaceTB/SmolLM2-135M-Instruct` was used as the first real trained chat
model. It proved that the P550 vLLM path could load a real model and return a
non-random answer. Answer quality was limited by model size.

### Qwen2.5 0.5B Instruct

`Qwen/Qwen2.5-0.5B-Instruct` is the current preferred manual validation model.
It was downloaded outside the repository and copied to a local board model
directory.

The following five chat sessions were validated through the P550 vLLM service:

```text
What is 1+2? Answer with only the number. -> 3
What is 2*3? Answer with only the number. -> 6
Translate to Chinese: hello. Answer with only the translation. -> ni hao
What is the capital of France? Answer with one word. -> Paris
Answer yes or no: Is water wet? -> Yes.
```

Observed latency was about 19 to 20 seconds per short request on the scalar
RISC-V CPU path.

One earlier prompt, "What animal says meow? Answer with one word.", returned an
incorrect answer from the 0.5B model. That is treated as model-quality behavior,
not a vLLM runtime failure.

## Manual Browser Validation

Start the P550 service first, then open the Windows-side browser page:

```text
http://127.0.0.1:8765/tools/p550_web_chat.html?v=qwen
```

Set the endpoint field to:

```text
http://<p550-board-host>:8000
```

The page should show:

```text
Model: qwen2.5-0.5b-instruct
Max Tokens: 16
```

Click `Check Health`. A successful response should display:

```text
Service OK: qwen2.5-0.5b-instruct
```

Then send one of the short prompts from the page, such as:

```text
What is 1+2? Answer with only the number.
```

The expected response is:

```text
3
```

## Current Limitations

- The current path validates functionality, not production throughput.
- The P550 scalar CPU path is slow for chat generation.
- RVV is not enabled in the validated baseline because the board did not expose
  a supported RVV vector length during probing.
- Larger models may require additional memory tuning, longer startup time, or a
  different quantization strategy.
- The minimal HTTP service is a validation tool, not the full vLLM OpenAI API
  server.

## Future Work

- Re-run CPU feature probing on newer board firmware or kernel builds to check
  whether RVV features become visible.
- If RVV is exposed, validate the generated `RVV` dispatch path and compare it
  against the scalar `VEC` path.
- Add a formal P550 CI-like smoke target that can run without committing model
  weights.
- Evaluate smaller quantized instruction models if the dependency stack supports
  them on RISC-V.
- Replace the minimal validation HTTP service with the full vLLM API server only
  after the runtime and dependency stack are stable.

## Copyright

This implementation guide and the P550-specific validation workflow include:

```text
Copyright (c) 2026 zyz
```

The vLLM project source code remains governed by the Apache-2.0 license and the
existing copyright notices in each source file.
