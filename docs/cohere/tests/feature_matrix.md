# Feature Matrix

This document tracks feature compatibility across the main Cohere feature areas.

## Related Documents

This matrix is part of a three-layer documentation structure:

| Layer | Document | Purpose |
| --- | --- | --- |
| **Registry** | [`observability_matrix.md`](./observability_matrix.md) | Central index of every test entry and benchmark metric. Each entry gets a unique `<cat>.<feat>.<seq>` ID. |
| **Compatibility** | This file (`feature_matrix.md`) | Cross-feature compatibility tables. Cells reference the registry via `T.<cat>.<feat>.<seq>` to record which test case verified compatibility. |
| **Detail** | [`features/*.md`](./features/) (e.g. [`c5_arch.md`](./features/c5_arch.md), [`fp32_logits.md`](./features/fp32_logits.md)) | Per-feature docs with full test case details: How it runs, Checks, Measurements, Compatibility, and Implementation. |

**How they connect:**

- Each per-feature doc has a `## Compatibility` section that classifies the test against input types, hardware, quantization, etc.
- Those classifications are propagated here as `T.<cat>.<feat>.<seq>` cell values (compatible), `❌` (not compatible), or blank (not checked).
- The `T.` prefix traces back to the matching entry in `observability_matrix.md`, which links to the full feature doc.

## How to Use

- Each table corresponds to one main feature category.
- The template tables in `## Feature List` define the full set of features
  (columns) across all categories. Every `### Section` under
  `## Feature Matrix` must correspond to exactly one column in the template
  tables — this forms the NxN compatibility matrix where each feature section
  records its compatibility with all other features.
- When adding a new feature section under `## Feature Matrix`, first add it as
  a column in the appropriate template table (and in all per-feature copies of
  that table). Conversely, when adding a new column, create a matching
  `### Section` once test coverage exists.
- Use the empty compatibility row to record which other features have been verified as compatible with that category.
- When compatibility is confirmed, fill the relevant cell with the test case number that verifies it. Leave the cell blank when compatibility has not been checked yet.
- Use `T.<category>.<feature>.<seq>` in this matrix to refer to the matching numbered entry in [`observability_matrix.md`](./observability_matrix.md).
- Example: `T.2.1.1` in the `FP32 Logits` matrix maps to test entry `2.1.1` in `observability_matrix.md`, so a `T.2.1.1` cell means that compatibility was verified by observability-matrix test case `2.1.1`.

## Feature List

| Input | Basic | Long Context | Multilingual | Multi Turn | Image | Audio |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

| Cohere Feature | Speculative Decoding | Guided Generation | Thinking Budget | Melody | Weight Reload | Auto-Config |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

| Model Architecture | C3 Arch | C4 Arch | C5 Arch | Reward | LoRA | ASR |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

| Quantization | BF16 | FP8 | MXFP8 | W4A16 | FP32 Logits | Online Config |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

| Hardware | A100 | H100 | B200 | GB200 | MI300x |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

| vLLM Feature | Chunked Prefill | Hybrid Memory Allocator | Asynchronous Scheduling | Torch Compile | CUDA Graphs | Sliding Window |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

## Feature Matrix

### FP32 Logits

| Input | Basic | Long Context | Multilingual | Multi Turn | Image | Audio |
| --- | --- | --- | --- | --- | --- | --- |
| | T.2.1.1 | | | | | |

| Cohere Feature | Speculative Decoding | Guided Generation | Thinking Budget | Melody | Weight Reload | Auto-Config |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

| Model Architecture | C3 Arch | C4 Arch | C5 Arch | Reward | LoRA | ASR |
| --- | --- | --- | --- | --- | --- | --- |
| | | | T.2.1.1 | | | |

| Quantization | BF16 | FP8 | MXFP8 | W4A16 | FP32 Logits | Online Config |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | T.2.1.1 | |

| Hardware | A100 | H100 | B200 | GB200 | MI300x |
| --- | --- | --- | --- | --- | --- |
| | T.2.1.2 | T.2.1.2 | T.2.1.2 | T.2.1.2 | ❌ |

| vLLM Feature | Chunked Prefill | Hybrid Memory Allocator | Asynchronous Scheduling | Torch Compile | CUDA Graphs | Sliding Window |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | ❌ | ❌ | |

### C5 Arch

| Input | Basic | Long Context | Multilingual | Multi Turn | Image | Audio |
| --- | --- | --- | --- | --- | --- | --- |
| | T.1.1.1 | T.1.1.1 | T.1.1.1 | | T.1.1.1 | |

| Cohere Feature | Speculative Decoding | Guided Generation | Thinking Budget | Melody | Weight Reload | Auto-Config |
| --- | --- | --- | --- | --- | --- | --- |
| | T.3.1.1 | T.3.1.1 | T.4.1.1 | | | |

| Model Architecture | C3 Arch | C4 Arch | C5 Arch | Reward | LoRA | ASR |
| --- | --- | --- | --- | --- | --- | --- |
| | | | T.1.1.1 | | | |

| Quantization | BF16 | FP8 | MXFP8 | W4A16 | FP32 Logits | Online Config |
| --- | --- | --- | --- | --- | --- | --- |
| | | T.1.1.1 | | | | |

| Hardware | A100 | H100 | B200 | GB200 | MI300x |
| --- | --- | --- | --- | --- | --- |
| | ❌ | T.1.1.1 | T.1.1.1 | T.1.1.1 | T.1.1.1 |

| vLLM Feature | Chunked Prefill | Hybrid Memory Allocator | Asynchronous Scheduling | Torch Compile | CUDA Graphs | Sliding Window |
| --- | --- | --- | --- | --- | --- | --- |
| | T.1.1.1 | | T.3.1.1 | | T.1.1.1 | T.6.2.1 |

### LoRA

| Input | Basic | Long Context | Multilingual | Multi Turn | Image | Audio |
| --- | --- | --- | --- | --- | --- | --- |
| | T.1.2.1 | | T.1.2.1 | | | |

| Cohere Feature | Speculative Decoding | Guided Generation | Thinking Budget | Melody | Weight Reload | Auto-Config |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

| Model Architecture | C3 Arch | C4 Arch | C5 Arch | Reward | LoRA | ASR |
| --- | --- | --- | --- | --- | --- | --- |
| | | | T.1.2.1 | | T.1.2.1 | |

| Quantization | BF16 | FP8 | MXFP8 | W4A16 | FP32 Logits | Online Config |
| --- | --- | --- | --- | --- | --- | --- |
| | T.1.2.1 | | | | | |

| Hardware | A100 | H100 | B200 | GB200 | MI300x |
| --- | --- | --- | --- | --- | --- |
| | ❌ | T.1.2.1 | T.1.2.1 | T.1.2.1 | ❌ |

| vLLM Feature | Chunked Prefill | Hybrid Memory Allocator | Asynchronous Scheduling | Torch Compile | CUDA Graphs | Sliding Window |
| --- | --- | --- | --- | --- | --- | --- |
| | T.1.2.1 | | | | T.1.2.1 | |

### Thinking Budget

| Input | Basic | Long Context | Multilingual | Multi Turn | Image | Audio |
| --- | --- | --- | --- | --- | --- | --- |
| | T.4.1.1, T.4.5.1 | T.4.1.1 | T.4.1.1 | | T.4.1.1 | |

| Cohere Feature | Speculative Decoding | Guided Generation | Thinking Budget | Melody | Weight Reload | Auto-Config |
| --- | --- | --- | --- | --- | --- | --- |
| | T.3.1.1 | T.3.1.1 | T.4.1.1, T.4.5.1 | | | |

| Model Architecture | C3 Arch | C4 Arch | C5 Arch | Reward | LoRA | ASR |
| --- | --- | --- | --- | --- | --- | --- |
| | | | T.4.1.1, T.4.5.1 | | | |

| Quantization | BF16 | FP8 | MXFP8 | W4A16 | FP32 Logits | Online Config |
| --- | --- | --- | --- | --- | --- | --- |
| | | T.4.1.1, T.4.5.1 | | | | |

| Hardware | A100 | H100 | B200 | GB200 | MI300x |
| --- | --- | --- | --- | --- | --- |
| | ❌ | T.4.1.1, T.4.5.1 | T.4.1.1, T.4.5.1 | T.4.1.1, T.4.5.1 | T.4.1.1, T.4.5.1 |

| vLLM Feature | Chunked Prefill | Hybrid Memory Allocator | Asynchronous Scheduling | Torch Compile | CUDA Graphs | Sliding Window |
| --- | --- | --- | --- | --- | --- | --- |
| | T.4.1.1, T.4.5.1 | | T.3.1.1 | | T.4.1.1, T.4.5.1 | |

### Weight Reload

| Input | Basic | Long Context | Multilingual | Multi Turn | Image | Audio |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | T.6.3.1 | |

| Cohere Feature | Speculative Decoding | Guided Generation | Thinking Budget | Melody | Weight Reload | Auto-Config |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | T.6.3.1 | |

| Model Architecture | C3 Arch | C4 Arch | C5 Arch | Reward | LoRA | ASR |
| --- | --- | --- | --- | --- | --- | --- |
| | | | T.6.3.1 | | | |

| Quantization | BF16 | FP8 | MXFP8 | W4A16 | FP32 Logits | Online Config |
| --- | --- | --- | --- | --- | --- | --- |
| | | T.6.3.1 | | | | |

| Hardware | A100 | H100 | B200 | GB200 | MI300x |
| --- | --- | --- | --- | --- | --- |
| | | | | T.6.3.1 | |

| vLLM Feature | Chunked Prefill | Hybrid Memory Allocator | Asynchronous Scheduling | Torch Compile | CUDA Graphs | Sliding Window |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | T.6.3.1 | |

### Auto-Config

Pure CPU unit suite -- input/quantization/hardware/vLLM-feature axes are
not exercised; entries are intentionally blank. The suite gates the
`VLLM_ENABLE_COHERE_AUTO_CONFIG` opt-in path, profile resolution, and
`EngineArgs.__post_init__` integration.

| Input | Basic | Long Context | Multilingual | Multi Turn | Image | Audio |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

| Cohere Feature | Speculative Decoding | Guided Generation | Thinking Budget | Melody | Weight Reload | Auto-Config |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | T.6.1.1 |

| Model Architecture | C3 Arch | C4 Arch | C5 Arch | Reward | LoRA | ASR |
| --- | --- | --- | --- | --- | --- | --- |
| | T.6.1.1 | T.6.1.1 | T.6.1.1 | | | |

| Quantization | BF16 | FP8 | MXFP8 | W4A16 | FP32 Logits | Online Config |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

| Hardware | A100 | H100 | B200 | GB200 | MI300x |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

| vLLM Feature | Chunked Prefill | Hybrid Memory Allocator | Asynchronous Scheduling | Torch Compile | CUDA Graphs | Sliding Window |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

### Sliding Window

Pure CPU unit suite — no model inputs, no GPU, no engine launch. Input,
Quantization, Hardware, and most vLLM Feature axes are not exercised; entries
are intentionally blank. The suite gates the NeMo-inclusive `+1` offset in
`CohereAttention` and the downstream FlashAttention and KV-cache eviction
formulas.

| Input | Basic | Long Context | Multilingual | Multi Turn | Image | Audio |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

| Cohere Feature | Speculative Decoding | Guided Generation | Thinking Budget | Melody | Weight Reload | Auto-Config |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

| Model Architecture | C3 Arch | C4 Arch | C5 Arch | Reward | LoRA | ASR |
| --- | --- | --- | --- | --- | --- | --- |
| | | | T.6.2.1 | | | |

| Quantization | BF16 | FP8 | MXFP8 | W4A16 | FP32 Logits | Online Config |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

| Hardware | A100 | H100 | B200 | GB200 | MI300x |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

| vLLM Feature | Chunked Prefill | Hybrid Memory Allocator | Asynchronous Scheduling | Torch Compile | CUDA Graphs | Sliding Window |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | T.6.2.1 |

### Online Config

| Input | Basic | Long Context | Multilingual | Multi Turn | Image | Audio |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

| Cohere Feature | Speculative Decoding | Guided Generation | Thinking Budget | Melody | Weight Reload | Auto-Config |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

| Model Architecture | C3 Arch | C4 Arch | C5 Arch | Reward | LoRA | ASR |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

| Quantization | BF16 | FP8 | MXFP8 | W4A16 | FP32 Logits | Online Config |
| --- | --- | --- | --- | --- | --- | --- |
| | | T.2.2.1 | T.2.2.1 | | | T.2.2.1 |

| Hardware | A100 | H100 | B200 | GB200 | MI300x |
| --- | --- | --- | --- | --- | --- |
| | ❌ | ❌ | ❌ | ❌ | ❌ |

| vLLM Feature | Chunked Prefill | Hybrid Memory Allocator | Asynchronous Scheduling | Torch Compile | CUDA Graphs | Sliding Window |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | ❌ | ❌ | |

### ASR

| Input | Basic | Long Context | Multilingual | Multi Turn | Image | Audio |
| --- | --- | --- | --- | --- | --- | --- |
| | | T.7.1.2 | | | | T.7.1.1 |

| Cohere Feature | Speculative Decoding | Guided Generation | Thinking Budget | Melody | Weight Reload | Auto-Config |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

| Model Architecture | C3 Arch | C4 Arch | C5 Arch | Reward | LoRA | ASR |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | T.7.1.1 |

| Quantization | BF16 | FP8 | MXFP8 | W4A16 | FP32 Logits | Online Config |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

| Hardware | A100 | H100 | B200 | GB200 | MI300x |
| --- | --- | --- | --- | --- | --- |
| | | T.7.1.1, T.7.1.2 | | | |

| vLLM Feature | Chunked Prefill | Hybrid Memory Allocator | Asynchronous Scheduling | Torch Compile | CUDA Graphs | Sliding Window |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

## Compatibility Sources

Where to check compatibility for each category when backfilling
`## Compatibility` in a feature doc:

| Category | Default source | Additional source |
| --- | --- | --- |
| Input | Test case code. For [`test_bee_samples.py`](../../tests/cohere/test_bee_samples.py) the eval tasks cover: **Basic** (mmlupro, aime, mbpp_plus), **Long Context** (niah), **Multilingual** (mgsm), **Image** (ocrbench, infovqa, mathvista). | |
| Cohere Feature | Test case code | |
| Model Architecture | Test case code | |
| Quantization | Test case code | |
| Hardware | [`tests/cohere/configs/runner_map.json`](../../tests/cohere/configs/runner_map.json) -- which GPUs have CI runners for the test group | Test case skip markers |
| vLLM Feature | [`vllm/cohere/hardware_profiles.yaml`](../../vllm/cohere/hardware_profiles.yaml) -- which vLLM features are enabled per GPU profile | Test case code |
