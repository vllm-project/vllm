<!-- markdownlint-disable MD001 MD041 -->
<div align="center" style="margin: 30px 0;">
  <img alt="Digital China" src="./docs/assets/logos/digital-china-logo.png" style="max-width: 180px; height: auto;">
</div>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---

## About

This is a vLLM fork based on v0.14.0 with **MoE Offload** feature, enabling efficient CPU offloading for Mixture-of-Experts (MoE) model inference.

## Design Overview

### Core Design Philosophy

The core design principle is that the GPU no longer stores all expert weights for each layer, but instead caches only a limited number of hot experts. The CPU maintains the complete set of experts and dynamically determines which experts need to be copied to the GPU and which should be computed directly on the CPU based on actual token routing behavior.

The entire mechanism revolves around:
- Expert cache management
- Miss buffer handling
- Copy policy decisions
- CPU/GPU computation overlap

### Key Components

1. **Python Offload Manager (CpuOffloadInfer)**: Orchestrates the offload process, manages expert cache state, and coordinates GPU-CPU interactions
2. **GPU Expert Cache**: Limited-capacity cache storing hot experts on GPU
3. **Miss Expert Buffer (double-buffered)**: Temporary buffer for experts that miss the cache during forward passes
4. **CPU MoE Execution Engine**: AVX/AMX-optimized kernels for computing expert forward passes on CPU
5. **GPU↔CPU Callback-based Synchronization**: Asynchronous communication mechanism for coordinating GPU and CPU execution

### Initialization Phase

During model initialization:
- All MoE expert weights for each layer are fully loaded and permanently resident in CPU pinned memory
- The GPU allocates an Expert Cache with capacity `cache_expert_num` for each layer, storing the most frequently accessed experts
- The GPU cache is not static; experts are dynamically managed based on runtime token routing behavior

To track the state of experts in the GPU cache, the system maintains per-layer metadata:
- `cache_map`: Maps expert IDs to their positions in the GPU cache
- `miss_map`: Tracks which experts are currently in the miss buffer
- `policy_sort`: Maintains priority ordering for expert replacement decisions

### Forward Pass Execution Flow

#### Step 1: Expert Cache Policy Matching

At the start of a forward pass, the model has already obtained `topk_ids` for each token from the router. The system calls `expert_cache_policy` to match these `topk_ids` against the current layer's cache state.

This process outputs two key pieces of information:
1. `cpu_topk_ids`: Which tokens' experts require CPU computation
2. `copy_map`: The set of experts that need to be copied from CPU to GPU in this forward pass

**Important**: `copy_map` does not directly correspond to "experts copied to GPU cache". It is simply a list of experts that need to be copied in this pass, and their final destination depends on the execution mode.

#### Step 2: Execution Mode Selection

The system operates in two primary execution modes:

**DBO Mode (Dual Batch Overlap)**

When the system is in DBO mode or in decode/small batch scenarios, the forward pass enters a fully parallel CPU-GPU execution path:

- Experts in `copy_map` are asynchronously copied to the GPU Expert Cache for subsequent `fused_experts` computation
- CPU immediately begins computing miss experts
- CPU computation, GPU computation, and expert copying are deliberately placed in different execution threads
- Overlap is achieved through vLLM's DBO scheduling mechanism: while the GPU computes fused experts for the current batch, the CPU is already working on miss experts for the next step or the same step, maximizing resource utilization and reducing decode latency

**Prefetch Mode**

In Prefetch mode (typically for larger prefill batches), system behavior adjusts based on the number of tokens in the batch:

- As token count increases, more experts are triggered in the forward pass
- The system dynamically calculates `n_copy` to limit the maximum number of experts copied in this pass
- If `n_copy` is less than the total number of experts:
  - CPU still participates in computation
  - Experts in `copy_map` are not placed in the GPU cache
  - Instead, they are copied to a dedicated Miss Expert Buffer (`temp_layer`)
  - GPU uses this temp buffer to execute `fused_experts`
  - CPU computes the remaining experts that were not copied
  - Results from both paths are merged at the output stage
- When batch size is extremely large and `n_copy` covers all or nearly all experts:
  - The system automatically degrades to "full GPU mode"
  - CPU no longer participates in computation
  - All experts are copied and `fused_experts` computation is completed on the GPU side
  - This is not an additional branch logic, but a natural consequence of the Prefetch strategy when copy count reaches the threshold

**Double-Buffered Miss Expert Buffer Management**: To prevent miss experts from being overwritten during cross-layer execution, the system globally maintains only two Miss Expert Buffers, using `layer_id % 2` for double-buffering:
- Even-numbered layers use buffer 0
- Odd-numbered layers use buffer 1

By coordinating with independent CUDA streams and events:
- Copy and computation on the same buffer are strictly serialized
- Different buffers can form a natural pipeline
- Expert copying and computation for adjacent layers can interleave, enabling efficient pipelining

## Installation

Install this version in development mode:

```bash
pip install -e .
```

## Usage

### Example 1: 4 GPU Setup (TP=4)

```bash
CUDA_VISIBLE_DEVICES='2,3,4,5' vllm serve /home/models/DeepSeek-R1/ \
--trust-remote-code --max-num-seqs 4 --tensor_parallel_size 4 --distributed-executor-backend "mp" \
--compilation-config '{"cudagraph_capture_sizes": [1,2,4]}' \
--enable-dbo --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 16384 --max-model-len 16384 --no-enable-chunked-prefill --no-enable-prefix-caching --moe-offload \
--moe-offload-cache-expert-num 32 --moe-offload-cache-topk 2 --moe-offload-update-expert-num 2 --moe-offload-context-num-threads 14
```

### Example 2: 8 GPU Setup (TP=8)

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' vllm serve /home/models/DeepSeek-R1/ \
--trust-remote-code --max-num-seqs 8 --tensor_parallel_size 8 --distributed-executor-backend "mp" \
--compilation-config '{"cudagraph_capture_sizes": [1,2,4,8]}' \
--enable-dbo --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 16384 --max-model-len 16384 --no-enable-chunked-prefill --no-enable-prefix-caching --moe-offload \
--moe-offload-cache-expert-num 104 --moe-offload-cache-topk 2 --moe-offload-update-expert-num 2 --moe-offload-context-num-threads 6
```

### MoE Offload Parameters

| Parameter | Description | Default | Recommended Values |
|-----------|-------------|---------|-------------------|
| `--moe-offload` | Enable MoE offload mode | `false` | Required to enable |
| `--moe-offload-cache-expert-num` | Number of MoE experts cached per layer on GPU | - | TP=4: 32, TP=8: 104 |
| `--moe-offload-cache-topk` | CPU cache computation strategy | `2` | 2 |
| `--moe-offload-update-expert-num` | Number of experts updated in CPU MoE | `2` | 2 |
| `--moe-offload-context-num-threads` | Number of threads per process for CPU computation | - | TP=4: 12-14, TP=8: 6 |

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

