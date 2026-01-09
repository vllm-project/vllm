<!-- markdownlint-disable MD001 MD041 -->
<div align="center" style="display: flex; justify-content: center; align-items: center; gap: 60px; flex-wrap: wrap; margin: 30px 0;">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" style="max-width: 300px; height: auto;">
  </picture>
  <div style="display: flex; align-items: center; gap: 15px;">
    <!-- 神州数码logo - 请将logo文件放置到 docs/assets/logos/digital-china-logo.png -->
    <img alt="Digital China" src="./docs/assets/logos/digital-china-logo.png" style="max-width: 250px; height: auto;" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
    <div style="display: none; flex-direction: column; align-items: flex-start; justify-content: center;">
      <div style="font-size: 28px; font-weight: bold; color: #000; line-height: 1.2;">神州数码</div>
      <div style="font-size: 16px; color: #666; margin-top: 5px;">Digital China</div>
    </div>
  </div>
</div>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

<div align="center" style="margin: 20px 0;">
  <button id="lang-en" onclick="switchLanguage('en')" style="padding: 8px 16px; margin: 0 5px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px;">English</button>
  <button id="lang-zh" onclick="switchLanguage('zh')" style="padding: 8px 16px; margin: 0 5px; background-color: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px;">简体中文</button>
</div>

<script>
function switchLanguage(lang) {
  const enContent = document.querySelectorAll('.lang-en');
  const zhContent = document.querySelectorAll('.lang-zh');
  const enBtn = document.getElementById('lang-en');
  const zhBtn = document.getElementById('lang-zh');
  
  if (lang === 'en') {
    enContent.forEach(el => el.style.display = 'block');
    zhContent.forEach(el => el.style.display = 'none');
    enBtn.style.backgroundColor = '#007bff';
    zhBtn.style.backgroundColor = '#6c757d';
  } else {
    enContent.forEach(el => el.style.display = 'none');
    zhContent.forEach(el => el.style.display = 'block');
    enBtn.style.backgroundColor = '#6c757d';
    zhBtn.style.backgroundColor = '#007bff';
  }
}

// Initialize to English by default
window.onload = function() {
  switchLanguage('en');
};
</script>

---

<div class="lang-en">

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

</div>

<div class="lang-zh" style="display: none;">

## 关于

这是基于 v0.14.0 版本的 vLLM 分支，带有 **MoE Offload** 特性，支持高效的混合专家（MoE）模型推理 CPU 卸载。

## 设计概述

### 核心设计理念

核心设计原则是 GPU 不再存储每一层的所有专家权重，而是仅缓存有限数量的热专家。CPU 维护完整的专家集合，并根据实际 token 路由行为动态决定哪些专家需要复制到 GPU，哪些应该直接在 CPU 上计算。

整个机制围绕以下几个方面：
- 专家缓存管理
- 缺失缓冲区处理
- 复制策略决策
- CPU/GPU 计算重叠

### 关键组件

1. **Python 卸载管理器 (CpuOffloadInfer)**: 协调卸载过程，管理专家缓存状态，并协调 GPU-CPU 交互
2. **GPU 专家缓存**: 在 GPU 上存储热专家的有限容量缓存
3. **缺失专家缓冲区（双缓冲）**: 在前向传播期间缓存未命中的专家的临时缓冲区
4. **CPU MoE 执行引擎**: 用于在 CPU 上计算专家前向传播的 AVX/AMX 优化内核
5. **GPU↔CPU 基于回调的同步**: 用于协调 GPU 和 CPU 执行的异步通信机制

### 初始化阶段

在模型初始化期间：
- 每一层的所有 MoE 专家权重完全加载并永久驻留在 CPU 固定内存中
- GPU 为每一层分配容量为 `cache_expert_num` 的专家缓存，存储最常访问的专家
- GPU 缓存不是静态的；专家根据运行时 token 路由行为动态管理

为了跟踪 GPU 缓存中专家的状态，系统维护每层元数据：
- `cache_map`: 将专家 ID 映射到其在 GPU 缓存中的位置
- `miss_map`: 跟踪当前在缺失缓冲区中的专家
- `policy_sort`: 维护专家替换决策的优先级排序

### 前向传播执行流程

#### 步骤 1: 专家缓存策略匹配

在前向传播开始时，模型已经从路由器获取了每个 token 的 `topk_ids`。系统调用 `expert_cache_policy` 将这些 `topk_ids` 与当前层的缓存状态进行匹配。

此过程输出两个关键信息：
1. `cpu_topk_ids`: 哪些 token 的专家需要 CPU 计算
2. `copy_map`: 在此次前向传播中需要从 CPU 复制到 GPU 的专家集合

**重要**: `copy_map` 并不直接对应于"复制到 GPU 缓存的专家"。它只是本次传播中需要复制的专家列表，它们的最终目的地取决于执行模式。

#### 步骤 2: 执行模式选择

系统有两种主要的执行模式：

**DBO 模式（双批次重叠）**

当系统处于 DBO 模式或解码/小批次场景时，前向传播进入完全并行的 CPU-GPU 执行路径：

- `copy_map` 中的专家异步复制到 GPU 专家缓存，用于后续的 `fused_experts` 计算
- CPU 立即开始计算缺失专家
- CPU 计算、GPU 计算和专家复制被有意放置在不同的执行线程中
- 通过 vLLM 的 DBO 调度机制实现重叠：当 GPU 计算当前批次融合专家时，CPU 已经在为下一步或同一步处理缺失专家，最大化资源利用率并减少解码延迟

**预取模式**

在预取模式（通常用于较大的预填充批次）中，系统行为根据批次中的 token 数量进行调整：

- 随着 token 数量增加，前向传播中触发的专家数量增加
- 系统动态计算 `n_copy` 以限制本次传播中复制的最大专家数量
- 如果 `n_copy` 小于专家总数：
  - CPU 仍参与计算
  - `copy_map` 中的专家不放置在 GPU 缓存中
  - 相反，它们被复制到专用的缺失专家缓冲区 (`temp_layer`)
  - GPU 使用此临时缓冲区执行 `fused_experts`
  - CPU 计算未复制的剩余专家
  - 两个路径的结果在输出阶段合并
- 当批次大小非常大且 `n_copy` 覆盖所有或几乎所有专家时：
  - 系统自动降级为"全 GPU 模式"
  - CPU 不再参与计算
  - 所有专家被复制，`fused_experts` 计算在 GPU 端完成
  - 这不是额外的分支逻辑，而是当复制计数达到阈值时预取策略的自然结果

**双缓冲缺失专家缓冲区管理**: 为了防止跨层执行期间缺失专家被覆盖，系统全局仅维护两个缺失专家缓冲区，使用 `layer_id % 2` 进行双缓冲：
- 偶数层使用缓冲区 0
- 奇数层使用缓冲区 1

通过与独立的 CUDA 流和事件协调：
- 同一缓冲区上的复制和计算严格串行化
- 不同的缓冲区可以形成自然流水线
- 相邻层的专家复制和计算可以交错，实现高效的流水线处理

## 安装

以开发模式安装此版本：

```bash
pip install -e .
```

## 使用方法

### 示例 1: 4 GPU 设置 (TP=4)

```bash
CUDA_VISIBLE_DEVICES='2,3,4,5' vllm serve /home/models/DeepSeek-R1/ \
--trust-remote-code --max-num-seqs 4 --tensor_parallel_size 4 --distributed-executor-backend "mp" \
--compilation-config '{"cudagraph_capture_sizes": [1,2,4]}' \
--enable-dbo --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 16384 --max-model-len 16384 --no-enable-chunked-prefill --no-enable-prefix-caching --moe-offload \
--moe-offload-cache-expert-num 32 --moe-offload-cache-topk 2 --moe-offload-update-expert-num 2 --moe-offload-context-num-threads 14
```

### 示例 2: 8 GPU 设置 (TP=8)

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' vllm serve /home/models/DeepSeek-R1/ \
--trust-remote-code --max-num-seqs 8 --tensor_parallel_size 8 --distributed-executor-backend "mp" \
--compilation-config '{"cudagraph_capture_sizes": [1,2,4,8]}' \
--enable-dbo --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 16384 --max-model-len 16384 --no-enable-chunked-prefill --no-enable-prefix-caching --moe-offload \
--moe-offload-cache-expert-num 104 --moe-offload-cache-topk 2 --moe-offload-update-expert-num 2 --moe-offload-context-num-threads 6
```

### MoE Offload 参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--moe-offload` | 开启 MoE offload 模式 | `false` | 需要启用 |
| `--moe-offload-cache-expert-num` | GPU 端每层存放的 MOE 专家个数 | - | TP=4: 32, TP=8: 104 |
| `--moe-offload-cache-topk` | CPU 端 cache 的计算策略 | `2` | 2 |
| `--moe-offload-update-expert-num` | CPU 端 MOE 更新的专家个数 | `2` | 2 |
| `--moe-offload-context-num-threads` | CPU 中一个进程参与 CPU 计算的线程个数 | - | TP=4: 12-14, TP=8: 6 |

## 引用

如果您在研究中使用了 vLLM，请引用我们的[论文](https://arxiv.org/abs/2309.06180)：

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

</div>
