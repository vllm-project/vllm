# vLLM 100 天学习计划

> 本计划旨在帮助开发者系统性地学习 vLLM 框架，从基础入门到高级特性，最终能够进行二次开发和贡献。

---

## 学习前准备（第 0 天）

### 环境搭建

```bash
# 1. 克隆仓库
git clone https://github.com/vllm-project/vllm.git
cd vllm

# 2. 安装依赖
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .

# 3. 安装测试和 lint 工具
uv pip install -r requirements/test.txt
uv pip install -r requirements/lint.txt
pre-commit install
```

### 基础知识准备

| 必需知识 | 推荐知识 |
|---------|---------|
| Python 编程 | CUDA 编程 |
| PyTorch 基础 | Triton kernel |
| Transformer 架构 | 分布式训练 |

### 学习资源

- **vLLM 官方文档**: https://docs.vllm.ai
- **GitHub 仓库**: https://github.com/vllm-project/vllm
- **PagedAttention 论文**: https://arxiv.org/abs/2309.06180

---

## 第一阶段：vLLM 基础入门（第 1-15 天）

### 第 1-3 天：vLLM 概览和核心概念

#### 学习目标
- 理解 vLLM 的核心功能和设计哲学
- 了解 PagedAttention 的基本原理
- 掌握 vLLM 的基本使用方式

#### 学习内容

1. **阅读官方文档**
   - 核心特性：PagedAttention、连续批处理、注意力后端等

2. **阅读关键论文**
   - PagedAttention 论文：理解虚拟内存在 KV 缓存管理中的应用

3. **代码探索**
   - 阅读 `vllm/v1/` 目录结构
   - 了解 v0 和 v1 架构的区别

#### 实践任务

```python
# 任务 1：使用 LLM 类运行一个简单的推理
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
outputs = llm.generate(["Hello, world!"], SamplingParams())
print(outputs[0].outputs[0].text)

# 任务 2：尝试不同的采样参数
params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    max_tokens=100
)

# 任务 3：尝试批处理推理
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
outputs = llm.generate(prompts, SamplingParams())
```

---

### 第 4-7 天：KV Cache 管理机制

#### 学习目标
- 深入理解 KV Cache 的工作原理
- 掌握 PagedAttention 的实现细节
- 了解 v1 架构中的 KV Cache 管理器

#### 核心文件阅读

| 文件 | 内容 |
|------|------|
| `vllm/v1/core/kv_cache_manager.py` | KVCacheManager 类：总体缓存管理逻辑 |
| `vllm/v1/core/block_pool.py` | BlockPool 类：物理块的分配和释放 |
| `vllm/v1/core/kv_cache_utils.py` | 缓存相关的工具函数 |
| `vllm/v1/core/single_type_kv_cache_manager.py` | 单一类型缓存管理器 |

#### 实践任务

1. **分析 KV Cache 内存使用**
   - 计算不同模型大小和序列长度下的 KV Cache 内存占用

2. **绘制 KV Cache 管理流程图**
   - 理解 allocate、append、fork、free 等操作

3. **实验不同 block_size 的影响**
   - 修改配置，观察性能变化

---

### 第 8-15 天：调度系统

#### 学习目标
- 理解 vLLM 的调度策略
- 掌握请求队列和批处理机制
- 了解上下文并行（Context Parallel）

#### 核心文件阅读

| 文件 | 内容 |
|------|------|
| `vllm/v1/core/sched/scheduler.py` | Scheduler 类：主调度器实现 |
| `vllm/v1/core/sched/request_queue.py` | RequestQueue 类：请求队列管理 |
| `vllm/v1/core/sched/output.py` | 调度输出数据结构 |
| `vllm/v1/core/sched/interface.py` | 调度器接口定义 |
| `vllm/v1/core/encoder_cache_manager.py` | 编码器缓存管理（多模态模型） |

#### 实践任务

1. **跟踪一个请求的完整调度生命周期**
   - 从提交到完成的整个流程

2. **实验不同的调度策略**
   - 观察不同批处理大小对吞吐量的影响

3. **分析抢占（preemption）机制**
   - 理解 vLLM 如何处理内存压力

---

## 第二阶段：注意力机制深入（第 16-40 天）

### 第 16-22 天：注意力后端架构

#### 学习目标
- 理解 vLLM 注意力后端的插件架构
- 掌握不同后端的特点和适用场景
- 了解 MLA（多头潜在注意力）实现

#### 核心文件阅读

| 文件 | 内容 |
|------|------|
| `vllm/v1/attention/backend.py` | AttentionBackend 基类、AttentionLayer、AttentionMetadata |
| `vllm/v1/attention/selector.py` | 后端选择和自动回退机制 |
| `vllm/v1/attention/backends/mla/` | 各种 MLA 后端实现 |
| `vllm/model_executor/layers/attention/mla_attention.py` | MLA 通用实现 |

#### MLA 后端文件详解

| 文件 | 描述 |
|------|------|
| `flashattn_mla.py` | FlashAttention MLA（Hopper 架构） |
| `flashinfer_mla.py` | FlashInfer MLA（SM100/Blackwell） |
| `flashmla.py` | DeepSeek FlashMLA（Hopper + Blackwell） |
| `cutlass_mla.py` | CUTLASS MLA（SM100） |
| `rocm_aiter_mla.py` | ROCm Aiter MLA |
| `triton_mla.py` | Triton MLA（通用） |

#### 实践任务

1. **绘制注意力后端类层次图**
   - 理解继承关系和接口设计

2. **实验不同后端**
   - 在支持的硬件上测试不同后端的性能

3. **分析 MLA 与标准注意力的区别**
   - 理解 DeepSeek 等模型的压缩机制

---

### 第 23-30 天：注意力 Operations

#### 学习目标
- 深入理解 Triton kernel 实现
- 掌握解码注意力和预填充注意力的区别
- 了解 Flash Attention 集成

#### 核心文件阅读

| 文件 | 内容 |
|------|------|
| `vllm/v1/attention/ops/triton_decode_attention.py` | 解码阶段的注意力 kernel，支持 GQA |
| `vllm/v1/attention/ops/triton_prefill_attention.py` | 预填充阶段的注意力 kernel |
| `vllm/v1/attention/ops/triton_unified_attention.py` | 统一的注意力实现 |
| `vllm/v1/attention/ops/vit_attn_wrappers.py` | Vision Transformer 注意力封装 |
| `vllm/v1/attention/ops/flashmla.py` | FlashMLA 操作 |
| `vllm/v1/attention/ops/xpu_mla_sparse.py` | XPU 稀疏 MLA |

#### 实践任务

1. **学习 Triton 基础**
   - 完成 Triton 官方教程

2. **分析 kernel 性能**
   - 使用 NVIDIA Nsight 或类似工具

3. **尝试修改 kernel 参数**
   - 观察 BLOCK_SIZE、NUM_WARPS 等参数的影响

---

### 第 31-40 天：稀疏注意力和 MLA

#### 学习目标
- 理解 DeepSeek-V3.2 的稀疏注意力机制
- 掌握 MLA 的压缩原理
- 了解索引器实现

#### 核心文件阅读

| 文件 | 内容 |
|------|------|
| `vllm/v1/attention/backends/mla/flashmla_sparse.py` | FlashMLA 稀疏实现，FP8 KV 缓存 |
| `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py` | FlashInfer 稀疏 MLA |
| `vllm/v1/attention/backends/mla/indexer.py` | DeepSeek-V3.2 索引器 |
| `vllm/v1/attention/backends/mla/sparse_utils.py` | 稀疏索引转换工具 |

#### FP8 KV 缓存格式（FlashMLA Sparse）

```
每个 token 的 KV 缓存 = 656 字节
├── 前 512 字节：量化的 NoPE 部分（512 个 float8_e4m3 值）
├── 接下来 16 字节：缩放因子（4 个 float32 值）
└── 最后 128 字节：RoPE 部分（64 个 bfloat16 值，未量化）
```

#### 实践任务

1. **理解稀疏注意力的索引机制**
   - 绘制索引转换流程图

2. **分析 FP8 KV 缓存格式**
   - 理解 656 字节结构

3. **比较稠密和稀疏 MLA 的性能差异**

---

## 第三阶段：执行引擎和采样（第 41-60 天）

### 第 41-48 天：执行引擎

#### 学习目标
- 理解 vLLM 的执行引擎架构
- 掌握模型执行流程
- 了解 CUDA Graph 优化

#### 核心文件阅读

| 文件 | 内容 |
|------|------|
| `vllm/v1/engine/llm_engine.py` | 主引擎类，协调所有组件 |
| `vllm/v1/engine/async_llm.py` | 异步 LLM 接口 |
| `vllm/v1/engine/core.py` | 核心执行逻辑 |
| `vllm/v1/engine/coordinator.py` | 协调器，管理分布式执行 |
| `vllm/v1/worker/gpu_model_runner.py` | GPU 模型执行器 |

#### 实践任务

1. **跟踪一个请求的完整执行路径**
   - 从 API 调用到模型前向传播

2. **分析 CUDA Graph 的捕获和重放**
   - 理解性能优化原理

3. **实验不同的执行配置**
   - tensor_parallel_size、pipeline_parallel_size

---

### 第 49-55 天：采样和 Logits 处理

#### 学习目标
- 掌握各种采样策略
- 理解 Logits 处理器机制
- 了解惩罚和约束实现

#### 核心文件阅读

| 文件 | 内容 |
|------|------|
| `vllm/v1/sample/sampler.py` | 采样器主类 |
| `vllm/v1/sample/ops/topk_topp_sampler.py` | Top-K 和 Top-P 采样 |
| `vllm/v1/sample/ops/penalties.py` | 频率惩罚、存在惩罚等 |
| `vllm/v1/sample/logits_processor/` | 各种 logits 处理器 |
| `vllm/v1/sample/rejection_sampler.py` | 拒绝采样（speculate decode） |

#### 实践任务

```python
# 任务 1：实现自定义 logits 处理器
from vllm import SamplingParams

def my_processor(logits, token_ids):
    # 自定义逻辑
    return logits

# 任务 2：实验不同的采样参数
# 观察输出质量和多样性

# 任务 3：实现简单的约束解码
# 如只允许特定词汇
```

---

### 第 56-60 天：结构化输出

#### 学习目标
- 理解语法约束解码
- 掌握 JSON/Regex 约束实现
- 了解 XGrammar 和 Outlines 集成

#### 核心文件阅读

| 文件 | 内容 |
|------|------|
| `vllm/v1/structured_output/__init__.py` | 结构化输出主模块 |
| `vllm/v1/structured_output/backend_xgrammar.py` | XGrammar 后端 |
| `vllm/v1/structured_output/backend_outlines.py` | Outlines 后端 |
| `vllm/v1/structured_output/backend_guidance.py` | Guidance 库集成 |

#### 实践任务

```python
# 任务 1：使用 JSON Schema 约束输出
from vllm import LLM, SamplingParams

llm = LLM(model="...")
sampling_params = SamplingParams(
    guided_json={"type": "object", "properties": {...}}
)

# 任务 2：尝试 Regex 约束
sampling_params = SamplingParams(
    guided_regex=r"\d{3}-\d{3}-\d{4}"  # 电话号码格式
)

# 任务 3：分析语法树匹配算法
```

---

## 第四阶段：高级特性（第 61-80 天）

### 第 61-67 天：推测解码（Speculative Decoding）

#### 学习目标
- 理解推测解码原理
- 掌握 EAGLE 和 Medusa 实现
- 了解草稿模型机制

#### 核心文件阅读

| 文件 | 内容 |
|------|------|
| `vllm/v1/spec_decode/eagle.py` | EAGLE 推测解码实现 |
| `vllm/v1/spec_decode/medusa.py` | Medusa 多头条解码 |
| `vllm/v1/spec_decode/ngram_proposer.py` | N-gram 提议器 |
| `vllm/v1/spec_decode/suffix_decoding.py` | 后缀解码 |
| `vllm/v1/spec_decode/metrics.py` | 推测解码指标 |

#### 实践任务

```python
# 任务 1：配置推测解码运行
llm = LLM(
    model="target_model",
    speculative_model="draft_model",
    num_speculative_tokens=5
)

# 任务 2：分析接受率
# 理解影响接受率的因素

# 任务 3：比较不同推测策略
```

---

### 第 68-73 天：分布式推理

#### 学习目标
- 理解张量并行和流水线并行
- 掌握 KV 缓存分布式管理
- 了解 PD 分离架构

#### 核心文件阅读

| 文件 | 内容 |
|------|------|
| `vllm/v1/worker/gpu/cp_utils.py` | 上下文并行工具 |
| `vllm/v1/worker/gpu/dp_utils.py` | 数据并行工具 |
| `vllm/v1/worker/gpu/pp_utils.py` | 流水线并行工具 |
| `vllm/v1/engine/parallel_sampling.py` | 并行采样 |
| `vllm/distributed/` | 分布式通信原语 |

#### 实践任务

```bash
# 任务 1：多 GPU 运行
python -m vllm.entrypoints.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4

# 任务 2：分析通信开销
# 理解 AllReduce、AllGather 等操作

# 任务 3：配置 PD 分离
# Prefill 和 Decode 分离部署
```

---

### 第 74-80 天：性能分析和优化

#### 学习目标
- 掌握性能分析工具
- 理解瓶颈识别方法
- 学习优化策略

#### 核心文件阅读

| 文件 | 内容 |
|------|------|
| `vllm/v1/metrics/` | 性能指标收集 |
| `vllm/v1/metrics/prometheus.py` | Prometheus 指标导出 |
| `vllm/v1/metrics/loggers.py` | 日志记录器 |
| `vllm/profiler/` | 性能分析工具 |

#### 实践任务

```python
# 任务 1：使用 PyTorch Profiler 分析
import torch.profiler as profiler

with profiler.profile(...) as prof:
    llm.generate(...)

# 任务 2：分析内存使用
# 识别内存瓶颈

# 任务 3：优化配置
# 调整 block_size、gpu_memory_utilization 等
```

---

## 第五阶段：扩展和贡献（第 81-100 天）

### 第 81-87 天：新模型适配

#### 学习目标
- 理解模型注册机制
- 掌握新模型添加流程
- 学习模型架构实现

#### 核心文件阅读

| 文件 | 内容 |
|------|------|
| `vllm/model_executor/model_loader/` | 模型加载器 |
| `vllm/model_executor/models/` | 各种模型实现示例 |
| `vllm/transformers_utils/config.py` | Transformers 配置处理 |

#### 建议阅读的模型实现

- `qwen2.py` / `qwen3_5.py`
- `llama.py`
- `mixtral.py`

#### 实践任务

1. **添加一个新的模型架构**
   - 或为一个现有模型添加变体

2. **实现自定义权重加载**
   - 处理非标准权重格式

3. **提交 PR**
   - 遵循项目贡献指南

---

### 第 88-93 天：自定义 Kernel 开发

#### 学习目标
- 掌握 Triton kernel 开发
- 理解 CUDA kernel 集成
- 学习性能优化技巧

#### 学习内容

1. **Triton 深入学习**
   - 官方文档和示例
   - 阅读 vLLM 中的 Triton kernel

2. **CUDA 基础（如果需要）**
   - CUDA C++ 编程
   - kernel 优化技巧

3. **阅读 `vllm/csrc/` 目录**
   - C++/CUDA 扩展

#### 实践任务

```python
# 任务 1：实现一个简单的 Triton kernel
@triton.jit
def my_kernel(...):
    # 自定义逻辑
    pass

# 任务 2：优化现有 kernel
# 尝试不同的 block size、warp 配置

# 任务 3：benchmark 对比
# 与 PyTorch 原生实现比较
```

---

### 第 94-100 天：综合项目和总结

#### 学习目标
- 整合所学知识
- 完成一个综合项目
- 总结学习经验

#### 项目选项

| 选项 | 描述 |
|------|------|
| 实现一个新特性 | 如新的采样策略、新的注意力后端等 |
| 性能优化 | 针对特定场景优化 vLLM |
| 新硬件支持 | 为 vLLM 添加新硬件后端 |
| 文档和教程 | 编写详细的学习指南 |

#### 实践任务

1. **设计和实现项目**
   - 应用所学知识

2. **编写测试**
   - 确保代码质量

3. **文档化**
   - 编写清晰的使用说明

---

## 每日学习时间安排建议

### 工作日（周一至周五）

| 时间段 | 时长 | 内容 |
|--------|------|------|
| 早上 | 1 小时 | 阅读文档/论文 |
| 晚上 | 2-3 小时 | 代码阅读和实践 |

### 周末

| 时间段 | 时长 | 内容 |
|--------|------|------|
| 全天 | 4-6 小时 | 综合实践和项目 |

**总计**: 约 20-25 小时/周

---

## 进度追踪

### 每周自我评估

1. 本周学到了什么？
2. 有哪些困惑？
3. 下周计划是什么？

### 笔记建议

使用笔记工具记录：
- 关键概念和原理
- 代码注释和理解
- 遇到的问题和解决方案

---

## 里程碑检查点

| 阶段 | 天数 | 检查点 |
|------|------|--------|
| 第一阶段 | 第 15 天 | 能够独立使用 vLLM 进行推理 |
| 第二阶段 | 第 40 天 | 理解注意力机制和 KV Cache 管理 |
| 第三阶段 | 第 60 天 | 掌握执行引擎和采样策略 |
| 第四阶段 | 第 80 天 | 能够配置分布式推理和性能优化 |
| 第五阶段 | 第 100 天 | 能够进行二次开发和贡献 |

---

## 常见问题解答

### Q1: 没有 GPU 可以学习吗？

可以阅读代码和理解架构，但实践任务需要 GPU。建议使用：
- Colab 免费版（T4）
- 云平台按需实例

### Q2: 需要多深的 CUDA 知识？

基础使用不需要，但要深入优化需要：
- 基础：Python + PyTorch
- 进阶：Triton
- 深入：CUDA C++

### Q3: 如何保持学习动力？

- 加入 vLLM 社区
- 参与 Issues 讨论
- 尝试贡献小 PR

---

*最后更新：2026-03-22*
