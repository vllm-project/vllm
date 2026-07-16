# 关键横切模块深度解析

## 概述

横切模块 (cross-cutting modules) 是指跨越多个层级、为整个系统提供通用能力的子系统。它们不属于某一个具体层，而是像"基础设施"一样被各层依赖。

```
横切模块
├── 采样系统 (v1/sample/)          ← logits → token 的完整流水线
├── 分布式通信 (distributed/)      ← TP/PP/DP/EP 并行通信
├── 投机解码 (v1/spec_decode/)     ← 加速推理的投机-验证机制
├── 多模态处理 (multimodal/)       ← 图像/音频/视频输入处理
├── LoRA 适配 (lora/)             ← 低秩适配器的动态加载
└── 量化 (layers/quantization/)   ← 模型压缩与低精度推理
```

---

## 1. 采样系统

采样系统负责把模型输出的 logits 张量转化为实际的 token id。这不只是一个 `argmax`，而是一套复杂的流水线。

### 1.1 采样流水线（按顺序）

```python
# vllm/v1/sample/sampler.py
class Sampler(nn.Module):
    def forward(self, logits, sampling_metadata):
        # 1. 计算原始 logprobs（如果请求了 logprobs）
        # 2. 转 float32
        # 3. 应用 allowed_token_ids 白名单
        # 4. 应用 bad_words 黑名单
        # 5. 应用 logit processors（min_tokens, logit_bias）
        # 6. 应用 penalties（repetition, frequency, presence）
        # 7. 采样（greedy 或 random）
        #    a) temperature 缩放
        #    b) min_p 过滤
        #    c) top_k / top_p 截断
        #    d) 从概率分布中采样
        # 8. 收集 top logprobs
        # 9. 返回 SamplerOutput
```

### 1.2 两种采样路径

```
all_greedy = True  → 直接 argmax，跳过 temperature/top_k/top_p
all_random = True  → 完整随机采样流程
混合模式           → 先 greedy 采样，再 random 采样，按请求级参数合并结果
```

### 1.3 关键组件

| 组件 | 文件 | 作用 |
|------|------|------|
| `Sampler` | `sampler.py` | 主采样器，编排整个流水线 |
| `TopKTopPSampler` | `ops/topk_topp_sampler.py` | top-k/top-p 截断 + 采样 |
| `apply_all_penalties` | `ops/penalties.py` | 频率/重复/存在惩罚 |
| `SamplingMetadata` | `metadata.py` | 每个请求的采样参数 |

### 1.4 SamplingMetadata：按请求的参数管理

```python
class SamplingMetadata:
    temperature: torch.Tensor      # [num_reqs] 每个请求的温度
    top_p: torch.Tensor           # [num_reqs]
    top_k: torch.Tensor           # [num_reqs]
    min_p: torch.Tensor           # [num_reqs]
    all_greedy: bool              # 是否所有请求都是 greedy
    all_random: bool              # 是否所有请求都是 random
    max_num_logprobs: int | None  # 最大 logprobs 数量
```

由于 continuous batching 中每个请求可以有不同的采样参数，Sampler 必须支持 batch 内的异构采样。

---

## 2. 分布式通信

分布式模块是 vLLM 多 GPU / 多节点推理的基础。

### 2.1 并行组管理

```python
# vllm/distributed/parallel_state.py
class GroupCoordinator:
    """管理一组进程之间的通信"""
    rank: int                        # 全局 rank
    world_size: int                  # 组大小
    cpu_group: ProcessGroup          # CPU 通信组
    device_group: ProcessGroup       # GPU 通信组
    device_communicator: DeviceCommunicatorBase  # 设备级通信器
```

vLLM 维护多个并行组，每个组负责不同维度的通信：

```python
get_tp_group()   # Tensor Parallel 组（同一个模型切片内的 ranks）
get_pp_group()   # Pipeline Parallel 组（不同 stage 的 ranks）
get_dp_group()   # Data Parallel 组（处理不同数据的 ranks）
get_ep_group()   # Expert Parallel 组（MoE 的专家分布）
```

### 2.2 通信操作

```python
# vllm/distributed/communication_op.py

# TP 内常用操作
tensor_model_parallel_all_reduce(tensor)   # 求和归约
tensor_model_parallel_all_gather(tensor)   # 收集拼接

# PP 之间的操作
get_pp_group().send_tensor_dict(tensors)   # 发送中间张量到下一个 stage
get_pp_group().recv_tensor_dict()          # 从上一个 stage 接收
```

### 2.3 设备通信器

底层通信走可插拔的 `DeviceCommunicator`：

| 通信器 | 场景 | 说明 |
|--------|------|------|
| PyNCCL | NVIDIA GPU 默认 | 高性能 GPU 集合通信 |
| CustomAllReduce | 单节点 TP | 基于共享内存的低延迟 all-reduce |
| FlashInfer AllReduce | 可选 | FlashInfer 提供的 all-reduce |
| Ray Communicator | 多节点 Ray | 通过 Ray Actor 中转 |
| CPU Communicator | CPU 推理 | Gloo 后端 |

### 2.4 四种并行的协作

```
一个 8 GPU、TP=4、PP=2 的部署：

PP Stage 0              PP Stage 1
┌───────────────────┐   ┌───────────────────┐
│ TP rank 0 │ rank 1│   │ TP rank 0 │ rank 1│
│ GPU 0     │ GPU 1 │   │ GPU 4     │ GPU 5 │
├───────────┼───────┤   ├───────────┼───────┤
│ TP rank 2 │ rank 3│   │ TP rank 2 │ rank 3│
│ GPU 2     │ GPU 3 │   │ GPU 6     │ GPU 7 │
└───────────────────┘   └───────────────────┘
       │                        ▲
       └── PP send/recv ────────┘

TP 通信：每一层 forward 内部的 all-reduce（高频、小数据量）
PP 通信：stage 之间传递 hidden_states（低频、大数据量）
```

---

## 3. 投机解码

投机解码 (Speculative Decoding) 是 vLLM 的加速手段：用小模型/启发式方法快速"猜"多个 token，再用大模型一次性验证。

### 3.1 核心思想

```
传统 autoregressive:
  每步生成 1 个 token，需要 N 步才能生成 N 个 token

投机解码:
  1. Draft 模型快速生成 K 个候选 token（成本低）
  2. Target 模型一次 forward 验证全部 K 个 token
  3. 接受正确前缀，拒绝错误的，从拒绝点开始重采样
  4. 一次 forward 可能产出 1~K+1 个 token
```

### 3.2 Proposer 体系

```python
# vllm/v1/spec_decode/llm_base_proposer.py
class SpecDecodeBaseProposer:
    def propose(self, ...):
        """生成 draft tokens"""
        # 使用 draft 模型或其他方法生成候选 token
```

| Proposer | 方法 | 特点 |
|----------|------|------|
| `EagleProposer` | EAGLE/EAGLE-2/3 | 利用 hidden states 的轻量级 draft head |
| `DraftModelProposer` | 独立小模型 | 标准的 draft model 方案 |
| `DFlashProposer` | DFlash | Qwen3 优化的投机方案 |
| `NgramProposerGPU` | N-gram 匹配 | 无需额外模型，基于上下文历史 |
| `MedusaProposer` | Medusa heads | 多个并行 draft head |
| `Gemma4Proposer` | Gemma4 MTP | 多 token 预测 |
| `SuffixDecodingProposer` | 后缀匹配 | 在已生成文本中找可复用序列 |

### 3.3 验证与接受

```python
# vllm/v1/sample/rejection_sampler.py
# 对每个 draft token：
#   - 如果 target 模型对该 token 的概率 >= draft 概率：直接接受
#   - 否则：按概率比率随机接受或拒绝
#   - 拒绝后：从修正后的分布中重新采样一个 token
```

### 3.4 在调度层面的协作

```
EngineCore.step():
  1. scheduler.schedule() → 给每个请求分配 draft token 信息
  2. executor.execute_model() → ModelRunner 同时验证 draft tokens
  3. executor.take_draft_token_ids() → 获取新的 draft tokens 给下一步
  4. scheduler.update_draft_token_ids() → 更新调度状态
```

---

## 4. 多模态处理

多模态模块负责将非文本输入（图像、音频、视频）转化为模型能理解的 token 或嵌入。

### 4.1 模块结构

```
multimodal/
├── registry.py    ← 多模态处理器的注册中心
├── inputs.py      ← 多模态输入的数据结构
├── processing.py  ← 基础处理器接口
├── image.py       ← 图像处理
├── audio.py       ← 音频处理
├── video.py       ← 视频处理
└── cache.py       ← 编码结果缓存
```

### 4.2 处理流程

```
用户请求（含图像 URL/base64）
│
├── InputProcessor (engine 层)
│   ├── 下载/解码多模态数据
│   ├── MULTIMODAL_REGISTRY.create_processor(model_config)
│   └── 生成 placeholder tokens + 原始多模态数据
│
├── ModelRunner (执行层)
│   ├── Vision Encoder forward（编码图像 → 嵌入向量）
│   ├── 结果缓存（避免重复编码同一图像）
│   └── 将嵌入插入到对应 placeholder 位置
│
└── 模型 forward（混合 text embedding + multimodal embedding）
```

### 4.3 注册机制

每个多模态模型通过装饰器注册自己的处理器：

```python
@MULTIMODAL_REGISTRY.register_processor(MyMultiModalProcessor)
class MyVisionLanguageModel(nn.Module, SupportsMultiModal):
    ...
```

注册表统一管理所有模型的多模态处理逻辑，使引擎层不需要关心具体模型的多模态实现细节。

---

## 5. LoRA 适配

LoRA (Low-Rank Adaptation) 允许在不修改基础模型权重的情况下，通过加载小型适配器来定制模型行为。

### 5.1 架构

```
lora/
├── layers/           ← LoRA 化的 Linear/Embedding 层
│   ├── column_parallel_linear.py
│   ├── row_parallel_linear.py
│   └── vocal_parallel_embedding.py
├── lora_model.py     ← LoRA 模型包装器
├── model_manager.py  ← 多 LoRA 适配器管理
├── worker_manager.py ← Worker 级别的 LoRA 管理
├── punica_wrapper/   ← Punica CUDA kernel（批量 LoRA 计算）
└── ops/              ← LoRA 计算算子（Triton/Torch/XPU）
```

### 5.2 核心机制

```python
# 基础 Linear forward:
output = W @ x

# LoRA Linear forward:
output = W @ x + (B @ A) @ x  # A: 降维, B: 升维, rank << hidden_size
```

vLLM 的 LoRA 支持**batch 内混合多个 LoRA 适配器**：

```
Request 1: 使用 LoRA-A
Request 2: 使用 LoRA-B
Request 3: 无 LoRA (基础模型)
Request 4: 使用 LoRA-A

→ 一个 batch forward 中，不同请求使用不同的 LoRA 权重
→ Punica kernel 高效处理这种 batched LoRA 计算
```

### 5.3 动态加载

```python
# LoRA 可以在运行时动态加载/卸载
executor.collective_rpc("add_lora", args=(lora_request,))
executor.collective_rpc("remove_lora", args=(lora_id,))
```

ModelManager 维护一个 LoRA 适配器的 LRU 缓存，热门适配器常驻 GPU，冷门的按需加载。

---

## 6. 量化系统

量化模块让 vLLM 能用更低精度（INT4/INT8/FP8）运行模型，减少显存占用并提升吞吐。

### 6.1 支持的量化方法

| 方法 | 精度 | 特点 |
|------|------|------|
| GPTQ | INT4/INT8 | Post-training，广泛支持 |
| AWQ | INT4 | Activation-aware，通常质量更好 |
| FP8 | E4M3/E5M2 | NVIDIA Hopper+ 硬件原生支持 |
| BitsAndBytes | NF4/INT8 | 简单易用，显存效率高 |
| CompressedTensors | 多种 | 统一格式 |
| NVFP4 | FP4 | NVIDIA 最新格式 |

### 6.2 透明注入机制

```python
# 量化对模型定义完全透明
class LinearBase:
    def __init__(self, ..., quant_config=None):
        # quant_config 决定使用哪种 quant_method
        self.quant_method = quant_config.get_quant_method(self)

    def forward(self, x):
        # quant_method 替换了实际的 gemm 实现
        return self.quant_method.apply(self, x)
```

### 6.3 KV Cache 量化

除了权重量化，vLLM 还支持 KV cache 的量化：

```python
# cache_config.cache_dtype = "fp8"
# → Attention 层在写入 KV cache 时自动量化
# → 读取时自动反量化
# → 显存节省约 50%（FP8 vs BF16）
```

---

## 7. 各模块的协作关系

```
                          ┌─────────────────┐
                          │   EngineCore    │
                          └────────┬────────┘
                                   │
               ┌───────────────────┼───────────────────┐
               │                   │                   │
        ┌──────▼──────┐     ┌─────▼─────┐     ┌──────▼──────┐
        │  Scheduler  │     │  Executor │     │  SpecDecode │
        │ (调度请求)   │     │ (编排GPU)  │     │ (投机提议)  │
        └──────┬──────┘     └─────┬─────┘     └──────┬──────┘
               │                   │                   │
               │            ┌──────▼──────┐            │
               │            │   Worker    │            │
               │            │ (GPU 管理)   │            │
               │            └──────┬──────┘            │
               │                   │                   │
               └───────────┬───────▼───────────────────┘
                           │
                    ┌──────▼──────┐
                    │ ModelRunner │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
 ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
 │    Model    │   │   Sampler   │   │ Multimodal  │
 │ (模型层)    │   │ (采样系统)   │   │ (多模态)    │
 └──────┬──────┘   └─────────────┘   └─────────────┘
        │
 ┌──────┼──────────────────┐
 │      │                  │
 │  ┌───▼───┐   ┌────────▼────────┐
 │  │ LoRA  │   │   Distributed   │
 │  │(适配器)│   │  (TP all-reduce) │
 │  └───────┘   └─────────────────┘
 │
 ┌▼───────────┐
 │Quantization│
 │(低精度计算) │
 └────────────┘
```

### 各模块的触发时机

| 模块 | 何时工作 | 工作频率 |
|------|----------|----------|
| 采样 | 每步 forward 之后 | 每步 |
| 分布式 | 模型 forward 内部（all-reduce），PP send/recv | 每步多次 |
| 投机解码 | execute_model 内部，与 forward 交织 | 每步 |
| 多模态 | 新请求到达时编码，结果缓存复用 | 按需 |
| LoRA | forward 时混入适配器计算 | 有 LoRA 请求时 |
| 量化 | 每次 gemm 计算时 | 每步多次 |

---

## 8. 关键设计决策总结

| 模块 | 设计选择 | 原因 |
|------|----------|------|
| 采样 | 全部在 GPU 上完成 | 避免 CPU-GPU 数据搬运 |
| 分布式 | 单独的 GroupCoordinator 抽象 | 隔离通信后端差异 |
| 投机解码 | Proposer 可插拔 | 不同模型适合不同投机策略 |
| 多模态 | 编码结果缓存 + 异步编码 | 避免重复编码同一输入 |
| LoRA | Punica batched kernel | 支持 batch 内混合多个适配器 |
| 量化 | 在 Linear 层抽象中透明替换 | 模型代码零修改 |

---

## 9. 后续深入方向

- 采样中 structured output (grammar-guided decoding) 的实现
- 分布式中 Expert Parallel 与 EPLB（Expert Parallel Load Balancing）的动态调度
- 投机解码中 EAGLE 如何利用 hidden states 做高质量 draft
- P/D 分离（Prefill-Decode Disaggregation）中的 KV Transfer 机制
- `torch.compile` 如何与量化 kernel 协作
