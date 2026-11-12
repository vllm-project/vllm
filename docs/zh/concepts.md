# 关键概念

本页把阅读 vLLM 源码时常遇到的概念串成一张地图。每个主题回答三个问题：它是什么、为什么需要、在 vLLM 中大致如何工作。参数默认值和硬件/模型支持范围可能随版本变化，本文用于建立心智模型，精确行为仍应核对对应配置和实现。

## V1 的统一执行方式

V1 的主要变化是重新组织请求调度、KV Cache 管理、Worker、采样和 API 执行路径，而不是重写所有模型和 kernel。这样连续批处理、分块 prefill 和前缀缓存可以共用一套请求状态与执行循环，同时把模型计算和设备算子保留在相对独立的层中。

```text
1. 客户端请求 -> API / Python 入口：规范化输入
2. Engine Core：保存请求状态并询问 Scheduler
3. Scheduler：选定本 step 的 token 与 KV Cache slots
4. Executor / Worker / Model Runner：执行模型
5. Engine Core：接收结果、更新状态、决定继续或完成
6. API / Python 入口：返回结果
```

例如 A、B 两个请求正在 decode，新请求 C 有 10 个 prompt token；假设本例每 step 的 token budget 为 6，Scheduler 可以先给 A、B 各安排 1 个 decode token，再把 C 的 prompt 分成 chunk：

```text
Step 1: A decode(1) + B decode(1) + C prefill(4/10) = 6 tokens
Step 2: A decode(1) + B decode(1) + C prefill(4/10) = 6 tokens
Step 3: A decode(1) + B decode(1) + C prefill(2/10) = 4 tokens
```

这是帮助理解 token budget 的简化例子，不代表所有策略下每一步都按这个顺序分配。重点是 prefill 和 decode 是同一个调度循环里的工作；Scheduler 每步都会结合运行请求、等待请求、KV Cache 空间和配置重新决定执行计划。

## 推理阶段与批处理

- **Prefill（提示词计算）是什么**：处理请求已有的 prompt token，计算模型状态并建立 KV Cache。长 prompt 会一次带来许多 token，通常需要较多计算。
- **Decode（逐步生成）是什么**：模型基于已有上下文预测后续 token，并把新 token 的 KV 状态追加到 cache。普通 decode 每步通常生成一个 token；speculative decoding 等路径可能一次验证多个 token。
- **Continuous Batching（连续批处理）为什么需要**：静态 batch 要等其中最慢的请求结束，短请求结束后的空位不能及时补上。连续批处理允许请求在运行期间加入或完成，Scheduler 每个 step 都可以重新组合工作。
- **Chunked Prefill（分块预填充）如何缓解延迟**：长 prompt 若一次占满执行预算，会让已在生成的请求等待。Scheduler 可先安排待 decode 工作，再用剩余 token budget 处理 prompt；放不下的 prompt 留到后续 step 继续计算。

```text
静态批处理（固定请求组）
请求 A: running ------------------------------> done
请求 B: running ------> done; 等待 A 结束期间有空位

连续批处理（每个 step 重新组合）
Step 1: A decode + B decode + C prefill chunk
Step 2: A decode + C prefill chunk + 新请求 D
Step 3: A done + C decode + D prefill
```

图中的每一行表示一个 step 中组合执行的工作，不代表固定 batch slot。`max_num_batched_tokens` 限制本 step 可计算的 token 总量：调大通常让更多 prefill 一起进入，可能改善 TTFT 或吞吐；调小可减少长 prefill 对 decode 的干扰，可能改善 ITL。实际取舍依赖请求长度、模型和硬件。FCFS/priority 等策略影响请求选择；KV Cache 不足时可能抢占请求，之后再恢复或重算。详见 [V1 Scheduler](../design/arch_overview.md#scheduler)、[V1 指南](../usage/v1_guide.md)和[优化配置](../configuration/optimization.md#chunked-prefill)。

## KV Cache、PagedAttention 与前缀缓存

### KV block 与 Block Table

**是什么**：KV Cache 保存注意力层对已处理 token 计算出的 Key/Value。新 token 计算时可以读取已有 KV，而不必重新计算整个历史上下文。

**为什么按 block 管理**：请求长度不同且会持续增长。如果每个请求都预留一段连续显存，空间容易浪费，也难以灵活扩展。vLLM 把 KV 切成固定大小的物理 block，再用 Block Table 将请求的逻辑 token 区间映射到物理 block。

```text
请求 token 范围 -> 逻辑 block
token 0..3 -> logical block 0
token 4..7 -> logical block 1
token 8..   -> logical block 2（需要时再分配）

Block Table -> 物理 GPU block
logical block 0 -> GPU block 7
logical block 1 -> GPU block 2
logical block 2 -> 尚未分配
```

比如一个请求先有 8 个 token，只需分配两个 block；继续生成时再按需申请下一个 block。物理 block 不必相邻，因此内存分配和回收更灵活。PagedAttention 常用来指代这种分页式 KV 管理及对应的注意力访问思路；当前 vLLM 有多种 attention backend 和 cache layout，[PagedAttention 设计文档](../design/paged_attention.md)描述的是历史设计，不应把其中旧 kernel 当作所有当前后端的实现。

### Automatic Prefix Caching（APC）

**是什么**：新请求含有之前已计算过的相同前缀时，复用该前缀对应的 KV block，跳过重复 prefill。

**如何识别相同前缀**：V1 对完整 block 建立链式哈希。每个 block 的身份包括父 block 哈希、本 block token，以及 LoRA、多模态输入哈希或 cache salt 等附加信息。只缓存完整 block；部分填充的尾块不能直接作为完整前缀命中。

```text
Block size = 4

请求 A block 0: [The sky is blue] -> hash H0
请求 A block 1: [today ...]      -> hash H1
请求 B block 0: [The sky is blue] -> hash H0 -> HIT, 复用 KV
请求 B block 1: [cloudy ...]      -> hash 与 H1 不同 -> MISS
结果: B 复用 block 0，只需 prefill 后续不同部分
```

**为什么有用**：它主要降低共享前缀的 prefill 成本，不直接加速新 token 的 decode；没有重复前缀就没有可复用项。多模态占位 token 一样不代表图片内容一样，因此媒体输入哈希也参与身份判定；多租户需要隔离复用范围时可以使用 cache salt。请求释放引用后，block 进入可复用/回收队列；内存压力下，缓存映射按 LRU 规则淘汰。详见[前缀缓存设计](../design/prefix_caching.md)和[用户指南](../features/automatic_prefix_caching.md)。

## 并行维度

并行策略回答的是“切分模型的哪一部分，或把请求分给哪一份模型副本”。它们切分对象不同，下面用简图分别说明。图里的 rank 表示分布式执行 rank，不一定等于某个固定 GPU 编号。

### TP：切分同一层

**为什么**：模型权重或单层计算量超过一张卡的承载能力时，把层内矩阵/张量切到多个 rank。

```text
同一请求、同一层的输入
input -> TP rank 0 使用权重分片 0 -> partial result 0
input -> TP rank 1 使用权重分片 1 -> partial result 1
partial results -> collective -> layer output
```

例如 `TP=2` 时两个 rank 协同完成同一次线性层/attention 计算；每个 rank 只持有相应权重分片，部分计算后需要 collective 汇总或交换结果。TP 能降低单卡权重占用，但会增加 rank 间通信。

### PP：切分不同层

**为什么**：模型层数或总权重规模需要更多设备，或者希望按节点/设备划分模型阶段。

```text
hidden states -> PP stage 0（layers 0..15）
PP stage 0 -> stage 间传递 activations
activations -> PP stage 1（layers 16..31）
PP stage 1 -> logits
```

例如 `PP=2` 时，前半模型在 stage 0，后半在 stage 1。PP 切的是层，不是同一层里的矩阵；stage 之间需要传递 activations。多个请求/微批次可用于填充流水线，但效果取决于模型和调度。

### DP：复制模型执行组，分请求

**为什么**：单个模型副本已经能放入一组 GPU，想让多组设备同时处理不同请求以提高总吞吐。

```text
incoming requests -> router -> DP rank 0 -> model replica 0 -> requests A, C
incoming requests -> router -> DP rank 1 -> model replica 1 -> requests B, D
```

DP rank 是数据并行组中一个模型副本/执行组的编号，不是 GPU 编号。`DP=2, TP=2` 通常表示两份模型副本，每份各由两个 TP rank 组成，总共四个 GPU rank；路由/负载均衡把不同请求送到不同 DP rank。

### EP 与 PCP：MoE 专家和长上下文

- **EP（Expert Parallel）**：MoE 中不同 expert 放在不同 rank。Router 按 token 选择专家，执行时把 token 派发到拥有对应 expert 的 rank，再汇集结果。适合专家参数需要分布到多设备的模型；通信和负载均衡是重要成本。

```text
tokens -> router
router -> expert rank 0 -> partial output 0
router -> expert rank 1 -> partial output 1
partial outputs -> combine -> next layer
```

- **PCP（Prefill Context Parallel）**：针对长 prompt，把上下文维度的 prefill 工作分配到多个 rank，以缩短单个长上下文的处理时间；通常需要按模型/attention 实现交换或归并中间结果。并非所有模型和后端都支持。

```text
长 prompt tokens 0..N -> 按上下文切分
context shard 0 -> PCP rank 0 计算
context shard 1 -> PCP rank 1 计算
各 rank 的 attention 信息 -> 按后端算法交换/归并
归并结果 -> 完成 prefill
```

TP、PP、DP、EP、PCP 可以在受支持的配置中组合，但组合规则由模型、硬件、网络和 executor 决定。部署选择见[并行与扩展指南](../serving/parallelism_scaling.md)；MoE 部署见[数据并行部署文档](../serving/data_parallel_deployment.md)。

## 编译与图执行

这几个名字容易混在一起。先记住它们处理的是不同问题：`torch.compile` 尝试优化“算子如何执行”；CUDA/HIP Graph 复用“已经准备好的设备操作序列”；Attention Backend 则决定“attention 本身用哪套实现”。

### `torch.compile`：优化计算图

**是什么**：PyTorch 的编译入口。运行模型 forward 时，Dynamo 观察张量运算并捕获为 FX graph；Inductor 再分析图、融合可融合的运算，并生成 Triton/C++ 等执行代码，或调用适合的底层库。vLLM 也会通过 custom op 把部分复杂操作（例如 attention）作为明确边界，避免编译器追踪其内部实现。

**为什么需要**：eager 模式下，Python 逐个执行 PyTorch 运算并提交设备工作。Transformer layer 包含大量小型 elementwise 运算、归一化和线性变换；CPU 调度开销和多个小 kernel 的 launch 成本可能很显眼。编译器可以减少 Python 调度、融合部分相邻运算，并针对已知形状选择更合适的实现。大矩阵乘法本身通常仍由优化过的库或专用 kernel 承担，不是所有运算都会融合成一个 kernel。

**例子**：假设某层要做 bias add 后接 GELU：

```text
Eager:
Python -> 提交 bias add kernel -> 再提交 GELU kernel
GPU 异步执行已提交的 kernel（通常不要求每步同步等待）

torch.compile:
Python forward -> Dynamo 捕获 FX graph
FX graph -> Inductor 检查可融合部分
生成代码 -> bias add + GELU 可能融合为一个 kernel
后续请求 -> 调用已编译代码，而不是重新解释这段 Python
```

这里“可能融合”取决于图边界、dtype、shape 和后端，不保证任何特定模型都恰好减少为一个 kernel。关键子概念如下：

- **Dynamo / FX graph**：Dynamo 捕获 Python 张量运算，FX graph 是编译器可分析的中间表示。遇到 Python 动态分支或不支持的操作时，可能产生 graph break，或由 custom op 将其包成不展开的边界。
- **Inductor**：优化图并生成或选择执行代码；形状已知时还可能进行针对形状的 autotuning。
- **动态形状**：请求的 token 数和 batch 会变化。编译器需要用符号形状处理可变尺寸，或者为常见尺寸生成特化版本；特化越多，可能提升对应形状的速度，也会增加编译时间和缓存项。
- **编译缓存**：保存编译产物，后续相同配置和代码可以复用。冷启动仍需编译；vLLM 会在开始服务前完成需要的编译，避免请求运行中突然遇到首次编译延迟。

因此，`torch.compile` 的代价主要体现在启动编译时间、缓存空间和图边界兼容性上；收益取决于模型、shape、算子和设备，不等于“总是把整个模型变成一个 kernel”。详见[`torch.compile` 设计文档](../design/torch_compile.md)。

### CUDA/HIP Graph：重放设备操作序列

**是什么**：CUDA Graph（ROCm 上有对应的 HIP Graph）把一段设备操作的依赖关系在 capture 阶段记录下来，之后使用预先准备的 buffer 重放这段操作序列。它不负责改写 attention 数学公式，也不等同于编译模型代码。

**为什么需要**：decode 每一步常只处理少量 token，但仍需运行 embedding、多个 Transformer layer、attention、采样相关计算等许多设备操作。若每一步都由 CPU 逐个提交这些短 kernel，CPU 发射开销可能占总延迟的较大部分。capture 一次、重复 replay，可以减少每一步重复提交 launch 的工作。

**例子**：同一批请求连续生成多个 token，模型结构和 batch 形状没有变化：

```text
普通执行（每个 decode step）:
CPU -> launch op 1 -> launch op 2 -> ... -> launch op N
CPU -> launch op 1 -> launch op 2 -> ... -> launch op N

Graph 执行:
预热/capture 一次: op 1 -> op 2 -> ... -> op N
每个 decode step:  更新输入 buffer -> replay 整张图
```

图重放对输入/输出 buffer 地址、张量形状和执行路径有要求；形状变化、attention backend 不支持或执行路径含不兼容操作时，不能强行复用同一张图。vLLM 的 dispatcher 会依据 batch 描述和 backend 能力选择：

```text
运行时 batch -> dispatcher
dispatcher -> FULL: 适用时重放完整模型图
dispatcher -> PIECEWISE: 重放可捕获片段，不兼容部分照常执行
dispatcher -> NONE: 没有可用图，走 eager 路径
```

捕获更多尺寸/路径会增加启动时间和显存占用，因此 Graph 不是“免费加速”，也不是每个模型、backend、batch 都能命中。CUDA Graph 与 `torch.compile` 可以组合：编译器优化可编译的计算，Graph 减少重复提交设备操作；也可以在某些配置中只启用其中一类。模式和兼容性详见[CUDA Graphs 设计](../design/cuda_graphs.md)。

### Attention Backend：attention 的计算实现

**是什么**：Attention Backend 是执行 attention、读取/写入 KV Cache 的具体实现及其 metadata/layout 约定。模型定义 attention 的逻辑；backend 负责把它映射到匹配硬件和输入形状的 kernel/库调用。

**为什么有多个**：attention 的工作负载差异很大：prefill 一次处理很多 query token，decode 通常处理少量 query 并读取很长的 KV；MHA/GQA 与 MLA 的数据布局不同；dtype、head size、block size、稀疏模式、GPU 架构和 Graph 能力也会改变可用实现。单个 backend 很难在所有组合下都正确、高效并支持同样的功能。

**常见家族示例**（不是完整列表，也不是固定推荐顺序）：

| 家族 | 常见名称 | 大致用途 |
| --- | --- | --- |
| 标准 MHA/MQA/GQA | `FLASH_ATTN`（FA2/FA3/FA4）、`FLASHINFER`、`TRITON_ATTN` | 常见 decoder attention；具体版本和路径依硬件/形状而定 |
| 组合/路由 | `TRITON_FLASH_ATTN`、`TRITON_FLASHINFER` | 兼容的多模态 prefix 等场景可按 query 类型在子 backend 间路由 |
| ROCm | `ROCM_AITER_FA`、`ROCM_AITER_UNIFIED_ATTN`、`ROCM_ATTN` | AMD GPU 上由 AITER 或 ROCm 实现提供的路径 |
| MLA | `FLASH_ATTN_MLA`、`FLASHINFER_MLA`、`TRITON_MLA`、`FLASHMLA`、`CUTLASS_MLA` | DeepSeek-style Multi-head Latent Attention；prefill 与 decode 可能分别选择 backend |
| 稀疏/模型专用 | `FLASHMLA_SPARSE`、`FLASHINFER_MLA_SPARSE`、MiniMax/DeepSeek 专用 backend | 稀疏 MLA、block-sparse attention 或特定模型结构 |

**例子**：标准 GQA decoder 可能在 FlashAttention、FlashInfer 或 Triton 路径中选择；DeepSeek-style MLA 则进入 MLA 专用实现，且 prefill/decode 可以使用不同 backend。兼容的多模态 prefix 场景还可能由组合 backend 在双向图片 attention 与因果文本 attention 之间路由。

**如何选择**：未指定时，vLLM 按平台和 backend 优先级逐项校验模型 dtype、head size、KV dtype/layout、block size、计算能力及所需功能，使用第一个兼容实现；显式指定但不兼容时会报错及原因。所有名称与实时兼容矩阵请查[Attention Backend Feature Support](../design/attention_backends.md)，不要仅凭名称推断“哪个一定最快”。

### 视觉 Encoder CUDA Graph

视觉 Encoder 也可独立捕获 CUDA Graph，与语言模型 Decoder 的图执行路径正交。它解决的是 Encoder forward 的 CPU kernel launch 开销，不改变图片预处理或模型语义。

图片/视频可能产生不同数量的 patch/frame token，无法只捕获一种固定输入尺寸。vLLM 可按 token budget 捕获多个图；运行时把输入装入最小可容纳预算，超预算或不满足形状约束时对相应路径 eager 执行：

```text
启动：捕获预算图 [512, 1024, 2048]
请求：图片 A -> 400 tokens -> 选择 512 图并 replay
请求：图片 B -> 900 tokens -> 选择 1024 图并 replay
请求：图片 C -> 2300 tokens -> 超预算，eager fallback
```

模型 protocol 提供 capture/replay 输入准备和 Encoder forward；通用 manager 负责预算、分批、capture/replay 与 fallback。详见[多模态 CUDA Graph 文档](../design/cuda_graphs_multimodal.md)。

## 多模态数据

**是什么**：多模态请求把文本与图片、视频、音频等媒体组合输入。HTTP 中的图片 URL/视频内容只是外部表示，不是模型直接消费的 GPU 张量；模型需要特定尺寸、裁剪、帧采样、音频切片和占位位置。

**为什么需要专门处理**：Tokenizer 负责文本，HF Processor 或 vLLM 模型逻辑负责媒体张量与 prompt 占位 token。Processor 可能把一个 `<image>` 展开成多个 feature placeholder；vLLM 必须保存占位位置和媒体输入/Encoder 输出的对应关系，才能正确合并 embeddings，并支持缓存和调度。

**如何工作**：以“这张图片里有什么？”为例，下面展示的是概念流程；实际 processor 可能在 Renderer 前后分阶段运行，具体由模型实现决定。

```text
输入: 文本“这张图片里有什么？” + image.jpg
媒体路径: 加载 / 解码 / resize / Processor -> pixel_values
文本路径: Renderer / Tokenizer -> token IDs + <image> 占位位置
对齐: Processor / 模型记录占位 token 与图片特征的对应关系
Encoder: pixel_values -> Vision Encoder -> image embeddings
合并: 文本 embeddings + image embeddings -> 语言模型输入
生成: Decoder -> sampler -> 文本回答
```

如果输入是视频，预处理还需采样帧；音频则会有音频解码/特征处理路径。相同媒体输入的 Processor 结果可以缓存；部分模型可把归一化/缩放融合到设备侧，减少 CPU 计算和 PCIe 传输。详见[多模态数据处理设计](../design/mm_processing.md)，再跟进具体模型的 Processor 和 `embed_multimodal` 路径。

## 继续定位

- 调度与请求状态：`vllm/v1/core/sched/`
- KV Cache 管理：`vllm/v1/core/` 中的 KV cache 相关模块
- 编译和 CUDA Graph：`vllm/compilation/`、`vllm/v1/cudagraph_dispatcher.py`
- 多模态输入与 processor：`vllm/multimodal/`、`vllm/model_executor/models/`