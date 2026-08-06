# 并行 Linear 层中的 Sequence Parallel 通信

本文档梳理 vLLM 在提交 `72cd5424d` 上的 Sequence Parallel（SP）实现，
并说明将 SP 通信收敛到并行 Linear 层这一重构的范围和边界。

## 现有实现

vLLM 当前有两套相互独立、都被称为 Sequence Parallel 的功能：

- `vllm/compilation/passes/fusion/sequence_parallelism.py` 中的编译 Pass
  会把 all-reduce 和 RMSNorm 模式改写成 reduce-scatter、本地归一化和
  all-gather。该实现与模型无关，但要求使用 full-graph 编译。
- MoE Sequence Parallel 由 `ParallelConfig.use_sequence_parallel_moe`
  控制。模型会沿 token 维度切分输入，使 tensor-parallel rank 同时作为
  sequence rank 执行 expert 计算。这条路径的大部分通信逻辑原本直接写在
  各个模型文件中。

模型侧的实现可以分为以下几类：

| 实现类型 | 模型系列 | 模型侧通信方式 |
| --- | --- | --- |
| 旧版 MoE Block | AXK1、GPT-OSS、GraniteMoE、InternS1 Pro、Llama 4、MiMo V2、Nemotron-H、OpenPangu、Qwen3-MoE | 进入 MoE 前调用 `sequence_parallel_chunk`，MoE 之后执行 all-gather |
| Attention 到 MoE 的衔接路径 | DeepSeek V2、Qwen3-Next、通过 Qwen3-Next 复用实现的 Qwen3.5、DeepSeek V3.2 | Attention 前执行 all-gather，row-parallel 输出后执行 reduce-scatter |
| Transformers backend | 通用 Transformers MoE fuser | 在 fuser 内调用 `sequence_parallel_chunk` 和 all-gather |
| 新模型实现 | Kimi K3 | 在 Attention 和切分权重的 dense MLP 前后执行 all-gather/reduce-scatter；模型边界和 MTP 边界还有额外 gather |
| 新模型实现 | DeepSeek V4 | 在 Attention 前后执行 all-gather/reduce-scatter；模型、dSPARK 和 MTP 边界还有额外 gather |

上述清单不包含那些不属于 token 维度 SP 的 collective，例如：

- vocabulary 或 logits all-gather；
- 视觉模型中的 Q/K all-gather；
- pipeline rank 之间的传输；
- Inkling 沿 hidden 维度执行的 reduce-scatter/all-gather。

模型边界的 gather 也不属于 parallel-linear 通信。即使完成 Linear 层迁移，
最终 hidden states、辅助 hidden states、MTP 输入输出以及 pipeline 边界仍可能
需要由模型显式编排。

## 统一入口

SP collective 统一放在 `vllm.distributed.communication_op` 中：

- `sequence_parallel_all_gather` 沿第 0 维收集 token shard，不改变收集后的
  shape；
- `sequence_parallel_reduce_scatter` 对各 rank 的部分结果求和，再沿第 0 维
  重新分发 token shard，不执行 padding；
- 两个操作都会优先使用 device communicator 提供的自定义 SP collective；
  自定义实现不可用时，再回退到普通 TP collective。

`vllm.models.common.ops.sequence_parallel` 暂时保留为兼容层，使现有 Kimi K3
和 DeepSeek V4 调用点可以继续工作。

并行 Linear 层统一暴露 `sequence_parallel` 开关：

```text
本地 token shard
  -> ColumnParallelLinear.prepare_input() -> all-gather
  -> column-parallel 计算
  -> row-parallel 计算
  -> RowParallelLinear.reduce_output() -> reduce-scatter
  -> 本地 token shard
```

该开关默认为 `False`，因此现有 Linear 层的 TP 行为保持不变。

开关启用后：

- ColumnParallelLinear 会在 quantization method 执行前收集输入 token shard；
- RowParallelLinear 会对部分输出执行 reduce-scatter，不再执行 all-reduce；
- MergedColumnParallelLinear 和 QKVParallelLinear 也提供相同的构造参数。

统一通信入口不会执行 padding、unpadding 或其他 token 数调整。调用方必须
提供符合 TP collective 要求的 token 维度，包括满足 reduce-scatter 的
整除条件。

LoRA 的 column 和 row wrapper 分别调用基础 Linear 层的 `prepare_input` 和
`reduce_output`。这样无论执行基础权重还是 LoRA 权重，通信策略都由并行
Linear 层统一管理。

## 模型迁移边界

当前改动只建立公共实现入口，尚未让任何模型启用新的 Linear SP 开关。
后续迁移每个模型时应完成以下工作：

1. 在第一个接收 token shard 的 column-parallel projection 上启用 SP；
2. 在将部分结果重新转换成 token shard 的 row-parallel projection 上启用 SP；
3. 删除对应的模型侧 all-gather、reduce-scatter，以及
   `reduce_results=False` 临时绕行逻辑；需要的布局校验应保留在 Linear
   通信入口之外；
4. 保留并单独审查那些不属于 column/row Linear 配对的模型边界 gather。
