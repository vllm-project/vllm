# `model.load_weights()` 直接 Reload 可行性实验

## 1. 问题定义

本文只回答一个问题：

> 模型完成首次加载和 PWAL、已经进入 serving 状态后，是否可以只调用一次
> `model.load_weights(checkpoint_weights)` 完成权重更新？

这里的“直接调用”严格指：

```python
model.load_weights(checkpoint_weights)
```

并且**不执行**：

- checkpoint schema 恢复；
- layerwise/modelwise reload prepare；
- `process_weights_after_loading`（PWAL）；
- runtime storage copy-back；
- derived state 重建。

这个定义非常重要。此前名称为 `direct_reload_smoke.py` 的脚本实际调用的是：

```python
self.model_runner.reload_weights(weights_path=path)
```

它走了完整 reload 流程，并不是直接调用运行态模型的
`model.load_weights()`。因此不能把该脚本的成功结果当作“直接 load_weights
成功”的证据。

## 2. 总结

| 模型运行态 | 只调用运行态 `model.load_weights()` | 结论 |
|---|---:|---|
| 普通 BF16 dense，运行态 schema 未被改变 | 理论上可以 | 尚无独立端到端实验结果 |
| 普通 BF16 MoE，运行态 expert schema 未被改变 | 理论上可以 | 尚无独立端到端实验结果 |
| BF16 MLA | 不完整 | checkpoint 参数会更新，但 MLA 派生权重不会自动重建 |
| 在线 FP8/INT8/MXFP8 | 不可以 | runtime 权重已经量化，shape/dtype/loader 语义不是 BF16 checkpoint schema |
| GPTQ/AWQ/Marlin 等需要 repack 的 checkpoint 量化 | 通常不可以 | 首次 PWAL 后参数已转换为 kernel layout |
| compressed-tensors 等带 packed/scale 派生状态的量化 | 通常不可以 | 运行态 schema 与 checkpoint schema 不同，且需要重新 PWAL |
| 已经是最终 runtime/kernel format 的权重 | 可以原位更新 | 应走按 runtime tensor 名称的 `copy_`，不是 checkpoint `model.load_weights()` |

截至 2026-08-07，已有端到端实验真正证明的是：

> 恢复 checkpoint schema 后，可以使用一次或多次 `model.load_weights()` 加载
> checkpoint 权重，再统一执行 PWAL 并提交回原 runtime storage。

实验没有证明所有模型都能对**已经完成 PWAL 的运行态模型**只调用
`model.load_weights()`。

## 3. 判断标准

直接 reload 是否成立，不应简单按“量化/非量化”划分，而应检查下面三个条件。

### 3.1 当前 Parameter/Buffer 是否仍是 checkpoint schema

`model.load_weights()` 的职责是：

- checkpoint 名称映射；
- TP/EP shard；
- QKV 和 gate/up fusion；
- 调用 parameter 的 `weight_loader`；
- 将输入权重写入模型当前暴露的加载目标。

如果首次加载和 PWAL 没有改变 Parameter/Buffer 的 shape、dtype、layout 和 loader
语义，那么直接再次调用 `model.load_weights()` 有可能成立。

如果 PWAL 已经把 BF16 checkpoint 参数替换为 FP8、INT8、INT4、Marlin packed
权重或 scale，那么当前模型不再是 checkpoint schema，直接加载就不成立。

### 3.2 Forward 是否只读取 Parameter/Buffer

有些模型在 PWAL 中根据 checkpoint 参数生成普通 tensor attribute 或其他派生状态。
这些对象不一定出现在：

```python
model.named_parameters()
model.named_buffers()
model.state_dict()
```

再次调用 `model.load_weights()` 只会更新加载目标，不会自动刷新这些派生状态。

### 3.3 更新后是否还需要 PWAL

如果 forward 使用的权重需要量化、repack、transpose、scale 生成或模型级派生，
那么 `model.load_weights()` 只是“加载 checkpoint”的中间步骤，不是完整 reload。

## 4. 各类流程分析

### 4.1 普通 BF16 Dense

候选流程：

```text
BF16 checkpoint
    -> model.load_weights()
    -> BF16 fused/sharded Parameter
    -> forward
```

如果模型首次加载后：

- 参数仍为 BF16；
- 参数 shape 和 loader 没有被 PWAL 改变；
- forward 不依赖额外派生权重；

那么 `model.load_weights()` 可通过现有 weight loader 覆盖 QKV fusion、gate/up
fusion 和 TP shard 对应的目标切片，理论上不需要 quant PWAL。

**实验状态：未形成有效的独立直接调用实验结论。**

此前安排过 Qwen dense BF16 的直接验证，但任务在模型测试启动前被中断，没有生成
模型加载日志、pytest 结果或推理对比。因此目前应标为“候选支持”，而不是“已验证
通过”。

### 4.2 普通 BF16 MoE

候选流程：

```text
BF16 expert checkpoint tensors
    -> model.load_weights()
    -> fused expert Parameter slices
    -> forward
```

MoE 的 expert 名称映射、EP shard 和 fused expert slice 本来就由模型
`load_weights()`/expert weight loader 负责。如果首次加载后 expert Parameter schema
没有改变，直接调用理论上可以覆盖目标 slice。

但以下情况仍需要额外处理：

- expert kernel 在 PWAL 中生成新的 packed layout；
- scale、transpose 或 workspace tensor 由首次加载派生；
- forward 使用不在 named parameters/buffers 中的派生 tensor。

**实验状态：未形成有效的独立直接调用实验结论。**

此前 MoE 直接验证尚未启动，因此不能将完整 `reload_weights()` 的成功结果归因于
直接 `model.load_weights()`。

### 4.3 BF16 MLA

直接调用不完整。

MLA 在 `process_weights_after_loading()` 中会从 `kv_b_proj` 等权重派生 forward
使用的运行时 tensor，例如：

```text
W_UK_T
W_UV
```

某些实现还会生成复制、聚合或 backend 专用状态。这些状态可能只是普通 tensor
attribute，而不是 checkpoint Parameter/Buffer。

直接再次调用：

```python
model.load_weights(new_checkpoint)
```

可以更新 `kv_b_proj`，但不会自动重建已经存在的 `W_UK_T/W_UV`。forward 仍可能
读取旧派生权重。

因此 BF16 并不保证可以直接 reload；MLA 至少还需要模型级 post-load hook。

### 4.4 在线 FP8、INT8 和 MXFP8

直接调用不可以。

以 BF16 checkpoint 到在线 FP8 为例，首次加载是：

```text
BF16 checkpoint Parameter
    -> model.load_weights()
    -> FP8 PWAL
    -> FP8 weight + scale + kernel runtime layout
```

完成 PWAL 后，运行态模型暴露的是 FP8 权重及 scale，而 trainer 发送的是 BF16
checkpoint tensor。二者可能存在：

- dtype 不一致；
- shape 或 padding 不一致；
- Parameter 已被替换；
- 原 checkpoint weight loader 已被转换或解包；
- scale 和 packed tensor 尚未生成。

所以不能把 BF16 checkpoint 直接加载进当前 FP8 runtime Parameter。正确流程必须
先恢复 BF16 checkpoint schema，再调用 `model.load_weights()`，最后重新运行 PWAL。

在线 `fp8_per_tensor`、`fp8_per_block`、`fp8_per_channel`、
`int8_per_channel_weight_only` 和在线 `mxfp8` 都属于这一类。

### 4.5 GPTQ、AWQ、Marlin 和 Compressed Tensors

这些格式通常也不能保证对运行态模型直接调用 `model.load_weights()`。

原因不是它们都采用同一种量化方式，而是首次加载后可能执行：

```text
checkpoint qweight/scales/zeros
    -> shard/fuse
    -> repack/transpose/padding
    -> Marlin 或其他 kernel layout
    -> runtime Parameter/Buffer
```

再次输入的是 checkpoint-format tensor，而加载目标已经是 kernel-format tensor。
如果没有恢复 checkpoint schema并重新执行 PWAL/repack，就可能发生：

- shape/dtype 不匹配；
- loader 找不到原 checkpoint 目标；
- checkpoint 数据被写入错误 layout；
- weight 更新但 scale/zero/derived packed state 仍是旧值。

我们验证过 GPTQ、GPTQ-Marlin、compressed-tensors FP8/W4A16/W4A8 的完整
modelwise reload 能通过，但这证明的是“恢复 schema + load_weights + PWAL”有效，
不是“直接 load_weights”有效。

### 4.6 Runtime/Kernel Format 更新

如果发送侧提供的已经是 receiver 当前使用的最终 tensor：

```text
名称一致 + shape 一致 + dtype 一致 + layout 一致
```

则可以直接：

```python
runtime_tensor.copy_(received_tensor)
```

这种路径无需 checkpoint `model.load_weights()`，也无需重新量化。但发送侧必须准确
生成各 rank 的 TP/EP shard、fused layout、packed weight 和 scale，通用性较差。

## 5. 实际通过的实验是什么

已有实验使用的是下列流程，而不是裸 `model.load_weights()`：

```text
start
    -> 保存 runtime Parameter/Buffer 与 storage
    -> 恢复 checkpoint schema

update(chunk) × N
    -> model.load_weights(chunk)
    -> 不运行 PWAL
    -> 不使用 numel 判断完成

finish
    -> 全模型 PWAL
    -> 校验 runtime shape/dtype
    -> copy 回原 runtime storage
    -> 恢复原 binding
    -> 清空 prefix cache
```

以下量化流程通过了该完整 modelwise reload：

| 流程 | 完整 modelwise reload 结果 | 为什么不能据此声称裸 `load_weights()` 可用 |
|---|---:|---|
| 在线 FP8 per-tensor | Pass | 实验先恢复 BF16 schema，finish 时重新量化 |
| 在线 FP8 per-block | Pass | 同上 |
| 在线 FP8 per-channel | Pass | 同上 |
| 在线 INT8 weight-only | Pass | finish 时重新生成 INT8 runtime 权重 |
| 在线 MXFP8 | Pass | finish 时重新生成 runtime 格式 |
| compressed-tensors FP8 block | Pass | 重新执行 checkpoint load 和 PWAL |
| compressed-tensors W4A16 | Pass | 重新执行 packed/quant post-processing |
| compressed-tensors W4A8 MoE | Pass | 重新执行 expert quant post-processing |
| GPTQ/AutoGPTQ/GPTQ-Marlin | Pass | 恢复 checkpoint schema 后重新 repack |
| experts INT8 MoE | Pass | 恢复 BF16 expert schema后重新量化 |

在线 FP8 per-tensor 还通过了以下传输实验：

| 后端 | 分块方式 | 结果 |
|---|---|---:|
| CUDA IPC | 四次 partial update | Pass |
| NCCL | 四次 partial update | Pass |
| CUDA IPC | 384 MiB packed buffer，三个 chunks | Pass |
| NCCL | 一次 update API，内部三个 packed chunks | Pass |

四组实验都在 finish 前保持 BF16 checkpoint schema，只在 finish 执行一次 PWAL，
最终推理 token 与冷加载一致，且 runtime storage identity 不变。

## 6. 失败或受限的量化项

以下结果也不能解释为 reload 本身失败：

| 项目 | 状态 | 原因 |
|---|---|---|
| 较老 TinyLlama compressed-tensors W4A8 | 初始加载失败 | checkpoint 含未注册的 `weight_chan_scale` 目标 |
| Humming | 环境受限 | 初始 PWAL 的 NVRTC 编译失败 |
| bitsandbytes | 未测试 | 当前离线环境没有依赖，未下载或安装 |
| TorchAO | 未测试 | 当前离线环境没有依赖，未下载或安装 |
| ModelOpt/Quark | 未测试 | 没有兼容的本地 checkpoint |
| MXFP4/GPT-OSS MXFP4 | 环境受限 | FlashInfer JIT 缺少 `cublasLt.h` |
| FP-Quant | 硬件不支持 | 要求 Blackwell，H200 compute capability 不满足 |

这些项目没有到达“对运行态模型直接调用 `model.load_weights()`”的实验阶段。

## 7. 最终结论

最准确的结论不是“量化模型不支持 `model.load_weights()`”，而是：

1. `model.load_weights()` 只负责把 checkpoint tensor 写入**当前模型结构**。
2. 如果当前运行态结构仍等于 checkpoint schema，并且没有派生状态，直接调用可能
   足够；普通 BF16 dense/MoE 属于候选，但尚缺独立端到端验证。
3. 如果首次 PWAL 改变了权重 dtype、shape、layout 或生成了派生 tensor，直接调用
   不足；在线量化、repack 量化和 MLA 属于这一类。
4. 已验证的通用方案是由 `ModelwiseReloadSession` 恢复 checkpoint schema，然后
   调用一次或多次 `model.load_weights()`，最后统一 PWAL 并原位提交。
5. packed chunk、partial update 和 tensor `numel` 都不应被当作模型更新完成边界；
   `finish_weight_update()` 才是唯一提交信号。
