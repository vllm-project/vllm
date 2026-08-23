# PCP O-Proj TP MVP 方案

实现基线：`vllm-project/vllm upstream/main@f8e0602713`。

## 背景

vllm-ascend 的细粒度 O-Proj TP 将 O-Proj 的输入维从 TP 继续切到更多
rank，以减少每 rank 常驻权重。DSA-CP 场景在运行时按 batch 类型切换：decode
使用更细的权重分片并归约部分和；包含 prefill 时，先聚合出原 TP 粒度的完整
权重，再执行原有 O-Proj。

vLLM 已有独立的 TP、PCP 和 DCP process group，但通用
`RowParallelLinear` 只按 TP 切权重。本 MVP 使用 PCP 作为 O-Proj 的第二个
权重切分轴，实现与上述动态切换等价的 `pcp_o_proj_tp`。

## MVP 限制

- `prefill_context_parallel_size > 1`。
- `decode_context_parallel_size = 1`。
- 仅支持 DeepSeek V2/V3/V3.2 和 GLM-4 MoE Lite MLA。GLM-4 MoE Lite 的
  attention 继承 DeepSeekV2 MLA 路径，因此复用相同的显式 prefetch 与 O-Proj
  动态路径。
- 仅支持 FP16/BF16 非量化权重、`bias=False`。
- 仅支持 eager，暂不支持 CUDA Graph、torch.compile 和 DBO。
- 暂不支持 LoRA。
- 暂不支持 CPU weight offload。
- attention module 必须在主 attention 计算前显式调用
  `o_proj.prefetch_full_weight_if_needed(has_prefill)`。

这些限制全部通过配置或线性层构造检查 fail fast，避免功能开关静默失效。

## 权重布局

设 O-Proj 权重为 `W[N, K]`，TP 大小为 `T`，PCP 大小为 `P`。
普通 row parallel 在 TP rank `t` 上常驻：

```text
W_tp[t] = W[:, t*K/T : (t+1)*K/T]
```

MVP 在同一个 TP shard 内继续按 PCP rank `p` 切分：

```text
W_local[t,p] = W[:, (t*P+p)*K/(T*P) : (t*P+p+1)*K/(T*P)]
```

checkpoint loader 使用展平 rank `t*P+p` 和 world size `T*P`，因此无需改变
checkpoint 格式。每 rank 的 O-Proj 常驻权重降为原 TP 方案的 `1/P`。

## 运行时路径

### Decode-only

PCP rank `p` 从 TP-sharded attention output `X_t[..., K/T]` 中取对应 feature
slice，与本地 `W_local[t,p]` 做 GEMM。各 PCP rank 的部分和先在 PCP group
all-reduce，恢复普通 TP rank 的 O-Proj 部分结果；随后保持原有 TP reduction
语义。

```text
Y_t = PCP-AllReduce(X_t,p @ W_local[t,p]^T)
Y   = TP-AllReduce(Y_t)                    # reduce_results=True 时
```

DeepSeek-V3.2 的 O-Proj 使用 `reduce_results=False`，因此只做 PCP reduction，
后续已有的 TP fused all-reduce/RMSNorm 仍负责 TP reduction。

### Prefill 或 mixed batch

attention module 根据该层 `MLACommonMetadata.num_prefills` 判断 batch。只要包含
prefill，就在 attention 主计算前异步执行 PCP all-gather：

```text
W_local[t,p]^T --PCP AllGather--> W_tp[t]^T
```

O-Proj forward 到达时才等待异步 handle，然后直接用 `W_tp[t]` 处理当前 PCP
rank 的本地 token。该路径不做 PCP output reduction，因为不同 PCP rank 处理的
是不同 token。原 TP reduction 保持不变。

为支持 `all_gather_into_tensor` 沿第 0 维拼接，通信输入使用连续的转置权重
`[K/(T*P), N]`，输出 buffer 为 `[K/T, N]`。完整 TP 权重只保留转置 view，
不再额外复制。

## Buffer 生命周期

完整 TP 权重 buffer 按 vLLM config scope、device、dtype 和 shape 复用。模型
构造时即分配并由各层持有同一个共享 buffer，使该开销进入启动显存预算；正常
层序执行中，上一层 O-Proj 已等待并消费 gather 后，下一层才能复用它。因此
常驻额外开销是一份 TP O-Proj 权重，而不是每层一份。

异步通信输入是当前层本地权重的连续转置临时 tensor，保持到 handle wait 完成
后释放。DBO 暂时禁用，以排除两个 microbatch 并发占用共享 buffer。

## 显式触发与正确性约束

通用 MLA wrapper 和模块化 DeepSeek-V3.2 attention 均显式触发 prefetch。
`PCPOProjRowParallelLinear.forward()` 要求每次调用前必须完成触发；遗漏触发会
直接报错，而不会猜测 batch 类型。这样可保证：

- prefill/mixed 的 gather 足够早，能够与 Q/K/V projection、RoPE、indexer 和
  attention 主计算重叠；
- decode-only 不误发 weight gather；
- O-Proj 不会在异步权重未完成时读取共享 buffer；
- metadata 为空的 profiling 路径按 decode-sharded 数学执行，零 attention
  output 的正确性不受影响。

## 配置

新增 CLI/config 开关：

```text
--enable-pcp-o-proj-tp
```

示例：

```bash
vllm serve <model> \
  --tensor-parallel-size 2 \
  --prefill-context-parallel-size 2 \
  --decode-context-parallel-size 1 \
  --enable-pcp-o-proj-tp \
  --enforce-eager \
  --dtype bfloat16
```

## MVP 验证结果

2026-08-21 在 4 张 H100 上使用 GLM-4.7-Flash BF16 和 GSM8K test
（1319 题）完成了 TP4、PCP4TP1、PCP2TP2 的在线精度验证。三组统一使用
eager、`max_model_len=8192`、5-shot、`temperature=0`、关闭 thinking、32
并发；32 题门禁使用 `max_tokens=256`，完整集使用
`max_tokens=4096`。

| 并行策略 | 32 题门禁 | 完整集正确数 | 完整集准确率 | 请求失败 | 无效答案 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP4 baseline | 25/32 | 1134/1319 | 85.9742% | 0 | 0 |
| PCP4TP1 | 25/32 | 1136/1319 | 86.1259% | 0 | 0 |
| PCP2TP2 | 26/32 | 1135/1319 | 86.0500% | 0 | 0 |

相对 TP4 baseline，PCP4TP1 有 35 题由错变对、33 题由对变错，净增 2
题；PCP2TP2 有 32 题由错变对、31 题由对变错，净增 1 题。三种归约
拓扑会产生足以改变部分自回归生成路径的数值扰动，因此逐题文本并不要求完全
一致；完整集结果未观察到精度回退。

按 checkpoint safetensors 元数据核算，48 层 O-Proj 共
`1,006,632,960` bytes，占模型参数字节的 `3.2242%`。特性开启后：

- PCP4TP1 每 rank 的 O-Proj 常驻权重从完整 O-Proj 的 100% 降到 25%，即减少
  75%；
- PCP2TP2 每 rank 的 O-Proj 常驻权重从原 TP2 shard 的 100% 降到 50%，即
  减少 50%。

该比例只描述参数权重，不包含 CUDA context、KV cache、通信 buffer、共享的
单层 prefetch buffer 和 allocator 碎片，不能等同于整机 HBM 降幅。

## 后续工作

1. 补充 TP×PCP 的逐层 prefill、mixed、decode tensor 数值对齐、多轮稳定性
   验证，并采集 weight gather/attention overlap trace。
2. 将共享 buffer lease 纳入可并发的模型级 workspace manager，解除 DBO 限制。
3. 增加 CUDA Graph/torch.compile 可捕获路径。
4. 为 FP8、block quant 等量化格式定义 weight、scale 和 metadata 的联合聚合。
5. 评估独立 PCP communicator/stream，避免与 attention PCP collective 互相串行。
