# 补上最后一块拼图：让 Kimi K3 的 fused KDA decode kernel 支持 DS conv-state 布局

> 一句话总结：vLLM 里 Kimi K3 的 69 层 KDA（Kimi Delta Attention）decode 早就有一条
> "单 kernel 融合" 快路径，但它只接受 SD 布局的 conv-state cache；而所有需要 P/D 分离
> （NIXL 点对点状态传输）的部署都被固定为 DS 布局，于是整条快路径在生产配置下被静默
> 旁路，回退到一条 3-op Triton 链。本 PR 把 kernel 的 conv-state 内层 stride 从编译期
> 常量改为启动参数，让两种布局都走融合路径，KDA decode 子链得到 1.16-1.58x 加速
> （耗时降低 14-37%，batch 越大绝对收益越小），E2E decode 吞吐 +2.2%、TPOT -2.3%。

## 背景：Kimi K3 与 KDA decode 的三条路径

Kimi K3 是一个 93 层的混合架构模型：69 层 KDA 线性注意力（门控 delta
net，hidden 7168、每个 TP rank 12 个 local head、head_dim 128）+ 24 层 MLA。
decode 阶段每层 KDA 的计算链条是：

```
in_proj GEMM -> causal conv1d (state cache 移位+卷积) -> gated delta rule
recurrence (状态 S: Hx128x128 fp32) -> gated RMS norm -> o_proj GEMM
```

vLLM 为这条链实现了三种执行方式：

1. **融合 CUDA kernel**（`ops.fused_kda_decode`，`csrc/libtorch_stable/kimi_k3/
   fused_kda_decode_kernel.cu`）：一个 kernel 内完成 conv 更新 + recurrence +
   output gate + norm，每步 decode 每层只剩 3 个 kernel launch（in_proj GEMM、
   融合 kernel、o_proj GEMM）。这是给纯 decode（无投机采样）准备的快路径。
2. **Triton 回退链**：`causal_conv1d_update` + `fused_recurrent_kda_packed_decode`
   + `rms_norm_gated`，外加两次 conv/state 切片拷贝，每层约 8-9 个 launch。
3. **deepspark/投机采样路径**：逐 token 的 chunk 处理，与本文无关。

其中路径 1 有一个隐藏的前置条件：`is_fused_kda_decode_supported()` 里排除了
`is_conv_state_dim_first()` —— 即 **DS 布局（每个 cache block 内按 (dim, state_len)
排布 conv state）直接禁用融合 kernel**。

而 DS 布局恰好是生产刚需：

- `VLLM_SSM_CONV_STATE_LAYOUT=DS` 让 conv state 的 block 内排布变为 (dim, state_len)，
  这是 **NIXL P/D 状态直传**（prefill/decode 分离部署里把 Mamba/线性注意力状态从
  prefill 节点传到 decode 节点）所要求的布局；我们规范化部署 Kimi K3 时全部 pin 了
  这个选项。
- 近期的相关修复也默认 DS 是生产形态；DS 已是事实上的生产默认。

结果就是：**生产条件下 69 层 KDA 一律走 Triton 回退链**，此前 benchmark 里的
conv+recurrent+norm 合计 ~3.5% 的 GPU kernel 时间白白多花，launch 数量也接近
3 倍。

## 根因：布局相关的编译期常量

看 kernel 里 conv-state 的寻址：

```cpp
// 修改前：kPackedDim = 3 * H_local * 128 是编译期常量
constexpr int kPackedDim = 3 * kLocalDim;
const __nv_bfloat16 q_state = cs_q_for_slot[hk + w * kPackedDim];
                                             │     └── tap 维 stride
                                             └──────── channel 维 stride (隐含 1)
```

这对应 SD 布局的内存排布（每个 slot 共 9*dim 个 bf16 元素）：

| 布局 | 每 slot 排布（[tap][plane][channel]，plane ∈ {q,k,v}） | channel stride | tap stride |
|------|-------------------------------------------------------|----------------|------------|
| SD   | `[w][q dim][k dim][v dim]`，tap 为主维                | 1              | 3·dim      |
| DS   | `[plane][channel][w]`，channel 为主维                 | 3              | 1          |

kernel 的其他所有部分（q/k/v 三个 plane 的基址 `conv_ptr + k*conv_segment_bytes`、
S state、gate/norm 部分）本来就和内层排布无关 —— 实际上 host 侧
`conv_segment_bytes = dim * stride(1) * sizeof(bf16)` 的写法在两种布局下都恰好给出
正确的 plane 间距（SD: dim×1、DS: dim×3），说明当初写代码时已经把布局意识到一半了，
只是 kernel 里把 tap/channel stride 写死了。

## 修改：把两个 stride 变成 launch 参数

改动本身非常机械、但每一步都要对得上：

1. `KdaDecodeStrides` 增加 `conv_channel` / `conv_tap` 两个字段（SD 填入
   `(1, 3*dim)`，DS 填入 `(3, 1)`），kernel 内 6 处 conv-state 读写全部换成
   `idx * conv_channel_stride + tap * conv_tap_stride`。
2. host wrapper 的合法性检查从 "必须是 SD" 放宽为 "SD 或 DS 之一"；
   `raw_strides` 数组从 5 个扩到 7 个，extern-C launcher 一并透传。
3. `is_fused_kda_decode_supported()` 删掉 `or is_conv_state_dim_first()`
   排除项；Python 侧 DS 模式下 `conv_state` 本来就不做 transpose，直接传进来
   的 stride 与新的检查恰好匹配。
4. ROCm 对应实现（`fused_kda_decode_kernel_rocm.cu`）做同样的镜像修改。
5. `tests/models/kimi_k3/test_kda.py::test_fused_kda_decode_correctness` 增加 DS
   参数化（5 → 10 个用例），参考链 `causal_conv1d_update` 本身接受任意 stride，
   所以 DS 对照零成本。

一个值得记录的一致性技巧：SD/DS 两条路径在每个 channel 上的乘加顺序完全一致，只是访存地址不同，因此同一份随机输入下两种布局的输出应当**逐位相同**。
这给了我们一个比容差检查更强的回归哨兵。

## 正确性验证

### 单元/算子层

- 新增 DS 用例后 `test_fused_kda_decode_correctness` 10/10 通过（SD 5 + DS 5，
  对照组为 Triton 参考链 `causal_conv1d_update + fused_recurrent_kda_packed_decode
  + 手写 gated norm`，容差与原有用例一致 atol=rtol=3e-2，conv state 精确相等）。
- 同一随机种子的 SD vs DS 运行：输出、conv state、recurrent state **逐位一致**
  （max|diff| = 0.0），证明 DS 泛化是纯布局参数化，没有引入任何数值路径变化。
- `test_fused_kda_decode_rejects_speculative_conv_state` 不受影响（投机采样仍走
  回退链）。

### 端到端 greedy A/B

在同构生产部署（16x B200、TP8+PP2、DS 布局 pin）上，对 24 个随机 prompt、
temperature=0、各生成 128 token 对比 base 与 patched 两个服务：

- **首 token logprob 逐位一致**（24/24，mean=max=0.0）。
- 生成文本的前缀一致长度：中位数 48/128 token、均值 42；文本差异全部是
  bf16 数值漂移在 borderline token 上的正常分叉，一次分叉后两侧条件上下文不同，
  后续文本不再可比。这与「同一算子换实现」的预期行为一致。
- 单元层（更强判据）：相同输入下 SD/DS 两种布局的输出、conv state、recurrent
  state 逐位一致，说明 DS 泛化没有引入任何数值路径变化。

### lm_eval GSM8K（patched 服务全量跑分）

在上述 patched 双节点服务上直接跑 lm_eval 0.4.12 的 GSM8K
（5-shot 默认，chat template，temperature=0，max_tokens=1500，全量 1319 条 test）：

| Filter | Metric | Value | Stderr |
|---|---|---|---|
| flexible-extract | exact_match | 0.9689 | ±0.0048 |
| strict-match | exact_match | 0.9697 | ±0.0047 |

~97% 的分数落在 Kimi K3 应有的前沿区间，说明 decode 改走 fused kernel + DS 布局后
模型质量没有任何回归。原始产物在共享盘 `lm_eval_gsm8k_patched/`（含逐条 samples
jsonl 和完整日志）。

## 性能

### Kernel 级微基准（CUDA graph replay，B200，H=12、K=V=128，生产 TP8 形状）

DS 布局、融合 vs 回退链（conv update + packed decode + gated norm）：

| batch | fallback (us) | fused (us) | 加速比 |
|------:|--------------:|-----------:|-------:|
|     1 |          8.97 |       6.15 |  1.46x |
|     8 |         11.00 |       8.96 |  1.23x |
|    32 |         17.99 |      11.42 |  1.58x |
|    64 |         28.24 |      20.54 |  1.37x |
|   128 |         56.33 |      45.39 |  1.24x |
|   256 |        104.41 |      88.87 |  1.17x |
|   512 |        199.06 |     171.60 |  1.16x |

（norm 单独 ~4.9us 被完全吸收进融合 kernel；B≥32 时主要收益来自 recurrence 部分
访存合并与一次 global round-trip 的消除。此表的 fused 数字与修改前 SD 布局下的
融合 kernel 数字一致 —— DS 泛化本身没有性能代价。）

### E2E（2 节点 x 8xB200，TP8+PP2，FP8 KV cache，FULL_DECODE_ONLY cudagraph，DS 布局）

bench 配置：`vllm bench serve`，随机数据 128 输入 / 512 输出，128 请求，
并发上限 64。两侧各跑 2 次取均值（不带 profiler）：

| 指标 | baseline (fallback 链) | patched (融合 kernel) | 变化 |
|------|-----------------------:|----------------------:|-----:|
| Mean TPOT | 28.50 ms | 27.85 ms | **-2.28%** |
| Median TPOT | 28.60 ms | 27.91 ms | -2.42% |
| Output throughput | 2130.9 tok/s | 2177.8 tok/s | **+2.20%** |
| Mean TTFT | 807.6 ms | 808.6 ms | +0.1%（符合预期：prefill 路径未动）|

两侧 run-to-run 噪声 <0.3%，2.2% 的吞吐收益显著高于噪声。

### 火焰分解（torch profiler trace）

patched 服务 profile 窗口内的 top kernel（rank0，PP 第一段；profile 窗口内
3 个请求 x 48 token ≈ 144 个 decode 步 x 本段 35 层 KDA ≈ 5076 次调用）：

| kernel | 总时长占比 | 调用数 | 均时 |
|--------|-----------:|-------:|-----:|
| `kda_decode_fusion_many_heads_kernel` | 1.26% | 5076 | 7.24us |

同时回退链 decode 侧的 `_causal_conv1d_update_kernel` /
`fused_recurrent_kda_packed_decode` / `layer_norm_gated`（prefill 的
`_causal_conv1d_fwd_kernel` 和 FlashKDA prefill 保留，属预期）在 decode 热路径中
消失 —— 证明 DS 布局下融合 kernel 已在 cudagraph 生产形态中真实激活。
baseline 侧对照 trace 里，整个 KDA decode 三件套合计约 3.5% kernel 时间
（conv 371ms + recurrent 761ms + norm 140ms），与理论预期（融合后省掉
2 次 global round-trip + 2/3 的 launch + norm 完全吸收）一致。

## 结论与后续方向

这项工作把一条已经写好、且被 benchmark 验证过的融合快路径对生产默认配置解锁了。
它不引入新 kernel、不改调度、不加环境变量，属于 "把现有能力的适用范围补齐" 的
低风险必要优化。

顺着 tokenSpeed 现有实现还能往前再走一步：TSN 的 `fused_recurrent_kda_megafuse`
把 f_b(128->1536) GEMV、conv、recurrence、norm 全合进一个 kernel。我们目前的
in_proj GEMM 还留在 cuBLAS/nvjet，下一步可以评估把 skinny GEMM 也拉进融合路径，
或者反过来像 PR#53040 对 DeepGEMM 做的那样，为这条小链条定制 dispatch。

## 附：复现

```bash
# 服务（双节点；片段，完整脚本见仓库脚本目录）
VLLM_SSM_CONV_STATE_LAYOUT=DS vllm serve $MODEL \
  --kv-cache-dtype fp8 --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --nnodes 2 --max-num-seqs 128 --max-cudagraph-capture-size 128 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' ...

# bench
vllm bench serve --backend openai-chat --dataset-name random --ignore-eos \
  --random-input-len 128 --random-output-len 512 \
  --num-prompts 128 --max-concurrency 64

# 单测
pytest tests/models/kimi_k3/test_kda.py -k fused_kda_decode -v
```
