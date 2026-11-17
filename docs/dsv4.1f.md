# DeepSeek-V4.1-Flash：aiter / vLLM PR 状态



Image: vllm/vllm-openai-rocm:nightly-96710245ece8fe57a0df3ea18cdc79cbb8ce93cf, digest sha256:6eeac5c6…e51b.

vLLM version: 0.34.0rc1.dev142+g96710245e.rocm100, which corresponds to upstream commit 96710245ece8fe57a0df3ea18cdc79cbb8ce93cf.



| **组件**     | **版本**                   | **来源**                                                                                                   |
| ---------- | ------------------------ | -------------------------------------------------------------------------------------------------------- |
| **aiter**  | `amd-aiter 0.1.28.post1` | 从 [ROCm/aiter](https://github.com/ROCm/aiter) 的 tag `v0.1.28.post1` 源码编译，编译架构 `gfx942;gfx950`            |
| **mori**   | `amd_mori 1.2.3.post1`   | 从 [ROCm/mori](https://github.com/ROCm/mori) 的 tag `v1.2.3.post1` 源码编译，针对 ROCm 10.0.0，NIC backend 为 `all` |
| **FlyDSL** | `flydsl 0.4.2`           | 从 PyPI 安装，作为 aiter 的依赖（aiter 的元数据要求 `flydsl==0.4.2`），不是源码编译                                              |



## vllm commit `96710245` 上尚未包含、有性能帮助的 PR

核对时间：2026-11-17。工作 commit：

`96710245ece8fe57a0df3ea18cdc79cbb8ce93cf`（2026-11-10 13:52 +0800 = `2026-11-10T05:52:43Z`，`[Rust Frontend] Stream DeepSeek DSML tool-call arguments (#66529)`）

对照 tip：`origin/main` `a4910b41280ae228bcc74ebc9d21144e09ef63ef`（2026-11-16）。

来源补充：[#66587](https://github.com/vllm-project/vllm/issues/66587)（MI355X / gfx950 DSv4.1-Flash 性能 tracker，body 更新至 2026-11-16）。issue 里 **merged_at 早于工作 commit** 的项视为镜像已含，不重复列出；下列只收 **晚于该 commit / 仍未合入**，且带性能或 tracker 宣称的 PR。

### 已合入 main / aiter，但晚于 `96710245`（或晚于镜像 aiter `0.1.28.post1`）

| PR | merge | 作用 / 宣称增益 |
|---|---|---|
| [#67377](https://github.com/vllm-project/vllm/pull/67377) dense MXFP8 / `wo_a` 走 AITER FlyDSL + 更快 MXFP8 quant | 2026-11-15 | 离线 dense GEMM：prefill 1.78x、decode 1.61x；E2E AgentX 一度约 +11% tok/s/GPU |
| [#66036](https://github.com/vllm-project/vllm/pull/66036) Sparse MLA 支持 per-tensor FP8 KV（`fp8_e4m3`） | 2026-11-12 | 同显存约 +12.5% KV 容量；TP2/TP4 + DSpark 输出吞吐最高约 +7.6%（#66587） |
| [#66596](https://github.com/vllm-project/vllm/pull/66596) AITER bump 到 v0.1.29 | 2026-11-10 07:49Z（晚于 base 05:52Z） | 带入新 all-reduce（lamport）等；多个后续 kernel PR 依赖它 |
| [#65992](https://github.com/vllm-project/vllm/pull/65992) fused mHC 走 AITER 单 launch Mega-MHC | 2026-11-15 | 主要是 launch/prefill 侧；DEP8 随机负载上 E2E 多在噪声内，4k/C64 input tok/s 约 +1.5% |
| [#66047](https://github.com/vllm-project/vllm/pull/66047) sparse indexer decode 行按 TP 分片 | 2026-11-12 | E2E 约 +1–2%（C64/C256）；单并发点自身不显著 |
| [#65461](https://github.com/vllm-project/vllm/pull/65461) Mono：TP8 上把 `K2(l)` 与 `K1(l+1)` 合成一次 launch | 2026-11-13 | TP8 mono decode launch 合并 |
| [#62769](https://github.com/vllm-project/vllm/pull/62769) adaptive DSpark 决策以 TP rank0 为准 | 2026-11-11 | 正确性（防 hang）；#66587 DSpark 表 |
| [#65926](https://github.com/vllm-project/vllm/pull/65926) K-cache gather 用 int64 row offset | 2026-11-15 | 正确性（大 cache 不溢 32-bit）；#66587 |
| [aiter #7656](https://github.com/ROCm/aiter/pull/7656) gfx950 FP8 fused-MoE 重调 TP4 prefill 行（4096–16384） | 2026-11-12 | MoE 约 −10～−16%（TP4）；镜像 aiter 仍是 `0.1.28.post1`，未含 |

### 仍未合入（open），有明确性能宣称

| PR | 状态 | 作用 / 宣称增益 |
|---|---|---|
| [#65421](https://github.com/vllm-project/vllm/pull/65421) Asm decode layers（17–256 rows，gfx950 TP4） | open | 相对仅 `--monokernel`：median TPOT 约 1.14–1.45x 更低（8k/200k，c32–c128）；#66587 |
| [#57756](https://github.com/vllm-project/vllm/pull/57756) 融合 decode metadata + final norm | open | TP8 c1 decode step 约 −3.2%；#66587 |
| [#63182](https://github.com/vllm-project/vllm/pull/63182) DSpark draft-prepare 快路径（dedup / pad / KV-only insert） | open | TP4 decode ITL 约 −5.0%；#66587 |
| [#60076](https://github.com/vllm-project/vllm/pull/60076) DSpark context KV 的 KV-only insert | open | 每 16k prefill chunk：163–602 µs → 19 µs；#66587 |
| [#61962](https://github.com/vllm-project/vllm/pull/61962) MoRI dispatch 前把 MoE activation 量化到 MXFP8 | open | DEP8 约 +18.6% 输出吞吐；DEP4/DEP2 约 +11.8% / +8.6%。本地有 cherry-pick `702b526af8`，不在该工作 commit |
| [#63029](https://github.com/vllm-project/vllm/pull/63029) A4W4 路径通过 MoRI 预量化 MXFP4 dispatch | open | 开 `VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4=1` 时 DEP8 约 +7.0–7.5% |
| [#63821](https://github.com/vllm-project/vllm/pull/63821) TP1 MXFP8 tile + 预量化 `wo_b` 输入（DEP8） | open | 叠在 FlyDSL/MXFP8 路径上；AgentX 单跑约 16.8k→17.6k tok/s |
| [#64235](https://github.com/vllm-project/vllm/pull/64235) MoRI combine 可选 FP4（`VLLM_MORI_FP4_COMBINE`） | open | DEP8 c256 约 +2.8% 输出吞吐；依赖 MoRI 侧配套 |
| [#65128](https://github.com/vllm-project/vllm/pull/65128) MoRI EPv2 + DBO（native / AsyncLL）及 ROCm DBO 修复 | open | V4-Pro DP8+EP8：prefill 相对 DBO off 约 +12–14%（native）/ +22%（AsyncLL）；decode 拆批会变慢，建议 prefill-only DBO；有冲突需 rebase |
| [#65146](https://github.com/vllm-project/vllm/pull/65146) MI355X fused mega-attention prefill | open | prefill 相对现有 ROCm sparse：16/128 tok 约 1.7x，512 约 1.29x，2048 约 1.15x；有冲突需 rebase |
| [#66572](https://github.com/vllm-project/vllm/pull/66572) q RMSNorm + MXFP8 quant 合成一次 launch | open | 2×(TP4+EP4) c64：约 +32–41 TPS / +1.2–1.5 tok/s |
| [#65742](https://github.com/vllm-project/vllm/pull/65742) ROCm gfx950 用单 kernel DSv4 top-k | open | DEP8：C128 TPOT p50 约 −2.5%，C256 约 −1.1% |
| [#65749](https://github.com/vllm-project/vllm/pull/65749) gfx950 decode 的 Split-K BF16 router GEMM | open | decode step 变短；C256 吞吐多在噪声内 |
| [#66128](https://github.com/vllm-project/vllm/pull/66128) adaptive verification 下 MXFP4 indexer 按 varlen 计分 | open | indexer 1.4–2.4x；E2E 多在噪声内；#66587 |
| [#67363](https://github.com/vllm-project/vllm/pull/67363) MXFP4 indexer decode metadata 一次 launch | open | C1：eager builder 17→4、约 −23 µs/step；DEP8 C128 估计约 −87 µs（~0.5%） |
| [#67364](https://github.com/vllm-project/vllm/pull/67364) 压缩 launch 内直接写 compressed KV cache | open | 每 decode step 少 4 次 launch；TP8 C1 微基准约 −7.5 µs/step |
| [#67365](https://github.com/vllm-project/vllm/pull/67365) decode sparse-MLA ragged top-k 从 row bounds 一次 pack | open | 微基准每 step 约 −41～−47 µs（7–336 rows） |
| [#67366](https://github.com/vllm-project/vllm/pull/67366) consumer remap 融进 top-512；所有 decode 行跳过 -1 fill | open | remap 约 −1.25 µs/call；高并发 fill 约 −16～−20 µs/step；多与 indexer 栈一起测 |
| [#67367](https://github.com/vllm-project/vllm/pull/67367) gfx950 top-512 少行调用在 ≤36K 时保持单 block | open | 32–35K context indexer top-k 约 −16～−24 µs/step；64K 约 −5 µs；128K 略慢 |
| [#67368](https://github.com/vllm-project/vllm/pull/67368) indexer head weights 在 MXFP4 Q quant launch 内算 | open | C1 微基准约 −12～−17 µs/step；依赖 #67364 一类 compress-insert 栈 |
| [aiter #5918](https://github.com/ROCm/aiter/pull/5918) fused all_reduce+add + gfx950 all-reduce 重调 | open | TP4 all-reduce median 约 −16%（kernel）；#66587 |
| [aiter #7657](https://github.com/ROCm/aiter/pull/7657) FlyDSL fused-MoE tuner（random routing / skip stage-1） | open | 调参基础设施；#66587 未给 E2E 数字 |
| [aiter #7349](https://github.com/ROCm/aiter/pull/7349) FlyDSL MoE sorting LDS race | open | 正确性（DSpark 下 NaN）；#66587 |
| [aiter #7997](https://github.com/ROCm/aiter/pull/7997) `sparse_mla_fwd` 可选融合 Q RoPE + fp8 cast（gfx950） | open | 32K SWA-only pipeline 约 −31%、two-list 约 −12%；vLLM 接线见 [#67800](https://github.com/vllm-project/vllm/pull/67800)；AgentX DEP8 单跑 E2E 在噪声内 |

### 仍是 draft / 已关闭，但列表里标过性能相关

| PR | 状态 | 备注 |
|---|---|---|
| [aiter #7917](https://github.com/ROCm/aiter/pull/7917) MoE v2 stage 2 跳过 `inter_dim` padding | open，draft | DSv4.1 TP4 a8w4 `fused_moe` 约 −4.5～−7.5%（256–16384 tok）；E2E TPOT 约 −1.7～−3.9%；可去掉 vLLM `intermediate_pad=0` 强制 |
| [aiter #7918](https://github.com/ROCm/aiter/pull/7918) MoE stage 1 XCD remap 只覆盖 valid rows | open，draft | 叠在 #7656 上：`xcd_swizzle=1` 时 fused_moe 最高约 −8%（4096 tok −7.8%）；2048–16384 行受益 |
| [#63326](https://github.com/vllm-project/vllm/pull/63326) Engram WKV chain 与 decoder 重叠 | open，draft | TP4 mean TPOT 约 −2.3%（12 点，未叠 #63367）；#66587；拟与已合入的 #63367 重测 |
| [#66856](https://github.com/vllm-project/vllm/pull/66856) 把 `wo_a` MXFP8 quant 融进 epilogue | open，draft | 宣称约 +1.2%；依赖 #66572 一类 q/quant 融合栈 |
| [#65243](https://github.com/vllm-project/vllm/pull/65243) 去掉 MoRI finalize 多余 copy | open，draft | 每层少一次 copy；单独 E2E 多在噪声内 |
| [#66001](https://github.com/vllm-project/vllm/pull/66001) router GEMM 放到 shared-expert fork 之前 | open，draft | 层内约 −3.6 µs；E2E A/B 仍 TODO |
| [#58352](https://github.com/vllm-project/vllm/pull/58352) sparse-MLA ragged decode metadata 打进 graph buffer | closed，draft | TPOT c1 −0.4%、DSpark verify −1.5%；#66587，2026-11-11 关 |
| [#66212](https://github.com/vllm-project/vllm/pull/66212) context WKV projection 合成一次 GEMM | closed，draft | kernel 2.2–2.9x；#66587，2026-11-11 关 |
| [#59782](https://github.com/vllm-project/vllm/pull/59782) 小 M 时 MXFP8 block32 内做 activation quant | closed，draft | 1–8 tok 每 call 1.14–1.61x；#66587，2026-11-10 关 |
| [#64917](https://github.com/vllm-project/vllm/pull/64917) FP8 paged decode indexer 按行负载均衡 | closed，draft | 无 E2E 数字；2026-11-13 关闭 |
| [#64135](https://github.com/vllm-project/vllm/pull/64135) DP 上 native ASM sparse attention | closed，draft | 作者称尚无 E2E 增益；AgentX 长跑曾 fault，2026-10-30 关闭 |
| [#64280](https://github.com/vllm-project/vllm/pull/64280) adaptive verification profile 间释放 allocator cache | closed | 正确性（#64275）；#66587，2026-11-12 关 |

优先建议（相对工作 commit / 镜像）：先吃已合入的 [#67377](https://github.com/vllm-project/vllm/pull/67377) + [#66596](https://github.com/vllm-project/vllm/pull/66596) + [#66036](https://github.com/vllm-project/vllm/pull/66036) + [aiter #7656](https://github.com/ROCm/aiter/pull/7656)；TP4/`--monokernel` 场景再追 [#65421](https://github.com/vllm-project/vllm/pull/65421) / [#63182](https://github.com/vllm-project/vllm/pull/63182) / [#57756](https://github.com/vllm-project/vllm/pull/57756)；DEP/MoRI 场景再追 [#61962](https://github.com/vllm-project/vllm/pull/61962) / [#63029](https://github.com/vllm-project/vllm/pull/63029) / [#64235](https://github.com/vllm-project/vllm/pull/64235)。

## 未合入的 PR

| PR | 状态 | 作用 |
|---|---|---|
| [aiter #6119](https://github.com/ROCm/aiter/pull/6119) mxfp4 moe sort skip invalid ids | open | `VLLM_MOE_SKIP_PADDING` 默认打开时，padding 行的 expert id 是 -1。mxfp4 moe sort 会把 -1 当成合法行读崩。这条让 sort、quant 和 scatter-reduce 丢掉 `[0, E)` 以外的 id，a4w4 才能在 skip-padding 下把 DSv4.1-Flash 服务拉起来。 |
| [vLLM #59812](https://github.com/vllm-project/vllm/pull/59812) Pass no MoE padding to AITER on the a4w4 path | open，draft | a4w4 路径把 `hidden_pad` / `intermediate_pad` 传 0。AITER 一旦看到 padding 就会丢掉 tuned inline-sort MXMOE config，512 token 以内都落到 heuristic kernel。仍由 `VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4=1` 控制。 |
| [vLLM #60834](https://github.com/vllm-project/vllm/pull/60834) paged MXFP4 sparse indexer dense path | **已合入** 2026-10-23（早于 `96710245`，镜像应已含） | 把 #58671 的 paged MXFP4 sparse indexer dense 路径接到 DeepSeek-V4；opt-in `--attention-config '{"indexer_kv_dtype": "mxfp4"}'`。 |
| [aiter #6154](https://github.com/ROCm/aiter/pull/6154) full-expert DSv4.1-Flash MegaMoE tuning | open，draft | 给 gfx950 上 DeepSeek-V4.1-Flash 的 MegaMoEV2 补全专家并行配置：H=5120，I=2304，E=384，top-k=6，EP2 每 rank 192 个专家、EP4 每 rank 96 个。把 peer dispatch 和 W13、clamp-10 SwiGLU、FP8 中间量化融合，再接 MXFP4 W2 和加权 peer combine。shared expert 仍分开。叠在已合入的 aiter #6127 上。 |

核对时间：2026-10-16。对照的远端 tip：

- `vllm-project/vllm` `main` `a956a8126e6bd184cb0a6b8f7318fe4717675c2f`（2026-10-15 18:00 -0700）
- `ROCm/aiter` `main` `c118fafad9ad2873bb63b9b122d59aef5c307a4f`（2026-10-15 13:28 -0400）
- `vllm-project/recipes` 只核对了 PR / merge commit，没有本地 clone

5 个 PR 不能收成一个 commit：两条还没合入，已合入的三条分属 vLLM 和 recipes。

## 总表

| PR | 仓库 | 是否 merge | main 上最早包含该 PR 的 commit |
|---|---|---|---|
| [#59454](https://github.com/vllm-project/vllm/pull/59454) [Bugfix][ROCm] Use a zero default for masked scales in the MXFP8 GEMM（cagrikymk） | vllm | 已合入，2026-10-01 16:32 UTC，dllehr-amd | `08e03df94028d5e7cff96bf569459d8d2b926d9e`（2026-10-01 11:32 -0500） |
| [#58819](https://github.com/vllm-project/vllm/pull/58819) [ROCm][Perf] Add opt-in a4w4 (FP4 activation) MoE for DeepSeek V4.1 on AITER（Fangzhou-Ai） | vllm | 已合入，2026-09-30 07:07 UTC，shen-shanshan | `77e8b02c8d69f25867227fb9883797996fbfcd7b`（2026-09-30 15:07 +0800） |
| [#1061](https://github.com/vllm-project/recipes/pull/1061) [DeepSeek][ROCm] V4.1-Flash: enable INT4 QuickReduce on MI355X（ahmed-bsod） | vllm-project/recipes | 已合入，2026-10-03 04:57 UTC，Fangzhou-Ai | `ca06dc0a165ea073b7122e65b9c2ab36d68a02c4` |
| [#6119](https://github.com/ROCm/aiter/pull/6119) [HIP] [Bugfix] mxfp4 moe sort skip invalid ids (-1s)（cagrikymk） | aiter | 未合入 | 无。`origin/main` 到 `c118fafad9` 的 log 里没有 `#6119` |
| [#59812](https://github.com/vllm-project/vllm/pull/59812) [ROCm][DSv4.1][Perf] Pass no MoE padding to AITER on the a4w4 path（cagrikymk） | vllm | 未合入（draft） | 无。`origin/main` 到 `a956a8126e` 的 log 里没有 `#59812` |

## vLLM 上同时包含已合入 PR 的最早 commit

`08e03df94028d5e7cff96bf569459d8d2b926d9e` 是 `main` 上最早同时包含 #58819 和 #59454 的 commit。`git merge-base --is-ancestor` 确认 `77e8b02` 是它的祖先；#59454 自己的 squash commit 就是这条。

这条不包含 #59812，也不包含 aiter #6119 或 recipes #1061。

## 和原问题的对应

### MXFP8 masked scale，以及 H200 CI

#59454 把 MXFP8 GEMM 里被 mask 掉的 e8m0 scale 默认值从 127（1.0）改成 0（2^-127）。数值侧对应 lane 已经是 0，scale 乘上去不影响精度；目的是避开 Triton 3.8 在 `other=127` 时读到脏 LDS（可能是 NaN）的问题。改动文件：`vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py`。

合入前作者跑过 `/ci run`（Buildkite #92235，commit `e525bb74`）和 `/ci retry`（Buildkite #92281，重跑 4 个失败 step，commit `9ad97905`）。Slack 里说的 H200 check 失败发生在合入之前。本次没有重拉 Buildkite 日志，不能从这里判断 H200 job 最后是通过还是被跳过；能确定的是 PR 已经在 2026-10-01 进入 `main`。

同一改动在 AITER 侧是 [#6007](https://github.com/ROCm/aiter/pull/6007)，已合入。`main` 上最早包含它的 commit 是 `fbbd9f6ac66e6e452d8c992e2a5b0b1d4b1f3b07`（2026-10-01 14:18 -0400，`[Triton] Use a zero default for masked scales in gemm_afp8wfp8`）。

### a4w4（MXFP4 x MXFP4）MoE 会不会默认打开

#58819 已经合入，但是 opt-in，不是默认。打开方式是 `VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4=1`，且只允许 `deepseek_v41` / `deepseek_v41_text`；其它 MXFP4 模型设了这个变量会在启动时报错。默认仍是 a8w4。

PR 里的 AgentX（TP2，DSpark）对比是：c1 output tok/s -1.0%，c8 持平，c64 +10.5%。作者因此没有把它做成默认。后续还想靠它吃到 tuned inline-sort kernel 的话，依赖下面两条还没合入的 PR。

### INT4 QuickReduce

#1061 改的是 recipe，不是 vLLM 运行时。`models/deepseek-ai/DeepSeek-V4.1-Flash.yaml` 的 ROCm strategy override 增加了 `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION: "INT4"`。镜像要包含这条，recipes 需要新于 `ca06dc0a165ea073b7122e65b9c2ab36d68a02c4`。

### 还没合入、a4w4 路径仍依赖的两条

- aiter #6119：`VLLM_MOE_SKIP_PADDING` 默认打开时，padding 行的 expert id 是 -1，mxfp4 moe sort 会把 -1 当成合法行读崩。PR 仍 open。2026-10-09 yzhou103 要求先解决冲突。assignee 是 yzhou103。
- vLLM #59812：a4w4 路径把 `hidden_pad` / `intermediate_pad` 传 0，避免 AITER 丢掉 tuned inline-sort MXMOE config。仍是 open draft，2026-10-02 之后没有合入记录。a4w4 本身仍由 `VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4=1` 控制。

## 第二批

同一对 `origin/main` tip。列表里的重复链接已合并。#59038 是 issue，不是 PR。

| PR | 是否 merge | main 上最早包含该 PR 的 commit |
|---|---|---|
| [#60834](https://github.com/vllm-project/vllm/pull/60834) [ROCm][Perf] Apply paged MXFP4 sparse indexer dense path for DeepSeek-V4（shen-shanshan） | 未合入（open） | 无 |
| [#57773](https://github.com/vllm-project/vllm/pull/57773) [ROCm][Perf] Enable DPA+ETP prefill compute/comm dual-batch-overlap for DeepSeek-V4（shen-shanshan） | 已合入，2026-10-15，shen-shanshan | `f46021a7faa62015016f2862d02f2a64c24bb960`（2026-10-15 11:16 +0800） |
| [#61378](https://github.com/vllm-project/vllm/pull/61378) [ROCm][Perf] Enqueue the default chain first in CSA/HCA at small batch sizes for DeepSeek-V4（shen-shanshan） | 已合入，2026-10-14，shen-shanshan | `2521446ac8c92937b5a32b7dc2a84ef7fa8e99c5`（2026-10-14 10:57 +0800） |
| [#58689](https://github.com/vllm-project/vllm/pull/58689) [ROCm][DSv4.1] Disambiguate ROCm attention class names（Tuukka Sarvi） | 已合入，2026-10-15 | `0951fc183cc7a86b54be45320dd42679c7065995`（2026-10-15 21:38 +0800） |
| [#58151](https://github.com/vllm-project/vllm/pull/58151) [ROCm][DSv4.1] Enable DSpark adaptive verification（tuukkjs） | 已合入，2026-10-15，shen-shanshan | `10d2c6083d3b7508f2409a127a1d03dd489e427c`（2026-10-15 12:05 +0800） |
| [#61004](https://github.com/vllm-project/vllm/pull/61004) [ROCm][DSv4] Fuse the shared expert into AITER MegaMoE（AMD-yanfeiwang） | 已合入，2026-10-12，shen-shanshan | `e3bbaf23695f7f30f144a288057b0419ab2b6a8d`（2026-10-12 15:53 +0800） |
| [#58897](https://github.com/vllm-project/vllm/pull/58897) [Perf][MoE] Cheaper triton_kernels 3.8 routing build in the OAI Triton MoE（ZhengGong-amd） | 已合入，2026-10-14，shen-shanshan | `2308b6f00be1b65c0b9159e922c529ead5299603`（2026-10-14 10:48 +0800） |
| [#60397](https://github.com/vllm-project/vllm/pull/60397) [ROCm][Perf][DSv4.1] Mono decode layer for DeepSeek-V4.1-Flash on MI355X（Fangzhou-Ai） | 已合入，2026-10-09，dllehr-amd | `193922d6ff5f7e213eacadd3cb7dd96612b30ed3`（2026-10-09 20:53 -0500） |
| [#59685](https://github.com/vllm-project/vllm/pull/59685) [ROCm][DSv4] AITER MegaMoEV2 Integration For DeepSeek V4（micah-wil） | 已合入，2026-10-07，AndreasKaratzas | `f6e3944e3100fcb526e5cc24248080c31812be0f`（2026-10-07 09:35 -0500） |
| [#6154](https://github.com/ROCm/aiter/pull/6154) [FlyDSL] Add full-expert DeepSeek V4.1 Flash MegaMoE tuning for gfx950（Fangzhou-Ai） | 未合入（draft） | 无。aiter `origin/main` 到 `c118fafad9` 没有 `#6154` |
| [#59039](https://github.com/vllm-project/vllm/pull/59039) [ROCm][Perf][DSv4.1] Enable decoder SWA bounded replay（frida-andersson） | 已合入，2026-10-13，shen-shanshan | `c077dc663d929fd19bdfb15ccf00f83cc490b8b0`（2026-10-13 02:19 +0000） |
| [#59038](https://github.com/vllm-project/vllm/issues/59038) [RFC] DeepSeek-V4.1-Flash performance on MI325X (gfx942)（frida-andersson） | issue，仍 open | 无 |

这 9 条已合入的 vLLM PR 都在 first-parent `main` 上。最早同时包含它们的 commit 是 `0951fc183cc7a86b54be45320dd42679c7065995`（#58689，2026-10-15 21:38 +0800）。它也包含上一节已合入的 #58819 和 #59454。它不包含 #60834、#59812、aiter #6119、aiter #6154，也不包含 recipes #1061。

[#59038](https://github.com/vllm-project/vllm/issues/59038) 跟踪的是 MI325X / gfx942 的 prefill。其中 decoder SWA bounded replay 对应的 [#59039](https://github.com/vllm-project/vllm/pull/59039) 已在 `c077dc66`。issue 正文里点名、本次列表没有单独给出的项：FlyDSL QuickReduce 接线 [#59478](https://github.com/vllm-project/vllm/pull/59478) 在写 issue 时尚未合入；sparse MLA prefill 候选是 aiter [#6002](https://github.com/ROCm/aiter/pull/6002)。这两条没有在本次对 `main` 复核。
