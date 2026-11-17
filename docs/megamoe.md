# aiter MegaMoE（gfx950 / 相关变体）

本地笔记：对照 [ROCm/aiter](https://github.com/ROCm/aiter) 源码（`aiter/ops/flydsl/kernels/mega_moe*`）。重点是 **gfx950 上 EP 用的 `MegaMoEV2`**：逻辑上常说 prepare / stage1 / stage2 / combine，但 **不是一律四个独立 kernel launch**；其他模型多半共用同一套多段 kernel 族，差在 `NetworkProfile` 与是否启用 Pro 专属路径。

## 家族一览

| 入口 | 目标 | 说明 |
|---|---|---|
| **`MegaMoEV2`** | gfx950 EP（单节点） | 主路径：多段 FlyDSL（prepare / stage1 / stage2，combine 常融进 stage2） |
| **`MegaMoEInternode`** | 跨节点 EP（如 EP16） | 在 MegaMoEV2 协议上扩展；stage2 复用 GEMM2 core |
| **`MegaMoeTP` / `MegaMoeTPEngine`** | gfx950 **TP** MoE | **另一套**：单 launch（+ big-M）做 AG/RS 或 AR + GEMM1/2，不是下面四个 EP stage |
| **`MegaMoEGfx1250`** | gfx1250 | 独立实现，不走 gfx950 MegaMoEV2 四段 |

公开 import（懒加载）：

- `aiter.ops.flydsl.kernels.mega_moe` → `MegaMoEV2`, `MegaMoEConfig`, `select_mega_moe_config`, …
- `aiter.ops.flydsl.kernels.mega_moe_internode` → `MegaMoEInternode`
- `aiter.ops.flydsl.kernels.mega_moe_gfx1250` → `MegaMoEGfx1250`
- `aiter.ops.flydsl.mega_moe_tp` → TP 封装

源码目录：

```
aiter/ops/flydsl/kernels/mega_moe/          # MegaMoEV2 + prepare/stage1/stage2 + large_m/small_m
aiter/ops/flydsl/kernels/mega_moe_internode/
aiter/ops/flydsl/kernels/mega_moe_tp/
aiter/ops/flydsl/kernels/mega_moe_gfx1250/
```

量化：`MegaMoEV2` 仅 **`a4w4` / `a8w4`**。依赖 FlyDSL + MORI shmem。

## gfx950 MegaMoEV2：逻辑阶段 vs 实际 launch

逻辑上四段：

| 阶段 | 作用 |
|---|---|
| **prepare** | compact 路径：路由整理 +（常含）activation quant；fixed-slot **不走**独立 prepare |
| **stage1** | peer dispatch + W13 GEMM1 + SwiGLU/SiTUv2 + 中间量化 |
| **stage2** | W2 GEMM2 + 加权 peer combine（多数配置下 **combine 融在 stage2**） |
| **combine** | 名义第四段；独立 launch 少见，多由 stage2 / small_m / large_m 一并做完 |

`forward` 调度要点（`mega_moe_v2.py`）：

1. 若 **large_m 打开且 eligible** → 整段交给 `LargeMRuntime`（见下）
2. **fixed-slot**：host `quantize` → `_run_joint`（stage1 + stage2），**无** `_run_compact_prepare`
3. **compact**：`_run_compact_prepare`（可带 `quant_input`）→ `_run_joint(..., prepared=True)`

因此「四个 kernel」只是命名习惯；实际常见是 **2～3 次 launch**（compact 有 prepare 时偏 3）。

### Dispatch 模式

| 模式 | 何时 | 典型场景 | launch 大致 |
|---|---|---|---|
| **fixed-slot** | `mtpr ≤ FIXED_SLOT_MAX_MTPR`（默认 env `AITER_MEGA_FIXED_SLOT_MAX_MTPR=255`）且 epr 等条件满足 | decode / 小 MTPR | quant（host）+ stage1 + stage2 |
| **compact** | 更大 MTPR | prefill | prepare + stage1 + stage2 |

`TOKEN_BUCKETS`：1…32768（2 的幂档）。配置按 bucket 选 `Stage1Config` / `Stage2Config`。

### Pro 专属：small_m / large_m（更少 kernel）

仅 **DeepSeek-V4-Pro 形状**：`H=7168, I=3072, E=384, topk=6, EP8, a8w4, silu, swiglu_limit=10`，且无 `fanout_masks`。Flash（5120×2304）在 aiter 0.1.29 原版里**进不了** small_m；打上 [patches/aiter-0.1.29-mega_moe-small_m-dsv41-flash.patch](./patches/aiter-0.1.29-mega_moe-small_m-dsv41-flash.patch) 后可用，vLLM 接线和实测见 [megamoe_single_kernel_dsv41f.md](./megamoe_single_kernel_dsv41f.md)。

| 路径 | 开关 | 条件补充 | Kernel 数 |
|---|---|---|---|
| **`small_m`** | `AITER_MEGA_SMALL_M=1` | gfx950、**256 CU**、每 rank ≤512 tok（默认 runtime 256/512） | **1**（quant+dispatch+GEMM1+GEMM2+combine） |
| **`large_m`** | `AITER_MEGA_LARGE_M=1`（或 ctor `large_m_enabled`） | `max_tok_per_rank==8192` | **2**：单独 quant + **一条** persistent（dispatch…combine） |

调用：`forward_small_m(...)`；large_m 走普通 `forward`（`set_large_m_enabled`）。small/large 与主路径 **不共享协议状态**，各 EP rank 必须选同一条。

## 注册网络（`NetworkProfile`）

按 **`(experts_per_rank, model_dim, inter_dim)`** 匹配；未命中用 **base config**（无专调表）。

| name | epr | H × I | 备注 |
|---|---|---|---|
| `r1` | 32 | 7168 × 2048 | `fp4_prelude` |
| `v4_pro` | 48（EPLB 族 52/56） | 7168 × 3072 | 可开 small_m / large_m；`fp4_prelude` |
| `kimi_k3` | 112 | 3584 × 3072 | act 常为 **situv2**；FP4/FP8 均可调 |
| `v41_flash_ep2` | 192 | 5120 × 2304 | Flash EP2；`fp4_prelude` |
| `v41_flash_ep8` | 48 | 5120 × 2304 | Flash EP8 |

模型差异：**同一 MegaMoEV2 多段 kernel 族**，差在 tuner（`_tune_r1` / `_tune_v4_pro` / `_tune_kimi_k3` / `_tune_v41_flash_*`）、形状、topk/act、是否 Pro 单/双 kernel 路径——**不是**另写一套「四分法」。

### DeepSeek-V4.1-Flash

- 形状：H=5120，I=2304，E=384，topk=6；EP2 → epr=192，EP8 → epr=48。
- prefill 走默认多段（prepare/stage1/stage2）。decode（每卡 ≤512 tok）在 aiter ≥ 0.1.29 并打上补丁后走 small_m 单 kernel（[megamoe_single_kernel_dsv41f.md](./megamoe_single_kernel_dsv41f.md)）；large_m 仍不可用。
- 相关 aiter PR：[aiter #6154](https://github.com/ROCm/aiter/pull/6154)（full-expert Flash MegaMoE tuning，draft）；vLLM 接线见 [#59685](https://github.com/vllm-project/vllm/pull/59685) 等。更广的 Flash PR 表见同目录 [dsv4.1f.md](./dsv4.1f.md)。

## 调参

选择入口：`select_mega_moe_config` / `build_mega_moe_bundle_plan`（`mega_moe_config.py`）。命中 profile 后跑对应 `_tune_*`。

主要旋钮：

| | 字段（示例） |
|---|---|
| **Stage1** | `sort_block_m`, `tile_n`/`tile_k`, `num_waves`, `grid_mult`, `num_dispatch_cu`, `async_a_copy`, `mfma_amajor`, `b_nt`, `prepare_quant_cu`, … |
| **Stage2** | `block_m`/`block_n`/`block_k`, `persist`/`persist_cu`, `use_nt`, `aligned_pair`, P2P quant（`none` / `fp8_blockwise_1x32` / `auto`）, … |

Benchmark：

- `op_tests/multigpu_tests/bench_mega_moe_v2.py`
- `op_tests/multigpu_tests/bench_mega_moe.py`
- 正确性：`test_mega_moe_v2.py`、`test_mega_moe_small_m.py` 等

加新网络：实现 `_tune_<name>(config, ctx)`，往 `_NETWORKS` 加一行 `NetworkProfile`。

## 环境变量（`mega_moe/envs.py`）

| 变量 | 默认 | 作用 |
|---|---|---|
| `AITER_MEGA_FIXED_SLOT_MAX_MTPR` | 255 | fixed-slot 上限 MTPR |
| `AITER_MEGA_COMBINE_MASK` | 0 | combine 跳过 top-k id `-1` |
| `AITER_MEGA_MOE_PRELOAD` | 0 | 构造时 preload AOT bundle |
| `AITER_MEGA_S1_EPI_EVEC` | 8 | stage1 epilogue 向量宽度 |
| `AITER_MEGA_S1_FIXED_KPAIR` | 1 | fixed-slot K-pair |
| `AITER_MEGA_DISPATCH_FAST_COPY` | 1 | compact dispatch fast copy |
| `AITER_MEGA_SMALL_M` | 0 | 启用 Pro small_m |
| `AITER_MEGA_LARGE_M` | 0 | 启用 Pro large_m |

## API 速查（`MegaMoEV2`）

```text
MegaMoEV2(rank, world_size, model_dim, inter_dim, experts, topk,
          quant, w1, w1_scale, w2, w2_scale, max_tok_per_rank, ...)

forward(x_bf16, wts, topk_ids, ...)           # 主入口；可能转 large_m
forward_prequant(x_q, scales, wts, topk_ids)  # 已量化 activation
forward_small_m(...)                          # 仅 small_m
quantize / bind_weights / bind_layer
preload_aot_bundles / set_large_m_enabled
```

## 和「单 kernel 融合全部」的关系

- **默认 EP MegaMoEV2**：多 launch（prepare/stage1/stage2），**不是** dispatch+gemm1+gemm2+combine 一条。
- **Pro `small_m`**：最接近全融合单 kernel。
- **Pro `large_m`**：quant 外再一条 persistent。
- **`MegaMoeTP`**：TP 场景下的另一套单/双路径融合，与 EP 四段无关。

## 一句话结论

gfx950 MegaMoE **不都是四个 kernel**。Flash / R1 / Kimi-K3 / 多数 Pro 共用 **MegaMoEV2 多段**（实际常 2～3 launch）；只有 **V4-Pro + env** 才走 1～2 kernel 的 small/large_m；模型间主要差 **NetworkProfile 与形状**，不是换另一套四分法。
