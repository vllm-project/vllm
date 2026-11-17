# DeepSeek-V4.1-Flash：decode 走 AITER 单 kernel MegaMoE（small_m）

2026-11-17，8× MI355X，DP8+EP8，`--moe-backend aiter_mega_moe`，DSpark nst5。

## 改动

decode 时（DP 间每卡最大 token 数 ≤ 512），MegaMoE 改走 AITER 的单 kernel 版本 `SmallMRuntime`：量化、dispatch、GEMM1、GEMM2、combine 在一次 launch 里完成。更大的 batch（prefill）仍走 `MegaMoEV2` 的多段 kernel。

- `vllm/models/deepseek_v4/amd/mega_moe.py`
  - `finalize_weights` 之后，另外构造容量 256 和 512 的 `SmallMRuntime`，所有层共享。每层 W2 的 dispatch 表在 CUDA graph 捕获前就建好（单 kernel 把 W2 指针放在这张表里，捕获期间不能新建）。
  - `forward` 按 DP 间最大 token 数选路径，所有 EP rank 走同一条路径。换层用 `bind_weights`。
  - 新增 `uses_single_kernel(num_tokens)`。
- `vllm/models/deepseek_v4/amd/model.py`：用单 kernel 时，共享专家不放到辅助流上，改为串行执行（见"已知问题"）。
- `vllm/envs.py`：新增 `VLLM_ROCM_AITER_MEGA_MOE_SINGLE_KERNEL`，默认 1，设为 0 回到多段 kernel。

以下情况自动回退到多段 kernel：
- AITER 没有单 kernel（< 0.1.29，例如本分支镜像自带的 0.1.28.post1）；
- AITER 的 `small_m_eligible` 不接受当前形状；
- 融合了共享专家（每卡 49 个专家）。

## 依赖：AITER ≥ 0.1.29 加一个补丁

AITER 0.1.29 的单 kernel 只对 DeepSeek-V4-Pro 的形状（7168 / 3072）开放。V4.1-Flash 需要 [`patches/aiter-0.1.29-mega_moe-small_m-dsv41-flash.patch`](./patches/aiter-0.1.29-mega_moe-small_m-dsv41-flash.patch)：

1. `small_m_eligible` 允许 5120 / 2304；
2. `small_m_config`：每卡 ≤ 8 token 的档位默认 `tile_n=256`，V4.1 上 GEMM1 只有 18 个 N-tile，触发 `assert N_TILES % 4 == 0`。改为退回 `tile_n=128`（36 个 N-tile，和其他档位一致）。

V4.1 不需要单独的调优配置：单 kernel 的配置规则与形状无关，所有可调的配置项和阈值扫描下来，默认值都在最优的 1% 以内。

不打这个补丁时，本 PR 对 V4.1 不生效，行为和改动前一样。

## 实测

8× MI355X，`vllm/vllm-openai-rocm:nightly-9bc16ae6a34b1e0fe260e8d12b98568478bf38c3`（AITER 0.1.29），在镜像上覆盖本 PR 的 3 个文件和上面的 AITER 补丁。

测法：真实文本 prompt，512 个请求一次性并发（每卡 64），temperature 0，`ignore_eos`，输出 8K。ITL 是同一请求相邻两个流式 chunk 的间隔（即每步间隔），取纯 decode 窗口内的值，2 次运行平均。单卡吞吐 = 512 × 前 1K 输出的接受长度 / ITL / 8。

| | 1K ITL 平均 / p99 (ms) | 1K 单卡 tok/s | 100K ITL 平均 / p99 (ms) | 100K 单卡 tok/s | gsm8k（1319 题） |
|---|---|---|---|---|---|
| 镜像 MegaMoE（改动前） | 38.92 / 49.42 | 6,026 | 40.35 / 52.75 | 5,805 | 0.907 |
| **本 PR** | **33.67 / 43.67** | **6,927（+15.0%）** | **35.05 / 46.45** | **6,637（+14.3%）** | 0.909 |
| 本 PR，AITER 不打补丁（回退） | 38.77（1 次） | — | — | — | — |

叠加在其他优化上（zovonoir 的 dsv41f-dp8ep8-opt 补丁 + dense GEMM 优化 + `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=1024`）：1K ITL 36.62 → 32.44 ms，单卡 6,362 → 7,192 tok/s；100K ITL 37.99 → 33.79 ms，单卡 6,124 → 6,904 tok/s；gsm8k 0.904。

- **单 kernel 本身**（8 卡单测，随机 FP4 权重，CUDA graph 计时，每卡 384 token，均匀路由）：多段 kernel 447 µs，单 kernel 282 µs。
  - 和 FP32 参考对比，单 kernel 误差 0.25%，多段 kernel 2.66%。原因是多段 kernel 在容量 16384 时把 GEMM2 的结果按 FP8 传回 combine，单 kernel 传的是 BF16。
  - 同样输入跑两次，结果逐位相同。每卡 0～512 token、各 rank token 数不同、全部 rank 空闲等情况都测过。
- **模型 trace**（1K，典型层）：MegaMoE 每层从 490 µs 降到 374 µs，每步周期从 36.9 ms 降到 32.8 ms。

## 启动命令

```bash
docker run -d --name dsv41f --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v <MODEL_DIR>:/models/DeepSeek-V4.1-Flash:ro \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=1 \
  -e VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 -e VLLM_USE_RUST_FRONTEND=1 \
  -e MORI_SHMEM_HEAP_SIZE=16G -e OMP_NUM_THREADS=1 \
  --entrypoint sleep <IMAGE: vLLM with this PR + AITER >= 0.1.29 with the patch above> infinity

docker exec -d dsv41f vllm serve /models/DeepSeek-V4.1-Flash \
  --host 0.0.0.0 --port 8000 --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --data-parallel-size 8 --tensor-parallel-size 1 --enable-expert-parallel \
  --all2all-backend mori_high_throughput --moe-backend aiter_mega_moe \
  --language-model-only --tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice --reasoning-parser deepseek_v41 \
  --gpu-memory-utilization 0.9 \
  --attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --block-size 128 --max-model-len 131072 --no-swa-bounded-replay \
  --engram-config '{"cpu_offload":true}' \
  --max-cudagraph-capture-size 384 --max-num-batched-tokens 16384 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'
```

启动日志里出现 `AITER MegaMoE: batches of up to 512 tokens per rank use the single-kernel path`，说明单 kernel 已启用。

## 已知问题

- **不能和辅助流上的共享专家同时运行。** 单 kernel 要求整个 grid 同时驻留在 GPU 上（每 CU 2 个 CTA）。共享专家在辅助流上并行时，第一次 CUDA graph 捕获就会挂死：GPU 100%，没有报错。所以用单 kernel 时共享专家改为串行。单 kernel 本来就占满所有 CU，并行也没有收益。具体是哪个 kernel 形成了循环等待，还没有查清。
- **草稿模型用不上单 kernel。** DSpark 草稿的 MoE 是 128 个专家（每卡 16 个）、top-3，单 kernel 只支持 384 个专家、top-6、每卡 48 个专家。
- **EPLB 没有收益。** 冗余专家会让每卡超过 48 个专家，单 kernel 不支持。不加冗余专家的 EPLB 只能移动专家，拆不开单个热点专家：实测 p50 只降 0.2 ms，平均值还因为重排停顿变差。

## 下一步

| 方向 | 推算可省（ms/步） | 依据 |
|---|---|---|
| 单 kernel 更好地处理热点专家：把热点专家的行拆给更多 CTA；或支持每卡 48 个以外的专家数，以便用冗余专家 | 最多约 3.6 | 模型里 374 µs/层，均匀路由下 282 µs/层 |
| 共享专家放进单 kernel：参考 DeepGEMM，给每个 token 加一个只在本卡计算的共享槽位，权重保持 FP8（B300 上的 `sm100_fp8_fp4_mega_moe_impl` 就是这样做的，`kNumSharedExperts=1`） | 约 1.9 | 共享专家串行，每层 48 µs |
| 草稿 MoE（每卡 16 个专家、top-3）的单 kernel 或小容量路径 | 约 1 | 草稿 MegaMoE 每次约 400 µs；B300 是 115 µs |
| 把 AITER 补丁提交到 ROCm/aiter | — | 去掉对本地补丁的依赖 |
| 查清和辅助流同时运行时挂死的原因 | — | 现在只是靠串行绕过 |

对比 B300（同一个 vLLM commit，DeepGEMM MegaMoE，已融合共享专家）：1K ITL 20.30 ms，单卡 11,413 tok/s；MegaMoE 加共享专家每层 243 µs，MI355X 是 374 + 48 = 422 µs。
