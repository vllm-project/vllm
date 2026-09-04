# ZoomKV K+V CPU Offload 启动与测速说明

本文描述 `feature/zoomkv-v020-integration` 工作树相对
`b3cb5f4d1` 的当前性能测试实现。该实现仍处于**性能测试中/实验性**
阶段，不能视为已经过生产验证。英文实现说明见
[README.md](./README.md)。

## 1. 本轮实现变化

### 1.1 K+V CPU offload 已可直接测试

- vLLM KV page 仍为 16 token，`block_size` 不变。
- offload 以 64 token logical unit（4 个 page）等待完整后异步 D2H。
- sink、local window 和正在写入的 in-flight page 始终留在 GPU。
- pinned CPU pool 同时保存完整精度 K 和 V。
- dense attention 读取前必须把 cold page 恢复到 GPU。

生命周期如下：

1. **GPU-only → warm**：完整 64-token retrieval-zone unit 异步复制到
   pinned CPU；GPU page 暂不清零，chunked prefill 的 dense FA 仍可读取。
2. **warm → cold**：进入 sparse decode 后，仅对已有 CPU 副本的
   retrieval-zone GPU K+V page 清零，不产生额外 PCIe 传输。
3. **cold sparse read**：hybrid gather 对 hot page 读 GPU，对 cold page
   直接读 mapped pinned memory，并在一个 kernel 中同时 gather K+V。
4. **cold → warm**：prefill、mixed batch、prefix-cache hit 等 dense read
   前 H2D 恢复。CPU 副本和 slot 映射保留，后续再次 cold 只需清零。
5. **free/reuse**：释放或复用 block 时同步清理 summary、offload mask、
   `physical_to_slot` 和 CPU slot。

`zoomkv_cpu_bytes_per_rank` 表示**每个 worker rank 的 pinned K+V 总预算**，
不是每层预算，也不是 K/V 各一份预算。容量耗尽时，新 block 安全地留在
GPU；不会破坏正确性，但显存节省会变少。CPU pool 只为 regular
full-attention KV cache 创建，不为 GDN/local-attention cache 创建。

### 1.2 Retrieval 已简化为单层 chunk mean

旧 hierarchical Quest parent/child 流程已删除。当前固定流程：

1. 将 GQA query heads 按 KV group 求均值。
2. 对 retrieval zone 的全部 16-token chunk 计算 `q·centroid`。
3. 保留 Top-200 chunk。
4. 前 60 个候选 chunk 各做 8-token KIVI，其余候选各做 4-token KIVI。
5. 合并后选最终 Top-100 token。
6. 最终 attention 范围为 `sink + local + retrieved Top-100`。

以下旧字段已经删除，**不能再传入**：

- `zoomkv_quest_chunk`
- `zoomkv_quest_large_chunk`
- `zoomkv_quest_large_ratio`
- `zoomkv_quest_small_ratio`
- `zoomkv_dense_ratio`

它们不会被静默兼容；旧 parent summary、Quest ratio 配置和基于旧字段的
启动命令都应清理。

### 1.3 Offload 关键性能路径

- 常驻 GPU `physical_to_slot` 映射，避免每层、每 decode step 重建整张
  slot tensor。
- direct physical retrieval，减少 logical/physical id 转换和中间操作。
- 单 kernel hybrid K+V gather，避免 K/V 分开 dispatch。
- head dimension 128/256 且 stride 满足要求时，使用 16-byte
  vectorized UVA load。
- 修复 hybrid cache 下 scheduler block id 到 physical block id
  扩展错误和复用失效。
- zero/invalidate/free hook 都会同步维护 offload cache 状态。

### 1.4 Qwen3 模型侧性能优化

当前性能树的 Qwen3 CUDA 路径直接调用 `fused_qk_norm_rope`，并让长上下文
FP32 RoPE cache 常驻，避免逐层/逐 token 转换整张 cache。需要分层定位时可
设置：

```bash
VLLM_ZOOMKV_LAYER_NVTX=1
```

这会增加 NVTX range，仅用于 profiling。

## 2. 当前性能测试参数

启动模板使用以下显式参数：

- `zoomkv_sink_size=64`
- `zoomkv_local_size=256`
- `zoomkv_chunk_size=16`
- `zoomkv_chunk_candidates=200`
- `zoomkv_dense_chunks=60`
- `zoomkv_dense_topk=8`
- `zoomkv_sparse_topk=4`
- `zoomkv_final_topk=100`
- `zoomkv_full_attention_threshold=3072`
- `zoomkv_enable_offload=true`
- `zoomkv_cpu_bytes_per_rank=25769803776`（每 rank 24 GiB，建议起点）
- `zoomkv_offload_unit_tokens=64`
- `zoomkv_strict_kernels=true`
- `zoomkv_dense_fallback=false`

代码默认 threshold 为 2000，测速模板固定传 3072，不依赖默认值。
`zoomkv_chunk_size` 必须为 16，且与 `--block-size 16` 一致；sink/local
必须能被 16 整除。`zoomkv_offload_unit_tokens` 必须是 16 的倍数。
offload 不能与 `zoomkv_dense_fallback=true` 同时使用。

`strict_kernels=true` 会在必需 kernel 不可用时直接失败；测速前先构建并
确认扩展：

```bash
cmake -S . -B build -DVLLM_BUILD_ZOOMKV_EXT=ON
cmake --build build --target _zoomkv_C
python -c "import vllm._zoomkv_C"
```

## 3. 启动方式

直接使用当前 offload 性能模板：

```bash
bash examples/features/zoomkv/serve_zoomkv_qwen36_example.sh
```

GPU、模型、TP、context 和 CPU pool 均可通过环境变量覆盖：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MODEL_PATH=/path/to/Qwen3.6-27B \
TENSOR_PARALLEL_SIZE=2 \
MAX_MODEL_LEN=131072 \
ZOOMKV_CPU_BYTES_PER_RANK=$((24 * 1024**3)) \
bash examples/features/zoomkv/serve_zoomkv_qwen36_example.sh
```

脚本使用 JSON encoder 构造 `--attention-config`，并校验
`ZOOMKV_CPU_BYTES_PER_RANK` 是正整数字节数，环境变量不会拼出非法 JSON。
不要再额外传 `--attention-backend`。

## 4. 禁用和对照

为了让对照有意义，固定同一 prompt、TP、batch size、输出 1024 token、
NUMA 绑定和 warmup：

- **GPU-only sparse 对照**：保持其他字段不变，仅设
  `zoomkv_enable_offload=false`。
- **Dense 对照**：先关闭 offload，再设 `zoomkv_dense_fallback=true`。
- **路由检查**：序列长度小于 threshold 时本来就走 dense；测速 prompt
  必须超过 3072。

不要用旧 Quest ratio 试图“对齐”GPU-only，它们已经不存在。

## 5. Benchmark 与 recall

GPU-only 端到端：

```bash
python examples/features/zoomkv/benchmark_zoomkv_gpu_only.py \
  --model /path/to/Qwen3.6-27B \
  --mode sparse \
  --tensor-parallel-size 2 \
  --threshold 3072 \
  --output-tokens 1024 \
  --runs 3 \
  --output-json /tmp/zoomkv-gpu-only.json
```

测试 K+V CPU offload 时，在相同命令中增加：

```bash
  --enable-offload \
  --cpu-bytes-per-rank $((24 * 1024**3))
```

128K retrieval 聚焦 profiling：

```bash
python examples/features/zoomkv/profile_retrieval_128k.py --help
```

当前 recall CLI 正确参数如下；不再支持 Quest ratio：

```bash
python examples/features/zoomkv/measure_topk_recall.py \
  --model /path/to/Qwen3.6-27B \
  --tensor-parallel-size 2 \
  --threshold 3072 \
  --chunk-candidates 200 \
  --dense-chunks 60 \
  --dense-topk 8 \
  --sparse-topk 4 \
  --final-topk 100 \
  --output-json /tmp/zoomkv-recall.json
```

首轮包含 JIT/CUDA Graph 等初始化，不计入稳态数据。`recall` probe 会同步
GPU，只用于调试。JSON、日志、trace、profiler 结果应写到 `/tmp` 或外部
结果目录，不写入仓库。

## 6. 2026-09-04 阶段性性能基线

同机、NUMA0、TP=2、BS=1、decode=1024 的临时数据：

- GPU-only：64K TPOT 14.03 ms；128K TPOT 14.28 ms。
- offload UVA vectorized：64K TPOT 27.72 ms；128K TPOT 33.70 ms。

当前 offload **明显慢于 GPU-only**。这些数字仅是当日优化过程的阶段性
基线，不能外推到其他机器、batch、NUMA 或模型，也不能表述为生产结论。

## 7. 限制

- 当前仅支持 NVIDIA CUDA、FP16/BF16 KV、16-token page。
- sparse 路径面向 single-token long-context decode。
- GPU-only mixed batch 可让满足阈值的 decode prefix 走 sparse、prefill
  suffix 走 dense；offload mixed batch 会恢复 cold page 并走 dense。
- multi-token/speculative decode 和不支持的 mixed shape 走 dense。
- KV connector 不支持。
- offload 不参与 full CUDA Graph capture。
- 16-byte vectorized UVA 路径仅覆盖 head dimension 128/256 和匹配 stride；
  其他布局会走标量/非向量化 fallback。
- pinned 内存大小和 NUMA 放置会直接影响性能；24 GiB/rank 只是建议起点。
- 当前处于性能测试中，尚未完成生产稳定性和质量验证。

## 8. 实现与测试索引

- 参数：`vllm/config/attention.py`
- 路由：`vllm/v1/attention/backends/zoomkv_attn.py`
- retrieval：`vllm/v1/attention/ops/zoomkv/retriever.py`
- summary/state：`vllm/v1/attention/ops/zoomkv/state.py`
- CPU pool/lifecycle：`vllm/v1/attention/ops/zoomkv/offload.py`
- paged gather/attention：`vllm/v1/attention/ops/zoomkv/paged.py`
- kernel dispatch：`vllm/v1/attention/ops/zoomkv/kernels.py`
- hybrid K+V gather：
  `vllm/v1/attention/ops/zoomkv/cuda/h2d_gather_tokens.cu`
- direct physical retrieval：
  `vllm/v1/attention/ops/zoomkv/cuda/physical_retrieval.cu`
- 构建：`cmake/zoomkv.cmake`
- 单测：`tests/v1/attention/zoomkv/test_zoomkv_ops.py`
- 128K profiler：`examples/features/zoomkv/profile_retrieval_128k.py`

轻量 CPU 测试：

```bash
pytest -q tests/v1/attention/zoomkv/test_zoomkv_ops.py -k cpu
```
