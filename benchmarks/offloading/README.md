# OffloadingConnector E2E Benchmark 使用说明

当前完整实测汇总位于
`benchmark-results/all-offloading-benchmark-results.md`，包含端到端性能、短
prefill CUDA Graph 曲线、cache source、传输字节数和稳定性统计。
使用 `--enforce-eager` 重跑的 400 个样本及独立审计结果位于
`benchmark-results/all-offloading-benchmark-results-eager.md`。两份报告不能
直接混用：Eager 模式同时关闭 Inductor compilation 和 CUDA Graph。
IQuest MTP、async scheduling 和 decode offload 的组合验证位于
`benchmark-results/offloading-mtp-async-smoke.md`。

本工具用于比较两条完整的 first-token 路径：

- `recompute`：请求无 HBM/external cache 命中，完整计算 prompt KV。
- `offload`：先计算并保存 target KV，再将其从 HBM 淘汰，最后从 CPU KV
  cache 加载相同请求。

统一入口为 `bench_offloading_e2e.py`。TTFT 样本使用 `/v1/completions`，每个请求
设置 `max_tokens=1`。Offload 模式还会默认执行一次独立的 decode-block 保存校验，
它不计入 TTFT 样本。

## 指标口径

- `server_ttft_seconds`：从请求到达 engine 到第一个输出 token，包含完整
  recompute 或异步 CPU-to-GPU load，是主要对比指标。
- `client_e2e_seconds`：客户端发送 HTTP 请求到收到完整响应，包含 TTFT、HTTP
  和序列化开销。
- `server_prefill_seconds`：vLLM 从 `SCHEDULED` 到 first token。异步 connector
  在 load 完成后才记录 `SCHEDULED`，因此该指标不包含 CPU-to-GPU 等待，不能
  作为 OffloadingConnector 的完整 load 时间。
- eviction 请求只用于构造 HBM 淘汰，不计入 replay latency。

## 前置条件

1. 测试期间 server 不应有其他流量。脚本使用 Prometheus counter delta 校验
   单个请求，其他请求会污染 TTFT 和 token-source 指标。
2. Recompute 与 offload server 必须使用相同模型、代码版本、TP/PP/DP、HBM
   block 数、`max_model_len` 和请求参数。
3. Recompute server 不配置 KV connector，但保持 prefix cache 开启。脚本通过
   每次生成不同的 `cache_salt` 保证 cold miss。
4. Offload server 配置 `OffloadingConnector`，例如：

```bash
--num-gpu-blocks-override 4352 \
--kv-transfer-config '{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "cpu_bytes_to_use": 68719476736,
    "offload_prompt_only": false
  }
}'
```

上例 HBM KV 容量为 `4352 * 16 = 69632` tokens，CPU cache 为 64 GiB。
Recompute server 使用相同的 `--num-gpu-blocks-override 4352`，但删除整个
`--kv-transfer-config`。

`cpu_bytes_to_use` 是每个 worker rank 的上限，不是整个 vLLM 实例的总量。
例如 TP=8 且每个 rank 配置 64 GiB，主机侧理论上最多需要约 512 GiB，再加上
模型加载、进程和 pinned-memory 等额外开销。

## 完整 Server 命令

以下所有命令都从 vLLM 仓库根目录执行。确保 `vllm` 位于 `PATH`，并在启动前将
`MODEL_PATH` 设置为待测试模型目录。下面两组只允许同时启动一组，并保持除
connector 外的参数完全一致。

Recompute server：

```bash
vllm serve "$MODEL_PATH" \
  --port 9999 \
  --host 0.0.0.0 \
  --served-model-name M1-0710 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 80000 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.9 \
  --num-gpu-blocks-override 4352 \
  --enable-prefix-caching \
  --load-format instanttensor \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser iquest_coder_v2 \
  --reasoning-parser iquest_coder_v2 \
  --nnodes 1 \
  --node-rank 0 \
  --uvicorn-log-level info \
  --enable-log-requests
```

OffloadingConnector server：

```bash
vllm serve "$MODEL_PATH" \
  --port 9999 \
  --host 0.0.0.0 \
  --served-model-name M1-0710 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 80000 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.9 \
  --num-gpu-blocks-override 4352 \
  --enable-prefix-caching \
  --load-format instanttensor \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser iquest_coder_v2 \
  --reasoning-parser iquest_coder_v2 \
  --nnodes 1 \
  --node-rank 0 \
  --uvicorn-log-level info \
  --enable-log-requests \
  --kv-transfer-config '{
    "kv_connector":"OffloadingConnector",
    "kv_role":"kv_both",
    "kv_connector_extra_config":{
      "cpu_bytes_to_use":68719476736,
      "offload_prompt_only":false
    }
  }'
```

启动后先确认 `curl -f http://127.0.0.1:9999/health` 成功，再运行 client。

如需 Eager 对照实验，在 Recompute 和 OffloadingConnector 两条 server 命令中
同时添加 `--enforce-eager`，其余参数保持完全一致。标准 vLLM 不导出逐请求
CUDA Graph runtime mode 的 Prometheus counter，因此结果会将该字段明确记录为
`unavailable`；server 启动参数仍需随测试结果一同保存。

## 推荐完整流程

### 1. Recompute 组

先启动无 connector server，然后执行：

```bash
python3 benchmarks/offloading/bench_offloading_e2e.py \
  --mode recompute \
  --host 127.0.0.1 \
  --port 9999 \
  --model M1-0710 \
  --sizes 256 512 1024 2048 4096 8192 16384 32768 65536 \
  --repeats 10 \
  --max-attempts 30 \
  --settle-seconds 0.2 \
  --output benchmark-results/recompute.json
```

每个有效 recompute 样本必须满足：

```text
local_compute_tokens   = context length
local_cache_hit_tokens = 0
external_kv_tokens     = 0
```

兼容入口也可以使用：

```bash
python3 benchmarks/offloading/bench_recompute_vs_kv_load.py \
  --host 127.0.0.1 \
  --port 9999 \
  --model M1-0710 \
  --sizes 256 512 1024 2048 4096 8192 \
  --repeats 10 \
  --max-attempts 30 \
  --output benchmark-results/recompute-short.json
```

### 2. OffloadingConnector 组

关闭 recompute server，使用相同公共参数启动 OffloadingConnector server。
对于 69,632-token HBM KV 容量，使用一条 69,616-token eviction 请求：

```bash
python3 benchmarks/offloading/bench_offloading_e2e.py \
  --mode offload \
  --host 127.0.0.1 \
  --port 9999 \
  --model M1-0710 \
  --sizes 256 512 1024 2048 4096 8192 16384 32768 65536 \
  --repeats 10 \
  --max-attempts 30 \
  --eviction-token-counts 69616 \
  --block-size 16 \
  --settle-seconds 0.2 \
  --output benchmark-results/offload.json
```

每个 sample 的执行顺序是：

1. Cold target，使用新的 target salt。
2. 按 `--eviction-token-counts` 顺序发送 eviction 请求，每条使用新的 salt 和
   不同的 repeated token ID。
3. Replay target，prompt 和 salt 与 cold target 完全相同。
4. 校验 cold、每条 eviction 和 replay 的 Prometheus counter delta。

正式采样前，默认的 `--verify-decode-blocks 4` 还会执行两条独立请求：

1. `1 prompt token + 17 decode tokens`，标定一个完整 block 的 GPU-to-CPU
   transfer 操作数和字节数。
2. `1 prompt token + 65 decode tokens`，要求操作数和字节数均严格等于标定值的
   4 倍。

额外的一个 decode token 保证每个目标 block 填满后还有下一 engine step，可将
延迟的 store job 真正提交，而不会把它遗留到下一条请求。上述两条请求的已计算
KV token 数分别是 `1 + 17 - 1 = 17` 和 `1 + 65 - 1 = 65`，其中完整 block
数恰好为 1 和 4。该校验可以消除 TP 每个 rank 分别上报 transfer 的固定倍数，
并验证 decode 每跨过一个完整 offload block 都产生一次保存任务。若 server 使用
`offload_prompt_only=true`，
由于 prompt 只有一个 token，该校验会失败。使用 `--verify-decode-blocks 0` 可以
显式关闭；建议正式结果不要关闭。

这里的“触发保存”不是在填满 block 的同一个 forward 内同步 memcpy。scheduler
在该 step 识别新完整 block，worker 将任务延迟到下一 engine step 开始时异步
提交，从而避开当前 step 的 sampling transfer；请求结束或 block 被复用前会执行
必要的完成等待。

启用 speculative decoding 时，每个 decode step 可能一次产生多个 accepted
tokens。此时需要将校验请求越过 block 边界的安全尾部设为
`num_speculative_tokens + 1`。例如 MTP 使用两个 speculative tokens 时添加：

```bash
--verify-decode-tail-tokens 3
```

校验器会在每条目标请求后自动发送一条不会形成完整 block 的 drain 请求，使
延迟到下一 engine step 汇报的异步 store 指标归属到正确样本。

对于 context 长度是 block size 整数倍的请求，有效 replay 必须满足：

```text
local_cache_hit_tokens = 0
external_kv_tokens     = context length
local_compute_tokens   = 1
cpu_to_gpu_bytes       > 0
cold output            = replay output
```

对于非 block 对齐的长度，external hit 预期为完整 block 数，剩余 token 本地
计算。例如 block size 为 16、context 为 2050 时，预期 external=2048、
local_compute=2。

### 3. 非 Block 对齐的 Mixed Replay 组

这组只需要增加 OffloadingConnector 测试，不需要增加 recompute 样本。固定增加
8 tokens，既避开 aligned request 为生成 logits 而保留的 1-token 特例，也能稳定
验证同一个请求同时执行 external KV load 和少量本地计算：

```bash
python3 benchmarks/offloading/bench_offloading_e2e.py \
  --mode offload \
  --host 127.0.0.1 \
  --port 9999 \
  --model M1-0710 \
  --sizes \
    256 512 1024 2048 4096 8192 16384 32768 65536 \
    264 520 1032 2056 4104 8200 16392 32776 65544 \
  --repeats 10 \
  --max-attempts 30 \
  --eviction-token-counts 69616 \
  --block-size 16 \
  --settle-seconds 0.2 \
  --output benchmark-results/offload-mixed.json
```

新增九档的严格预期为：

| Context | External KV tokens | Local compute tokens | HBM hit tokens |
|---:|---:|---:|---:|
| 264 | 256 | 8 | 0 |
| 520 | 512 | 8 | 0 |
| 1,032 | 1,024 | 8 | 0 |
| 2,056 | 2,048 | 8 | 0 |
| 4,104 | 4,096 | 8 | 0 |
| 8,200 | 8,192 | 8 | 0 |
| 16,392 | 16,384 | 8 | 0 |
| 32,776 | 32,768 | 8 | 0 |
| 65,544 | 65,536 | 8 | 0 |

使用 aligned recompute 文件和 `offload-mixed.json` 生成报告时，公共 aligned
context 仍用于 TTFT/E2E speedup；新增九档会进入 `Offload-Only Mixed Replay
Validation` 表，只报告 cache source 和传输字节，不计算没有严格对照组的 speedup。

### 4. 生成对比报告

```bash
python3 benchmarks/offloading/summarize_offloading_benchmark.py \
  --recompute benchmark-results/recompute.json \
  --offload benchmark-results/offload.json \
  --output benchmark-results/comparison.md
```

工具会拒绝以下输入：

- 任一 benchmark 未完成。
- mode 不正确。
- 模型、DP engine、prompt token、block size 或 `max_tokens` 不一致。
- 两个结果没有共同 context size。

## 短上下文配置

如果 HBM 被限制为 768 blocks，即 12,288 tokens，可以使用两条 12K eviction：

```bash
python3 benchmarks/offloading/bench_offloading_e2e.py \
  --mode offload \
  --host 127.0.0.1 \
  --port 9999 \
  --model M1-0710 \
  --sizes 256 512 1024 2048 4096 8192 \
  --repeats 10 \
  --max-attempts 30 \
  --eviction-token-counts 12000 12000 \
  --output benchmark-results/offload-short.json
```

旧参数仍受支持，下面两种写法等价：

```text
--eviction-token-counts 12000 12000
--eviction-tokens 12000 --eviction-requests 2
```

## 短 Prefill 计算曲线

需要验证少量本地计算 token 的真实 prefill 时间时，在无 connector server 上
使用同一进程完成预热后运行：

```bash
python3 benchmarks/offloading/bench_offloading_e2e.py \
  --mode recompute \
  --host 127.0.0.1 \
  --port 9999 \
  --model M1-0710 \
  --sizes 8 16 32 64 128 256 \
  --repeats 30 \
  --max-attempts 30 \
  --settle-seconds 0.2 \
  --output benchmark-results/recompute-short-prefill.json
```

解释这组数据时必须同时记录 CUDA Graph 配置。默认
`max_cudagraph_capture_size=min(max_num_seqs * 2, 512)`；例如
`max_num_seqs=64` 时阈值为 128。scheduler 单步计算 token 数超过该阈值后，
dispatcher 返回 `CUDAGraphMode.NONE`，因此 128→256 可能出现执行路径切换，
不能将该跳变解释成纯 FLOPs 线性增长。

benchmark 默认不依赖额外的 CUDA Graph Prometheus counter。若 server 未提供
`vllm:cudagraph_dispatch_total{runtime_mode=...}`，JSON 和汇总报告会将执行模式
记录为 `unavailable`，而不是错误地记成未使用 CUDA Graph。只有在测试的 server
分支确实导出该 counter 时，才使用 `--require-cudagraph-metrics` 强制逐请求校验；
该选项不是 OffloadingConnector 正确性或性能测试的前置条件。

一 token 请求存在 source counter 的特殊记账：可能同时上报一个 local compute
和一个 local cache hit。需要严格互斥 source accounting 时，应从 8 tokens 开始。

## Checkpoint 和断点续跑

脚本在以下时机原子更新 output JSON：

- 创建 run。
- 每个有效或无效 attempt 结束。
- 每个 context 完成。
- 完成、失败或 Ctrl-C 中断。

中断后使用相同参数并增加 `--resume`：

```bash
python3 benchmarks/offloading/bench_offloading_e2e.py \
  --mode offload \
  --host 127.0.0.1 \
  --port 9999 \
  --model M1-0710 \
  --sizes 16384 32768 65536 \
  --repeats 10 \
  --max-attempts 30 \
  --eviction-token-counts 69616 \
  --output benchmark-results/offload.json \
  --resume
```

可以在 resume 时增加 `--repeats` 或 `--max-attempts`。影响请求语义或校验的
参数必须与原 run 一致，否则脚本拒绝 resume。

如果确实需要覆盖已有结果，显式使用 `--overwrite`。默认不会覆盖文件。

## Output JSON

Schema version 为 2。顶层关键字段：

```text
status       running / completed / interrupted / failed
mode         recompute / offload
run_id       本轮稳定 ID
config       完整 client 配置和 eviction plan
results      每个 context 的 attempts、有效样本数和统计
decode_block_verification  decode 完整 block 增量保存校验（仅 offload）
```

每个 attempt 都会保留：

- `valid` 和全部 `validation_errors`。
- 请求 salt、prompt token、request ID 和输出。
- client E2E、server TTFT、server prefill。
- local compute、HBM hit、external hit。
- CPU-to-GPU 和 GPU-to-CPU counter delta。
- CUDA Graph runtime mode、各 mode 的 dispatch 次数和是否实际使用 CUDA Graph。
- offload 模式下的 cold、每条 eviction 和 replay 完整记录。
- decode 校验的一 block 标定值、多 block 实测值、操作数、字节数和校验错误。

脚本还会自动读取 `/version` 和 `/v1/models`，在顶层 `server` 字段保存 vLLM
版本、served model ID、模型根路径和 `max_model_len`。Resume 时 server metadata
必须与原 run 一致；生成对比报告时 recompute/offload 的 metadata 也必须一致。

统计只使用 `valid=true` 的 attempt，包含 count、mean、median、P95、min、max、
标准差和 CV。

## 常见问题

### Replay 仍有 HBM hit

增加 eviction 总量或减小 `--num-gpu-blocks-override`。脚本会将该 attempt 标记
为无效并保存实际 `local_cache_hit_tokens`，不会把它计入统计。

### Replay external hit 不完整

常见原因是 CPU cache 太小，target 在 replay 前已被 CPU LRU 淘汰。CPU cache
至少需要同时容纳 target 和当前 eviction plan 的 KV，并预留 allocator 开销。

### Eviction 请求无法调度

不要让 eviction prompt 加上 lookahead slot 恰好超过 HBM block 容量。例如
69,632-token HBM 容量下，69,632-token prompt 配合 `max_tokens=1` 可能停在
waiting；本次验证使用 69,616 tokens。

### `cpu_to_gpu_bytes` 指标不存在

默认要求 replay 的 CPU-to-GPU counter delta 大于 0。旧分支没有该指标时可以
使用 `--no-require-transfer-bytes`，但仍会严格校验 external token hit。

### 重复 token 是否代表真实业务

默认 prompt 是同一个 token ID 重复 N 次，这能精确控制长度和 cache key。
对于 MoE 模型，token 内容可能影响 expert routing，因此建议在确认 connector
链路后，再使用固定真实语料做生产代表性测试。

## 手动单请求验证

需要逐条执行 cold、eviction、replay 时，使用：

```bash
python3 benchmarks/offloading/send_one_offloading_request.py --help
```

该工具每次只发送一条 completion 请求并打印该请求的 counter delta。

## 开发者测试

本工具的定向单元测试命令是：

```bash
python3 -m unittest -v tests.benchmarks.test_offloading_e2e
python3 -m py_compile \
  benchmarks/offloading/bench_offloading_e2e.py \
  benchmarks/offloading/bench_recompute_vs_kv_load.py \
  benchmarks/offloading/send_one_offloading_request.py \
  benchmarks/offloading/summarize_offloading_benchmark.py \
  tests/benchmarks/test_offloading_e2e.py

python3 -m pytest -q \
  tests/v1/kv_connector/unit/test_offloading_connector.py
```

不要在仓库根目录使用不带目标的 `python3 -m unittest discover`。该命令会收集
整个 vLLM 测试树，其中大量测试要求 `pytest`、CUDA driver、LMCache、Helion、
IPEX 等额外环境，与本 benchmark 的测试结果无关。

真实 GPU 集成校验至少应包含：

1. decode block 1→4 倍验证通过。
2. cold target、所有 eviction 请求均为 local compute，cache hit 为 0。
3. replay 的 `local_cache_hit_tokens=0`。
4. replay 的 external token 数等于完整 block 覆盖的 context。
5. CPU-to-GPU bytes 大于 0，cold/replay 输出相同。
6. output JSON 为 `status=completed`，报告生成器能够成功读取。

2026-07-27 的 M1-0710、TP=8、block size=16 实测中，一 block decode store 为
8 次 rank-level 操作、5,767,168 bytes；四 block 为 32 次操作、23,068,672
bytes。2K replay 实测 HBM hit=0、external hit=2048、CPU-to-GPU bytes 为
738,197,504；1K replay 对应 external hit=1024、CPU-to-GPU bytes 为
369,098,752；256/512-token replay 分别对应 92,274,688 和 184,549,376 bytes。
完整 256/512/1K/2K/4K/8K/16K/32K/64K 正式数据仍应分别在无 connector
和 OffloadingConnector 两个 server 上按“推荐完整流程”重新采集，不能用单次
smoke 数值替代多次稳定性结果。
