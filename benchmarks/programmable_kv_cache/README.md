# Programmable KV Cache 测试命令

- [分支实现与验证结果](../../docs/programmable-kvc/testing.md)
- [已执行实验报告](../../docs/programmable-kvc/rollout-report-2026-10-12.md)
- [控制开销与扩大工作集实验](../../docs/programmable-kvc/rollout-report-2026-10-18.md)
- [暂停时间与准入协同实验](../../docs/programmable-kvc/rollout-report-2026-10-19.md)
- [当前与历史证据索引](../../openspec/changes/programmable-kv-cache/validation.md)
- [原语与生命周期契约](../../vllm/v1/kv_hints/README.md)

以下命令从仓库根目录执行，Python 使用 `.venv/bin/python`。
本地只使用 GPU 0、3；性能、模型正确性和其他回归安排在独立窗口。
模型/数据路径可替换为对应本地副本。
`results/` 保存本地实验输出，已加入 `.gitignore`，原始轨迹、日志和采样文件不随代码提交。
复现命令中的 `--trace` 指向本地已有轨迹；新环境需用下面的在线采集命令生成轨迹，并替换该路径。

## 在线 rollout 与重复性能对照

`verl_rollout.py` 使用真实 verl ToolAgentLoop、题目查询与算术工具，保存 token 对数概率、响应掩码、奖励分数和控制结果。
需要可导入的 verl 仓库及其 Agent Loop 依赖；本轮使用的提交为 `fbb4b3a8bf636f290c9c59fc346f756849e9c241`。

实验设置与计时说明见[实验报告](../../docs/programmable-kvc/rollout-report-2026-10-12.md)。

对已配置的服务采集在线轨迹：

```bash
PYTHONPATH=../verl \
  CUDA_VISIBLE_DEVICES=0,3 \
  OMP_NUM_THREADS=1 \
  .venv/bin/python -m benchmarks.programmable_kv_cache.verl_rollout \
  --base-url http://127.0.0.1:18083 \
  --model /home/youwei/bzh/model/Qwen/Qwen3.8-27B \
  --dataset /home/youwei/bzh/dataset/openai/gsm8k \
  --mode native \
  --prompts 8 \
  --samples 4 \
  --families 8 \
  --fewshot 48 \
  --concurrency 16 \
  --max-tokens 256 \
  --temperature 0.6 \
  --seed 0 \
  --output /tmp/kv-live-trace.json.gz
```

`run_rollout_matrix.py` 创建并清理本次测试的服务、Mooncake Store 和 GPU 采样进程；所需端口被占用时拒绝启动。
它排除启动与显式预热时间，检查缓存重置成功，并记录配置、命令、源码摘要和逐轮数据。
主要性能对照默认采用 `gpu_memory_utilization=0.5`，不覆盖 KV 字节预算。
省略 `--trace` 时，工具先运行真实在线工具循环采集轨迹；采集时间不计入后面的固定工作量回放。
以下配置包含 256 条 rollout 和 32 个前缀家族，实际缓存压力以日志与指标为准：

```bash
.venv/bin/python -m benchmarks.programmable_kv_cache.run_rollout_matrix \
  --model /home/youwei/bzh/model/Qwen/Qwen3.8-27B \
  --dataset /home/youwei/bzh/dataset/openai/gsm8k \
  --gpus 0,3 \
  --gpu-memory-utilization 0.5 \
  --prompts 64 --samples 4 --families 32 --fewshot 48 \
  --concurrency 64 --max-num-seqs 64 \
  --arms native,idle,noop,policy \
  --repeats 3 \
  --output-dir /tmp/kv-deployment-comparison
```

| 参数/模式 | 用途 |
| --- | --- |
| `native` / `idle` / `noop` | 原生 APC；仅启用控制器；同时启用控制器和连接器。后两项不发送 hint |
| `priority` / `movement` / `composed` | 应用端显式保留、搬运、二者组合 |
| `policy` | 信号驱动的本地策略与调度 |
| `--policy-components quota` | 仅启用配额调度；不加载搬运连接器，不发送无用关联与工具信号 |
| `--policy-components retention` | 通过 importance 信号触发保留策略，并检查真实优先级操作；不加载搬运连接器。可同时启用 quota |
| `--policy-max-wait-seconds` | 策略准入/次序延期上限，默认 30 秒；同时设置 quota 自身等待阈值，实验时需报告尾延迟 |
| `--quota-aging-seconds` / `--quota-reuse-score-weight` | quota 等待老化时间尺度和缓存复用评分权重，默认 1 秒和 0.5 |
| `--policy-decode-lookahead-tokens` | quota 估计运行请求后续 KV 需求的 token 上限，默认 512 |
| `--gpu-memory-utilization 0.5` | 默认主要性能预算，各对照记录实际分配出的 KV 容量 |
| `--cache-gib 2` | 历史人工配额压力诊断；不要当作正常资源主基线 |
| `--live-after-replay` | 缓存重置后按 trace 的原在线配置运行真实工具循环，结果单列 |
| `--trace` | 固定原输入、输出长度和暂停，重新生成；不执行工具或报告回放奖励分数 |
| `--tool-delay` / `--replay-pause-floor` | 明确标记的人为工具等待；不能代替真实服务延迟 |
| `--prediction-mode adaptive` | 默认使用已完成工具批次更新预测；在线按工具组合区分，旧 trace 无工具名时使用统一估计 |
| `--prediction-mode fixed` / `--prediction-seconds` | 固定预测对照；`prediction-seconds` 也作为自适应估计的初始值 |
| `--oracle` | 回放时使用真实暂停时长，明确标为 oracle；不能作为应用未知未来时的成绩 |
| `--movement-min-pause-seconds` | movement 的最低预计暂停窗口，默认 0.5 秒；还须覆盖已观测的存出加加载耗时 |
| `--movement-min-pressure` | 搬运压力阈值，默认 0.8；短暂停利用排队窗口时还检查当前引用占用 |
| `--movement-pressure-only` | 忽略暂停和等待成本的压力消融；容量充足时可能产生大量无效存出 |
| `--prefetch-on-admission` | 有排队需求时等调度器选中对应 session 再准备，失败后回退重算；结果收集期限包含准入等待预算 |
| `--enforce-eager` / `--disable-custom-all-reduce` | 显式诊断选项；不与默认编译/图执行的速度混算 |
| `--profile` | 服务预热后启用 Yappi 多线程 CPU 计时；GPU 默认跳过 128 个迭代后采集 32 个迭代，由 `--profile-delay-iterations` / `--profile-iterations` 调整。与普通计时实验分开 |

准入协同配置将应用暂停窗口、已观测搬运耗时、当前 KV 占用和持续等待用于搬运判断。
短工具使用排队窗口时，要求实际占用达到配置阈值，且已观察等待超过搬运成本。
选中对应请求、整组容量足够时准备前缀。
实测收益与容量范围见[2026-10-19 报告](../../docs/programmable-kvc/rollout-report-2026-10-19.md)。

```bash
.venv/bin/python -m benchmarks.programmable_kv_cache.run_rollout_matrix \
  --model /home/youwei/bzh/model/Qwen/Qwen3.8-27B \
  --dataset /home/youwei/bzh/dataset/openai/gsm8k \
  --trace benchmarks/programmable_kv_cache/results/rollout-2026-10-18/baseline/trace.json.gz \
  --gpus 0,3 --gpu-memory-utilization 0.5 \
  --concurrency 64 --max-num-seqs 64 \
  --policy-max-wait-seconds 120 --quota-aging-seconds 30 \
  --prefetch-on-admission --movement-min-pressure 0.8 \
  --arms native,policy --repeats 3 \
  --output-dir /tmp/kv-admission-pressure
```

quota 对照另设 `--policy-components quota` 和独立输出目录。
V3 强制压力消融配置使用 `--movement-pressure-only --movement-min-pressure 0.6`，结果单列。
整组准备期间，已完成和原有驻留成员持续持有引用；取消/reset 继续等待已派发设备访问结束。
任务结束时先取消无消费者的准备，再收齐结果；普通计时包含设备搬运排空。
超时仍判失败，取消与 ACK 仅回收控制记录。

`--concurrency` 设置客户端并发；服务并发由 `--max-num-seqs` 设置，省略时沿用客户端并发。
CPU profiling 需要 `uv pip install yappi`。
诊断记录包含 Python 线程的函数统计、各线程 CPU 计数和线程名；Store master 的 CPU 也单独记录。
原生 C++ 后台线程仍需结合 CPU 计数与传输结果分析；累计函数时间存在嵌套，不能直接相加。
GPU trace 在负载期间保存为未压缩 JSON，压缩应在服务结束后执行，避免同步压缩阻塞推理和控制进度。
各轮记录等价工作量摘要、控制 RPC 次数与耗时、准备等待分位数，以及服务端排队、prefill、decode、inference 的计数和时间总和。
RPC 与逐请求耗时存在并发重叠，不能相加当作整轮耗时。
客户端在后台收集控制结果，无需收齐结果才发送下一请求；任务结束时仍收齐控制结果并清理，相关时间包含在整轮计时内。
启用按准入准备后，请求在引擎内等待对应加载发布，等待计入服务端排队和 TTFT。
真实加载和缓存发布继续遵守引擎资源屏障。

`analyze_rollout.py --run LABEL=REPORT.json.gz --baseline LABEL --output /tmp/comparison.json`
可以合并多个完整结果，核对输入/输出长度指纹、服务参数，并输出耗时分布、吞吐差值与家族尾延迟。
同一标签可传入多轮；profile 轮不参与普通吞吐对比。
仍须检查各轮 GPU 归属与并行活动；工具不会根据 token 指纹认定测量环境有效。

## 功能测试的服务配置

文本/Qwen TP=2 功能服务示例，固定 4 GiB KV/卡用于正确性检查：

```bash
CUDA_VISIBLE_DEVICES=0,3 \
  OMP_NUM_THREADS=1 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  .venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/youwei/bzh/model/Qwen/Qwen3.8-27B \
  --served-model-name qwen-validation \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --kv-cache-memory-bytes 4294967296 \
  --mamba-cache-mode align \
  --dtype bfloat16 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enforce-eager \
  --disable-custom-all-reduce \
  --kv-control-config '{"engine_id":"kv-validation"}' \
  --host 127.0.0.1 \
  --port 18083
```

Mooncake 数据面另启动隔离 Store：

```bash
.venv/bin/mooncake_master \
  --rpc_address=127.0.0.1 \
  --rpc_port=19151 \
  --metrics_host=127.0.0.1 \
  --metrics_port=19153 \
  --enable_metric_reporting=false
```

将客户端配置保存为 `/tmp/programmable-kv-mooncake-config.json`：

```json
{
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:19151",
  "protocol": "tcp",
  "device_name": "",
  "global_segment_size": "8GB",
  "local_buffer_size": "256MB"
}
```

启动功能服务时设置 `MOONCAKE_CONFIG_PATH` 为该文件，并增加：

```bash
--kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"enable_lookup":false,"enable_save":false,"save_decode_cache":false}}'
```

关闭普通请求的缓存查找与写入，使显式 COPY/MOVE 的效果可单独测量。
CPU 功能检查则使用以下连接器，无需 Mooncake Store：

```bash
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":8589934592,"blocks_per_chunk":1}}'
```

原生 Mooncake 路径要求目标与完整的主缓存项及各缓存组边界对齐，预填充和解码的上下文并行度均为 1（PCP=DCP=1）。
通用 CPU 路径要求每个传输单元（chunk）包含一个缓存块，暂不支持 CPU Share。
其他布局应通过 `capabilities` 查询实际支持范围。

## 功能测试驱动

`http_control.py` 同时检查请求完成时 MOVE、下次入口 COPY、独立 COPY/MOVE、优先级与真实缓存命中。
原 `primitive_control.py` 已合并到此入口；协议错误、请求关联增长和重放边界由 pytest 覆盖。

```bash
CUDA_VISIBLE_DEVICES=0,3 \
  .venv/bin/python -m benchmarks.programmable_kv_cache.http_control \
  --base-url http://127.0.0.1:18083 \
  --engine-id kv-validation \
  --dataset /home/youwei/bzh/dataset/openai/gsm8k \
  --output /tmp/kv-attached.json
```

默认使用 Mooncake；CPU 服务使用 `--tier cpu`，该驱动在推理时禁用自动 CPU 加载。
加 `--evaluation-samples 128` 做质量门槛，加 `--destination-url` 验证兼容目的引擎的实际命中。
两个服务须采用相同模型、布局、Store 与并行设置，各有独立引擎 ID；同用 GPU 0/3 时两者均可采用 `--gpu-memory-utilization 0.45`，目的端口用 18084。

| 入口 | 参数和前置条件 |
| --- | --- |
| `lifecycle.py` | `--base-url --trace --output`；原生控制与 Mooncake，检查共享分支、TTL、取消和恢复竞争 |
| `weight_update.py` | `--base-url --model --trace --output`；同机 GPU、对应模型检查点、IPC 权重传输后端和 Store |
| `evaluate.py` | `--model --dataset --output`，可加 `--enable-kv-control`；仅运行重算/APC 质量基线，控制协议由 pytest 检查 |
| `control_overhead.py` | `--output`；测量 CPU 元数据操作，不代表推理服务或 RL 吞吐 |
| `run_recovery.py` | 自建服务/Store，`--model --dataset --trace --output-dir`，可加 `--evaluation-samples 128`；新组合测试驱动尚未执行 |

IPC 权重检查要求服务额外启用 `--weight-transfer-config '{"backend":"ipc"}'`，设置 `VLLM_SERVER_DEV_MODE=1` 与既有同机 IPC 序列化配置 `VLLM_ALLOW_INSECURE_SERIALIZATION=1`，服务保持绑定本地地址。
脚本修改真实 GPU 参数并回滚，未执行优化器更新。
其他示例不需要这些权重更新设置。

策略在启动时通过 `--kv-control-config` 配置，空 `policies` 使信号不产生策略动作。
`http_control.py --policy` 的专用服务可用：

```json
{
  "engine_id": "kv-policy",
  "policies": [
    {
      "name": "retention",
      "implementation": "retention"
    },
    {
      "name": "movement",
      "implementation": "tool_wait",
      "options": {
        "tier": "mooncake"
      }
    }
  ]
}
```

对该服务使用 `--engine-id kv-policy --policy`。
`--policy` 额外检查工具等待与恢复信号产生的派生搬运结果。
`signal` 完成表示策略处理了信号；派生 COPY 等操作必须分别查询，完成后才表示缓存准备就绪。
收集同一消息中的全部动作及派生操作结果后再 ACK。
`residency` 返回查询时的 GPU 状态快照，历史 COPY 成功不保证之后持续驻留。
`client.py` 与 `rollout.py` 共享提示消息和结果协议。

## 回归检查

测试范围与执行证据见[分支汇报](../../docs/programmable-kvc/testing.md)，分阶段回归结果见[实验报告](../../docs/programmable-kvc/rollout-report-2026-10-12.md)。
删除的协议探针自测由直接 API 行为测试替代，实际事件缺口/路由/重放/ACK 测试继续保留。

```bash
.venv/bin/python -m pytest tests/entrypoints/serve/kv_control/test_api.py -q
.venv/bin/pre-commit run \
  --files tests/entrypoints/serve/kv_control/test_api.py
.venv/bin/pre-commit run mypy-3.12 \
  --all-files \
  --hook-stage manual
openspec validate programmable-kv-cache \
  --strict
```

完整回归和模型门槛的执行状态以实验报告为准。
命令展示不代表已执行；日志、源代码、工作量与质量必须一起归档。
