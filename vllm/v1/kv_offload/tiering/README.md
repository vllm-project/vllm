# KV Cache Tiering and Lifecycle Management

该目录基于 vLLM 的 `OffloadingConnector` 提供 KV Cache 分层驻留：设备侧
KV Cache 作为热数据层，CPU 作为一级卸载层，文件系统、对象存储或 P2P
节点作为二级存储层。

## 执行路径

Store 路径由压力水位触发：

```text
HBM pressure: NPU/GPU KV block -> CPU primary tier
CPU pressure: CPU primary tier -> first secondary tier
```

Load 路径：

```text
secondary tier hit -> CPU promotion -> NPU/GPU restore
```

`TieringOffloadingManager` 负责块级查询、压力判断、CPU 空间分配、按需降级、
二级层提升以及异步任务完成处理。HBM 充足时不创建 Device→CPU Store Job；
CPU 驻留率超过水位后，冷 Block 才写入第一个 secondary tier，并在写入成功后
释放 CPU 副本。Ascend 环境由 CPU Worker 工厂自动选择
`AscendCPUOffloadingWorker`，数据搬运使用独立 NPU Stream 上的
`swap_blocks_batch` 完成。Store Job 在下一 engine step 提交，默认只在 decode
阶段分批搬运，从而将传输与后续 token 计算重叠。CPU→NPU 回载使用固定上限的
NPU staging buffer，并将长前缀拆分为多个连续 chunk，避免按完整上下文长度
分配双份临时 HBM。可通过 `VLLM_ASCEND_KV_LOAD_STAGING_BYTES` 调整 staging
上限，默认值为 64 MiB。

一次 HBM 压力不会再提交全部待卸载 Block。策略按调度步、请求、会话压力周期和
全局压力周期分配迁移额度，并限制同时在途的 Device→CPU Job 数。候选 Block
按逻辑前缀连续选择，多个 KV group 交错分配额度，因此未选中的 Block 不会被
错误跳过。携带稳定 Session ID 的请求会根据回访次数和经验复用概率筛选；没有
稳定 Session ID 的普通请求在 HBM 压力下仍可卸载，避免默认配置无法释放设备
前缀缓存。请求可通过 `force_kv_offload=true` 显式覆盖热度和阶段筛选，但仍受
迁移额度保护。

## 生命周期策略

生命周期模块按照稳定的 Session ID 管理以下状态：

```text
ACTIVE -> IDLE_RETAINED -> EXPIRED -> DELETED
```

Session ID 按顺序读取 `session_id`、`conversation_id` 和 `kv_session_id`，
没有提供时回退到请求 ID。请求结束后保留已访问的 KV Block 元数据；空闲时间
超过 TTL 后清理生命周期状态。系统通过 Session 到 Block 的反向引用索引判断
共享关系，只在最后一个 Session 引用释放后回收 CPU 副本；启用二级文件删除时，
再删除对应 FS 副本。正在执行迁移或仍被活跃 Session 使用的 Block 不参与回收。

异步 Store 完成后会立即移除该轮请求 ID 的反向索引。没有稳定 Session ID 的
一次性请求不保留 Session 元数据；稳定 Session 即使关闭 TTL，也受
`lifecycle_max_sessions` 上限约束，超过上限时优先裁剪最久未访问的空闲 Session。
因此历史请求数量和 Session 数量不会随服务运行时间无界增长。

CPU 层支持两类主动降级策略：Block 空闲超过设定时间，或 CPU 驻留率超过高水位。
未持有二级副本的 Block 会先异步写入 secondary tier；写入成功后才释放 CPU
副本。已有二级副本的 Block 可以直接回收，因此后续请求仍可按
`FS -> CPU -> NPU/GPU` 路径恢复。

相关配置：

| 配置 | 默认值 | 说明 |
| --- | ---: | --- |
| `tiering_hbm_pressure_aware` | `true` | 是否只在设备 KV Block 池有压力时卸载到 CPU |
| `tiering_hbm_high_watermark` | `0.70` | 启用 Device→CPU 卸载的 KV Block 使用率 |
| `tiering_hbm_low_watermark` | `0.50` | 停止 Device→CPU 卸载的 KV Block 使用率 |
| `tiering_secondary_pressure_aware` | `true` | 是否只在 CPU 压力下写入 secondary tier |
| `tiering_bypass_unknown_secondary_when_relaxed` | `true` | HBM 低压且无本地驻留记录时跳过二级查询 |
| `tiering_min_session_requests_for_offload` | `2` | 自动卸载所需的最少 Session 请求次数 |
| `tiering_min_reuse_probability` | `0.5` | Session 经验回访概率下限 |
| `tiering_store_during_decode_only` | `true` | 常规迁移延迟到 decode 阶段，预抢占时自动解除 |
| `tiering_max_device_store_blocks_per_step` | `16` | 每个调度步最多提交的 Device→CPU Block 数 |
| `tiering_max_device_store_blocks_per_request` | `64` | 单请求最多提交的 Device→CPU Block 数，`0` 不限 |
| `tiering_max_device_store_blocks_per_pressure_episode` | `128` | 单次 HBM 压力周期的全局迁移上限，`0` 不限 |
| `tiering_max_device_store_blocks_per_session_episode` | `64` | 单会话在压力周期内的迁移上限，`0` 不限 |
| `tiering_max_inflight_device_store_jobs` | `2` | 同时在途的 Device→CPU Store Job 上限，`0` 不限 |
| `tiering_reclaim_device_cache_after_store` | `true` | 压力卸载完成后驱逐本地前缀缓存映射；活跃请求仍持有的 Block 会在请求释放时归还池中 |
| `tiering_use_pinned_cpu_primary` | 自动 | 未配置 Secondary Tier 时使用页锁定 CPU 主层；配置 FS 等 Secondary Tier 时保留共享 mmap 主层 |
| `lifecycle_idle_ttl_sec` | `0` | 空闲 Session 的 TTL，`0` 表示关闭 |
| `lifecycle_delete_expired_secondary` | `false` | 过期时是否删除 FS 二级副本 |
| `lifecycle_cpu_demote_after_sec` | `0` | Block 空闲多久后释放 CPU 副本，`0` 表示关闭 |
| `lifecycle_cpu_high_watermark` | `0.9` | 触发 CPU→secondary 降级的驻留率高水位 |
| `lifecycle_cpu_low_watermark` | `0.7` | CPU 降级后的目标驻留率 |
| `lifecycle_reclaim_batch_size` | `64` | 单次调度周期最多主动回收的 Block 数 |
| `lifecycle_max_sessions` | `4096` | 最多保留的稳定 Session 元数据数量，超限时裁剪最旧空闲 Session |
| `residency_tracking_enabled` | `false` | 无 TTL/水位策略时单独启用驻留观测 |
| `residency_max_entries` | `64000` | Block 驻留与引用元数据数量上限 |

指标包括各层驻留 Block 数、活跃/空闲 Session 数、共享 Block 数、分层查询结果、
迁移 Block/字节/时延、迁移预算占用、复用信号、过期/裁剪 Session 数和 CPU
主动回收数量。指标统一使用 `vllm:kv_tiering_*` 前缀，可通过 vLLM 的
`/metrics` 端点观察。

## Ascend Serve 示例

在仓库根目录执行：

```bash
bash vllm/v1/kv_offload/tiering/my_tests/run_serve.sh
```

常用覆盖参数：

```bash
ASCEND_DEVICE=1 \
MODEL=/root/models/Qwen2.5-7B-Instruct \
PORT=8081 \
NPU_KV_BYTES=268435456 \
CPU_KV_BYTES=1073741824 \
MAX_MODEL_LEN=4096 \
FS_ROOT=/tmp/vllm_kv_tiering \
LIFECYCLE_TTL=10 \
CPU_DEMOTE_AFTER=2 \
CPU_HIGH_WATERMARK=0.9 \
CPU_LOW_WATERMARK=0.7 \
HBM_HIGH_WATERMARK=0.70 \
HBM_LOW_WATERMARK=0.50 \
MIN_SESSION_REQUESTS=2 \
MAX_STORE_BLOCKS_PER_STEP=16 \
MAX_STORE_BLOCKS_PER_REQUEST=64 \
VLLM_ASCEND_KV_LOAD_STAGING_BYTES=67108864 \
bash vllm/v1/kv_offload/tiering/my_tests/run_serve.sh
```

服务启动后运行：

```bash
python3 vllm/v1/kv_offload/tiering/my_tests/serve_lifecycle_client.py \
  --url http://127.0.0.1:8081/v1/chat/completions \
  --fs-root /tmp/vllm_kv_tiering
```

验证重点：

- 服务日志中出现 Ascend KV offload Worker 初始化信息。
- HBM 低压时 CPU/FS 迁移指标保持为零。
- HBM 高压时出现 `device -> cpu` 迁移指标。
- CPU 高压时才出现 FS 文件和 `cpu -> fs:0` 迁移指标。
- 重复前缀请求出现外部 KV Cache 命中和 promotion/load 指标。
- 请求空闲后 `kv_tiering_resident_blocks{tier="cpu"}` 下降，FS 驻留保持。
- 再次请求相同前缀时出现 `fs:0 -> cpu -> device` 迁移指标。
- TTL 到期后日志出现 `Expired ... idle KV lifecycle session(s)`。

## 测试

生命周期和 Worker 工厂单元测试：

```bash
python3 -m pytest -q --confcutdir=tests/v1/kv_offload \
  tests/v1/kv_offload/tiering/test_lifecycle.py \
  tests/v1/kv_offload/cpu/test_worker_factory.py
```

Ascend NPU/CPU 数据回环：

```bash
python3 vllm/v1/kv_offload/tiering/my_tests/npu_worker_roundtrip.py
```

已有 Benchmark JSON 可以使用以下命令比较：

```bash
python3 vllm/v1/kv_offload/tiering/my_tests/summarize_benchmark.py \
  /tmp/baseline-256m-round2.json \
  /tmp/tiering-256m-round2.json
```

Tiering 的主要收益指标应优先报告可复用 KV 驻留容量、外部 KV 命中率和可承载
并发规模。CPU/SSD 数据搬运会增加单请求 TTFT，不能在没有实测数据时声明吞吐或
时延提升。
