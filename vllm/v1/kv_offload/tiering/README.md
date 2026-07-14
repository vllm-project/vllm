# KV Cache Tiering and Lifecycle Management

该目录基于 vLLM 的 `OffloadingConnector` 提供 KV Cache 分层驻留：设备侧
KV Cache 作为热数据层，CPU 作为一级卸载层，文件系统、对象存储或 P2P
节点作为二级存储层。

## 执行路径

Store 路径：

```text
NPU/GPU KV block -> CPU primary tier -> secondary tiers
```

Load 路径：

```text
secondary tier hit -> CPU promotion -> NPU/GPU restore
```

`TieringOffloadingManager` 负责块级查询、CPU 空间分配、级联写入、二级层提升
以及异步任务完成处理。Ascend 环境由 CPU Worker 工厂自动选择
`AscendCPUOffloadingWorker`，数据搬运使用 `swap_blocks_batch` 完成。

## 生命周期策略

生命周期模块按照稳定的 Session ID 管理以下状态：

```text
ACTIVE -> IDLE_RETAINED -> EXPIRED -> DELETED
```

Session ID 按顺序读取 `session_id`、`conversation_id` 和 `kv_session_id`，
没有提供时回退到请求 ID。请求结束后保留已访问的 KV Block 元数据；空闲时间
超过 TTL 后清理生命周期状态。启用二级文件删除时，系统会先检查共享 Block
引用以及正在执行的传输，避免删除其他 Session 仍在使用的数据。

相关配置：

| 配置 | 默认值 | 说明 |
| --- | ---: | --- |
| `lifecycle_idle_ttl_sec` | `0` | 空闲 Session 的 TTL，`0` 表示关闭 |
| `lifecycle_delete_expired_secondary` | `false` | 过期时是否删除 FS 二级副本 |

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
- 长请求后 FS 数据文件数量和容量增加。
- 重复前缀请求出现外部 KV Cache 命中和 promotion/load 指标。
- TTL 到期后日志出现 `Expired ... idle KV lifecycle session(s)`。

## 测试

生命周期和 Worker 工厂单元测试：

```bash
python -m pytest -q --confcutdir=tests/v1/kv_offload \
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
