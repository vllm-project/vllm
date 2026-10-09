# UMBP 分支合并与提交拆分方案

## 1. 目标

本文档描述以下两个 UMBP 分支的合并方案：

- `compare/charxwu-umbp-rebased`
- `compare/yukio-umbp-standalone`

两个分支已经对齐到相同 upstream 基线：

```text
b558f160a2c0abcb5902acc3c91a14c38a4af173
```

计划实现以下目标：

1. 合并两个分支的 UMBP 功能，而不是简单拼接 Git 历史。
2. 保留 charxwu 分支的 embedded 和 lazy offload 能力。
3. 引入 Yukio 分支的 correctness、bulk transfer、hybrid、事件、指标和
   standalone runtime。
4. 将共享 base 和具体 runtime 分层。
5. 按功能构建可独立 review、测试、bisect 和 revert 的提交序列。

本文档是实施预览，不代表对应提交已经完成或通过测试。

## 实施状态

本方案已经在本地仓库中落地，尚未 push：

- 工作目录：`/home/charwu/vllm-umbp-merged-series`
- 本地分支：`review/umbp-merged-series`
- 基线：`b558f160a2c0abcb5902acc3c91a14c38a4af173`
- 最终代码树与 `wip/umbp-merged-convergence` 一致。
- UMBP unit tests：223 passed。
- CPU integration tests：6 passed，4 个 GPU/MORI 用例 skipped。
- Ruff：通过。
- `git diff --check`：通过。

实际提交序列见第 8 节。GPU/MORI 实机验证仍需在对应环境执行。

## 2. 当前分支状态

### 2.1 charxwu rebased

```text
f6d667798a [UMBP] Checkpoint slim embedded connector and lookup/load optimizations
92fdfb4d9e [UMBP] Add shared connector and embedded DRAM offloading
b558f160a2 upstream base
```

主要能力：

- Embedded MORI DRAM runtime。
- Eager store 和 free-queue-driven lazy offload。
- Compact whole-block load metadata。
- Layer-wise load 路径。
- Hybrid、partial hit、events 和 metrics 的早期实现。
- 已记录的 MI355、TP 和 GPT-OSS 历史验证结果。

### 2.2 Yukio standalone

```text
4ac52ea966 [UMBP] Add standalone mode backed by a node-local server
0ef2b64e11 [UMBP] Report KV cache events and transfer metrics
2d2615af79 [UMBP] Support hybrid KV cache groups
e794073e26 [UMBP] Optionally look up the pool off the scheduler thread
0fa9872515 [UMBP] Bound transfer time and cancel queued transfers on preemption
63a3e756ec [UMBP] Transfer whole blocks in bulk and skip resident restored blocks
f700ac6a9d [UMBP] Add UMBPStoreConnector with embedded DRAM offloading
b558f160a2 upstream base
```

主要能力：

- 精简后的 embedded scheduler 和 worker lifecycle。
- Whole-block bulk load/store。
- Transfer timeout、queued cancellation 和 MORI eviction retry。
- 当前 upstream hybrid/Mamba API 适配。
- 当前 KV event 和 metrics API 适配。
- Embedded lookup endpoint 隔离。
- Node-local standalone server runtime。

## 3. 合并策略

不建议执行普通的 `git merge`，也不建议直接 cherry-pick Yukio 的全部提交。

原因如下：

- 两个分支包含同源但独立修改过的 connector、scheduler 和 worker。
- `f700ac6a9d` 与 charxwu 的两个提交大面积重叠。
- Yukio 后续提交以其精简版 worker 为前提，直接 cherry-pick 可能删除
  charxwu 的 lazy offload。
- 普通 merge 即使文本冲突不多，也可能产生重复状态机和不一致的完成语义。

建议使用两个阶段。

```text
charxwu rebased ─┐
                 ├──► WIP convergence branch ──► verified final tree
Yukio standalone ┘                                  │
                                                    ▼
                                            curated commit series
                                                    │
                                                    ▼
                                             reviewable branch
```

### 3.1 阶段一：功能收敛

从 charxwu rebased 创建 WIP 分支：

```bash
git switch compare/charxwu-umbp-rebased
git switch -c wip/umbp-merged-convergence
```

在该分支上逐项移植 Yukio 功能。这个阶段允许临时提交，目标是先得到一个
行为明确且测试通过的最终代码树。

推荐移植顺序：

1. Shared bulk-transfer contract。
2. Embedded whole-block bulk store 和 residency skip。
3. Timeout、preemption 和 cancellation hardening。
4. Async lookup lifecycle 和 endpoint isolation。
5. Hybrid/Mamba 和 partial hit。
6. KV events 和 metrics。
7. Standalone runtime。
8. 文档与 CI。

### 3.2 阶段二：重建正式提交历史

功能收敛并完成验证后，从统一 upstream 创建正式分支：

```bash
git switch -c review/umbp-merged-series \
  b558f160a2c0abcb5902acc3c91a14c38a4af173
```

按照第 7 节的 commit preview 重建提交。每个提交应同时包含对应的最小测试，
而不是在最后集中补测试。

## 4. Base 与 Runtime 的职责边界

### 4.1 Shared/Base 层

```text
vllm/distributed/kv_transfer/kv_connector/v1/umbp/
├── connector.py
├── data.py
├── scheduler.py
├── worker.py
├── stats.py
└── runtime/
    ├── base.py
    └── factory.py
```

Base 层负责：

- KV object identity 和 namespace。
- Rank-local physical key 与 logical object identity。
- KV layout、range 和 whole-block transfer plan。
- External lookup 和 hit coordination。
- Load/store scheduling。
- Eager 和 lazy offload 策略。
- Request、preemption 和 block ownership lifecycle。
- Hybrid/Mamba group coordination。
- Runtime-independent failure mapping。
- KV events 和 connector stats。
- Runtime protocol 和 runtime factory。

Base 层不应负责：

- MORI client 创建。
- DRAM allocator 参数。
- Unix socket listener。
- HIP IPC registration。
- Standalone server endpoint。
- MORI-specific flush、retry 或 shutdown API。

Base 层应能通过 fake runtime 完成绝大多数单元测试。

### 4.2 Runtime 层

```text
runtime/
├── embedded.py
└── standalone.py
```

Runtime 层负责：

- GPU KV buffer registration。
- MORI load/store API 调用。
- Whole-block 和 ranged I/O 的具体实现。
- Transfer executor、poll、wait、timeout 和 cancel。
- MORI eviction 查询与 flush/retry。
- Embedded lookup socket。
- Standalone gRPC/Unix-domain endpoint。
- HIP IPC registration。
- DRAM capacity、watermark、NUMA、huge page 和 prefault 配置。
- Runtime-specific startup、failure 和 shutdown 行为。

Runtime 不应复制 scheduler、lazy offload 或 hybrid hit coordination。

## 5. 关键设计决策

| 冲突点 | 合并后的选择 |
| --- | --- |
| Lazy offload | 保留 charxwu 实现，默认关闭。 |
| Whole-block load | 默认使用 bulk path。 |
| Whole-block store | 增加 `store_blocks()` runtime API。 |
| Layer-wise load | 仅作为特殊 layout fallback。 |
| Layer-wise store | 删除；当前 MORI 不支持安全 staged publication。 |
| `hash_block_size` | 使用 upstream `prefix_match_unit`。 |
| 旧 `hash_block_size` 配置 | 先 deprecated；与 resolved value 不一致时明确报错。 |
| Hybrid/Mamba | 使用 Yukio 对当前 upstream 的实现。 |
| 多 group load failure | 使用 request-level reporting。 |
| Events/metrics | 使用 Yukio 对当前 API 的实现。 |
| `kv_consumer` | 只允许 lookup/load，不允许 store。 |
| Embedded endpoint | namespace、instance、DP rank 和 local rank 共同隔离。 |
| Standalone object key | 不包含 DP rank，允许相同 layout 的 engine 共享。 |
| `distributed` mode | 删除，直到存在实际 runtime。 |
| Runtime capabilities | 只保留实际需要的 capability。 |

### 5.1 Object namespace 与 endpoint identity

必须区分两个概念：

```text
Object namespace
└── 决定两个 engine 是否能共享同一个 KV object

Lookup endpoint identity
└── 决定两个 embedded worker 是否竞争同一个本地 socket
```

Standalone 模式下，DP rank 不进入 object namespace，否则不同 DP rank 不能共享
缓存。Embedded 模式下，DP rank 和 `lookup_instance` 必须进入 lookup endpoint
identity，否则相同模型和拓扑的两个 engine 会产生 socket 冲突。

### 5.2 Lazy offload 与 standalone

Lazy offload 属于 scheduler 策略，不属于 embedded runtime。因此合并后原则上应
同时支持：

```text
lazy scheduler ──► embedded runtime
lazy scheduler ──► standalone runtime
```

Standalone server 是否已经持有某个 object，仍由 authoritative lookup 决定。
Lazy candidate selection 不应维护独立的长期 residency cache。

### 5.3 Transfer ownership

任何仍可能访问 GPU KV block 的 transfer 都必须继续持有 block ownership：

```text
queued transfer
├── 可以 cancel
└── cancel 完成后释放 block

running transfer
├── wait/poll 到 terminal state
├── timeout 时不能静默释放 block
└── step failure 或 engine failure 必须显式上报
```

该规则必须同时覆盖 eager、lazy、bulk 和 fallback range transfer。

## 6. 最终文件归属 Preview

| 文件 | 主要来源 | 合并策略 |
| --- | --- | --- |
| `connector.py` | 两边 | 以 charxwu lifecycle 为底，移植 Yukio `kv_consumer` 和当前 event API。 |
| `data.py` | 两边 | 保留 compact metadata，采用 Yukio 简化后的 identity 和 failure model。 |
| `scheduler.py` | charxwu + Yukio | 保留 lazy；采用 Yukio hybrid/Mamba、residency skip 和当前 upstream API。 |
| `worker.py` | charxwu + Yukio | 保留 lazy jobs；采用 Yukio bulk store、timeout 和 request-level failure。 |
| `stats.py` | Yukio 为主 | 对齐当前 connector metrics API。 |
| `runtime/base.py` | 两边 | 定义 embedded/standalone 共用的最小 runtime protocol。 |
| `runtime/factory.py` | Yukio 为主 | 只支持实际存在的 `embedded` 和 `standalone`。 |
| `runtime/embedded.py` | 两边 | 保留 DRAM 配置；采用 Yukio bulk、retry、endpoint isolation。 |
| `runtime/standalone.py` | Yukio | 基本完整移植，并验证 lazy store。 |
| `test_umbp_shared.py` | charxwu | 拆分到 core、lazy、hybrid、events 测试。 |
| standalone tests | Yukio | 保留并增加 lazy/cross-engine 场景。 |

## 7. Commit Preview

### Commit 1: Shared runtime contracts and KV object model

```text
[UMBP] Define shared runtime contracts and KV object model
```

主要文件：

```text
umbp/data.py
umbp/runtime/base.py
umbp/runtime/factory.py
umbp/__init__.py
tests/v1/kv_connector/unit/test_umbp_core.py
```

内容：

- `RankTopology`。
- `UMBPNamespace`。
- `BlockIdentityCodec`。
- `KVLayoutDescriptor` 和 `KVLayoutPlanner`。
- `BlockTransferPlan` 和 `BlockLoadBatch`。
- `TransferJobState`。
- `load`、`load_blocks`、`store` 和 `store_blocks` runtime contract。
- `wait`、`poll`、`cancel`、`publish` 和 eviction contract。

该提交不引入 MORI，也不注册实际 connector。

### Commit 2: Shared scheduler and worker lifecycle

```text
[UMBP] Add shared connector scheduling and worker lifecycle
```

主要文件：

```text
umbp/connector.py
umbp/scheduler.py
umbp/worker.py
tests/v1/kv_connector/unit/test_umbp_core.py
```

内容：

- External lookup。
- Local hit 之后继续查询 external cache。
- Load/store plan construction。
- GPU block pinning。
- Worker job submission 和 completion。
- Load/store failure mapping。
- Request finished 和 cache reset lifecycle。
- Fake runtime unit tests。

### Commit 3: Lazy offload

```text
[UMBP] Add lazy GPU cache offloading
```

主要文件：

```text
umbp/scheduler.py
umbp/worker.py
umbp/data.py
tests/v1/kv_connector/unit/test_umbp_lazy.py
```

内容：

- `lazy_offload`。
- `lazy_offload_max_blocks`。
- Free-queue candidate selection。
- Prefix-chain preservation。
- In-flight key filtering。
- Lazy block pinning 和 completion。
- Eager/lazy 公共 store submission。

### Commit 4: Hybrid and boundary state

```text
[UMBP] Coordinate hybrid cache groups and boundary states
```

主要文件：

```text
umbp/scheduler.py
umbp/worker.py
umbp/data.py
tests/v1/kv_connector/unit/test_umbp_hybrid.py
```

内容：

- 多 KV cache group hit coordination。
- 当前 upstream `KVCacheCoordinator`。
- Group-specific block size。
- Draft cache group restore。
- Request-level multi-group failure reporting。
- Mamba align boundary handoff。
- Finished partial boundary state。
- Partial hash hit 和 upstream `prefix_match_unit`。
- Mamba、hybrid 和 sparse indexer async load requirement。

### Commit 5: Events and metrics

```text
[UMBP] Report KV cache events and transfer metrics
```

主要文件：

```text
umbp/connector.py
umbp/worker.py
umbp/stats.py
umbp/data.py
tests/v1/kv_connector/unit/test_umbp_events.py
```

内容：

- `BlockStored` 和 `BlockRemoved`。
- All-worker logical completion。
- Submitted/completed/failed counters。
- Transferred bytes。
- Prometheus metrics。
- Eager/lazy 公共 event 语义。
- `kv_consumer` 只读行为。

### Commit 6: Embedded runtime

```text
[UMBP] Add MORI embedded DRAM runtime
```

主要文件：

```text
umbp/runtime/embedded.py
umbp/runtime/__init__.py
umbp/runtime/factory.py
umbp/connector.py
kv_connector/factory.py
tests/v1/kv_connector/unit/test_umbp_embedded.py
tests/v1/kv_connector/umbp_integration/test_embedded_runtime.py
```

内容：

- MORI client 创建。
- GPU buffer registration。
- Ranged load/store。
- Embedded lookup socket。
- DRAM capacity、watermark、NUMA、huge page、prefault 和 shared memory。
- Embedded runtime 和 connector 注册。

这是第一个真实可运行的 MORI embedded commit。

### Commit 7: Whole-block optimization

```text
[UMBP] Optimize whole-block transfers and skip resident stores
```

主要文件：

```text
umbp/runtime/embedded.py
umbp/scheduler.py
umbp/worker.py
umbp/data.py
tests/v1/kv_connector/unit/test_umbp_embedded.py
```

内容：

- 注册固定 cache-group layout。
- 批量构建 GPU layer address。
- Whole-block `load_blocks` 和 `store_blocks`。
- 特殊 layout fallback 到 ranged I/O。
- Restored/resident object 不重复 store。
- Lazy store 使用相同 bulk path。

### Commit 8: Timeout and preemption

```text
[UMBP] Bound transfer waits and harden preemption
```

主要文件：

```text
umbp/runtime/embedded.py
umbp/worker.py
umbp/scheduler.py
umbp/data.py
tests/v1/kv_connector/unit/test_umbp_core.py
tests/v1/kv_connector/unit/test_umbp_embedded.py
```

内容：

- `timeout_ms`。
- Queued transfer cancellation。
- Running transfer ownership。
- Preemption block-reuse protection。
- MORI put flush/retry。
- Shutdown drain。
- Partial failure 不 publish。
- Lazy store cancellation。

### Commit 9: Async lookup isolation

```text
[UMBP] Isolate embedded lookup endpoints across engines
```

主要文件：

```text
umbp/runtime/embedded.py
umbp/scheduler.py
umbp/runtime/factory.py
tests/v1/kv_connector/unit/test_umbp_async_lookup.py
tests/v1/kv_connector/unit/test_umbp_embedded.py
```

内容：

- `lookup_async` lifecycle。
- Stale lookup generation。
- Finish/preemption/reset cancellation。
- `lookup_instance`。
- DP-specific socket identity。
- Live socket collision detection。
- Executor shutdown。
- Structured lookup diagnostics。

### Commit 10: Standalone runtime

```text
[UMBP] Add standalone mode backed by a node-local server
```

主要文件：

```text
umbp/runtime/standalone.py
umbp/runtime/__init__.py
umbp/runtime/factory.py
tests/v1/kv_connector/unit/test_umbp_standalone.py
tests/v1/kv_connector/umbp_integration/test_mori_standalone_gpu.py
docs/features/umbp_standalone.md
```

内容：

- External `umbp_standalone_server`。
- Unix-domain endpoint。
- HIP IPC registration。
- Server-owned DRAM pool。
- Cross-engine 和 cross-DP sharing。
- Engine restart persistence。
- Server stall、death 和 restart 行为。
- `startup_timeout_ms`。
- Standalone DRAM option rejection。

### Commit 11: Documentation

```text
[UMBP] Document embedded and standalone connector configuration
```

主要文件：

```text
docs/features/umbp_embedded_dram.md
docs/features/umbp_standalone.md
```

文档应包含：

- Embedded 和 standalone 使用方法。
- Lazy offload 配置。
- Failure policy。
- Capability matrix。
- 已验证和未验证场景。
- Embedded 与 standalone 的 sharing scope。

避免使用难以持续维护的主观完成度百分比。

### Commit 12: CI

```text
[CI][ROCm] Add UMBP connector integration coverage
```

主要文件：

```text
.buildkite/test-amd.yaml
```

覆盖：

- UMBP unit tests。
- Embedded MORI GPU integration。
- Embedded token restore。
- Lazy offload。
- TP restore。
- Standalone GPU integration。
- Standalone cross-engine restore。

## 8. 实际提交历史

```text
e6c7d896fb [Docs][UMBP] Document branch integration plan
557eb8a967 [Test][UMBP] Stub MORI memory locations in transfer ownership test
6dc9a40027 [UMBP] Add lazy GPU cache offloading
3ba8e8d3cf [UMBP] Add standalone mode backed by a node-local server
bf596e3b88 [UMBP] Report KV cache events and transfer metrics
7ef33c3944 [UMBP] Support hybrid KV cache groups
53fabd4013 [UMBP] Optionally look up the pool off the scheduler thread
fd449e32e3 [UMBP] Bound transfer time and cancel queued transfers on preemption
09655db62a [UMBP] Transfer whole blocks in bulk and skip resident restored blocks
a94fbff294 [UMBP] Add MORI embedded DRAM runtime
efb3069443 [UMBP] Add shared connector scheduling and worker lifecycle
9cc31cc2f7 [UMBP] Define runtime contracts and KV object model
b558f160a2 upstream base
```

与原 Yukio 历史相比，初始 `f700ac6a9d` 被实际拆分成前三个提交：

1. Runtime-neutral contracts 和 data model。
2. Shared connector、scheduler 和 worker。
3. Embedded MORI runtime、测试和 CI wiring。

后续 Yukio 功能提交保留原作者和原提交边界。charxwu 的 lazy offload 作为独立
提交重新适配到最终 scheduler/runtime contract 上。

## 9. 测试和验收流程

### 9.1 每个提交

按照仓库 `AGENTS.md`，Python 命令必须通过项目虚拟环境执行。

```bash
git diff --check HEAD^
pre-commit run --files <changed-files>
.venv/bin/python -m pytest <tests-for-this-commit> -v
```

Base 层优先运行：

```text
test_umbp_core.py
test_umbp_lazy.py
test_umbp_hybrid.py
test_umbp_events.py
test_umbp_async_lookup.py
```

Embedded runtime 加入后运行：

```text
test_umbp_embedded.py
test_embedded_runtime.py
test_mori_embedded_gpu.py
test_mori_token_e2e.py
```

Standalone 加入后运行：

```text
test_umbp_standalone.py
test_mori_standalone_gpu.py
```

### 9.2 GPU 验收矩阵

至少覆盖：

1. TP1 eager restore。
2. TP1 lazy restore。
3. TP2 restore。
4. DRAM eviction 后 restore。
5. Preemption 后立即复用 GPU block。
6. Multiple KV cache groups。
7. Mamba boundary state。
8. Standalone 两个 engine 共享。
9. Engine restart 后 restore。
10. Standalone server stall/death。
11. Embedded 和 standalone transfer bandwidth。
12. External hit 后 TTFT。

### 9.3 正确性要求

- Cold run 和 restored run token 序列一致。
- Failed load 按配置 recompute，不读取部分恢复的数据。
- Failed store 不发布 logical object。
- 所有 rank 完成后才能发布 `BlockStored`。
- Transfer 未完成时不能复用相关 GPU block。
- Reset 不得遗留 lookup、transfer 或 socket state。
- Lazy 和 eager 路径使用一致的 object identity。

### 9.4 PIT GPU/MORI 实测记录

2026-10-09 在 PIT `pit2-p03-g10` 上通过直接 SSH 运行测试，没有创建
Slurm allocation。测试使用镜像：

```text
charles5/vllm-openai-rocm:umbp-k3-dcp-af1c01499-async-load-fix
```

环境信息：

- 8 张 MI355X GPU。
- PyTorch `2.12.0+rocm10.0.0`。
- MORI Python bindings 和 `umbp_standalone_server` 可用。

结果：

- Embedded MORI GPU integration：4 passed。
    - TP2/PP2 四进程 store/lookup/load roundtrip。
    - Store 与 GPU compute overlap。
    - DRAM eviction 后 ranged restore。
    - DRAM eviction 后 bulk restore。
- Standalone MORI GPU integration：3 passed。
    - Writer 退出后由另一进程和另一张 GPU restore。
    - 多 client 可见的全局 cache clear。
    - Server loss 降级为 lookup miss 和 transfer failure。
- Qwen3-0.6B token-level restore：2 passed。
    - Eager offload。
    - Lazy offload。

节点虽然被 Slurm 标记为 idle，但 GPU 0 当时只有约 30--67 GiB 空闲，因此
token-level 测试显式使用 `gpu_memory_utilization=0.05`。该测试同时固定
`num_gpu_blocks_override=128`，降低启动预算不会改变测试所需的 KV block 数量。

测试结束后，临时源码目录、模型缓存和测试容器均已清理。

## 10. 作者和来源记录

最终提交不是原样 cherry-pick，因此 commit message 应记录实际来源。例如：

```text
Parts adapted from:
- YukioZzz/vllm@63a3e756ec
- charxwu/vllm@37832fa995

Co-authored-by: Yichao Zhu <Yichao.Zhu@amd.com>
Co-authored-by: CharlesWu <Charles.Wu@amd.com>
```

仅当提交实际包含对应作者的代码时添加 `Co-authored-by`，不要给全部提交机械添加
相同 trailer。

## 11. 风险分析

### 11.1 Lazy offload 与 Yukio worker 精简冲突

Yukio 删除了 charxwu 的部分 lazy、layer-wise 和 generation tracking 状态。不能
整体替换 scheduler 或 worker 文件，必须按功能移植。

### 11.2 相同 upstream 不代表 API 一定兼容

Rebase 无文本冲突只能说明 patch 能应用，不能证明 runtime behavior 正确。必须通过
unit、GPU 和 model-level restore 测试确认。

### 11.3 Multi-group failure 语义

Block ID 在多个 cache group 中不一定能无歧义映射到 request。合并后应采用
request-level failure reporting，并强制相关模型走 async load。

### 11.4 Standalone failure scope

Standalone pool 是 node-global resource：

- Reset 可能影响所有连接的 engine。
- 一个 engine 可以驱逐另一个 engine 的 object。
- Server restart 会使旧 GPU registration 失效。
- Server stall 时不能释放仍被 transfer 引用的 GPU block。

这些行为必须在文档和测试中明确。

### 11.5 Commit 粒度

过大的提交难以 review；过度拆分又会产生无法运行的中间状态。建议每个提交满足：

- 导入成功。
- 对应单元测试通过。
- 不依赖未来提交修复明显错误。
- 只引入一个可描述的行为或基础能力。

## 12. 完成标准

合并工作完成需要同时满足：

- Shared base 不依赖具体 MORI runtime 实现。
- Embedded 和 standalone 使用同一 scheduler/worker。
- Lazy offload 在 embedded 下通过真实 GPU restore 测试。
- Lazy offload 在 standalone 下有明确支持或明确拒绝。
- Hybrid/Mamba failure 和 boundary state 测试通过。
- Preemption 后不存在 GPU block use-after-reuse。
- Events 和 metrics 与实际 logical completion 一致。
- 文档中的配置均有测试或 validation。
- 最终 commit stack 可以逐个构建和测试。
- PR 描述记录 AI assistance、测试结果和 model evaluation 结果。
