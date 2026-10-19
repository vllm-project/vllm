# Validation evidence index

测试方法与下一轮设计见[testing.md](../../../docs/programmable-kvc/testing.md)。
当前结论、逐轮数据及未完成验收见[本轮实验报告](../../../docs/programmable-kvc/rollout-report-2026-10-18.md)，任务状态见[tasks.md](tasks.md)。
本文只定位证据，不重复实验叙述。
下文的 `results/` 文件均为本地实验数据，该目录已加入 `.gitignore`，Git 不跟踪这些原始文件。
版本库保留测试方法和结果报告；读取逐请求证据需要对应的本地数据或实验归档。

## Current local round

原始数据位于[rollout-2026-10-18](../../../benchmarks/programmable_kv_cache/results/rollout-2026-10-18/)。
只使用本地 GPU 0、3；主要性能场景为利用率 0.5，无固定 KV 字节覆盖。

| 数据 | 范围 |
| --- | --- |
| `baseline/trace.json.gz`、`baseline/native-0.json.gz` | 256 条真实 verl 轨迹及一轮固定工作量原生回放；并发 64、32 个前缀家族 |
| `profile-device-cold-policy/` | 完整策略诊断、四进程 Yappi、两卡 CUDA trace 和逐请求/逐操作记录；诊断时间独立于普通性能对照 |
| `profile-device-cold-policy/post-validation.json` | 原矩阵的函数限定名检查失败后，独立核验客户端、指纹、清理、GPU 归属和 CPU 时间；原失败矩阵保持原样 |
| `control-coverage-before-after.json` | 共享冷内容修正前后的完整 Mamba 组准备、实际命中和传输计数 |
| `cpu-regression-merged.log`、`cpu-layout-region.log` | 合并后 1,973 项 CPU 回归通过；另有 60 项布局/区域检查通过，范围不重叠 |
| `gpu-buffers-merged.log` | 合并后实际 GPU/CPU/Mooncake 搬运 56 项通过，1 项 ROCm 专用检查跳过 |
| `precommit-driver-staged.log`、`mypy-3.12-driver-staged.log` | 包括新 profiling bootstrap 的 staged 默认检查与 Python 3.12 全量类型检查通过 |
| `commit-core.log`、`commit-driver.log` | 核心修复 `af2c0d7ab2` 与异步驱动/性能工具 `10121d7920`；提交 hook 和 sign-off 检查通过 |
| `profile-*/` 的其他失败记录 | 采样停滞、无效 CPU 时间、共享 GPU 干扰和同步压缩造成的超时；保留原始失败，不作为性能通过记录 |

`functional/` 已完成当时版本的恢复、权重与 128 题 Mooncake 质量检查；合并后脚本的重验边界与失败另列在本轮报告。
按用户最新要求，优先运行 `priority-quota/` 的普通压力计时并逐次提交，细化输出校验暂缓。
重复压力/容量充足对照和预测实验仍待完成，结果逐项记录，不从旧的小配额数据推断。

## October 11 historical round

原始数据位于[rollout-2026-10-11](../../../benchmarks/programmable_kv_cache/results/rollout-2026-10-11/)。

| 数据 | 范围 |
| --- | --- |
| `pressure/matrix.json` 与逐轮 `*.json.gz` | 2 GiB 原生/完整策略，各三轮 |
| `fit/matrix.json` 与逐轮 `*.json.gz` | 8 GiB 原生/完整策略，各三轮，含退化 |
| `ablations/` | noop、priority、搬运、composed 单轮诊断 |
| `quota-only/`、`quota-verify/` | 三轮配额回放及一次真实配额在线运行 |
| `optimized-native-live.json.gz`、`optimized-policy-live.json.gz` | 原生、完整策略真实在线工具循环 |
| `lifecycle.json.gz`、`weight-update.json.gz` | 本轮实际生命周期和 IPC 参数更新/回滚 |
| `kv-rollout-summary.json`、`cache-fit-diagnosis.json` | 汇总与停止后只读诊断；不能代替原始请求 |
| `round-status.json` | 停止边界、已执行/未执行项和当时检查状态 |
| `*.log.gz`、`gpu.csv.gz`、`*.pressure-source.gz`、`runtime.patch.gz` | 服务/GPU 采样和测量对应源码 |

回放共 19 轮，另有三次在线运行；回放摘要一致。
脚本运行成功与 25% 性能目标通过分别判定。
当前 rollout 性能结果不代表最终整套回归、128 题、完整恢复或外部模型已通过。

## Earlier stages

历史文件保存在[results](../../../benchmarks/programmable_kv_cache/results/)，属于各自阶段的模型/配置/源码记录，不作为当前工作区的新验收。

| 文件族 | 检查范围 |
| --- | --- |
| `qwen3.8-27b-tp2-{baseline,native-control}.json.gz` | 重算/APC 基线和原生元数据控制 |
| `qwen3.8-27b-tp2-{http-control,request-associations}.json.gz` | HTTP 控制与请求关联/增长 |
| `qwen3.8-27b-tp2-mooncake-{control,cross-engine,evaluation}.json.gz` | Store 搬运、目的引擎复用及质量 |
| `qwen3.8-27b-tp2-cpu-{control,evaluation}.json.gz` | 通用 CPU 搬运及质量 |
| `qwen3.8-27b-tp2-quality-{head-http,quota-cpu}.json.gz` | 相应阶段的最终进程/策略正确性记录 |
| `rfc59708-{attached,policy,composed,cross-engine,evaluation}.json.gz` | 基本操作生命周期、策略和质量 |
| `control-overhead-2026-10-06.json.gz` | CPU 元数据微测量，非 serving 吞吐 |

花括号表示列出的多个文件名。
执行上下文以各 JSON 内容为准。

完整逐阶段日志、被替代的报告/审计/草稿和清理前源码已移至仓库外实验归档，归档 manifest 保留原路径与 SHA256。
提交 `8b1a1b0f2d` 中的旧版本也保留早期日志。
当前仓库只维护测试方法、实验报告和本索引；未执行的验收仍在 tasks 中。
