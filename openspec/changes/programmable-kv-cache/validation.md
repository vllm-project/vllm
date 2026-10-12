# Validation evidence index

测试方法与下一轮设计见[testing.md](../../../docs/programmable-kvc/testing.md)。
当前结论、逐轮数据及未完成验收见[实验报告](../../../docs/programmable-kvc/rollout-report-2026-10-12.md)，任务状态见[tasks.md](tasks.md)。
本文只定位证据，不重复实验叙述。
下文的 `results/` 文件均为本地实验数据，该目录已加入 `.gitignore`，Git 不跟踪这些原始文件。
版本库保留测试方法和结果报告；读取逐请求证据需要对应的本地数据或实验归档。

## Current local round

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
