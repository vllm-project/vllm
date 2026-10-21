# Validation evidence index

测试方法与下一轮设计见[testing.md](../../../docs/programmable-kvc/testing.md)。
当前验收见[剩余验收报告](../../../docs/programmable-kvc/acceptance-report-2026-10-20.md)，主性能数据见[优化报告](../../../docs/programmable-kvc/rollout-report-2026-10-19.md)，任务状态见[tasks.md](tasks.md)。
本文只定位证据，不重复实验叙述。
下文的 `results/` 文件均为本地实验数据，该目录已加入 `.gitignore`，Git 不跟踪这些原始文件。
版本库保留测试方法和结果报告；读取逐请求证据需要对应的本地数据或实验归档。

## Current acceptance

原始记录位于 `benchmarks/programmable_kv_cache/results/acceptance-2026-10-19/`。
任务 7.4/7.5/7.9/11.4 及本地 PR 描述 8.4 已完成，整体为 79/82。
用户已在收尾后重新指定 GPU 0/2，只执行新实验；显式和组件矩阵、低压力预测及其修复验证已完成，
native/quota 各两轮在线运行已完成；协同在线两次被外部 GPU 进程干扰，
队列已退出，等待独占窗口后再执行协同两轮和独立 profile。

| 数据 | 已验证范围 |
| --- | --- |
| `cpu-offline.log`、`api-sampling.log`、`gpu-buffers-driver.log` | 1,999 项相关 CPU、40 项更新 HTTP/工具检查（重叠）、56 项 GPU 通过，1 项 ROCm 跳过 |
| `functional-summary.json`、`recovery/` | 双引擎实际命中 784 token；共享分支/TTL/取消；真实参数更新及回滚；Store 失败/重启与旧 epoch 拒绝 |
| `recovery/{model-quality,cpu-model-quality}.json` | 两路径各 128 题重算/预取，均 124/128；零答案差异、零长度截断，命中 0/784 token，结束记录与在途工作清空 |
| `capacity-deferral/` | 在途预算为 1，partial/partial/partial/completed、延期 3/2/1/0；真实重试命中及普通重算回退，清理完成 |
| `runtime-identity.json`、`model-identity.json` | 18 个 vLLM 运行文件匹配此前版本；权重/配置等 21 个文件 SHA256 与环境版本 |
| `main-analysis.json`、`shared-load-analysis.json`、`cooperative-final-normal-preparation-audit.json` | 主实验逐家族尾延迟、预测误差、共享物理加载 ID 和操作范围覆盖；不能将所有命中归因于搬运 |
| `attempt-index.json`、`recovery-before-ack/`、`recovery-ack-replay-order/` | 未完成/失败的工具验证尝试，未计入通过数据 |
| `fit-overhead-summary.json`、`pressure-prediction-comparison.json` | 六轮空载开销及六轮压力预测对照；输入/输出长度相同、GPU 审计通过；30 s 预测为失准诊断输入 |
| `sampling-gap-diagnostic.json`、`pressure-direct-sampling-gap/` | 模型/控制完成但 11.072 s 采样空档超过 10 s 标准的无效性能轮，待重跑 |
| `mypy-sampling-final.log`、`hooks-sampling-staged.log` | 采样加密后 Python 3.12 类型、默认 hooks 通过；拒绝空档的标准保持不变 |
| `user-wrap-up.json` | 用户要求的停止边界、已完成/未完成任务、进程退出及无矩阵完成标记；后续需显式恢复 |

[本地 PR 描述](../../../docs/programmable-kvc/pr-description.md) 已整理当前实现、已测结果和未完成验收，
并包含重复工作边界、AI 使用说明及人工审阅/测试要求；未发布到远端。

GPU 0/2 的新记录位于 `benchmarks/programmable_kv_cache/results/acceptance-2026-10-20-gpu02/`。

| 数据 | 已验证范围 |
| --- | --- |
| `forecast-regression-before-fix.json`、`forecast-fix-comparison.json` | 长预测在低压力下产生无加载的存出和命中损失；修复后两轮零搬运、恢复完整命中，中位耗时减少 4.10% |
| `long-forecast-{before,after}-fix.log`、`forecast-guard-related.log` | 修复前两个失败行为复现；修复后 11 个针对性和 681 个完整相关检查通过 |
| `mypy-forecast-guard.log`、`hooks-forecast-guard.log` | 修复后全量 Python 3.12 类型和默认 hooks 通过 |
| `pressure-direct/`、`direct-preparation-audit.json` | 显式 movement/composed 各一轮，完整工作量与 GPU 审计通过；区分混合组覆盖与调用者归因歧义 |
| `direct-shutdown-classification.json` | 最后请求与计时结束后，主动退出阶段的 EngineDeadError；原日志保留 |
| `component-comparison.json`、五个 `pressure-*/matrix.json` | 五个单项/成对策略组全部完成，保留负面结果、逻辑 missing、物理搬运、CPU 与 32 个家族尾延迟 |
| `frozen-source{,-before-forecast-guard}.json`、`forecast-fix-comparison.json` | 本轮仅改变 `ToolWaitPolicy` 的当前压力判断，前后策略源码与记录提交对齐 |
| `verl-identity.json`、`health-checks.jsonl` | 实际依赖提交/源文件身份和约十分钟一次的轻量检查，检查自身的只读 RPC 单列 |
| `live-native/`、`live-quota/`、`completed-analysis.json` | 各两轮实际在线，GPU 审计和清理通过；合计 19 个有效运行的原始摘要、源码、时钟和工作量复核，完整矩阵仍未完成 |
| `live-quality-audit.json`、`quality-audit-{before,tests}.log` | 离线区分原始 reward 与完整最终答案；两组完整正确数均 249/248，保留不同的错误题；41 项检查通过，输入报告未修改 |
| `mypy-quality-audit.log`、`hooks-quality-audit.log` | 离线质量审计新增后的全量 Python 3.12 类型及默认 hooks 通过 |
| `live-cooperative-gpu-sharing{,-2}/`、`gpu-sharing-diagnostic{,-2}.json` | 外部 SGLang 进程分别干扰首轮和重试，计时均不采用；第二次提前中止，仅停止本任务进程 |
| `thermal-monitor.json`、`thermal-gpu.csv` | 用户要求保持设置，发现温控后每 10 s 采样；GPU 2 最高 86°C，记录软件限频；无历史温度回填 |

原生/主策略重复性能证据仍来自 GPU 0/3；新组不与其混算收益。
本轮未补跑 priority-only 无效轮或 native/quota/完整策略参考组。

## October 19 optimization round

原始数据位于[rollout-2026-10-19](../../../benchmarks/programmable_kv_cache/results/rollout-2026-10-19/)。
采用 GPU 0/3、Qwen3.8-27B BF16 TP=2、利用率 0.5、每卡实际 KV 12.12 GiB，未覆盖 KV 字节预算。
最终正常配置的三轮压力中位吞吐相对 native 高 68.0%、相对 quota 高 16.6%；
两轮容量充足计时接近 native，零搬运，命中 token 数相同。不能据小差值宣称通用零开销。

| 数据 | 范围 |
| --- | --- |
| `final-audit.json`、`final-pressure-comparison.json`、`final-fit-comparison.json` | 所选 9 个压力与 6 个容量回放的 SHA256、源码、等工作量、条件、逐轮指标和资源验证 |
| `native-v3-{0,1,2}/`、`quota-v3-{0,1,2}/` | 无 profiler 的三轮基线/配额，256 条轨迹、920 次调用、60,136 输出 token |
| `cooperative-v4-probe/`、`cooperative-v4-verify-1/`、`cooperative-final-normal/policy-0.json.gz` | 正常暂停/传输成本、占用阈值 0.8、按准入准备，所选三轮协同策略 |
| `fit-native-v3/`、`fit-quota-v3/`、`fit-cooperative-v4/` | 各两轮容量充足负载，4 个家族、256 条轨迹、864 次调用、66,160 输出 token，命中相同且无抢占 |
| `cooperative-final-normal/policy-live.json.gz`、`live-process-tree*.json`、`live-gpu-audit.json` | 真实 verl 工具循环 256 条轨迹、923 次调用、60,231 token；256 次查询/528 次算术调用，reward 248/256；在线窗口两卡各 216 个样本，无非实验进程，清理完成 |
| `v3-cpu.log`、`v3-gpu.log` | 扩展 CPU 1,993 项通过；实际 GPU 搬运 56 项通过、1 项 ROCm 专用检查跳过 |
| `v4-related-final.log`、`v4-mypy-final.log`、`v4-hooks-final.log` | 采用的正常决策与两阶段观察：671 项完整相关检查、Python 3.12 全量类型、默认 hooks 通过 |
| `restored-code-identity.json`、`v4-baseline-identity.json`、`final-normal-validation-ready.json` | 恢复后 29 个运行文件逐字节匹配 B5；quota/retention 类及未变的基线执行路径核对，源码/命令冻结 |
| `pressure-only-v2-0/`、`native-repeat/` | 控制超时/记录残留失败和 GPU 干扰失败，原始失败保留，排除于性能采用数据 |
| `pressure-only-v3-{0,1,2}/`、`fit-cooperative-v3/` | 强制压力消融：压力组快，容量充足组 483–503 s 的有效退化保留，未混入正常策略中位数 |
| `fit-cooperative-v4-p70/`、`cooperative-v4-p70-probe/` | 阈值 0.7 的探索性结果，单轮小差值不作稳定优势结论，最终仍用 0.8 |
| `fit-cooperative-v5-copy/`、`cooperative-v5-copy-probe/` | COPY 备份候选及负面/不确定结果；源提交 `2d8899bfd6`，选项已撤回，现有公共 COPY 原语保留 |
| `v3-negative.runs.json`、`v4-repeat-live.runs.json` | 实现切换时在模型加载阶段取消的服务，没有产生计时结果，不计作通过 |

实际循环是功能和自动 reward 验证，不与固定回放直接计算吞吐提升。
有 12 个生成请求达到长度上限，一条算术工具参数格式警告；本轮未逐 token 比较输出质量。
idle/noop、oracle/预测误差系统矩阵及部分故障/参数更新重验保持未完成。
本轮自建模型、Store、采样、观察进程均已结束，GPU 0/3 无本轮计算进程。

## October 18 historical round

原始数据位于[rollout-2026-10-18](../../../benchmarks/programmable_kv_cache/results/rollout-2026-10-18/)。
只使用本地 GPU 0、3；主要性能场景为利用率 0.5，无固定 KV 字节覆盖。

| 数据 | 范围 |
| --- | --- |
| `baseline/trace.json.gz`、`baseline/native-0.json.gz` | 256 条真实 verl 轨迹及一轮固定工作量原生回放；并发 64、32 个前缀家族 |
| `priority-quota/`、`quota-wait120-aging30/` | 相同利用率 0.5、固定工作量的普通 quota 30/1 与 120/30，各一轮；矩阵、资源归属与清理检查通过 |
| `stop-boundary-summary.json` | 三个普通轮的指纹、SHA256、吞吐、排队/阶段、抢占、RPC、进程 CPU 和家族延迟对比；单轮结果不代表稳定中位数 |
| `profile-device-cold-policy/` | 完整策略诊断、四进程 Yappi、两卡 CUDA trace 和逐请求/逐操作记录；诊断时间独立于普通性能对照 |
| `profile-device-cold-policy/post-validation.json` | 原矩阵的函数限定名检查失败后，独立核验客户端、指纹、清理、GPU 归属和 CPU 时间；原失败矩阵保持原样 |
| `control-coverage-before-after.json` | 共享冷内容修正前后的完整 Mamba 组准备、实际命中和传输计数 |
| `cpu-regression-merged.log`、`cpu-layout-region.log` | 合并后 1,973 项 CPU 回归通过；另有 60 项布局/区域检查通过，范围不重叠 |
| `gpu-buffers-merged.log` | 合并后实际 GPU/CPU/Mooncake 搬运 56 项通过，1 项 ROCm 专用检查跳过 |
| `precommit-driver-staged.log`、`mypy-3.12-driver-staged.log` | 包括新 profiling bootstrap 的 staged 默认检查与 Python 3.12 全量类型检查通过 |
| `quota-parameter-behavior.log`、`precommit-policy-parameters-final.log`、`mypy-3.12-policy-parameters.log` | 参数化后的 9 项 quota 行为检查、staged 默认检查与 Python 3.12 类型检查通过 |
| `commit-core.log`、`commit-driver.log`、`commit-policy-parameters.log` | 核心修复 `af2c0d7ab2`、异步驱动/性能工具 `10121d7920`、有界准入参数 `6d63f4eaa9`；提交 hook 和 sign-off 检查通过 |
| `user-stop-boundary.json`、`round-status.json` | 用户要求完成末轮后停止；实际完成/未执行范围、清理与检查状态 |
| `profile-*/` 的其他失败记录 | 采样停滞、无效 CPU 时间、共享 GPU 干扰和同步压缩造成的超时；保留原始失败，不作为性能通过记录 |

`functional/` 已完成当时版本的恢复、权重与 128 题 Mooncake 质量检查；合并后脚本的重验边界与失败另列在本轮报告。
按用户要求，完成 `quota-wait120-aging30/` 后停止；本轮相对原生单轮吞吐高 45.6%，尚未验证重复稳定性。
重复压力/容量充足对照、消融和预测保持未完成，细化输出校验暂缓；不从旧的小配额数据推断这些项目通过。

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
当前 rollout 性能结果不代表最终整套回归、128 题或完整恢复已通过。

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
