# 实验档案：应用协同 KV 缓存

本文件定位[技术报告](acceptance-report-2026-10-20.md)使用的原始证据。报告正文说明设计、比较方法与结论；执行命令、源代码身份、失败尝试和检查信息在这里及原始目录中保存。
原始结果位于本地忽略目录，不随版本库分发。本文没有新增实验或修改原始数据。

## 数据目录

| 目录 | 范围 |
| --- | --- |
| [主实验与候选记录](../../benchmarks/programmable_kv_cache/results/rollout-2026-10-19/) | 主压力对照、常规容量对照、早期压力搬运与保留源副本候选，以及未采用结果 |
| [补充验证](../../benchmarks/programmable_kv_cache/results/acceptance-2026-10-19/) | 空载成本、压力预测、模型质量、共享生命周期、故障和参数更新 |
| [后续诊断与采样](../../benchmarks/programmable_kv_cache/results/acceptance-2026-10-20-gpu02/) | 容量预测及占用门槛修正、组件与直接控制、重复在线运行、完整 CPU/GPU 采样和最终核对 |

目录名称是稳定的历史标识，不代表其中所有测量使用同一物理设备分配。实际分配、运行时间与设备 UUID 保存在每组矩阵、资源日志及续跑记录中；报告不跨批次构造固定工作量收益。

## 按实验定位证据

| 报告内容 | 主要证据 |
| --- | --- |
| 主压力对照与常规容量对照 | 主实验目录中的 `final-pressure-comparison.json`、`final-fit-comparison.json`；补充验证目录中的 `main-analysis.json` 保存逐家族延迟和请求阶段统计 |
| 压力下的恢复预测 | 补充验证目录中的 `pressure-prediction-comparison.json`，以及相应预测条件的原始报告 |
| 容量充足时的基础设施成本 | 补充验证目录中的 `fit-overhead/`、`fit-overhead-summary.json` |
| 占用门槛前后的容量预测 | 后续诊断目录中的 `forecast-regression-before-fix.json`、`forecast-fix-comparison.json`、`fit-forecast-*/` |
| 组件与应用直接控制 | 后续诊断目录中的 `component-comparison.json`、`pressure-direct/`、`direct-preparation-audit.json` |
| 真实工具循环与答案口径 | 后续诊断目录中的 `live-native/`、`live-quota/`、`live-cooperative/`、`live-comparison-audit.json` |
| CPU/GPU 成本与温控 | 后续诊断目录中的 `pressure-profile/`、`profile-analysis.json`、`profile-interpretation.json`、`thermal-*.csv` |
| 独立质量与恢复验证 | 补充验证目录中的 `recovery/`、`capacity-deferral/`、`functional-summary.json` |
| 共享物理加载与准备范围 | 补充验证目录中的 `shared-load-analysis.json`、`cooperative-final-normal-preparation-audit.json`；后续诊断目录中的 `direct-preparation-audit.json` |
| 工作量、源码及清理复核 | 后续诊断目录中的 `final-analysis.json`、`frozen-source*.json`、`final-source-cleanup.json`；模型与依赖身份见各批次的身份记录 |

后续诊断目录核对了 22 次有效运行：15 次普通固定回放、6 次真实在线和 1 次独立性能采样。此前通过的 12 个矩阵摘要保持不变。该计数不包含其他目录的主对照，也不包含失败或提前中止的尝试。

## 配置与统计口径

主性能、常规容量及压力预测对照先于当前引用占用门槛的加入；容量预测实验保留前后两版，组件、在线与性能采样使用增加门槛后的规则。具体源码身份以原始记录及冻结文件摘要为准，不能把主对照的历史增益标成最终规则重新测得的增益。

固定压力回放的输入/输出长度指纹为 `454762c34448412ab351a00747072a582f38eab0d5a4e94fec7c4ead6ca8e445`。真实工具循环使用相同初始问题，但后续输入与输出可变，不具备相同工作量指纹。
普通计时包含并发等待与控制清理，排除服务启动、预热和结果文件导出。采样轮不参与普通吞吐比例；GPU 事件累计时长可能重叠。

原始在线报告保留按回答末尾数字抽取的评分。技术报告统一使用完整最终答案口径：最终请求以 `stop` 结束、含 `####` 标记，且数值正确。原生两轮的原始分数为 250/249，完整答案分数为 249/248；另两组计数在这两个口径下相同。未完成与截断答案不从分母中删除，原始报告也未被改写。

搬运字节数使用后端物理传输计数。同一物理加载可以被多个逻辑操作引用，不能按调用者再次累加。操作只覆盖部分缓存组时，其他已驻留内容可能补齐；缺少查找瞬间的完整驻留快照，不能把操作自身的覆盖缺口直接计为无效传输。

## 未采用测量与检查

共享计算干扰、采样空档超过预设 10 秒的测量不纳入性能数据。相关记录保留在 `pressure-direct-sampling-gap/`、`live-cooperative-gpu-sharing*`、`live-cooperative-gpu03-interrupted/` 及对应诊断文件中。加载阶段中止、端口检查失败和工具校验失败也保留原状，不计为完成实验。
服务主动退出后的错误与计时期间的推理失败分开；具体判定见关闭阶段诊断。中止后的遗留进程与清理顺序问题记录在 `interrupted-cleanup-diagnostic.json`，没有因此修改已通过的计时结果。

离线广泛回归为 1,999 项通过，占用门槛修改相关回归为 681 项通过，更新后的 HTTP/分析工具检查为 41 项通过，范围重叠。真实 GPU buffer 检查为 56 项通过、1 项 ROCm 专用检查跳过。对应源码的类型与代码检查通过；这些记录不替代端到端模型与性能结果。

完整启动参数和命令入口见[基准运行指南](../../benchmarks/programmable_kv_cache/README.md)，分项开发验证索引见[验证记录](testing.md)。
