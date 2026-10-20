# 实验矩阵索引

此表列出初始快照中的全部 35 个矩阵，包括未完成、失败和已撤回候选。
“passed”是该矩阵文件的原始字段，不能单独证明它适合性能比较；采用范围以技术报告为准。
计划重复数为每个配置的目标次数，完成数为矩阵已记录的 replay/live 轮数。客户端已有结果但审计未通过时，有效轮数可能仍为 0。

| 实验目录 | 原始 passed | 计划重复/配置 | 已记录 replay / live | 报告用途 |
| --- | --- | ---: | ---: | --- |
| [acceptance-2026-10-19/fit-overhead](raw/results/acceptance-2026-10-19/fit-overhead/matrix.json) | true | 2 | 6 / 0 | 容量充足 native/idle/noop 对照 |
| [acceptance-2026-10-19/pressure-direct](raw/results/acceptance-2026-10-19/pressure-direct/matrix.json) | false | 1 | 0 / 0 | 采样断档失败；客户端 256 条已完成，排除性能结论 |
| [acceptance-2026-10-19/pressure-forecast-early](raw/results/acceptance-2026-10-19/pressure-forecast-early/matrix.json) | true | 2 | 2 / 0 | 两轮预测敏感性；30 秒组未选为默认 |
| [acceptance-2026-10-19/pressure-forecast-late](raw/results/acceptance-2026-10-19/pressure-forecast-late/matrix.json) | true | 2 | 2 / 0 | 两轮预测敏感性；30 秒组未选为默认 |
| [acceptance-2026-10-19/pressure-oracle](raw/results/acceptance-2026-10-19/pressure-oracle/matrix.json) | true | 2 | 2 / 0 | 两轮预测敏感性；30 秒组未选为默认 |
| [rollout-2026-10-18/baseline](raw/results/rollout-2026-10-18/baseline/matrix.json) | false | 3 | 1 / 0 | 历史基线只完成 1/3；保留 trace，不计入三轮主对照 |
| [rollout-2026-10-18/priority-quota](raw/results/rollout-2026-10-18/priority-quota/matrix.json) | true | 1 | 1 / 0 | 阶段性对照或候选；详见技术报告与原始矩阵 |
| [rollout-2026-10-18/quota-wait120-aging30](raw/results/rollout-2026-10-18/quota-wait120-aging30/matrix.json) | true | 1 | 1 / 0 | 阶段性对照或候选；详见技术报告与原始矩阵 |
| [rollout-2026-10-19/cooperative-final-normal](raw/results/rollout-2026-10-19/cooperative-final-normal/matrix.json) | true | 1 | 1 / 1 | 已采用三轮主对照的组成数据；真实工具循环单列 |
| [rollout-2026-10-19/cooperative-v2-0](raw/results/rollout-2026-10-19/cooperative-v2-0/matrix.json) | true | 1 | 1 / 0 | 阶段性对照或候选；详见技术报告与原始矩阵 |
| [rollout-2026-10-19/cooperative-v4-p70-probe](raw/results/rollout-2026-10-19/cooperative-v4-p70-probe/matrix.json) | true | 1 | 1 / 0 | 未采用候选，单轮压力结果不构成稳定收益 |
| [rollout-2026-10-19/cooperative-v4-probe](raw/results/rollout-2026-10-19/cooperative-v4-probe/matrix.json) | true | 1 | 1 / 0 | 已采用三轮主对照的组成数据；真实工具循环单列 |
| [rollout-2026-10-19/cooperative-v4-verify-1](raw/results/rollout-2026-10-19/cooperative-v4-verify-1/matrix.json) | true | 1 | 1 / 0 | 已采用三轮主对照的组成数据；真实工具循环单列 |
| [rollout-2026-10-19/cooperative-v5-copy-probe](raw/results/rollout-2026-10-19/cooperative-v5-copy-probe/matrix.json) | true | 1 | 1 / 0 | 未采用候选，单轮压力结果不构成稳定收益 |
| [rollout-2026-10-19/fit-cooperative-v3](raw/results/rollout-2026-10-19/fit-cooperative-v3/matrix.json) | true | 2 | 2 / 0 | 有效负面结果：容量充足时搬运导致退化 |
| [rollout-2026-10-19/fit-cooperative-v4](raw/results/rollout-2026-10-19/fit-cooperative-v4/matrix.json) | true | 2 | 2 / 0 | 阶段性对照或候选；详见技术报告与原始矩阵 |
| [rollout-2026-10-19/fit-cooperative-v4-p70](raw/results/rollout-2026-10-19/fit-cooperative-v4-p70/matrix.json) | true | 2 | 2 / 0 | 未采用候选，单轮压力结果不构成稳定收益 |
| [rollout-2026-10-19/fit-cooperative-v5-copy](raw/results/rollout-2026-10-19/fit-cooperative-v5-copy/matrix.json) | true | 2 | 2 / 0 | 未采用候选，单轮压力结果不构成稳定收益 |
| [rollout-2026-10-19/fit-native-v3](raw/results/rollout-2026-10-19/fit-native-v3/matrix.json) | true | 2 | 2 / 0 | 阶段性对照或候选；详见技术报告与原始矩阵 |
| [rollout-2026-10-19/fit-quota-v3](raw/results/rollout-2026-10-19/fit-quota-v3/matrix.json) | true | 2 | 2 / 0 | 阶段性对照或候选；详见技术报告与原始矩阵 |
| [rollout-2026-10-19/live-cooperative-v3](raw/results/rollout-2026-10-19/live-cooperative-v3/matrix.json) | false | 1 | 0 / 0 | 历史失败或中止记录；排除主性能结论 |
| [rollout-2026-10-19/native-clean](raw/results/rollout-2026-10-19/native-clean/matrix.json) | true | 1 | 1 / 0 | 阶段性对照或候选；详见技术报告与原始矩阵 |
| [rollout-2026-10-19/native-repeat](raw/results/rollout-2026-10-19/native-repeat/matrix.json) | false | 2 | 0 / 0 | 历史失败或中止记录；排除主性能结论 |
| [rollout-2026-10-19/native-v3-0](raw/results/rollout-2026-10-19/native-v3-0/matrix.json) | true | 1 | 1 / 0 | 已采用三轮主对照的组成数据；真实工具循环单列 |
| [rollout-2026-10-19/native-v3-1](raw/results/rollout-2026-10-19/native-v3-1/matrix.json) | true | 1 | 1 / 0 | 已采用三轮主对照的组成数据；真实工具循环单列 |
| [rollout-2026-10-19/native-v3-2](raw/results/rollout-2026-10-19/native-v3-2/matrix.json) | true | 1 | 1 / 0 | 已采用三轮主对照的组成数据；真实工具循环单列 |
| [rollout-2026-10-19/pressure-only-v2-0](raw/results/rollout-2026-10-19/pressure-only-v2-0/matrix.json) | false | 1 | 0 / 0 | 历史失败或中止记录；排除主性能结论 |
| [rollout-2026-10-19/pressure-only-v3-0](raw/results/rollout-2026-10-19/pressure-only-v3-0/matrix.json) | true | 1 | 1 / 0 | 阶段性对照或候选；详见技术报告与原始矩阵 |
| [rollout-2026-10-19/pressure-only-v3-1](raw/results/rollout-2026-10-19/pressure-only-v3-1/matrix.json) | true | 1 | 1 / 0 | 阶段性对照或候选；详见技术报告与原始矩阵 |
| [rollout-2026-10-19/pressure-only-v3-2](raw/results/rollout-2026-10-19/pressure-only-v3-2/matrix.json) | true | 1 | 1 / 0 | 阶段性对照或候选；详见技术报告与原始矩阵 |
| [rollout-2026-10-19/quota-0](raw/results/rollout-2026-10-19/quota-0/matrix.json) | true | 1 | 1 / 0 | 阶段性对照或候选；详见技术报告与原始矩阵 |
| [rollout-2026-10-19/quota-v2-0](raw/results/rollout-2026-10-19/quota-v2-0/matrix.json) | true | 1 | 1 / 0 | 阶段性对照或候选；详见技术报告与原始矩阵 |
| [rollout-2026-10-19/quota-v3-0](raw/results/rollout-2026-10-19/quota-v3-0/matrix.json) | true | 1 | 1 / 0 | 已采用三轮主对照的组成数据；真实工具循环单列 |
| [rollout-2026-10-19/quota-v3-1](raw/results/rollout-2026-10-19/quota-v3-1/matrix.json) | true | 1 | 1 / 0 | 已采用三轮主对照的组成数据；真实工具循环单列 |
| [rollout-2026-10-19/quota-v3-2](raw/results/rollout-2026-10-19/quota-v3-2/matrix.json) | true | 1 | 1 / 0 | 已采用三轮主对照的组成数据；真实工具循环单列 |

主性能的精确采用名单、逐轮原文件 SHA256 和版本见 [压力比较](raw/results/rollout-2026-10-19/final-pressure-comparison.json)；
质量/故障的失败尝试另见 [尝试索引](raw/results/acceptance-2026-10-19/attempt-index.json)。

初始采集后的队列恢复、采样修正、任务清单与补充分析保存于 [增量入口](latest-update.json)。
待启动的单项/组合消融、profile、在线重复与容量预测实验列在 [技术报告](technical-report.md)；没有结果的组不填入性能数值。
