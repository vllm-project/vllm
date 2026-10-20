# Programmable KV Cache 实验结果快照

本目录保存 2026-10-20 00:07:26–00:08:07（America/Phoenix，UTC−07:00）采集的结果。
初始归档包含 917 个证据文件、665,307,876 字节，约 634.5 MiB。
同日 01:38:05–01:38:22 补充保存 48 个新增或变化的证据文件，原副本未覆盖。
两次合计 **965 个证据文件、684,894,533 字节，约 653.2 MiB**，含同一文件不同时点的版本。
这是独立副本，原实验目录后续产生的新结果不会自动更新本目录。

- [对外技术报告 HTML](technical-report.html)：图表和样式已内嵌，可单独发送；[Markdown 版本](technical-report.md)用于编辑。
- [英文技术报告 HTML](technical-report-en.html)：英文正文与图表，可单独发送；[英文 Markdown 版本](technical-report-en.md)用于编辑。
- [实验索引](experiment-index.md)：35 个矩阵的重复计划、已有轮数、失败和采用范围。
- [机器可读汇总](summary.json)：逐轮耗时、指标、预测实验、验收状态及未采用结果。
- [证据清单](evidence-manifest.json)：每个归档文件的原路径、保存位置、长度和 SHA256。
- [补充证据清单](updates/2026-10-20-013805/evidence-manifest.json)及[补充状态](updates/2026-10-20-013805/status.json)：队列恢复、采样修正、77/82 任务状态和新增分析。
- [性能图 SVG](performance.svg) / [PNG](performance.png)：主对照、预测敏感性与容量退化结果。
- [英文性能图 SVG](performance-en.svg) / [PNG](performance-en.png)：对应英文报告，数据与中文版相同。
- [归档脚本](capture_snapshot.py)和[汇总脚本](summarize_snapshot.py)：记录采集与计算方法。
- [增量采集脚本](capture_update.py)：另建时间目录保存变化文件，复用未变化证据；[最新增量入口](latest-update.json)记录路径。
- [校验结果](verification.json)及[校验脚本](verify_snapshot.py)：逐文件重算 SHA256、检查本报告链接并复算吞吐比例。

原始数据副本放在 `raw/`：

```text
raw/
├── results/
│   ├── acceptance-2026-10-19/     本轮全部现有文件，包括失败及尚未完成的矩阵
│   ├── rollout-2026-10-19/        主性能、容量对照、负面候选、真实工具循环
│   └── rollout-2026-10-18/        基础 trace、历史 quota 对照与顶层元数据
└── source/                       RFC、OpenSpec、实现及相关测试的源码副本
updates/2026-10-20-013805/          增量清单、状态及变化文件的独立副本
```

`raw/` 及增量目录内的 `raw/` 按本目录的 `.gitignore` 排除出 Git，文件已实际保存在磁盘。
移交完整实验时应复制整个目录，包括这两处原始数据；单独提交 Markdown 不包含原始证据。
模型权重、原始数据集与虚拟环境未复制；模型/数据集标识和哈希在归档元数据中。
10 月 18 日体积较大的历史 profiler 子目录未复制，原位置仍保留；其耗时不用于本报告的主要收益计算。

采集按文件读取当时的长度，记录源文件是否在复制中变化；这次没有短读、不可解析 JSON 或复制中变化。
它属于一个明确时间窗口内的快照，不能视为对运行系统的全局原子冻结。
归档脚本只创建本目录内的文件，没有修改运行代码、实验队列或 OpenSpec 状态。

初始采集时 OpenSpec 为 76/82。`pressure-direct` 的 priority 客户端已完成全部轨迹，
但 GPU 采样最大间隔 11.072 秒超过 10 秒门槛，矩阵失败并停止主队列。
该轮作为诊断数据保存；movement/composed 尚无完成结果。其余未启动实验也不记为完成。

补充采集时 OpenSpec 为 **77/82**，7.5 的综合评估已归档。
队列已恢复，正在等待 GPU 0、3 空闲；采样周期从 5 秒改为 1 秒，最大允许间隔仍为 10 秒。
新增同采样参考组和显式指令重跑尚未执行，没有新性能结果替换原三轮结论。
对外技术报告介绍设计与实验结论，内部任务和队列状态保留在本说明与归档中；`summary.json` 保持初始快照的独立可复算汇总。

校验采用结果的原有 SHA256 与归档清单一致，18 个运行文件与已验证运行版本的标识一致。
汇总脚本只读取本目录副本，可在仓库根目录执行：

```bash
.venv/bin/python docs/programmable-kvc/experiment-reports/2026-10-20-000635-snapshot/summarize_snapshot.py
.venv/bin/python docs/programmable-kvc/experiment-reports/2026-10-20-000635-snapshot/verify_snapshot.py
```

采集脚本拒绝覆盖已有快照；重新采集应使用新目录。
