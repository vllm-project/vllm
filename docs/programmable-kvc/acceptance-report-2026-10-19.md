# Programmable KV cache 剩余验收（2026-10-19 启动）

本轮按用户要求继续完成 OpenSpec 剩余任务。外部两个模型的验收要求已移除，
范围为本机 GPU 0/3、Qwen3.8-27B BF16 TP=2；结论限定于实际验证的配置。
原始记录位于 `benchmarks/programmable_kv_cache/results/acceptance-2026-10-19/`。
该目录按仓库约定不纳入 Git；本文记录复现入口、结果与使用限制。

引擎运行代码保持上一轮版本：`runtime-identity.json` 核对先前快照中的 18 个
vLLM 文件逐字节一致，当前修改集中在验收工具。上一轮的三轮主性能与两轮容量对照
仍见[优化报告](rollout-report-2026-10-19.md)，本轮增加消融、预测和恢复证据。
`model-identity.json` 保存权重、配置等 21 个文件的 SHA256、下载元数据和依赖版本。

## 最终版本正确性与恢复

功能服务采用 TP=2、最大上下文 4096、并发 8、eager 执行。
两套引擎共用 GPU 0/3 时，每套显存利用率为 0.45；此配置用于正确性验证，
不参与主要性能收益计算。主要性能矩阵保持利用率 0.5，未设置固定 KV 字节上限。

| 验证 | 实测结果 |
| --- | --- |
| 离线相关 CPU 回归 | 1,999 项通过，633.87 s；覆盖控制、状态、引擎、HTTP、调度、前缀、Mooncake、offloading 和 tiering |
| 更新后的 HTTP/工具回归 | 36 项通过，11.37 s；与上述套件重叠，额外覆盖相反预测误差及 ACK 后保留失败结果 |
| 实际 GPU buffer 搬运 | 56 项通过，1 项 ROCm 专用检查跳过，23.74 s；CPU 两方向、布局、batch/Triton swap 和两卡 Mooncake |
| 双引擎 Mooncake | 目的端实际命中 784 token，短贪心输出一致 |
| 共享分支与生命周期 | root/B/C 清理保留外部共享任务；优先级 90→20→0；重复提交、COPY/resume 竞争及取消后的恢复通过；结束记录和关联为 0 |
| 真实参数更新/回滚 | IPC 修改 `model.language_model.layers.0.input_layernorm.weight`；新 namespace 不命中旧 KV，输出 logprob 改变；回滚后恢复旧 Store 内容复用；旧 epoch 被拒绝 |
| Store 停机与引擎重启 | 停机后的 COPY 明确失败，无虚假 GPU 驻留；重算零命中且短贪心输出一致；重启拒绝旧 epoch、快照为空，重算后重建关联并清理 |
| 有限加载预算与恢复 | 真实在途预算为 1，四次 COPY 状态为 partial/partial/partial/completed，延期目标数为 3/2/1/0；重试后命中完整 784-token 前缀；部分准备时直接恢复可正确重算；结束无记录、关联或在途工作 |

有限预算限制的是同时在途的加载块数，未人为缩小 GPU KV 字节容量。
权重检查使用 trace 的一个完整可复用边界加 128 token，共 912 token，
实际修改 GPU 参数并回滚；未执行 optimizer 更新。

128 题使用固定 seed=0、相同样本及 8-shot 提示，各路径分别比较强制重算与实际搬运后预取：

| 路径 | 重算正确 | 预取正确 | 答案差异 | 长度截断（两组） | 实际命中（重算/预取） |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mooncake | 124/128 | 124/128 | 0 | 0/0 | 每题 0/784 token |
| CPU offload | 124/128 | 124/128 | 0 | 0/0 | 每题 0/784 token |

两条路径的准确率均为 96.875%。非故障注入的两组质量检查，后端失败、缺失和丢弃均为 0，
结束时结果记录为 0、策略及设备搬运排空。质量结果说明这批题目未观察到退化，
不代表未测模型、上下文和执行配置的普遍保证。

原始证据为 `recovery/{cross-engine,model-quality,cpu-model-quality}.json`、
`recovery/{lifecycle,weight-update,recovery}.json.gz`、
`capacity-deferral/capacity-deferral.json.gz`、`functional-summary.json` 及相关日志。
执行命令保存在 `queue.json` 与恢复报告中；完整驱动入口见[基准指南](../../benchmarks/programmable_kv_cache/README.md)。

## 控制成本与原始数据复核

三轮主实验重新统计后，quota 和正常协同策略的 32 个家族 P95 中位数均低于 native。
逐家族 P95 相对 native 的最大比值分别为 0.7855 和 0.7293；每轮每家族只有 8 条轨迹，仍属小样本描述。
`main-analysis.json` 保留逐家族比值、预测的有符号/绝对误差与客户端等待分布。
相反方向误差不能通过平均值相抵后被隐藏，新增行为回归覆盖这一点。

客户端 CPU 中位数为 native/quota/协同 71.81/54.73/127.31 s。
正常协同每轮 submit/operation/ACK/association 的次数中位数为
1,584/15,070/4,420/257。异步控制降低关键路径等待，网络与 CPU 成本仍然存在。
RPC 时间互有重叠，不能相加作为整轮耗时。

`shared-load-analysis.json` 按物理 transfer ID 对齐不同逻辑 COPY：
三轮分别有 748、814、688 个加载被多个逻辑操作引用，最多 3 个操作共享一次加载。
搬运量使用后端物理计数，不重复累加各调用者的结果。

`cooperative-final-normal-preparation-audit.json` 的三轮操作范围审计显示，
报告覆盖兼容整组前缀的 COPY 为 135/139/95 次，随后完整前缀命中为 128/134/89 次。
其余操作范围存在覆盖间隙或检查点不一致；部分请求仍有缓存命中。
这些结果缺少每次查找时的完整驻留快照，不能将这些操作全部判为浪费，
也不能将每次命中全部归因于搬运。

CPU-only 元数据测量 `control-overhead.json` 为五轮：
64/4096 条结果的字典查询中位数约 0.194/0.187 μs；
1 万/10 万块条件下的空请求观察约 3.7 μs，缓存 guidance 约 0.15 μs。
这类测量未执行推理或真实请求扫描，不能代替服务端 profile 或宣称通用零开销。

## 验收工具修正与未采用记录

同步 HTTP 示例收齐终态后 ACK；重复提交检查安排在 ACK 前，ACK 后明确验证
`acknowledged` 拒绝再次执行。终态失败内容在 ACK 前保存，回归检查防止错误被成功 ACK 覆盖。
生命周期和重启用例额外核对清理后的结果、关联及在途工作。
在线矩阵增加独立重复与 GPU 归属/采样覆盖审计。

`cpu-regression.log` 的首个回归命令混入需下载模型的集成文件，已中止；
`model-download-diagnostic.log` 确认未缓存 Llama 的离线快照缺失。
通过的相关离线回归与本机 Qwen CPU/Mooncake 模型验证单独记录，未将下载失败计为通过。

`recovery-before-ack/` 为修正同步示例未 ACK 时中止的轮次；
`recovery-ack-replay-order/` 保留 ACK 后仍期待重放原结果而触发 409 的失败。
`attempt-index.json` 对齐原执行时间和归档位置，二者均不作为完成验收的证据。
上一轮强制压力的容量退化与 COPY 备份候选仍保留在优化报告和原始目录中。

## 尚在执行的矩阵

已完成任务 7.9 和 11.4。其余任务继续执行：容量充足 native/idle/noop 各两轮；
压力下显式 priority/movement/composed、单项及成对本地策略；
分别标记的记录暂停 oracle 与过早/过晚预测各两轮；
独立 CPU/GPU profile；真实 verl native/quota 各两轮与协同补充轮。
沿用完整 256 条轨迹与指定资源，不将单轮消融写成稳定收益。
全部结果汇总前，相关广义验收项继续保持未完成。
