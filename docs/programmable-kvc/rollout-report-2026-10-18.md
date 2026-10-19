# 控制开销与较大工作集实验（2026-10-18）

本轮按用户要求恢复实验和代码优化，按完整修改单元在相关检查通过后逐次提交本地 Git commit。
主要性能实验使用 GPU 0、3、Qwen3.8-27B TP=2、`gpu_memory_utilization=0.5`，不指定固定的 2/8 GiB KV 上限。
通过增加问题数、前缀家族、采样数与并发形成压力；容量充足对照在本轮停止前尚未执行。
本轮未开始发布 PR。

本轮按用户指定边界停止。普通压力点的原生、quota 30/1、quota 120/30 各完成一轮相同工作量，耗时分别为 **1,994.30、1,898.86、1,369.48 秒**。
120 秒等待上限、30 秒老化尺度的组合相对原生吞吐提高 **45.6%**，抢占从 9 次降至 2 次，TTFT P99 从 123.42 秒降至 71.21 秒。
该单轮超过 25% 目标；重复稳定性、容量充足时的边际开销和外部模型仍未验证。参数组合保留为实验选项，引擎默认值未改。

## 测试设计

控制客户端负责提交 hint、收集真实异步结果、推进暂停/恢复和完成清理；输入为真实请求与支持的动作，输出为推理轨迹和操作结果。
主要防护目标是移出等待后出现的遗漏结果、过早清理、重复 ACK 与加载竞态。
最小行为检查复用 HTTP suite，使用受控的异步响应验证推理与控制重叠、结束时仍等待控制，以及提交回执不能掩盖未完成子操作。
服务端复用前缀缓存 suite，验证无人消费信号不扫描请求，第三方策略仍收到默认信号。
完整单元、CPU/Mooncake adapter、真实 GPU 恢复、权重与模型检查分别记录；未执行项保持未完成。

## 执行记录

原始日志、命令、源代码摘要、逐请求结果与 GPU 采样保存到 `benchmarks/programmable_kv_cache/results/rollout-2026-10-18/`。
该目录被 Git 忽略；本文件保留可审阅结论。
本轮实验已结束，原始成功、失败与用户指定中断的记录均保留；未执行项见文末停止边界。
主要版本为 PyTorch `2.13.0+cu132`、Triton `3.7.1`、NCCL `2.29.7`、Mooncake CUDA 13 包 `0.3.13.post1`、Transformers `5.17.0`；初始完整依赖清单见 `environment.txt`，后加的可选 CPU profiler 为 Yappi `1.7.6`。

### 已执行的回归

所有命令使用 `.venv/bin/python`、`CUDA_VISIBLE_DEVICES=0,3` 和 `OMP_NUM_THREADS=1`。

| 日志 | 范围与结果 |
| --- | --- |
| `cpu-regression-merged.log` | 合并脚本/测试整理后的完整相关 CPU 回归：1,973 项通过，626.83 秒 |
| `gpu-buffers-merged.log` | 合并后的五个实际 GPU 搬运文件：56 项通过，1 项 ROCm 专用检查跳过，29.03 秒 |
| `precommit-merged-lifetime.log`、`mypy-3.12-merged-lifetime.log` | 合并代码及 Store host 生命周期修正后的默认 hook 与 Python 3.12 全部类型检查通过 |
| `cpu-regression-device-cold.log` | 修正线程设备与共享冷副本线索后的完整相关 CPU 回归：1,976 项通过，646.58 秒 |
| `gpu-buffers-device-cold.log` | GPU/CPU 与 Mooncake 实际搬运完整文件重跑：94 项通过，1 项 ROCm 专用检查跳过，39.08 秒；原生控制用例同时断言后台线程所在设备 |
| `shared-cold-regression-final.log` | 两个会话共享冷内容，清理原 owner 后仍可恢复完整前缀；容量 2 时实际命中 8 token，容量 1 淘汰线索后不虚报命中，2 项通过 |
| `cpu-regression-current.log` | 当前代码的控制、索引、engine、HTTP、prefix/scheduler、Mooncake、offloading、CPU manager/policies 和 tiering：1,970 项通过，673.16 秒 |
| `cpu-layout-region.log` | Mooncake hybrid 布局与共享内存区域：60 项通过，41.54 秒；与上行范围不重叠 |
| `gpu-buffers.log` | CPU offload worker、canonical layout、batch/Triton swap 及真实 Mooncake GPU buffer：94 项通过，1 项 ROCm 专用检查跳过，40.74 秒 |
| `control-retention-suite.log` | 修正 retention 消融驱动后，HTTP 与 prefix caching 完整文件 254 项通过，81.70 秒；其中 253 项与主回归重叠 |
| `control-resource-suite.log` | 加入 GPU 进程审计后，上述完整文件 257 项通过，79.41 秒；新增 3 项覆盖测量窗口内外的共享进程判定 |
| `unit.log` | 控制、索引、engine、HTTP、prefix caching、scheduler 与 Mooncake scheduler：684 项通过，374.35 秒 |
| `adapter-diagnosis.log` | Mooncake layout/connector/coordinator/worker/stats：292 项通过；随后通用集成用例访问受限 Llama 模型得到 HF 403，运行中止 |
| `adapter-offline.log` | offloading connector、CPU manager/policies、tiering/KVCR：988 项通过，389.64 秒 |
| `control-regressions.log` | 终态回执与异步子操作、推理/控制重叠与清理、无人消费信号、第三方策略兼容：4 项通过 |
| `control-regressions-final.log` | 上述行为及长生成期间收集结果、失败时等待其余控制完成：6 项通过 |
| `control-regressions-complete.log` | 加入跨轮次控制顺序与重复暂停事件检查：8 项通过 |
| `control-regressions-with-cleanup.log` | 再验证实际 HTTP/service 返回的清理完成状态、session revision 与其他 session 的独立生命周期：9 项通过 |
| `control-regressions-failure-cleanup.log` | 加入推理失败后取消并等待其余 episode 的行为检查：10 项通过，与上述用例重叠，不重复计数 |
| `runner-interrupt-check.log` | SIGTERM 触发 finally，测试自建进程组已停止并回收 |
| `precommit-retention-final.log` | retention 修正后的全部默认 pre-commit hook 通过；前一轮发现的超长测试签名已由格式钩子修正 |
| `mypy-3.12-retention.log` | `pre-commit run mypy-3.12 --all-files --hook-stage manual` 通过，检查期间未修改受检文件 |

首轮更宽的 adapter 命令因模型下载错误出现失败，保留在 `adapters.log`，诊断后按离线与真实 GPU 集成分别执行。
未将受限模型用例标为通过，也未为绕过模型权限修改测试。
本地 Qwen 的 CPU/Mooncake 实际搬运与模型质量验证另行记录。

当前 CPU 完整回归命令如下；另两项布局/区域测试使用同一环境单独运行。

```bash
CUDA_VISIBLE_DEVICES=0,3 OMP_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest \
  tests/v1/test_kv_hint_control.py tests/v1/test_kv_hint_state.py \
  tests/v1/engine/test_kv_control.py tests/entrypoints/serve/kv_control/test_api.py \
  tests/v1/core/test_prefix_caching.py tests/v1/core/test_scheduler.py \
  tests/v1/kv_connector/unit/test_mooncake_store_scheduler.py \
  tests/v1/kv_connector/unit/test_mooncake_store_layout.py \
  tests/v1/kv_connector/unit/test_mooncake_store_connector.py \
  tests/v1/kv_connector/unit/test_mooncake_store_coordinator.py \
  tests/v1/kv_connector/unit/test_mooncake_store_worker.py \
  tests/v1/kv_connector/unit/test_mooncake_stats.py \
  tests/v1/kv_connector/unit/offloading_connector \
  tests/v1/kv_offload/cpu/test_manager.py tests/v1/kv_offload/cpu/policies \
  tests/v1/kv_offload/tiering -q
```

真实 GPU 搬运检查使用同一环境运行以下完整文件：
`tests/v1/kv_offload/cpu/{test_gpu_worker,test_canonical_layout,test_swap_blocks_batch,test_swap_blocks_triton}.py`
与 `tests/v1/kv_connector/mooncake_integration/test_store_buffers.py`。
覆盖两个方向的值保真、partial miss、跨 rank 完成后发布、reset、共享布局和 compute stream 写入后加载；Mooncake 在物理 GPU 0 与 3 间实际传输。

审查发现只启用 retention 的消融驱动仍发送 `tool_wait`；该策略只消费 importance/resume，因此这组配置没有触发保留。
驱动增加 retention 模式发送 importance，关闭无搬运组件所需的 connector，并要求结果中存在保留策略产生的 `priority_applied`。
新增检查通过实际 HTTP/service 与缓存管理器观察到优先级 50、零搬运，以及任务结束后的优先级清理。
独立运行该用例时，首轮因未初始化跨文件 helper 依赖的空块 hash 失败；显式初始化后通过，254 项完整文件回归也通过。

### 工作集压力

首个服务使用 `gpu_memory_utilization=0.5`、TP=2、最大服务并发 64、batch token budget 4096。
每卡权重占用 25.69 GiB，可用 KV 预算为 12.12 GiB，引擎报告 307,048 token 容量。
在线负载为 64 个 GSM8K 问题、每题 4 次采样、32 个前缀家族，每个家族 48 个参考例题，并发 64。
完整在线运行的 KV 占用峰值为 100%，出现实际抢占，缓存命中较低。

首轮原生在线采集已完成：256 条轨迹、920 次调用、60,136 个输出 token，耗时约 2,010.64 秒。
这是 64 个独立问题各 4 次采样；奖励正确 249/256，12 次模型调用达到长度上限。
日志保留缺少工具参数和畸形工具调用的反馈，不将这些轨迹删除或替换。
输入共 8,781,870 token，报告命中 813,792 token（约 9.27%）；KV 占用峰值 100%，13 次抢占。
TTFT P50/P95/P99 为 81.06/119.06/130.24 秒。
服务端逐请求平均排队、prefill、decode 区间为 74.11、4.58、54.14 秒；decode 区间包含与其他请求预填充交错时的等待，不能当作纯 GPU 解码算子耗时。
该组是在线轨迹采集与质量记录；普通固定工作量单轮比较见文末，多次重复未在停止前执行。

原始服务日志使用执行主机的 UTC+08:00 时间；本报告和目录沿用会话开始时的用户日期。

### 评分标签核查

原始奖励分数保持 249/256。逐条核查七条 0 分轨迹后发现：四条属于 test[454]，另外三条的末次输出达到长度上限。
test[454] 的题面为两人每天各吃 4 个苹果、共 30 天，模型四次均回答 240；数据答案按每天共 5 个计算并标注 150，与题面冲突。
[OpenAI 上游 GSM8K 数据](https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl) 中也存在同一题面与答案组合。
本地文件 SHA256 与下载缓存中的 ETag 一致：`ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59`，下载 revision 为 `740312add88f781978c0658806c59bc2815b9866`。
原始数据、提示词、标签和正式评分均未修改；该审计单列，不能将四条标签冲突解读为新增控制造成的模型错误，也不将手工修正后的分数替代正式分数。

### 首轮固定工作量与采样

`baseline/native-0.json.gz` 为首个完整固定回放：256 条轨迹、920 次调用、60,136 输出 token，耗时 **1,994.30 秒**。
输入与输出长度指纹与原在线轨迹一致（`454762c34448412ab351a00747072a582f38eab0d5a4e94fec7c4ead6ca8e445`）。
命中 791,840 / 8,781,870 token（9.02%），9 次抢占；TTFT P95/P99 为 115.87/123.42 秒。
回放使用 ignore_eos 并输出固定 token 数，920 个 length 结束标记属于固定工作量设定，不作为在线答案截断率。
完成本轮后切换到独立采样；`baseline/round-status.json` 记录该边界，原计划的余下两轮普通原生回放仍需补齐。
原 matrix 的整组 passed 未置为 true；本轮单独的 passed 和指纹已核验。

独立采样的配置、日志、CPU 栈及 PyTorch trace 位于 `profile-pressure/`；采样时间不混入普通吞吐比较。
延后 32 个引擎迭代再采集 32 个迭代，每卡约 29.7 秒、65,384 个 GPU 活动事件。

| 采样 GPU 活动 | rank 0（物理 GPU 0） | rank 1（物理 GPU 3） |
| --- | --- | --- |
| 通信 | 48.85%，14.48 秒 | 49.63%，14.73 秒 |
| 矩阵乘法 | 39.88%，11.82 秒 | 39.16%，11.63 秒 |
| 拷贝 | 0.15%，43.60 ms | 0.15%，43.55 ms |
| GPU 活动区间并集 / 采样窗口 | 99.67% | 99.72% |

该比例仅描述采样窗口，各活动类别按 kernel 名称归类；原始名称和时间同时保留在 `profile-breakdown.json`。
`gpu-topology.txt` 确认 GPU 0/3 连接为 SYS，经过跨 NUMA 的主机互连。
按共同时间基准配对 4,128 次 NCCL AllReduce，两卡处于同一 collective 的重叠区间合计 **14.39 秒**；进入时间差中位数 **0.066 ms**、P95 **0.120 ms**。
这支持继续检查通信路径及前缀重算量；不能仅凭这段采样把整轮耗时归为 GPU 算力或控制 CPU 开销。
py-spy 在加载阶段有 21 条采样滞后提示，服务 ready 后尚未出现；CPU 栈分析按 engine busy loop 筛选，排除初始化。

### 失败采样与资源等待

`profile-pressure/native-0.json.gz` 的整轮采样失败，不能纳入性能或稳定性通过记录。
该轮未启用控制器或 connector；已完成 614 次调用、148 条轨迹时停止推进。
13:37:58 后物理 GPU 0 利用率降至 0%，GPU 3 保持 100%，两卡显存仍约 40 GiB，功耗降至约 75/85 W。
13:42:56 引擎因 `RPC call to sample_tokens timed out` 退出，日志未报告 CUDA 非法访存或 OOM。
GPU trace 已在 13:15:12 完成导出，因此导出与停滞相隔约 23 分钟；保存的 32 迭代 trace 可用于该窗口的算子分析，不能证明整轮通过。

默认 py-spy 采样会暂停目标进程，可能影响 TP worker 的协同；当时未抓到停滞线程状态，该解释仍是待验证假设。
对照改用 [py-spy 官方说明](https://github.com/benfred/py-spy#how-can-i-avoid-pausing-the-python-program) 的 `--nonblocking`，并在推理计数停滞时保存自建进程的状态、TracerPid、线程状态与 Python 栈。
非暂停采样可能读到不完整栈，CPU 样本份额仍需与 RPC、进程 CPU 时间及 GPU trace 交叉核对。

故障还暴露出 driver 的清理问题：一个 episode 失败后，普通 `gather` 留下其他推理继续运行，HTTP client 关闭时产生大量次生异常。
driver 已改为持有全部 episode task，在异常时取消并等待其余 task，再关闭 client；行为检查验证保留原始错误并完成其余 task 的 finally。

`profile-nonblocking/` 重试未进入推理：另一项 fasp-dev 任务占用 GPU 0/3，启动时空闲显存 38.33 GiB，低于 0.5 配置要求的 39.63 GiB。
该次记录属于资源不足的启动失败，与上面的推理停滞分开记录。
按用户选择等待任务自然结束；资源检查等待其父进程退出且两卡连续空闲 60 秒，再以原 256 条轨迹、并发 64、利用率 0.5 启动 `profile-nonblocking-native/`。
未终止该任务，未改用其他 GPU，也未降低主实验的显存利用率。

`profile-nonblocking-native/` 已完成整轮：256 条轨迹、920 次调用、60,136 个输出 token，指纹与普通基线一致，耗时 2,062.21 秒，18 次抢占。
这一轮包括 profiler 采集和导出，不并入普通性能重复。
32 迭代 GPU 窗口约 29.70 秒，两卡通信占 48.77%/49.46%、矩阵乘法占 39.96%/39.29%，GPU 活动区间并集约占窗口 99.70%。
该窗口位于早期 prefill，不能作为全轮或持续满缓存阶段的算子比例。
watcher 在统计表生成时抓到两卡空闲、worker 主线程执行 `_parse_kineto_results`，随后推理恢复；本轮未复现先前的单卡停滞。

CPU py-spy 文件存在一帧无效 UTF-8，原文件保持不变，完整性审计保存在 `cpu-file-integrity.json`。
剔除涉及该帧的一条样本后，仍只有初始化样本，没有 `EngineCore.run_busy_loop` 的运行期样本；该文件不作为 CPU 开销证据。
独立 CPU 诊断中，进程改名前后分别采到 85/143 条样本，排除了改名直接导致漏采的解释。
采样器的缺失原因尚未确认，不能据此声称控制器 CPU 占用低。

曾尝试预热后通过受控信号启用 cProfile，并以 `time.thread_time` 计时。
`profile-cprofile-pressure/` 在共享 GPU 干扰下主动停止；保存了 CPU 文件，但引擎统计有 18 项负时间，scheduler 自身时间甚至超过全部 CPU 时间，计时无效。
单线程设施检查通过不能证明多线程计时有效；该批原始文件保留，不用来估计控制开销。

当前诊断改用 [Yappi](https://github.com/sumerc/yappi) 1.7.6 的多线程 CPU 时钟，保存 API/engine/worker 的聚合统计与逐 Python 线程统计。
驱动核对 PID 的启动标识，等待 profiler 启停确认，并要求结果包含引擎 scheduler 的实际调用，防止将初始化文件当作运行期数据。
同时保存服务与 Store master 的逐线程 CPU 快照及 Python 线程名，区分主线程、后台传输与通信线程；结束时间对应 CPU 快照，排除之后的 profile 文件导出。
CPU 快照覆盖客户端进程启动至退出，包含输入准备和报告写入，不能当作纯推理窗口 CPU 量。
未设置诊断环境变量时，该 bootstrap 不启用 profiler；普通性能对照不加载此 bootstrap。
补充的多线程工具检查验证初始化排除、三个线程调用记录、PID 身份拒绝、非负函数时间及函数 CPU 总量不超过进程 CPU 预算；完整诊断也执行相同计时检查。
这些检查属于诊断设施验证，不替代模型测试。
原始 CUDA trace 继续保留，关闭已观察到较长停顿的统计表生成。

策略采样的 `profile-nonblocking-policy/` 未进入推理：新一项 fasp-dev 任务在服务初始化期间占用 GPU 0 约 57 GiB，剩余 21.99 GiB，启动因此失败。
继续按用户选择等待其任务自然结束，后续完整对照使用 `profile-cprofile-pressure/`；未将启动失败计为策略运行结果。

另一用户的 syminfer 进程随后在 `profile-cprofile-pressure/` 期间进入 GPU 0/3，本轮吞吐约减半。
已先保存 profiler 文件，再停止自己的实验；其他用户的进程未受终止操作影响。
`resource-overlap.csv` 与 `interference.json` 保存该边界，本轮不作为完整压力或性能通过记录。
驱动增加 UTC 时间的 `gpu-processes.csv`，在实际工作窗口核对 GPU 进程归属与采样覆盖；出现未知进程或超过 10 秒的采样缺口时拒绝接受性能结果。
新的完整策略诊断使用 `profile-yappi-policy/`，等待 GPU 0/3 自然释放后启动。

### 首轮完整策略诊断与修正

`profile-yappi-policy/` 已执行完 256 条轨迹、920 次调用和 60,136 个输出 token，耗时 2,080.24 秒。
末尾检查发现两条 `client_error`，因此整轮 `passed=false`，不作为性能通过记录。
离线 GPU 进程审计覆盖实际工作窗口，每卡 412 个样本，最大间隔约 9.84 秒，未发现外来计算进程。
两条超时都发生在 GPU profiler 自动停止时；7 个控制 RPC 同时等待约 69.5 秒，HTTP 最终均为 200。
Yappi 显示 `export_chrome_trace` 耗用约 64–67 秒 CPU，其中一张卡有 11,634,713 次 `GzipFile.write`、累计约 54 秒 CPU。
这是同步导出造成的诊断扰动；修正为负载期间写未压缩 trace，正常控制的 30 秒超时保持不变。

传输计数为加载 204,750,192,640 字节、写入 366,135,476,224 字节、完成 11,111 次、失败 0 次、缺失 213 次；结束时结果记录为 0，策略异常为空。
644 次准备成功的恢复中，593 次缺少至少一个 Mamba 组，587 次后续请求没有报告前缀命中。
51 次完整 Mamba 准备的检查点与后续输入兼容；其中仍有准备与首 token 相隔约 70–80 秒而未命中的情况，需结合排队、30 秒保留 TTL 与实际淘汰解释。
不能仅用拷贝完成或大传输量证明复用收益。

共享冷内容审计发现：352 次恢复遗漏了此前其他会话成功写入的 1,080 个 Mamba 块。
旧策略仅保留当前 owner 刚写入的子集，即使同一内容曾由其他 owner 存入 Store，也不会出现在准备快照中。
修正增加按 namespace/content 寻址的有界冷副本线索，并将声明范围内已知的共享冷内容纳入快照；线索独立于 owner 清理。
它不提供驻留保证，实际加载仍经过现有 backend 与所有 rank 完成检查，缺失时按原语义回退。
原始目标中未被缓存的历史状态仍不能凭空恢复，不能将全部 Mamba 缺失归为这一项修复。

另在首次异步搬运后观察到 rank 1 为 GPU 0 创建约 450 MiB 上下文，早于 GPU trace 开始。
搬运线程原来没有绑定 GPU；修正为在同步计算事件与调用 Store 前选择该事件所属设备。
CPU worker 190 项检查通过，实际 GPU 搬运重跑也通过设备断言；额外上下文的变化由新整轮占用日志再核对。

association 原来限定在输入的整块边界，限制了 `observe_growth` 的解码扩展，已改为无固定结束位置。
另审计固定长度回放的 664 个轮间转换，44 个最终检查点推进超过输入边界，17 个最终检查点超过实际后续输入的公共前缀。
固定长度回放不强制输出 token 内容；相关状态复用结论需结合这一差异，真实在线工具循环另行验证。
修正后的完整策略诊断位于 `profile-device-cold-policy/`，完成 256 条轨迹、920 次调用、60,136 输出 token，输入与输出长度指纹一致，客户端 `passed=true`。
整轮耗时 1,954.13 秒，命中 1,046,640 / 8,781,870 token（11.92%），11 次抢占，KV 峰值 100%；TTFT P95/P99 为 104.46/114.73 秒。
该轮启用诊断采样，不并入普通吞吐重复轮次，也不能据此判定 25% 目标。

控制错误为 0，后端完成 12,314 次传输，失败与缺失均为 0；加载 262,193,283,072 字节、写入 370,502,795,264 字节。
结束时结果记录和逻辑关联均为 0。
GPU 审计每卡覆盖 391 个样本，最大间隔约 5.02 秒，未发现外来计算进程；rank 1 在 GPU 0 的额外上下文未再出现。
未压缩 trace 的停止和导出约 7 秒，旧压缩导出约 74 秒。

| 诊断覆盖 | 修正前 | 修正后 |
| --- | ---: | ---: |
| 准备完成的恢复 | 644 | 646 |
| 所有 Mamba 组均已准备 | 51 | 484 |
| 后续请求报告非零缓存命中 | 57 | 78 |
| 所有 Mamba 组准备且后续实际命中 | 9 | 65 |

逐操作核查结果保存在 `control-coverage-before-after.json`。
修正显著增加完整组的准备覆盖；大量已准备内容仍未产生实际命中，不能将 484 次准备等同于 484 次复用。
排队期间的保留时机与容量压力仍须通过普通压力、quota 消融和容量充足对照判断。
`priority-copy-order-audit.json` 核查同一信号产生的 654 对 COPY/SET_PRIORITY：优先级结果序号全部晚于 COPY，8 组延期 COPY 对应零优先级作用。
已有执行器会等待同一信号的兼容 COPY；不能将残余低命中直接归为优先级先于复制执行。
优先级目标还包含已不存在的历史 Mamba 状态，`partial` 状态需结合实际块范围解释。
更严格地要求四个 cache group 都报告注册成功时，共有 479 次准备，其 Mamba 检查点均在下一次实际输入的公共前缀内；65 次后续命中，414 次未命中。
其中 388 次未命中在“控制完成并 ACK”至首 token 之间已超过 30 秒；未命中组的该间隔中位数为 66.92 秒。
命中组中位数也有 63.33 秒，且首 token 晚于准入和缓存查询，故这些时间只能表明 TTL 到期的机会，不能证明每次 miss 的淘汰原因。
该审计保存在 `preparation-timing-audit.json`，不将首 token 时间冒充调度准入时间。

四个进程的 Yappi 函数时间均非负，累计 self 时间均在对应进程 CPU 预算内；引擎记录到 2,556 次实际 scheduler 调用。
引擎控制模块的 self CPU 合计 17.33 秒；`observation` 的累计 CPU 为 46.17 秒，`observe_requests` 为 30.68 秒，`on_signal` 为 24.32 秒，`KVControlService.call` 为 26.76 秒。
这些累计值存在嵌套，不能相加；结果包含 profiler 扰动，不能当作普通运行中的边际开销。
客户端准备等待 P99 为 75.97 微秒，说明移出推理路径的控制等待已很小；异步控制与服务端策略仍有成本，普通 idle/noop/quota 对照另行测量。

新的 32 迭代 GPU 窗口约 28.12 秒，两卡 GPU 活动区间并集占 99.68% / 99.64%。
通信活动占 47.27% / 47.70%，矩阵乘法占 38.92% / 38.25%，拷贝占 2.36% / 2.62%。
这是带搬运的窗口，不能由原生与策略两个不同迭代窗口的活动比例直接推算整轮性能差值。

原 `matrix.json` 的后置检查只匹配未限定的函数名 `schedule`，而 Yappi 导出名为 `Scheduler.schedule`，故原记录保留 `passed=false`。
检查器已修正为匹配限定名的末段；`post-validation.json` 单独记录上述全部客户端、指纹、清理、资源与四进程 CPU 检查通过，未回写原始失败记录。

### 查重与现有交付

已重新读取 #57103 的正文和九条评论，运行按 issue body、programmable-cache/KV-hint 关键词的 open PR 查询，并读取相关 PR 正文。
旧 gh 的默认 issue 查询触发废弃 projectCards 字段错误；指定 JSON 字段后成功读取正文和评论。
本轮开始时，工作分支与 [既有草稿 #62382](https://github.com/vllm-project/vllm/pull/62382) head `ac5094f1776aa7fad0d5c0a2327bea6e1421ce34` 一致。
期间用户提交了文档清理等更改，验证基点推进至 `067332e667`；本轮本地提交沿用该分支，各轮另外保存实际源文件摘要。

- [#58198](https://github.com/vllm-project/vllm/pull/58198) 提供最小 CPU→GPU 独立预取执行原语，当前范围为单 KV cache group；本实现包含统一入口、逻辑目标、异步结果、CPU/Mooncake 及 hybrid 场景。分配、发布、取消等重叠执行机制仍须在上游整合。
- [#59714](https://github.com/vllm-project/vllm/pull/59714) 提供请求生命周期分发与 G1 SET_PRIORITY；本实现还包含 COPY/MOVE、独立控制及真实 rollout。公共动作、分发器和优先级机制需共享实现。
- [#38514](https://github.com/vllm-project/vllm/pull/38514) 提供 full-attention 范围优先级、TTL 和软保留；本实现还处理 session/continuation、hybrid、搬运与结果。淘汰优先级属于复用/整合边界。

上述差异已由提交者在 [RFC #57103](https://github.com/vllm-project/vllm/issues/57103) 的既有评论中说明。查询和正文原文保存在本轮 results 的 `issue-57103.json`、`prs-*.log`、`pr-*.json`。

### 功能验证与脚本合并边界

`functional/` 已完成实际双引擎缓存命中、128 题 Mooncake 质量、Store 停机/引擎重启、共享生命周期和 IPC 参数更新/回滚。
重算与预取准确率均为 124/128，均无长度截断，答案差异为 0；预取每题命中至少 784 token。
逐 token 审计有 41 条生成序列不同，不能声称逐 token 一致；原生 APC 的历史质量记录也有序列差异。
用户随后要求优先提高速度、暂缓细化输出校验；本轮停止扩展原生 APC/逐 token 质量对照，已有质量记录保留。
实际 Store 停机时四组加载失败、无 GPU 驻留或 loading 遗留，回退重算与故障前短贪心输出相同；重启后旧 epoch 返回 409，关联为空并可由新请求重建。
生命周期检查得到共享优先级 90→20→0，重复 COPY/恢复竞争与取消均通过；真实修改第 0 层 norm 参数后首次推理零命中，回滚后旧 Store 内容可复用。

用户随后要求合并同时进行的脚本与测试整理，统一使用原生 primitive CLI，并重新验证。
旧队列在 CPU 服务启动阶段停止，未执行 CPU 质量检查；该边界记录在 `functional/round-status.json`，不将整份 manifest 置为通过。
新 CLI 合并请求附带与独立 COPY/MOVE、G1 优先级和跨引擎准备；元数据协议/关联的拒绝和增长行为由 pytest 覆盖。
`merge-boundary.json` 保存合并授权、源码摘要与重验范围。

`functional-merged/` 的新跨引擎检查通过，但后续质量驱动首次 MOVE 发生真实 `TRANSFER_FAIL (-800)`，整轮保留失败。
19:05:31 目的引擎被原驱动关闭；19:05:34 源端写入仍连接其 Store 传输端口 15262/15809，出现 EOF、Broken pipe 和 Connection refused。
这两个端口与目的 worker 的监听日志完全对应；目的引擎同时贡献 Store 内存，关闭它使已有分配目标失效。
修正将目的引擎的生命周期延长到故障注入结束，并在启动新 Store 之前关闭；不增加等待常量、不忽略 partial 失败，也不放宽完成断言。
修正后的 `functional-merged-v2/` 再次完成新 CLI 跨引擎检查，后续首次 MOVE 后已进入重算质量阶段，未再出现上述断连。
按用户新的优先级，在质量评测期间停止自建服务，整轮未标记通过；`speed-priority-boundary.json` 记录该停止边界。
合并后的完整相关 CPU 回归为 1,973 项通过（626.83 秒），修改后的五个 GPU 搬运文件为 56 项通过、1 项 ROCm 专用检查跳过；默认 hook 和 Python 3.12 类型检查通过。
普通计时完成 `priority-quota/` 与 `quota-wait120-aging30/`：利用率 0.5、并发 64、256 条完整轨迹、920 次调用与 60,136 输出 token；多次重复及容量充足对照保持未完成。

核心修改已提交为 `af2c0d7ab2`：共享冷内容线索、无消费者信号快速返回、搬运线程设备绑定及相应行为覆盖。
异步控制、合并后的实验驱动与性能工具已提交为 `10121d7920`。
两次提交时默认 hook 与 sign-off hook 均通过；第二次提交前，包含新 bootstrap 的 Python 3.12 全量类型检查也通过。
证据索引与 OpenSpec 修订为 `e3f526abd5da4a575d500b61176f4d2affc4019d`，可调有界准入参数及首轮普通数据为 `6d63f4eaa9`。
各完整修改单元均在相关检查通过后提交。

### 普通压力计时与准入参数

`priority-quota/` 已完整通过：256 条轨迹、920 次调用、60,136 输出 token，输入/输出长度指纹与原生相同。
利用率 0.5、并发 64、每卡 KV 12.12 GiB，与原生对照一致；未启用 profiler、搬运连接器和逐请求控制。
每卡 GPU 进程审计覆盖 378 个样本，最大间隔 6.70/6.45 秒，未发现外来进程。

| 普通固定工作量 | 原生 | quota 30/1 | quota 120/30 |
| --- | ---: | ---: | ---: |
| 耗时（秒） | 1,994.30 | 1,898.86 | 1,369.48 |
| 相对原生吞吐 | 1.000 | 1.050 | 1.456 |
| 报告命中 token | 791,840 | 1,266,160 | 3,382,960 |
| 命中 / 输入 token | 9.02% | 14.42% | 38.52% |
| 抢占 | 9 | 18 | 2 |
| TTFT P95/P99（秒） | 115.87 / 123.42 | 87.97 / 107.29 | 64.04 / 71.21 |
| 服务端平均排队（秒） | 73.72 | 66.03 | 38.72 |
| 服务端平均 prefill/decode（秒） | 4.53 / 53.67 | 5.01 / 54.11 | 3.32 / 47.10 |
| rollout P95（秒） | 764.66 | 732.35 | 536.94 |
| 最慢前缀家族 rollout P95（秒） | 947.37 | 946.15 | 678.17 |
| 客户端 CPU（秒） | 73.01 | 89.20 | 72.48 |

每个配置只运行一轮，无法给出稳定中位收益或置信区间。30/1 未达到 25% 目标；120/30 在这一轮达到目标。
两种 quota 客户端都只进行两次 capacity 和一次结束关联查询，没有逐请求 hint、operation 查询或 ACK；服务端索引与策略成本仍存在。
不能由零逐请求 RPC 推断零控制开销，idle/noop 与容量充足对照尚未完成。
30/1 服务进程 CPU 启动窗口中，API、engine、两张卡 worker 分别为 74.90、1,899.48、1,924.97、1,922.35 秒；120/30 为 67.12、1,369.92、1,388.53、1,386.57 秒。
这些窗口包含客户端导入与报告写入，不能直接解读为净控制成本；原生轮没有同等进程 CPU 快照，不推算其边际差值。

三组的 32 个家族均各完成 8 条轨迹。30/1 有 11 个家族 P95 略高于原生，最大约 2.48%；120/30 的所有家族平均延迟和 P95 均低于原生，最慢家族 P95 为 678.17 秒。
同一单轮中未观察到用某个家族延期换取总吞吐的退化；这不能证明其他负载下的公平性。

`native-queue-horizon-audit.json` 对原生轮的服务端排队直方图作前后差分：867/920（94.24%）请求排队超过 30 秒。
首个 quota 配置的策略等待上限为 30 秒，超时后按公平性规则优先处理老请求，复用评分的影响因此受限。
驱动现将策略等待上限、quota 老化时间尺度、复用评分权重及解码前瞻暴露为实验参数；引擎默认行为不变。
120/30 同时延长等待预算并减慢等待分数增长，为复用评分与运行请求的增长预留留下作用时间。
结果表现为更多实际复用、更少抢占以及更低的排队和 prefill 时间，支持继续围绕准入与复用组织负载；本轮没有单变量消融，不能区分两个参数各自的贡献，也没有证明这是最优配置。

`quota-wait120-aging30/matrix.json` 整轮通过，输入/输出长度指纹与原生相同，零控制错误、零遗留结果和逻辑关联。
每卡 GPU 归属审计覆盖 273 个样本，最大间隔 5.24 秒，无外来计算进程。结束后自建服务和采样进程均已退出，GPU 0/3 上未留下本轮进程。
三个普通轮的逐项汇总、原始文件 SHA256 与家族对比保存在 `stop-boundary-summary.json`。

复现末轮使用同一 `baseline/trace.json.gz`，完整命令保存在该目录 `matrix.json` 的 `commands` 中：

```bash
.venv/bin/python -m benchmarks.programmable_kv_cache.run_rollout_matrix \
  --model /home/youwei/bzh/model/Qwen/Qwen3.8-27B \
  --dataset /home/youwei/bzh/dataset/openai/gsm8k \
  --trace benchmarks/programmable_kv_cache/results/rollout-2026-10-18/baseline/trace.json.gz \
  --gpus 0,3 --gpu-memory-utilization 0.5 --concurrency 64 --max-num-seqs 64 \
  --arms policy --policy-components quota --repeats 1 \
  --policy-max-wait-seconds 120 --quota-aging-seconds 30 \
  --output-dir /tmp/kv-quota-wait120-aging30
```

### 停止边界

按用户要求，完成 120/30 本轮后停止，未启动其他实验。
`user-stop-boundary.json` 与 `round-status.json` 记录完成范围、未执行项和进程清理结果。
尚未完成利用率 0.5 的多次重复、容量充足/idle/noop 对照、完整普通策略与消融、oracle/早晚预测、合并后完整恢复/CPU 质量重验及外部大模型评估。
细化输出检查按用户要求暂缓；已有 128 题与真实生命周期结果保留其版本和执行边界。
这些项目继续保留在 OpenSpec 未完成项中，未用历史 2/8 GiB 结果代替，也未归档整个 change。
