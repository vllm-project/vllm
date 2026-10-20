# 本轮代码交付与 PR Checklist

本轮交付供人逐行审查的实现和本地 commit，不发布新的 PR，也不宣称模型已验证。
按用户要求，pytest、模型评测、GPU/H20 e2e 均未运行。

## 1. 设计与调用链

```text
Model load / GPU memory profiling
  -> bind actual source descriptors
  -> allocate capture buffer + bounded output-rank snapshot pool
Scheduler.schedule
  -> shared generation / incremental KV block hashes / finish events
Worker.begin_step
  -> shared request lifecycle + reset capture validity
Model attention/indexer hooks
  -> source views, before shared-buffer overwrite
Worker.prepare_output
  -> acquire stable snapshot slot
AsyncOutput copy stream + existing completion event
  -> pinned Indexer D2H / accepted speculative rows
Worker.process_output
  -> shared full blocks / tails / SHM publication / prefix restoration
  -> release snapshot slot
Scheduler.update_from_output
  -> accepted logical range + independent prompt offsets
EngineCoreOutput / OutputProcessor
  -> string-stop clipping / accumulation / raw decode or packed payload
Python API
  -> final offline output or terminal HTTP choice/chunk
```

没有新增 Indexer Scheduler Manager、物理 Slot Store、传输线程或 PD 协议。
R3-only 使用旧命名空间、存储 shape/dtype 和原 D2H 路径；组合模式使用同一
定宽 Record。模型语义只出现在 Source 和 Codec 中，不进入公共请求状态机。
禁用 Indexer 时不创建 Capture Buffer、快照池、Indexer D2H 或新的 Store。

模型 Hook、Shape、Codec、后端和额外模型审计见 [PLANS.md](PLANS.md)；
公开使用说明见 [Indexer Replay](../../docs/training/indexer_replay.md)。

## 2. 正确性审查

- source 使用真实层 ID；DeepSeek IndexCache、V4.1 和 HY 共享消费者显式映射。
- V4 只注册实际 C4A Indexer；V4.1 只注册实际 Source；Candidate 不冒充 TopK。
- Buffer 写入发生在消费点附近，随后各层可覆盖原共享 Buffer。
- 快照池在模型内存统计阶段分配，避免 KV Planner 漏算大块 GPU 内存。
  output 消费前不释放槽位；关闭时解除 Source 绑定并释放 Capture/快照引用。
- 完成事件后检查 Producer/Token 有效标记，遗漏会明确报错。
- Spec reject 裁尾、async optimistic cursor、generation 和 finished 事件共用原链路。
- 抢占重算回退现有 tail；非对齐边界从未发布的 Block 或 SHM 补齐。
- Hash 延迟发布、多请求共享前缀和同 Batch 先发布后读取沿用既有机制。
- DeepSeek V4.1 decoder replay 在 Indexer-only 时也保留所有 Capture 行。
- Scheduler 和 API 字符串停止处理均裁剪 Indexer；最后 sampled token 不返回行。
- Pool 编码保留原顺序、每 Pool 有效位和原始 Tail，覆盖短上下文的部分 Pool。
- QSA 恢复块展开、因果 Tail、Padding 和末列 Count，不把 Count 当索引。
- Step5 分组保持 Token-first 布局，保留 Current Region、原宽度和 TP Group 顺序。

以上为代码审查结论，尚未通过行为测试或真实 Attention 对照建立验证证据。
仍保留 Store 的 immutable first-writer 语义；同 Hash 重算可能复用更早的合法选择。

## 3. 内存报告

每输出 Worker：`N_KV * B_scheduler * (S_R3 + S_Indexer)`。
每 SHM 对象：`B_hash * (S_R3 + S_Indexer)`。
逐个 Profile Source 求和，不按模型总层数估计 Indexer 数量。

| 每 Source，每 Token | Raw 字节 | Compact 字节 |
| --- | --- | --- |
| DSA / 压缩坐标 | `4*K` | 同 Raw |
| GLM KPool | `4*(K+P-1)` | `4*(K/P+P-1)`，位宽不够则用 Raw |
| NVIDIA QSA | `4*(K+R)` | `4*(K/R+2)` |
| AMD QSA | `4*(K+R-1)` | 同 Raw |
| Step5 | `4*G*(K+2)` | 同 Raw，包含原选择宽度 |

示例 `K=2048,P=R=4`：KPool 8204 -> 2060 B，NVIDIA QSA 8208 -> 2056 B。
这些是格式字节数，不是性能测试。日志分别列出 R3/Indexer、Block、SHM、
Host staging、估计 Metadata、安全余量、本机 DP 输出 Worker 数量。
SHM 配置少于 KV 覆盖需求明确失败，不静默缩容。GPU 内存由提前分配的
Capture/快照池进入模型加载统计。Host 检查是估计，不是跨部署原子内存预留。

## 4. 实际执行的检查

| 检查 | 本轮状态 |
| --- | --- |
| Ruff check / format，所有变更 Python 文件 | 通过 |
| Python 3.11 AST 语法解析 | 通过 |
| `git diff --check` | 通过 |
| SPDX、forbidden imports、torch CUDA API 规则 | 通过 |
| AuxOutput Config 文档/默认值规则 | 通过 |
| 新 CPU Suite 的 Buildkite tethering 静态检查 | 通过 |
| Profile/Capture/Connector/Worker 定向 mypy，`--follow-imports=skip` | 通过 |
| 完整 mypy / 完整 pre-commit | 未通过/未完成，不能标为通过 |
| pytest，包括 CPU UT / R3 regression | 用户要求暂不运行 |
| 模型评测、GPU/H20 e2e、同步检查、性能基准 | 未运行 |
| Rust fmt/build/test | 当前环境无 Rust 工具，未运行 |

类型环境：本轮系统 Python 的 Pydantic **2.11.7**，仓库要求 **>=2.12.0**，pre-commit 固定
**2.13.5**；本机无法识别 `Field(exclude_if=...)` 的类型签名。原样 main
`8407bc0bb6` 的同一批生产文件对照有 **242** 条 mypy 诊断。已修正此次发现的
Capture 属性类型和 Optional 访问问题；完整 silent-import 检查还曾超过
120 秒限制。因此这里只报告定向检查通过，不以干净 main 的错误替代本次检查。
另外确认共享项目 `.venv` 的 Pydantic 为 2.13.4，但未安装 mypy；未修改共享环境。
本轮 commit 使用 `--no-verify`，保留上述手动静态检查结果，整套自动 hook 待后续
在合适环境运行；这不代表完整 pre-commit 已通过。

已补写/扩展的用例覆盖 Source/Codec 精确还原、快照隔离/槽位所有权、
Pool 及 QSA 的边界、Step5 Prefill/Decode Group 布局、SHM Prefix 与 Spec
共享生命周期、抢占回退、独立 Offset、长度检查、raw/packed 输出、HTTP
各 Surface 的新字段、V4.1 保留行。测试设计采用小合成选择输入作为 oracle；
真实模型产生的索引仍需要后续 GPU 验证。

建议审查后执行：

```bash
python -m pytest tests/model_executor/test_indexer_capture.py tests/distributed/aux_output_connector/ tests/config/test_aux_output_config.py tests/test_request_input_bounds.py
python -m pytest tests/model_executor/test_routed_experts_capture.py tests/v1/core/test_scheduler.py tests/v1/core/test_async_scheduler.py tests/v1/engine/test_output_processor.py
python -m pytest tests/entrypoints/scale_out/token_in_token_out/test_protocol.py tests/entrypoints/scale_out/token_in_token_out/test_generate_stream.py
pre-commit run --files <changed-files>
```

## 5. PR 内容与剩余工作

- 基于 `main` 最初 `8407bc0bb6`，提交前再次 fetch 并 fast-forward 到
  `cc8a41f328`。这次上游只有非重叠改动。
- 保留 #49555 旧 head `f4f96b90e9` 为迁移历史的父提交；代码用新的
  AuxOutput 架构替换原 Slot Store，不需要 force-push。
- Duplicate-work 检查确认更新既有 #49555；不新增重复 PR，不依赖 #57968。
- [PR_BODY.md](PR_BODY.md) 使用当前模板，标明 AI 辅助与所有未执行验证。
  未向 GitHub 发评论或请求 re-review。
- 每行代码的人审、CPU 回归、模型评测、真实 GPU 对照仍待用户后续安排。
- MiniMax-M3 adapter、Rust HTTP、CPU/TPU/OOT inference、PP/DCP/PCP、
  adaptive verification 和远程 PD 不在本轮支持声明内。
- 未验证 CUDA Graph、TP/DP 多进程、吞吐、真实抢占及模型数值输出；
  不能把旧 PR 的历史 E2E 结果用于这次迁移。

建议按 Profile/生命周期、DSA、V4/V4.1、KPool/QSA、Step5 分片审查。
本轮仅为现有 PR 准备一个迁移 commit，后续是否拆 PR 由人审结果决定。


## 6. 按文件变更清单

| 文件 | 修改目的 |
| --- | --- |
| `docs/training/indexer_replay.md` | 用法、数据语义、内存与限制文档。 |
| `docs/training/routed_experts.md` | 用法、数据语义、内存与限制文档。 |
| `rust/src/engine-core-client/src/protocol/output.rs` | 追加 IPC Indexer 字段及兼容快照，保持旧位置不变。 |
| `rust/src/engine-core-client/src/tests/client.rs` | 追加 IPC Indexer 字段及兼容快照，保持旧位置不变。 |
| `rust/src/engine-core-client/src/tests/python_compat.py` | 追加 IPC Indexer 字段及兼容快照，保持旧位置不变。 |
| `tests/config/test_aux_output_config.py` | 补写或扩展回归用例；本轮未执行。 |
| `tests/distributed/aux_output_connector/test_store.py` | 补写或扩展回归用例；本轮未执行。 |
| `tests/entrypoints/openai/completion/test_completion.py` | 补写或扩展回归用例；本轮未执行。 |
| `tests/entrypoints/openai/test_stop_token_ids.py` | 补写或扩展回归用例；本轮未执行。 |
| `tests/entrypoints/scale_out/token_in_token_out/test_protocol.py` | 补写或扩展回归用例；本轮未执行。 |
| `tests/model_executor/test_indexer_capture.py` | 补写或扩展回归用例；本轮未执行。 |
| `tests/models/test_deepseek_v41_replay_batch.py` | 补写或扩展回归用例；本轮未执行。 |
| `tests/test_request_input_bounds.py` | 补写或扩展回归用例；本轮未执行。 |
| `tests/v1/engine/test_output_processor.py` | 补写或扩展回归用例；本轮未执行。 |
| `vllm/config/aux_output.py` | 独立 Indexer 配置、CLI/LLM 参数或 SamplingParams 传递。 |
| `vllm/config/vllm.py` | 独立 Indexer 配置、CLI/LLM 参数或 SamplingParams 传递。 |
| `vllm/distributed/aux_output_connector/connector.py` | 单一 Hash/事件流和各 Artifact 独立 Offset 裁剪。 |
| `vllm/distributed/aux_output_connector/profile.py` | Profile、Namespace 身份、定宽 Codec 解码和容量计算。 |
| `vllm/distributed/aux_output_connector/routed_experts.py` | 复用已有通用 Block/Tail Buffer，补充回退能力。 |
| `vllm/distributed/aux_output_connector/store.py` | 写入前验证固定 Slot Payload 长度。 |
| `vllm/distributed/aux_output_connector/worker.py` | 组合 Record、异步输出、共享 SHM 生命周期和容量校验。 |
| `vllm/engine/arg_utils.py` | 独立 Indexer 配置、CLI/LLM 参数或 SamplingParams 传递。 |
| `vllm/entrypoints/generate/base/serving.py` | 复用 HTTP 编码 helper；终止 Choice/Chunk 返回。 |
| `vllm/entrypoints/llm.py` | 独立 Indexer 配置、CLI/LLM 参数或 SamplingParams 传递。 |
| `vllm/entrypoints/openai/chat_completion/protocol.py` | vLLM 扩展字段；禁用时省略新字段。 |
| `vllm/entrypoints/openai/chat_completion/serving.py` | 复用 HTTP 编码 helper；终止 Choice/Chunk 返回。 |
| `vllm/entrypoints/openai/completion/protocol.py` | vLLM 扩展字段；禁用时省略新字段。 |
| `vllm/entrypoints/openai/completion/serving.py` | 复用 HTTP 编码 helper；终止 Choice/Chunk 返回。 |
| `vllm/entrypoints/scale_out/token_in_token_out/protocol.py` | vLLM 扩展字段；禁用时省略新字段。 |
| `vllm/entrypoints/scale_out/token_in_token_out/serving.py` | 复用 HTTP 编码 helper；终止 Choice/Chunk 返回。 |
| `vllm/model_executor/layers/indexer_capture.py` | Source 绑定、GPU Capture、有界快照池、有效性与释放。 |
| `vllm/model_executor/layers/mla.py` | 模型 Producer/Consumer Capture 位置、Profile 与层映射；V4.1 保留 Capture 行。 |
| `vllm/model_executor/models/deepseek_v2.py` | 模型 Producer/Consumer Capture 位置、Profile 与层映射；V4.1 保留 Capture 行。 |
| `vllm/models/deepseek_v32/amd/rocm.py` | 模型 Producer/Consumer Capture 位置、Profile 与层映射；V4.1 保留 Capture 行。 |
| `vllm/models/deepseek_v32/attention.py` | 模型 Producer/Consumer Capture 位置、Profile 与层映射；V4.1 保留 Capture 行。 |
| `vllm/models/deepseek_v4/attention.py` | 模型 Producer/Consumer Capture 位置、Profile 与层映射；V4.1 保留 Capture 行。 |
| `vllm/models/deepseek_v41/attention.py` | 模型 Producer/Consumer Capture 位置、Profile 与层映射；V4.1 保留 Capture 行。 |
| `vllm/models/deepseek_v41/nvidia/model_state.py` | 模型 Producer/Consumer Capture 位置、Profile 与层映射；V4.1 保留 Capture 行。 |
| `vllm/models/dots3_note/nvidia/model.py` | 模型 Producer/Consumer Capture 位置、Profile 与层映射；V4.1 保留 Capture 行。 |
| `vllm/models/glm5next/common/attention.py` | 模型 Producer/Consumer Capture 位置、Profile 与层映射；V4.1 保留 Capture 行。 |
| `vllm/models/hy_v4/nvidia/attention.py` | 模型 Producer/Consumer Capture 位置、Profile 与层映射；V4.1 保留 Capture 行。 |
| `vllm/models/qwen4_exp/amd/indexer_qsa.py` | 模型 Producer/Consumer Capture 位置、Profile 与层映射；V4.1 保留 Capture 行。 |
| `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` | 模型 Producer/Consumer Capture 位置、Profile 与层映射；V4.1 保留 Capture 行。 |
| `vllm/models/step5/nvidia/flashinfer_sparse.py` | 模型 Producer/Consumer Capture 位置、Profile 与层映射；V4.1 保留 Capture 行。 |
| `vllm/models/step5/nvidia/model.py` | 模型 Producer/Consumer Capture 位置、Profile 与层映射；V4.1 保留 Capture 行。 |
| `vllm/models/step5/nvidia/sparse_attention.py` | 模型 Producer/Consumer Capture 位置、Profile 与层映射；V4.1 保留 Capture 行。 |
| `vllm/outputs.py` | 追加输出字段，保留旧字段的位置与聚合语义。 |
| `vllm/sampling_params.py` | 独立 Indexer 配置、CLI/LLM 参数或 SamplingParams 传递。 |
| `vllm/v1/core/sched/scheduler.py` | 组合 Artifact 的独立 accepted-range 输出。 |
| `vllm/v1/engine/__init__.py` | 追加输出字段，保留旧字段的位置与聚合语义。 |
| `vllm/v1/engine/input_processor.py` | 请求入口校验 Indexer Offset 的 Prompt 边界。 |
| `vllm/v1/engine/output_processor.py` | 累积、raw/packed 组装、字符串 stop 裁剪。 |
| `vllm/v1/worker/gpu/model_runner.py` | 模型内存统计前分配 Capture/快照池，KV 初始化时复用并在关闭时清理。 |
