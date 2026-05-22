# Slime vLLM 后端设计（重整 v1）

> 这是基于 `slime_vllm_backend_learning_report.md`（§2/§3/§4）和 `skyrl_slime_vllm_design_reference.md` 的重整草案。**原文件保留不动，本文件是另一种组织方式的尝试。**
>
> 组织约定：
>
> - 顶层是 6 个**大主题**（§0 项目背景 / §1 协议与端点 / §2 权重同步 / §3 并行拓扑 / §4 engine 启动与运行时 / §5 参数层）。
> - 每个大主题下分**小主题**，每个小主题固定按 **背景知识 → 备注 → 参考 → 结论 / 待讨论** 四段组织。空段直接省略，不留空标题。
> - 原文已有的 §3.1–3.3 横扫决策表和 §4 待拍板问题都被拆散到对应小主题的「结论」/「待讨论」段。

---

## 0. 序言 / 项目背景

### 0.1 vLLM-only 目标

#### 背景知识

最终目标是把 Slime 收敛成一个 **vLLM-only** 的仓库，SGLang 只作为历史参考，不保留为最终后端路径。在参数设计上，目标是尽量复用 SGLang 的做法，把 vLLM 的 server args 也做成**全量注入**，而不是只保留少量手写开关。

这一阶段的目标不是把 vLLM 做成 SGLang 的完全替代品，而是先把它接入 slime 的现有训练/rollout 主链路，形成一个可验证、可迭代的后端实现。具体只看三件事：

1. 能通过 `--rollout-backend vllm` 走到 vLLM 路径。
2. 能正常启动、通过健康检查，并完成一次 rollout。
3. 能在一次权重更新之后继续下一轮，不崩、不挂、不破坏主训练闭环。

#### 结论

| 主题 | 结论 | 备注 |
| --- | --- | --- |
| 最终后端 | **vLLM-only** | SGLang 只作为历史参考，不进入最终仓库主路径。 |
| backend 抽象 | **不保留 backend-agnostic client / sidecar 抽象** | 不再引入 `RolloutBackendClient` 这种中间层。 |
| SGLang 兼容层 | **不作为最终目标** | 不再为了兼容 SGLang 请求而长期保留 `/generate` 翻译层。 |

如果只用一句话概括当前共识：

> 最终目标是一个 vLLM-only 的 slime，数据面优先走 vLLM OpenAI-compatible 接口，控制面与数据面分离，共卡必须保留，分离部署必须能跑，PD separation / DP>1 / 多 model / 多 router 都要按真实拓扑单独设计，而不是靠 SGLang 兼容层糊过去。

---

### 0.2 Calvin / Samit fork 对比

#### 背景知识

两个参考分支相对 slime 主线的改动：

- `CalvinXKY/slime` 的 `vllm-backend` 更像一个**最小可用的 vLLM backend**。
- `SamitHuang/slime` 的 `dev_vllm` 更像一次**backend + router + weight sync + 配置体系**的联合重构。

**Calvin 改动清单：**

| 文件 | 改动内容 | 这件事的意义 |
| --- | --- | --- |
| `slime/utils/arguments.py` | 新增 `--rollout-backend {sglang,vllm}`，补齐一组 `vllm-*` 参数，vLLM 路径下切换 rollout function，并禁止 `prefill_num_servers`。 | 把 backend 分叉提前到参数层，避免把 SGLang 专用参数混进 vLLM 主链路。 |
| `slime/ray/rollout.py` | 根据 `rollout_backend` 选择 `VLLMEngine` 或 `SGLangEngine`，vLLM 模式下自动启用 slime router。 | 让 rollout 调度层真正支持"同一套入口，不同 backend 实现"。 |
| `slime/backends/vllm_utils/vllm_engine.py` | 新增 RayActor 包装器，负责启动本地 `vllm.entrypoints.openai.api_server`，做健康检查、router 注册、权重更新、sleep/wake、崩溃模拟等控制面动作。 | 这是 vLLM backend 的核心生命周期管理层。 |
| `slime/rollout/vllm_rollout.py` | 新增 vLLM rollout 逻辑，负责发请求、收 completion/logprob、处理 abort、做 evaluation rollout。 | 这是业务流入口，不管引擎怎么起，只管样本怎么走。 |
| `slime/backends/fsdp_utils/update_weight_utils.py` | 扩展 distributed weight sync，加入 vLLM reload/native 两条路，增加 rollout lock、master 地址解析和 native weight transfer 兼容。 | 解决"训练侧更新权重后，vLLM 侧怎么继续跑"的问题。 |
| `slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py` | pause/flush/continue 改成作用于全部 rollout engines，量化后处理也改成对完整 engine 集合操作。 | 避免只暂停局部 engine 导致状态不一致。 |
| `slime/ray/actor_group.py` | 传播更多 NCCL 相关环境变量。 | 让训练 actor 和 rollout actor 在同一套分布式环境里更稳定。 |
| `slime/utils/distributed_utils.py` | 修正 PyTorch 版本比较逻辑。 | 这是分布式初始化的兼容修复，和 vLLM 后端一起落地。 |
| `scripts/run-qwen3-0.6B-vllm-backend.sh` | 增加一个可直接跑的 vLLM backend 示例脚本。 | 方便快速验证整条链路。 |

**Samit 改动清单：**

| 文件 | 改动内容 | 这件事的意义 |
| --- | --- | --- |
| `slime/rollout/base_types.py` | 新增 backend-agnostic 的 `RolloutBackendRequest` / `RolloutBackendResponse`。 | 先把"请求/响应协议"抽出来，给不同 backend 共用。 |
| `slime/rollout/backends/base_client.py` | 定义 `RolloutBackendClient` 和 `BackendCapabilities`。 | 用能力描述把 SGLang/vLLM 的差异显式化。 |
| `slime/rollout/backends/sglang_client.py` | 封装 SGLang 请求、返回、abort 逻辑。 | 保持 SGLang 行为，同时让 rollout 代码不直接依赖 SGLang HTTP 细节。 |
| `slime/rollout/backends/vllm_client.py` | 同时支持直连 vLLM `/v1/completions` 和 router mode。 | vLLM 不只是一条直连路径，还可以被 router 统一接管。 |
| `slime/rollout/sglang_rollout.py` | 改成按 backend client 发请求，并统一把 backend response 应用回 `Sample`。 | rollout 业务层从"只会 SGLang"变成"backend 无关"。 |
| `slime/ray/rollout.py` | 引入 `SglangConfig`、`ModelConfig`、`EngineGroupConfig`、`EngineGroup`，支持 placeholder group、PD disaggregation，并新增 `_start_vllm_rollout_servers()`。 | 这条分支把 rollout 启动层抽象成"模型组 / engine 组 / router 组"。 |
| `slime/backends/vllm_utils/vllm_engine.py` | 负责启动 vLLM server + translation sidecar，注册 sidecar 到 router，维护 weight version，并提供 pause/resume/sleep 等控制。 | 这里的关键不是"起一个 vLLM"，而是"起一个能吃 slime 协议的 vLLM 节点"。 |
| `slime/backends/vllm_utils/vllm_translation_sidecar.py` | 把 SGLang `/generate` 翻译成 vLLM `/v1/completions`，并代理 health / abort / flush_cache / weight_version。 | 这是这条分支最像"兼容层"的核心文件。 |
| `slime/backends/megatron_utils/update_weight/common.py` | 修正 TP gather，对 `partition_stride` 做更完整的重组。 | 说明这条分支不仅改 backend，还碰到了权重布局问题。 |
| `slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py` | 大幅重写分布式权重更新，加入 vLLM packed transfer、bridge worker、锁、native protocol、post-process 流。 | 这是训练侧和 vLLM 侧真正对齐的地方。 |
| `slime/backends/megatron_utils/weight_sync_utils.py` | 本地重实现 sglang 的 weight-sync 辅助类。 | 让 vLLM-only 场景也能跑，不强依赖 sglang 安装。 |
| `slime/backends/megatron_utils/sglang.py` | 变成"优先用 sglang，找不到就回退到本地实现"。 | 降低环境依赖，增强可移植性。 |
| `slime/backends/megatron_utils/actor.py` / `arguments.py` / `model_provider.py` | 做兼容性修复，避免 provider / args / actor 接口在新体系下断掉。 | 这些是重构后常见的粘合层修补。 |
| `slime/backends/fsdp_utils/update_weight_utils.py` | 为 vLLM reload/native 路径补权重同步和锁。 | 把 vLLM 纳入同一套 FSDP 权重更新流。 |
| `slime/router/router.py` / `slime/utils/http_utils.py` | 统一把并发参数改成 `rollout_server_concurrency`。 | 说明 router 与 HTTP client 已经被纳入 backend-agnostic 调度层。 |
| `docs/en/vllm/ROUTER_DESIGN.md` 等 RFC 与文档 | 补了大量架构说明、RFC、部署说明。 | 这条分支不只改代码，也补齐了决策记录和使用文档。 |
| `run-qwen2.5-0.5B-vllm.sh` 等示例脚本 | 补了可运行示例和模型复现实验脚本。 | 说明这条分支更关注"端到端可用性"和"可复现性"。 |

#### 备注

主干一句话：

- `CalvinXKY/slime`：先把 vLLM 作为 slime 的一个 backend 跑起来。
- `SamitHuang/slime`：把 backend 抽象、router 协议、权重同步、文档和示例一起重构成一套更通用的体系。

参数层的具体差异详见 §5.1。

---

### 0.3 FSDP 排除

#### 背景知识

如果目标只是"Megatron + vLLM rollout backend"，下面这些内容可以直接不看或只当历史参考：

| 参考分支里的内容 | 是否合并 | 原因 |
| --- | --- | --- |
| `slime/backends/fsdp_utils/*` | 不合并 | 最新主线没有这个目录，不支持 FSDP training backend。 |
| `fsdp_utils/update_weight_utils.py` 里的 vLLM native / reload 改造 | 不合并 | 是 FSDP 权重同步路径，不是当前 Megatron 主线入口。 |
| `fsdp_utils/actor.py` 里的模型 wrap / FSDP 兼容修复 | 不合并 | 和 vLLM rollout backend 主目标无关。 |
| FSDP smoke / debug 脚本 | 不合并 | 会扩大验证面，和 `qwen2.5-0.5B` 最小闭环无关。 |

#### 备注

不考虑 FSDP **不等于不考虑权重同步**。最新 slime 的 Megatron 路径仍然有两条权重更新入口：

- `slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py`（non-colocate）
- `slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py`（colocate）

详见 §2.1（共卡）/ §2.2（分离）。

#### 结论

| 主题 | 结论 |
| --- | --- |
| FSDP | **不考虑**，这次合并目标里直接剔除。 |

---

### 0.4 rollout.py 合并方向

#### 背景知识

当前 slime 主线（commit `41dc3b6d`）的 `slime/ray/rollout.py` 已经从"一个扁平的 `all_rollout_engines` 列表"演进成了 `RolloutServer / ServerGroup / ModelConfig` 这一套拓扑结构：

- **RolloutServer**：一个模型对应一个 router。
- **ServerGroup**：一个 RolloutServer 里可以有多个 server group。
- **server group**：每个 group 代表一类同构 engine，可能是 prefill、decode、regular，或者 placeholder。
- **node-0 engine**：多节点部署时，真正向 router 暴露的通常是 node 0 上的 engine。
- **args.sglang_model_routers**：最新主线把每个 model 的 router 地址收敛到一个 map 里。

后端切换发生在三个层次：

- 上层用 `--rollout-backend` 选择 backend（`slime/utils/arguments.py:181-239`）；
- 中层在 `slime/ray/rollout.py:468-514` 选择 engine 类（vLLM 模式下自动启用 slime router）；
- 下层在 `slime/rollout/vllm_rollout.py:1-220` 负责请求/响应；
- `slime/backends/vllm_utils/vllm_engine.py:45-152` 负责 vLLM server 生命周期和控制面。

#### 备注

对 rollout.py 的合并判断：

1. 最新主线方向比 Calvin 分支更新，已经是 `RolloutServer / ServerGroup / ModelConfig` 拓扑。
2. Samit 的 `_start_vllm_rollout_servers()` 说明 vLLM 启动路径需要单独梳理，但其当前实现更像单模型 prototype（默认只返回 `{"default": RolloutServer(...)}`）。
3. 最终合并不应该拿 Calvin 的扁平启动方式覆盖主线，也不应该直接照搬 Samit 的单模型 vLLM 启动函数。

更合理的目标是：

```text
最新主线的多 model / 多 router / ServerGroup 拓扑
  +
Samit 版 vLLM 独立启动路径的思路
  -
Samit 版只支持 default 单模型的限制
  -
Calvin 版扁平 all_rollout_engines 结构
```

如果最终仓库已经是 **vLLM-only**，不要保留这种没有信息量的空 wrapper：

```python
def start_rollout_servers(args, pg):
    return _start_vllm_rollout_servers(args, pg)
```

更好的做法二选一：

- 保留 `start_rollout_servers(args, pg)` 这个外部名字，但让它本身就是 vLLM-only 的真实实现；
- 或者把调用方改成 `start_vllm_rollout_servers(args, pg)`，彻底删除泛化名字。

无论选哪种，函数内部都应该支持：

- 多 model；
- 每个 model 一个 router；
- 每个 model 多个 engine group；
- 给 rollout 函数暴露 per-model router 映射；
- 未来把 `args.sglang_model_routers` 改名成更中性的 `args.rollout_model_routers` 或 `args.vllm_model_routers`。

#### 结论

| 主题 | 结论 |
| --- | --- |
| 多 model / 多 router | **必须保留**，不能退化成单 model、单入口的扁平结构。 |
| engine 形态 | **RayActor + 本地 vLLM server subprocess**，不把推理逻辑直接塞进训练进程。 |

---

### 0.5 第一阶段范围

#### 背景知识

第一阶段不把下面这些纳入主目标：

- PD / EPD 之类的复杂 disaggregation；
- 量化模型的 `post_process_weights` 收尾；
- 外部 engine 的兼容模式；
- 复杂 router 适配层；
- 和 SGLang 完全等价的全部能力对齐。

标准不是"功能全"，而是"**能跑通、能更新、能继续跑**"。

#### 备注

以最新 slime 为基准的最小合并面：

| 模块 | 需要做什么 | 参考谁 |
| --- | --- | --- |
| `slime/utils/arguments.py` | 增加 `--rollout-backend vllm` 和少量 vLLM 参数。 | Calvin 的参数更轻，可作起点。 |
| `slime/ray/rollout.py` | 在 `ServerGroup.start_engines()` 里根据 `args.rollout_backend` 选择 `SGLangEngine` 或 `VLLMEngine`。 | Calvin 的分支选择思路即可，但要适配最新 `ServerGroup` 结构。 |
| `slime/backends/vllm_utils/vllm_engine.py` | 新增 `VLLMEngine` RayActor，接口要尽量对齐当前 `SGLangEngine`。 | Calvin 的 wrapper 简单，Samit 的 sidecar 注册思路更适合 router 兼容。 |
| `slime/backends/vllm_utils/vllm_translation_sidecar.py` | 不作为最终合并目标。 | 目标是 vLLM-only，rollout 侧直接使用 vLLM OpenAI-compatible endpoint，不需要 SGLang-shaped `/generate` 兼容层。 |
| `slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py` | 只改 Megatron distributed weight update，不碰 FSDP。 | Samit 的 vLLM packed/native 思路更有价值。 |
| `slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py` | 必须适配 vLLM colocate。 | 共卡是目标能力，不能显式 unsupported。 |
| `slime/backends/megatron_utils/weight_sync_utils.py` | 保留为 vLLM-only colocate 所需的通用权重搬运工具。 | 详见 §2.1。 |
| 测试 / 脚本 | 只保留 `qwen2.5-0.5B` 的最小 smoke。 | 两个分支都可参考，但不要引入 FSDP 测试面。 |

#### 结论

目前最推荐的第一阶段目标：

1. 不合并任何 `fsdp_utils`。
2. 不引入完整 backend-client 重构。
3. 不引入 translation sidecar。
4. 新增 `VLLMEngine`，让它对齐当前 `SGLangEngine` 的 RayActor 方法表。
5. 新增 / 收敛 vLLM rollout 业务入口，直接构造 vLLM OpenAI-compatible 请求。
6. 只适配 Megatron 的 `update_weight_from_distributed.py`。
7. colocate / `update_weight_from_tensor.py` 是必选项，必须实现 tensor / IPC 权重更新到共卡 vLLM engine。

合并目标从"几十个文件的大重构"变成"少数入口文件 + 一个 engine wrapper + 一个直接 vLLM rollout 入口 + Megatron distributed/colocate 两条权重同步适配"。

最终路径应该让 slime **直接理解 vLLM 的请求/响应格式**，而不是让 vLLM 伪装成 SGLang worker。这样会多改一点 rollout 业务层，但能删掉一整层兼容协议，也符合"最终没有 SGLang 后端"的方向。

这也意味着 Samit 分支里的 backend client / sidecar 设计可以作为历史参考，但不应该原样合并进最终主线。

---

## 1. 协议与端点

### 1.1 数据面端点选型

#### 背景知识

router 不应该承担：

- SGLang 风格请求体翻译；
- vLLM OpenAI response 的再加工；
- 权重传输的主路径；
- 训练步骤本身的状态管理。

理想情况是：router 对外说自己的路由协议；slime 直接发 OpenAI-compatible 请求；worker / engine 自己吃 `/v1/completions`；训练侧完全不需要先转成 `/generate` 再翻译回来。

**vLLM 主仓 + SGLang 主仓的 6 个数据面 endpoint 横向对比：**

##### 端点注册位置

| 端点 | 框架 | 路由声明 |
| --- | --- | --- |
| `/generate` | vLLM | `vllm/entrypoints/api_server.py:46` |
| `/v1/completions` | vLLM | `vllm/entrypoints/openai/completion/api_router.py:35` |
| `/v1/chat/completions` | vLLM | `vllm/entrypoints/openai/chat_completion/api_router.py:41` |
| `/inference/v1/generate` | vLLM | `vllm/entrypoints/serve/disagg/api_router.py:50` |
| `/generate` | SGLang | `python/sglang/srt/entrypoints/http_server.py:702` |
| `/v1/completions` | SGLang | `python/sglang/srt/entrypoints/http_server.py:1482` |

##### Request 协议关键差异

| 维度 | `/generate`<br/>(vLLM) | `/v1/completions`<br/>(vLLM) | `/v1/chat/completions`<br/>(vLLM) | `/inference/v1/generate`<br/>(vLLM) | `/generate`<br/>(SGLang) | `/v1/completions`<br/>(SGLang) |
| --- | --- | --- | --- | --- | --- | --- |
| Schema 类 | 无 Pydantic（裸 dict → `SamplingParams`） | `CompletionRequest` | `ChatCompletionRequest` | `GenerateRequest` | `GenerateReqInput`（`io_struct.py:135`） | `CompletionRequest`（`openai/protocol.py:251`） |
| Prompt 形态 | `prompt: str \| list[str]`，只支持文本 | `prompt: list[int] \| list[list[int]] \| str \| list[str]`，token-in ✅ | `messages: list[ChatMessage]` | `token_ids: list[int]`，token-only ✅ | `text` 或 `input_ids: list[int]` 或 `input_embeds`，token-in ✅ | `prompt: list[int] \| list[list[int]] \| str \| list[str]`，token-in ✅ |
| Sampling params 布局 | 与 prompt 平铺 | OpenAI 平铺 | OpenAI 平铺（`max_completion_tokens`） | 嵌套 `sampling_params` 对象 | 嵌套 `sampling_params: dict` | OpenAI 平铺 + SGLang 扩展 |
| `logprobs` 入口 | 在 `SamplingParams` 内 | 顶层 `logprobs: int` + `echo: bool` + `prompt_logprobs: int` | 顶层 `logprobs` + `prompt_logprobs` | 通过 `SamplingParams` 传 | `return_logprob` / `top_logprobs_num` / `logprob_start_len` / `token_ids_logprob` | OpenAI 风格 `logprobs: int`（只 top-N） |
| `routed_experts` request 入口 | ❌ | ❌ | ❌ | ❌ | ✅ `return_routed_experts: bool` + `routed_experts_start_len: int` | ✅ `return_routed_experts: bool` |
| Streaming | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

##### Response 协议关键差异

| 维度 | `/generate`<br/>(vLLM) | `/v1/completions`<br/>(vLLM) | `/v1/chat/completions`<br/>(vLLM) | `/inference/v1/generate`<br/>(vLLM) | `/generate`<br/>(SGLang) | `/v1/completions`<br/>(SGLang) |
| --- | --- | --- | --- | --- | --- | --- |
| Schema 类 | `JSONResponse({"text": [...]})` | `CompletionResponse` + `CompletionResponseChoice` | `ChatCompletionResponse` + `ChatCompletionResponseChoice` | `GenerateResponse` + `GenerateResponseChoice` | dict（无正式 class，结构在 `tokenizer_manager.py:1737` 附近） | `CompletionResponse`（`openai/protocol.py:368`） |
| 生成 token IDs | ❌ 只返回 text | ✅ `choices[].token_ids` | ✅ `choices[].token_ids` | ✅ `choices[].token_ids` | ✅ `output_ids: list[int]`（顶层） | ❌ 只 `choices[].text`，无 `output_ids` |
| Logprobs | ❌ | ✅ `choices[].logprobs` + 顶层 `prompt_logprobs` | ✅ `choices[].logprobs` + 顶层 `prompt_logprobs` | ✅ `choices[].logprobs` | ✅ `meta_info["input_token_logprobs"]` / `meta_info["output_token_logprobs"]` + `*_top_logprobs` | ✅ `choices[].logprobs`（OpenAI 风格，无 prompt 段除非 `echo=True`） |
| MoE `routed_experts` | ❌ | ✅ `choices[].routed_experts` + 顶层 `prompt_routed_experts` | ✅ `choices[].routed_experts` + 顶层 `prompt_routed_experts` | ❌ | ✅ `meta_info["routed_experts"]`（base64 编码） | ✅ `sglext.routed_experts`（顶层 `SglExt` 字段，base64 编码） |
| `finish_reason` | ❌ | ✅ | ✅ | ✅ | ✅ `meta_info["finish_reason"]`（str 或 error dict） | ✅ `choices[].finish_reason` |
| `usage` | ❌ | ✅ | ✅ | ✅ | ✅ 散落在 `meta_info["prompt_tokens"]` / `["completion_tokens"]` / `["cached_tokens"]` | ✅ `usage: UsageInfo` |

源码定位（response schema 上 `routed_experts` 字段是否存在）：

- vLLM `/v1/completions`：`vllm/entrypoints/openai/completion/protocol.py:471` + `:478`。
- vLLM `/v1/chat/completions`：`vllm/entrypoints/openai/chat_completion/protocol.py:95` + `:102`。
- vLLM `/inference/v1/generate`：`vllm/entrypoints/serve/disagg/protocol.py` grep 不到这两个字段，确认缺失。
- SGLang `/generate`：`meta_info["routed_experts"]` base64 字符串（构造在 `tokenizer_manager.py:1712` 附近）。
- SGLang `/v1/completions`：`sglext.routed_experts` 字段在 `openai/protocol.py:342`（`SglExt` 类），同样是 base64。

##### 调用栈分叉与收敛

两个框架各自有一个收敛点：

**vLLM 4 个 endpoint → `AsyncLLM.generate(...)`**：

```text
/generate
  → api_server.py:60  handler
  → engine.generate(prompt, sampling_params, request_id)
                                                                ↓
/v1/completions                                                 ↓
  → completion/api_router.py:46  handler                        ↓
  → OpenAIServingCompletion.create_completion()                 ↓
  → engine_client.generate(...)  [completion/serving.py:191] ───┤
                                                                ↓
/v1/chat/completions                                            ↓
  → chat_completion/api_router.py:53  handler                   ↓
  → OpenAIServingChat.create_chat_completion()                  ↓
  → engine_client.generate(...)  [chat_completion/serving.py:341] ──┤
                                                                ↓
/inference/v1/generate                                          ↓
  → serve/disagg/api_router.py:61  handler                      ↓
  → ServingTokens.serve_tokens()                                ↓
  → engine_client.generate(...)  [serve/disagg/serving.py:171] ─┤
                                                                ↓
                                ┌───────────────────────────────┘
                                ↓
                AsyncLLM.generate(...)  [vllm/v1/engine/async_llm.py:524]
                                ↓
                AsyncLLM.add_request(...)  [line 559]
                                ↓
        (共用) InputPreprocessor / Scheduler / Workers
```

**SGLang 2 个 endpoint → `TokenizerManager.generate_request(...)`**：

```text
/generate
  → http_server.py:706  generate_request handler
  → TokenizerManager.generate_request(obj, request) ─────────┐
                                                             ↓
/v1/completions                                              ↓
  → http_server.py:1483  openai_v1_completions handler       ↓
  → OpenAIServingCompletion.handle_request()                 ↓
  → _convert_to_internal_request()  (CompletionRequest → GenerateReqInput)
  → TokenizerManager.generate_request() ─────────────────────┤
                                                             ↓
                          ┌──────────────────────────────────┘
                          ↓
        TokenizerManager.generate_request(...)
        [python/sglang/srt/managers/tokenizer_manager.py:511]
                          ↓
        (共用) tokenize → Scheduler → Detokenizer
```

##### RL backend 就绪度评分

| 端点 | token-in | token-out | per-token logprobs | prompt logprobs | routed_experts |
| --- | :---: | :---: | :---: | :---: | :---: |
| `/generate` (vLLM) | ❌ | ❌ | ❌ | ❌ | ❌ |
| `/v1/completions` (vLLM) | ✅ | ✅ | ✅ | ✅ | ✅ |
| `/v1/chat/completions` (vLLM) | ❌（messages-only） | ✅ | ✅ | ✅ | ✅ |
| `/inference/v1/generate` (vLLM) | ✅ | ✅ | ✅ | ✅ | ❌ |
| `/generate` (SGLang) | ✅ | ✅ | ✅ | ✅ | ✅ |
| `/v1/completions` (SGLang) | ✅ | ❌（无 `output_ids`） | ✅ | ⚠️ 需 `echo` | ✅ |

#### 备注

**`routed_experts` 的开关在 vLLM 和 SGLang 是两种范式：**

- **vLLM**：没有任何 endpoint 在 request 上提供 opt-in，开关在 server 启动侧——`ModelConfig.enable_return_routed_experts: bool`（`vllm/config/model.py:215`，CLI 用 `--enable-return-routed-experts`）。设上以后引擎对所有 request 都生产这条数据，**能不能传到 client 取决于该 endpoint 的 response schema 是否带这个字段**。
- **SGLang**：`/generate` 和 `/v1/completions` 都把 `return_routed_experts` 暴露成 per-request opt-in。`/generate` 额外有 `routed_experts_start_len` 控制从 prompt 哪个位置开始收集（跟 logprob 的 `logprob_start_len` 是一对的设计）。

**收敛结构有几个直接含义：**

- **6 个端点的差异完全发生在协议层**，不是引擎能力差异。任何 endpoint 能否拿到 `routed_experts`，纯粹看 schema 把这个字段透不透——引擎本身都能生产。
- **选哪个 endpoint 的判断依据是协议契合度**，不是性能或能力。
- **SGLang `/v1/completions` 是反 RL 友好的那一个**：它在 SGLang 内部把 OpenAI 请求转回 `GenerateReqInput`，但 response 走的是 OpenAI `CompletionResponse` 路径，所以丢掉了 `output_ids`（OpenAI 协议里没这个字段）——同样的引擎、同样的内部状态，只因 response schema 不同就少返一组数据。slime 当前用 SGLang `/generate` 是对的，不应该退回到 SGLang `/v1/completions`。

#### 参考

**SkyRL 选型已过时**：SkyRL 自加 `/skyrl/v1/generate` 那个 workaround 在现版本 vLLM 下没必要再做。SkyRL 源码注释里写 *"native endpoint `/inference/v1/generate` does not support returning routed expert IDs. TODO: Migrate back once this is fixed on the vllm side"*——他们等的是 `/inference/v1/generate` 补 `routed_experts`，但 `/v1/completions` 早就有这两个字段了。slime 不应复制 SkyRL 这个 workaround。

#### 结论

1. **vLLM `/v1/completions` 是当前 vLLM 主仓里唯一同时打通三件套（token-in/out、logprobs、routed_experts）的 endpoint。** slime 选它为主数据面是干净路径。
2. **vLLM `/inference/v1/generate` 的定位**：disagg（PD 分离）路径上的 token-only endpoint，schema 比 `/v1/completions` 更紧凑（无 OpenAI 平铺，sampling_params 嵌套），但代价是缺 `routed_experts`。第一阶段不推荐用，除非 PD 场景明确需要。
3. **vLLM `/generate` 已经不是合格的 RL 数据面端点**：只返回 `{"text": [...]}`，三件套全不支持。slime 不应该走这条。
4. **vLLM `/v1/chat/completions` 不能作为 token-in 主路径**：request 只接受 `messages`。如果未来需要 chat-template 主导的 rollout（multi-turn agent），它是合适入口，但默认还是 `/v1/completions`。
5. **slime 当前的 SGLang `/generate` 路径功能上跟 vLLM `/v1/completions` 是平迁的**：

   | 字段 | SGLang `/generate` | vLLM `/v1/completions` |
   | --- | --- | --- |
   | token-in | `input_ids` | `prompt: list[int]` |
   | token-out | `output_ids`（顶层） | `choices[].token_ids` |
   | per-token logprobs | `meta_info["output_token_logprobs"]` | `choices[].logprobs` |
   | prompt logprobs | `meta_info["input_token_logprobs"]` | 顶层 `prompt_logprobs` |
   | routed_experts | `meta_info["routed_experts"]`（base64） | `choices[].routed_experts`（list of list of list） |
   | routed_experts 触发方式 | per-request `return_routed_experts: bool` | server 启动 `--enable-return-routed-experts` |

   迁移面没有功能丢失，但有两个**协议层不可绕过的差异**需要 slime 这边做适配：

   - SGLang 的 `routed_experts` 是 base64 字符串，vLLM 是直接的嵌套 list；rollout 客户端解码逻辑不同。
   - SGLang 是 per-request 开关，vLLM 是 server-launch 开关。**这意味着 vLLM 启动 flag 一旦设了 `--enable-return-routed-experts`，所有 request 都会产生 routed_experts 数据**——非 MoE 模型也会带这个字段（值是 None 或空），不能像 SGLang 那样按需开关减少负载。

6. **不要退回到 SGLang `/v1/completions`**：丢掉 `output_ids`，还需要 `echo=True` 才能拿到 prompt logprobs。SGLang 体系内 `/generate` 才是 RL 完全打通的那条；slime 现状是对的。

7. **结论 #1 的"打通三件套"只对纯文本 RL 成立**——VLM 场景见 §1.2。

| 主题 | 已确认 |
| --- | --- |
| 主数据面 | `/v1/completions`（vLLM） |
| 采样接口 | 优先 `/v1/completions`（暂定，是否同时保留 `/v1/chat/completions` 待定，见待讨论） |

#### 待讨论

- 是否还需要保留 `POST /v1/chat/completions` 作为可选路径？
- 是否彻底删除 `/generate` 兼容层？
- `prompt` 允许 token id list 还是只接受 string？
- `logprobs`、`return_token_ids`、`prompt_logprobs` 是否都要支持？
- 路由或 MoE 专家信息是否要透传到 response？
- router 是否需要识别 `routed_experts` / `prompt_routed_experts` 这类元数据？

---

### 1.2 多模态输入路径

#### 背景知识

`/v1/completions` 没有任何原生多模态字段。VLM rollout 不能"`/v1/completions` 一把梭"。

| Endpoint | MM 输入 | 形式 |
| --- | :---: | --- |
| `/generate` (vLLM) | ❌ | — |
| `/v1/completions` (vLLM) | ❌ 无原生 | 只有 `prompt_embeds: bytes`（pre-encoded 文本侧 embedding）。要做 MM 必须 client 端自己跑 vision encoder + 把 image features 嵌进 prompt embedding 序列正确位置，整条 MM 前处理流水线搬到 client——现实里几乎不可行 |
| `/v1/chat/completions` (vLLM) | ✅ 完整 | messages content parts: `image_url` / `image_pil` / `image_embeds` / `audio_url` / `input_audio` / `audio_embeds` / `video_url`；顶层 `audio: OpenAIChatCompletionAudio`；`mm_processor_kwargs: dict` |
| `/inference/v1/generate` (vLLM) | ✅ disagg 风格 | `features: MultiModalFeatures`，含 `mm_hashes` / `mm_placeholders` / `kwargs_data`（base64 编码的 `MultiModalKwargsItem`）——**预处理过**的 features，不接受 raw image |
| `/generate` (SGLang) | ✅ raw | `image_data` / `video_data` / `audio_data: MultimodalDataInputFormat`，接受 file path / URL / base64 / image instance（`io_struct.py:152-156`）；加 `modalities: List[str]` 和 `use_audio_in_video: bool` |
| `/v1/completions` (SGLang) | ❌ | 跟 vLLM 同 endpoint 一样无原生 MM |

#### 备注

`MultiModalFeatures` 在 `vllm/entrypoints/serve/disagg/protocol.py:32` 的注释把 vLLM 的 disagg MM 流水线讲得很清楚：

> *Lightweight multimodal metadata produced by the render step. Carries hashes (for cache lookup / identification) and placeholder positions so the downstream `/generate` service knows where in the token sequence each multimodal item lives.*

vLLM 的 disagg MM 是**两段式**：

```text
client / raw image
  → POST /v1/chat/completions/render
  → MultiModalFeatures { mm_hashes, mm_placeholders, kwargs_data }
  → POST /inference/v1/generate (传 features)
  → token output
```

跟 vLLM 对照，SGLang 在 MM 上的设计哲学完全不同：**SGLang `/generate` 直接接 raw image / video / audio**（URL、base64、文件路径都行），server 端一站式处理。代价是 server 必须挂全套 MM 预处理 pipeline；好处是 client 端逻辑极其简单。

##### slime 的三条 VLM 走法

slime 现状是 SGLang `/generate` + raw `image_data`，迁到 vLLM 时**这条路径在 vLLM 主仓没有同形入口**。三条候选：

| 路径 | 优点 | 代价 |
| --- | --- | --- |
| A. vLLM `/v1/chat/completions` + messages 挂 `image_url` | vLLM 完整托管 MM 前处理 | 失去 token-in（messages-only），prompt 段精细控制弱；但 `output_ids` / `logprobs` / `routed_experts` 仍能拿（§1.1 表已确认） |
| B. Client 算 prompt_embeds → vLLM `/v1/completions` | 表面上 token-in/out + routed_experts 全打通 | 把 MM 整条前处理搬到 slime 训练侧（vision encoder + placeholder 对齐 + prefix cache 兼容），现实里几乎不可行 |
| C. 两段式 vLLM `/v1/chat/completions/render` → `/inference/v1/generate` | 利用 vLLM 原生 disagg MM 流水线，token-in/out 干净；跟 SkyRL 现有做法对齐 | `/inference/v1/generate` 没有 `routed_experts`；HTTP 多一跳；render 必须在同一 server 跑（共享 tokenizer / MM processor） |

场景矩阵：

| 场景 | 推荐路径 | 理由 |
| --- | --- | --- |
| 纯文本 RL | vLLM `/v1/completions` | 三件套全打通（见 §1.1 结论 #1） |
| VLM RL（非 MoE） | A 或 C | A 简单上手，C 干净且贴近 SkyRL 实际工程 |
| VLM + MoE（同时需要 routed_experts） | **当前 vLLM 主仓无干净路径** | A 能拿 routed_experts 但失 token-in；C token-in 干净但 routed_experts 没字段——这是真正的 vLLM 上游 gap |

#### 参考

SkyRL 已经在用 `/v1/chat/completions/render`（`remote_inference_client.py:768`）——这是 disagg MM 的标配 entry。

#### 待讨论

- VLM + MoE 同时需要时，要么等 vLLM `/inference/v1/generate` 补 `routed_experts`，要么 fork。这条要不要立项？
- VLM 第一阶段是 A 还是 C？

---

### 1.3 控制面端点

#### 背景知识

vLLM 控制面 endpoint：

- `/health`
- `/pause` / `/resume`
- `/sleep` / `/wake_up`
- `/reset_prefix_cache`
- `/start_profile` / `/stop_profile`
- `/init_weight_transfer_engine`
- `/update_weights`
- `/abort_request`

这些接口的共同点：**不属于生成请求本身**。它们处理生命周期、显存管理、权重更新、调试和中断，不应和 `prompt → completion` 主链路混在一起。

#### 结论

| 主题 | 已确认 |
| --- | --- |
| 控制面 | **控制面和数据面分离**——权重更新、pause/resume、sleep/wake、flush cache、profile 不和生成请求混在一起。 |

控制面操作的运行时语义（sleep tag、flush_cache 副作用等）见 §4.2。

#### 待讨论

- 控制面接口是否要统一通过 engine 本体而不是 router？

---

### 1.4 router 角色边界

#### 背景知识

router 只需要解决两件事：

1. **请求分发**：把一个 rollout 请求送到合适的 worker / engine / replica。
2. **拓扑协调**：知道当前有哪些 worker、哪些 worker 可用、哪些 worker 需要被剔除。

它不应该承担：SGLang 风格请求体翻译；vLLM OpenAI response 的再加工；权重传输的主路径；训练步骤本身的状态管理。

理想情况是：

- router 对外说自己的路由协议；
- slime 直接发 OpenAI-compatible 请求；
- worker / engine 自己吃 `/v1/completions`；
- 训练侧完全不需要先转成 `/generate` 再翻译回来。

如果后面确实需要兼容历史形态，那也应该只作为薄适配层存在，而不是主协议。

#### 结论

| 主题 | 已确认 | 备注 |
| --- | --- | --- |
| router 协议 | **主协议优先按 vLLM 的 OpenAI-compatible 接口走** | 数据面优先 `/v1/completions`，不是 SGLang `/generate`。 |
| 多 model / 多 router | **必须保留** | 不能退化成单 model、单入口的扁平结构。 |

| 主题 | 暂定 | 仍需确认 |
| --- | --- | --- |
| router 端点 | 优先让 router 直接接 OpenAI-compatible 请求 | `vllm-router` 是否需要额外包装一层 slime 自定义协议。 |

#### 待讨论

- `vllm-router` 是否只负责分发，还是还要承担部分协议适配（待和 router 侧作者确认）？
- router 是否需要识别 `routed_experts` / `prompt_routed_experts` 这类元数据？
- 多 model / 多 router 是否和当前 slime 主线保持一致？

---

## 2. 权重同步

### 2.1 共卡 (colocate) 路径

#### 背景知识

共卡是必须保留的路径。核心含义：

- 训练进程和 rollout engine 共享同一组 GPU 资源，或者在同一节点上紧密共址；
- 权重更新不能依赖"远端拉取整个 checkpoint"这种慢路径；
- 更适合走 tensor / IPC / 本地进程间通信；
- 本质目标是让训练和推理在同一台机器上的切换成本尽量低。

**共卡不是"特殊优化"，而是必须支持的主能力之一。**

##### 共卡权重同步路径示意（Megatron + vLLM colocate）

```text
Megatron training rank
  -> HfWeightIterator: Megatron layout -> HF/vLLM named tensors
  -> update_weight_from_tensor._send_to_colocated_engine()
  -> FlattenedTensorBucket: 多个 tensor 打平成一块 flattened_tensor
  -> MultiprocessingSerializer: 序列化 flattened_tensor + metadata
  -> dist.gather_object: 每个 engine 本地组汇总到 source rank
  -> VLLMEngine.update_weights_from_tensor(...)
  -> vLLM server native / tensor load path
```

##### 共卡需要的 IPC 工具集

`slime/backends/megatron_utils/weight_sync_utils.py` 在 vLLM-only + colocate 目标下应被重新理解成 **通用 weight sync IPC 工具**：

- `FlattenedTensorBucket`：减少大量小 tensor 的 IPC 往返，把权重按 bucket 搬运。
- `MultiprocessingSerializer` / `SafeUnpickler`：让包含 CUDA tensor 的对象能跨进程安全传递。
- `monkey_patch_torch_reductions()`：修复 multiprocessing 传 CUDA tensor 时设备编号不稳定的问题。

#### 备注

##### 为什么 Samit 分支会新增 `weight_sync_utils.py`

Samit 这个新文件的直接原因是：Calvin 分支仍然把这几个底层工具当成 **SGLang 包提供的工具** 来 import。例如 Calvin 分支里：

```python
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
```

这些 import 不只出现在 FSDP 路径里，也出现在 Megatron 的 `slime/backends/megatron_utils/sglang.py` 里。在 Calvin 的设计里，即使 rollout backend 是 vLLM，只要权重同步还复用这套工具，运行环境里仍然最好安装 SGLang。

Samit 新增 `slime/backends/megatron_utils/weight_sync_utils.py` 就是为了切断这个运行时依赖：

```text
原来:
  slime weight sync
    -> import sglang.srt.* helper
    -> 需要环境里有 sglang

Samit:
  slime weight sync
    -> 优先 import sglang.srt.* helper
    -> 如果没有 sglang，fallback 到本地 weight_sync_utils.py
    -> vLLM-only 环境也能跑
```

如果最终仓库真的完全删除 SGLang，理想合并方式不是继续保留"优先 import SGLang，失败再 fallback"的写法，而是把这几个工具正式变成 slime 自己的通用模块（如 `slime/backends/megatron_utils/weight_sync_utils.py` 或 `slime/backends/common/weight_sync_ipc.py`）。

##### 后续专题：Megatron + vLLM 共卡如何真正实现

这一块需要单独研究，不能只靠"把 SGLang 的共卡路径换成 VLLMEngine"来拍脑袋合并。原因是 colocate 不只是一个启动参数，它会同时影响三件事：

1. **进程拓扑**：Megatron training actor 和 vLLM rollout engine 是否在同一组 GPU placement 上，RayActor 如何绑定 GPU，vLLM server subprocess 如何继承可见设备。
2. **显存生命周期**：训练 step、rollout generation、KV cache、vLLM sleep/release memory、Megatron 参数/optimizer state 之间如何错峰占用显存。
3. **权重同步方式**：是走 tensor / IPC 直接把当前训练权重送入共卡 engine，还是走 vLLM native weight update / reload，哪些 rank 负责打包，哪些 engine endpoint 负责接收。

因此后续要专门读一遍：

- slime 当前 SGLang colocate 是怎么做的；
- Calvin / Samit 两个分支的 `update_weight_from_tensor.py` 如何把训练权重送到 rollout engine；
- vLLM server 在共卡时支持哪些 sleep / wake / weight update 能力；
- verl 里 Megatron + vLLM 或类似 colocate rollout 的实现方式，尤其是它怎么处理显存释放、权重同步和 actor/engine placement。

暂时的判断：

- `weight_sync_utils.py` 可以作为共卡 IPC 工具候选；
- `update_weight_from_tensor.py` 是 slime 侧必须重点适配的入口；
- `VLLMEngine.update_weights_from_tensor(...)` 是 vLLM engine 侧必须补齐的接口；
- 是否采用 vLLM native weight transfer，需要等看完 vLLM/verl 的共卡实现后再定。

##### 共卡的三种实现风格（对照 verl）

| 方式 | `verl` 里类似什么 | slime 里可能对应什么 |
| --- | --- | --- |
| 直接 NCCL 广播 | `sync_rollout_weights` 里的 broadcast | 训练进程直连 rollout group 的同步通道 |
| IPC / ZMQ bucket transfer | `ServerAdapter.update_weights()` | 训练侧把权重分桶送到本地 rollout engine/helper |
| shared memory fallback | `BucketedWeightSender` fallback | IPC 不可用时退到共享内存或同机缓冲 |

这三种方式里，第一版最该优先的是本地共卡路径，而不是远端广域同步。

#### 参考

##### SkyRL：权重同步要做成独立策略层，并拆成 `start -> chunk* -> finish`

SkyRL 在 `SkyRL/skyrl/backends/skyrl_train/weight_sync/*` 和 `SkyRL/skyrl/backends/skyrl_train/inference_servers/new_inference_worker_wrap.py` 里，把一次权重更新拆成三段：

- `start_weight_update()`：先准备加载环境
- `update_weights_chunk()`：按 chunk 传输并装载
- `finish_weight_update()`：最后做收尾和后处理

这条最值得抄，因为它把"传输方式"和"后处理"分开了：

- 共卡 / 非共卡可以走不同传输实现
- packed / native update 可以挂在不同策略后面
- 峰值内存和加载时序更容易控制

更具体地说：

- **SkyRL 现有的 chunked IPC 路径已经在用 `start_weight_update -> update_weights_chunk -> finish_weight_update`。**
- 这条路径正好适合后面实现 Slime 的 IPC 共卡方案时参考。
- 但它现在只覆盖 SkyRL 的 IPC chunked 路径，不代表所有权重同步后端都已经统一成三阶段。

但 Slime + vLLM-only 当前**没有这么做**——更偏向先把 vLLM-only 主路径跑稳，权重更新链路还没有像 SkyRL 这样显式拆成一套独立策略层。

##### SkyRL：把"抽权重 / 转格式"和"传权重"彻底拆开

SkyRL 里这层已经很清楚：

- `WeightExtractor` 只负责把模型整理成 `WeightChunk`
- `get_weight_metadata()` 只拿名字、dtype、shape，不碰 tensor 本体
- `WeightTransferStrategy` 只负责怎么传
- `BroadcastTransferStrategy` / `CudaIpcTransferStrategy` 只做 transport，不关心 Megatron 内部怎么切分参数

对应到 Samit 的 `update_weight_from_distributed.py`，现在还是揉在一起的：

- `named_params_and_buffers()` / `all_gather_param()` / `convert_to_hf()` 在同一个文件里
- pause / flush / expert 分流 / packed 分支 / bridge 发送也都在一起

如果后面要继续简化 Slime vLLM-only，建议优先学 SkyRL 这一点：**先把"权重是什么形状、怎么被拼起来"做成 extractor，再把"怎么送"做成 strategy。**

##### SkyRL：把"拓扑 / 部署参数"显式化，不要在发送时临时推

SkyRL 的思路是：**先把拓扑信息算成一个显式的 init_info，再让 sender / receiver 按这个 info 干活。**

在 `skyrl/backends/skyrl_train/weight_sync/transfer_strategy.py` 里：

```python
init_info = WeightSyncInitInfo.for_engine(
    engine_index=...,
    tp_size=...,
    pp_size=...,
    dp_size=...,
)
```

```python
init_info = BroadcastInitInfo.for_servers(
    world_size_per_server=...,
    num_servers=...,
    dp_size=...,
)
```

**"每台 server 有多大、总共有几台、DP 怎么切"先被写死在 init_info 里**。后面的 `create_sender()` / `create_receiver()` 只消费这个对象，不重新算拓扑。

对照 Samit 的 `update_weight_from_distributed.py`：

- 先在 `connect_rollout_engines_from_distributed()` 里临时拼 `engine_gpu_counts`
- 再算 `cumulative`、`world_size`
- 再决定每个 rollout group 怎么建 bridge、怎么连 NCCL
- 最后在 `update_weights_from_distributed()` 里继续做 packed / expert / non-packed 的分支发送

这能跑，但会让"拓扑推导"和"权重发送"缠在一起。一旦支持多 engine、不同 engine 拓扑不完全一致、DP > 1、未来 PD 分离、colocated / non-colocated 共存，就很容易把 `connect` / `send` / `broadcast` 揉成一锅。

Slime vLLM-only 该怎么学：

1. 启动阶段先构造一个 `WeightSyncTopology` / `RolloutTopology` 之类的对象
2. 这个对象里只放部署和并行信息，不放 tensor
3. sender / receiver / router / engine wrapper 都只读这个对象
4. 需要连多少个 engine、要不要 broadcast、要不要分组，都从这个对象读
5. 真正发权重的函数里不再重复计算 `rank_offset`、`world_size`、`group_name`

一句话：**部署拓扑应该先抽成显式配置，再让同步流程消费它；不要让权重发送函数顺手兼任"拓扑推导器"。**

##### CudaIpcTransferStrategy 不是 Samit 的对标

要把可比关系说准：

- `CudaIpcTransferStrategy` **不是** Samit `update_weight_from_distributed.py` 的对标实现。
- Samit 当前那份文件主要在做 **Megatron → rollout engine 的分离 NCCL 权重同步总控**——对标应该是 SkyRL 的 `BroadcastTransferStrategy` / vLLM native `NCCLWeightTransferEngine`（详见 §2.2 参考段）。
- `CudaIpcTransferStrategy` 只是在 SkyRL 里和 `BroadcastTransferStrategy` 并列的另一种 strategy，用于 **colocated / CUDA IPC 共卡路径**，只跟未来 IPC 共卡方案有关。

#### 结论

| 主题 | 结论 |
| --- | --- |
| 共卡 | **必须保留 colocate**，不能把共卡当成第一阶段可删项。 |
| 权重同步生命周期 | **一次 init + 后续复用**，不要每轮同步都重新 init 通信组。 |

#### 待讨论

- 共卡路径具体走 NCCL / IPC / helper process / HTTP bridge 哪种？
- native weight update 和 reload 模式的优先级如何？
- `post_process_weights` 是不是只在量化/特殊后处理场景启用？
- Megatron + vLLM 共卡专题（见上面"后续专题"）什么时候立项？

---

### 2.2 分离 (non-colocate / separated) 路径

#### 背景知识

这是训练和推理解耦部署时的路径。一般会出现这些要求：

- 权重从训练侧广播或传给独立的 rollout 侧；
- 需要一层明确的同步协议；
- 可能需要 helper worker、bridge worker、NCCL group 或者其它传输通道；
- 不能把训练进程直接绑死在推理 runtime 内部实现上。

这条路更适合大规模部署，但设计会比共卡更重。

##### 四种候选传输实现

| 候选实现 | 传输边界 | 优点 | 代价 |
| --- | --- | --- | --- |
| NCCL group | 训练与推理都在可互联的分布式域里 | 性能高，语义直观 | 对拓扑和环境要求高 |
| IPC / helper process | 训练进程只跟本地 helper 说话 | 训练主进程解耦，边界清楚 | 多一个进程层，状态复杂 |
| bucketed network transfer | 分块发送，减少单次内存峰值 | 更适合大模型权重 | 需要定义清楚 bucket 协议 |
| HTTP bridge | 纯 HTTP 调度 / 适配层 | 简单，易调试 | 不是高性能权重传输方案 |

##### 同步通信拓扑：PP=4 时 helper / NCCL world 到底是什么

先给结论：

1. **helper 子进程不是"每个 tensor 一个"，也不是"每个 broadcast 一个"**。
2. **helper 子进程是"每个权重同步域 / source rank 一套"**。
3. 如果 `PP=4`，并且这 4 个 pipeline stage 都各自有自己的 source rank、各自要把权重发到 rollout 端，**通常就会有 4 个 helper 子进程 + 4 套独立的 NCCL world**。
4. 独立 world 不是为了"分裂成很多小世界"，而是为了让每个 PP stage 的权重同步互不干扰。

```text
                训练侧
          +-------------------+
          | PP stage 0 source |---- helper0 ---- NCCL world0 ---- rollout group0
          +-------------------+
          | PP stage 1 source |---- helper1 ---- NCCL world1 ---- rollout group1
          +-------------------+
          | PP stage 2 source |---- helper2 ---- NCCL world2 ---- rollout group2
          +-------------------+
          | PP stage 3 source |---- helper3 ---- NCCL world3 ---- rollout group3
          +-------------------+

                推理侧
          rollout engines / server ranks
```

#### 备注

##### helper 是谁起的？

不是所有 rank 都起。helper 是**权重同步发起侧**起的——每个参与某个同步域的 **source rank** 起一个。

- 每个 PP stage 都有一个"本 stage 的发起点"；
- 这个发起点负责把本 stage 的权重整理好；
- 然后再通过 helper 把 tensor 送到对应的 rollout group。

helper 不是"训练集群统一多开几个线程"这么简单，而是跟**同步域**绑定的。

##### NCCL world 为什么会有多套？

因为每个 source rank 对应的接收集合可能不同。`world_size` 不是拍脑袋写死的，而是按这条同步域里的成员数算出来的：

```text
world_size = 训练 source rank 1 个 + rollout engine 内所有 GPU rank
```

代码里 `rank_offset` 和 `world_size` 就是在告诉 rollout 端：你在这个同步域里从哪个 rank 开始算；这个域里总共有多少参与者；你是哪个 `group_name` 的一部分。

所以当 PP stage 数量变多时，不是"一个超大 NCCL world 扛所有 stage"，而是**每个 stage 自己有自己的同步组**。

##### 这和 SGLang 原路有什么区别？

SGLang 原来的思路更直接：

```text
训练侧 source rank
   -> 直接 init_weights_update_group
   -> 直接 dist.broadcast / all_gather
   -> SGLang engine ranks 接收
```

它没有再额外套一层训练侧 helper 子进程。

- **SGLang**：训练进程直接拿 `torch.distributed` 的同步组。
- **bridge 版 vLLM**：训练进程先把 raw NCCL 放到 helper 子进程里，再由 helper 去跟 rollout 端同步。

##### 两种方式各自的利弊

###### 直连 SGLang 式

- 优点：路径短，概念少，调试时少看一层进程；
- 缺点：训练进程要同时承担训练通信和权重同步通信，耦合更高。

###### helper / bridge 式

- 优点：raw NCCL 从训练主进程里隔离出去，出问题时边界更清楚；
- 缺点：多一个子进程，多一层状态管理，整体更复杂。

##### 最实用的心智模型

```text
PP stage 是逻辑分片
source rank 是发起点
helper 是传输代理
NCCL world 是这次同步的通信域
```

把这四个概念分开后，就不会把"PP=4"误解成"随便开 4 个子进程 / 4 个 group"。真正的判断标准永远是：谁在发这组权重；这组权重要发给谁；这组权重属于哪个同步域；这组同步是不是独立于别的 PP stage。如果这些答案都独立，那么它们就应该有自己的 helper 和自己的 world。

#### 参考

##### Samit 的 NCCL 分离传输

Samit 这边的"分离"做得更重一点，主要体现在两层：

- `connect_rollout_engines_from_distributed()` 里，训练侧 rank0 先给每个 rollout engine 建好一条统一的 NCCL 通道，但真正的 raw NCCL 不放在训练主进程里，而是交给 `_NcclBridge` 子进程（见 `samithuang-slime-dev_vllm/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py:143`）。
- `update_weights_from_distributed()` 先通过 Ray 发 metadata，再让 bridge 通过 NCCL 发 tensor 本体。也就是说，**控制面 / 数据面是显式分开的**。

packed 还会进一步走 `NCCLTrainerSendWeightsArgs(packed=True)`，普通分支则逐 tensor broadcast，专家参数又是另一条 EP all-gather + flush 路径。Samit 的 NCCL 实现不是"一个简单 broadcast"，而是"控制面 + raw NCCL bridge + packed / non-packed / expert 三路分流"。

##### SkyRL 的 NCCL 分离传输

SkyRL 也有"metadata 先行、tensor 后发"的分离，但实现更轻：

- `BroadcastTransferStrategy.create_sender()` 在 rank0 上只建一条 model update communicator，不额外为 TP / EP / PP 各起一套 world group。
- `BroadcastWeightTransferSender._send_chunks_vllm_native()` 先把 metadata 交给 vLLM 的 `update_named_weights()`，再把 tensor 交给 `NCCLWeightTransferEngine.trainer_send_weights()`。
- 它没有像 Samit 那样把 raw NCCL 再剥一层 bridge 子进程；真正被拆成 start / chunk / finish 的，是 IPC 那条会话型路径。

更具体地说，SkyRL 在训练 rank0 里直接调用 vLLM 的 `NCCLWeightTransferEngine.trainer_init()`，这个函数内部会创建 `StatelessProcessGroup`，然后返回 `PyNcclCommunicator`。SkyRL 的选择是相信 vLLM 的 stateless process group 足够把这条 weight-transfer communicator 和训练侧已有的 `torch.distributed` 状态隔开；Samit 的选择则更保守，把 `PyNcclCommunicator` 放进 `_NcclBridge` 子进程里，训练主进程只通过 Pipe 发命令。

两边的 NCCL 差别可以直接记成一句话：

- **Samit**：训练侧总控 + raw NCCL bridge + packed / expert 分流。
- **SkyRL**：策略层更薄，直接用 vLLM native weight transfer / NCCL engine，不再额外包一层 bridge。

##### Samit 的 update_weight_from_distributed.py 不只是传输后端

它更像"训练侧总控台"：

- `update_weights()` 先让训练侧 rank0 去 `pause_generation()`、`flush_cache()`，再开始真正的参数同步。
- 普通参数先走 `all_gather_param()`，再 `convert_to_hf()`；expert 参数则单独走 EP all-gather，再单独 flush。
- 最终都是由 `update_weights_from_distributed()` 统一把 metadata 先交给 rollout，再通过 NCCL / bridge 把 tensor 本体送过去。

"packed / bucket"更像一次训练 step 内的 full update 流程，不是 IPC 那种跨多个 chunk 维持同一段 reload 会话的协议。这也正好解释了为什么 SkyRL 里 **NCCL 不需要 start/finish**，而 **IPC chunked 路径需要**：前者是一轮更新里的整批发送，后者才是真正的多 chunk 会话。

#### 结论

| 主题 | 结论 |
| --- | --- |
| HTTP bridge | **只能当兼容层**，不应该成为高频权重同步的主方案。 |

#### 待讨论

- 分离路径是否允许 helper / bridge / bucket transfer？
- 共卡和分离两条路径，具体走 NCCL、IPC、helper process 还是 HTTP bridge？（待和训练侧 / 推理侧一起确认）

---

## 3. 并行拓扑

并行维度判断表（适用于所有 §3.x）：

| 维度 | 它控制什么 | 它不控制什么 |
| --- | --- | --- |
| `TP` | 单个 engine 里模型怎么切 | 不决定 router 怎么分发 |
| `DP` | 同一个模型副本有多少组 | 不决定单个 engine 内部怎么切 |
| `PP` | 模型层怎么跨 stage 切分 | 不等于多 router |
| `EP` | MoE experts 怎么布置 | 不等于数据面协议 |
| `PD separation` | prefill / decode 是否拆开 | 不等于 TP，也不等于普通 DP |

总原则：

- `TP` 是 engine 内部问题；
- `DP` 是部署拓扑和请求分发问题；
- `PD separation` 是角色拆分问题；
- `colocate` / `separated` 是权重传输和资源共址问题（见 §2）。

如果有人问"`num_gpus_per_engine = 4` 时，到底哪个并行配置自动变成 4"，正确问法是：

1. 这个 4 是 engine 的物理 GPU 数，还是逻辑副本数？
2. 这 4 是给 `TP`、`DP`、`EP` 里的哪一个？
3. 如果还有 `PP` 或 `PD separation`，这 4 是落在哪个 stage 上？
4. router 注册时，register 的是整个 engine 还是内部某个 rank？

---

### 3.1 TP（engine 内部并行）

#### 背景知识

`TP > 1`：

- 单个 engine 内部模型切分；
- 主要影响模型加载和前向执行；
- 不决定 router 怎么分发。

#### 待讨论

- `TP > 1` 时 vLLM engine 内部怎么组织？

---

### 3.2 DP > 1（副本与路由）

#### 背景知识

`DP > 1`：

- 多个推理副本或多组 worker 共同承担请求；
- 需要 router / worker 注册机制；
- 需要明确权重更新是对所有副本同步，还是对某个 group 同步。

最容易误判的就是 `DP > 1`，它有可能意味着：

- 多个独立 engine 副本；
- 多个 router 后端 worker；
- 每个 worker 还可能有自己的 TP / EP / PP 结构。

所以"dp 等于几"不是一个孤立数值问题，而是要回到拓扑图里看它对应的是哪一层复制。

#### 结论

| 主题 | 暂定 | 仍需确认 |
| --- | --- | --- |
| dp > 1 | 必须支持，但拓扑需要再落细 | 是"一个 router 对多个 engine group"，还是"一个 engine group 内部有内部 dp"。 |

#### 待讨论

- `DP > 1` 时 router 怎么知道多个副本？
- dp 维度如何映射到 engine group / router / worker 注册？

---

### 3.3 PP / PD separation

#### 背景知识

`PP / PD separation`：

- 更复杂的多阶段拓扑；
- 是否支持、怎么支持，需要独立确认；
- 不能和普通 DP 混成一个概念。

#### 结论

| 主题 | 暂定 | 仍需确认 |
| --- | --- | --- |
| PD separation | 可支持，但不作为第一阶段默认前提 | vLLM router / engine 侧是否需要显式区分 prefill/decode 角色。 |

#### 待讨论

- `PP / PD separation` 是否真的进入第一阶段范围？
- vLLM 侧是否真的要支持 prefill/decode 分离？

---

## 4. engine 启动与运行时

### 4.1 启动时序与生命周期

#### 背景知识

默认方向：

1. 外层由 RayActor 管理生命周期。
2. 每个 engine actor 内部拉起一个本地 vLLM server subprocess。
3. 训练侧通过 HTTP / RPC / 本地代理去驱动这个 server。
4. 权重同步只是在这个生命周期里插入的一环，不是整个 engine 的全部。

engine 启动不是纯 HTTP server 的事情，而是"actor + subprocess + control plane + data plane"四层一起看。

##### 明确时序

```text
1. rollout manager 决定要启动几个 engine group / router group
2. 每个 engine group 对应一个 RayActor
3. RayActor 在本地选端口，起 vLLM subprocess
4. subprocess 提供 `/v1/completions` 之类的数据面接口
5. RayActor 再对外暴露控制面方法：health / pause / resume / sleep / wake_up / update_weights
6. 如果有 router，就把当前可服务地址注册到 router
7. 如果有权重同步组，就初始化一次通信域，然后反复复用
```

要特别分清两层地址：

- **engine 地址**：真正处理请求的 vLLM HTTP server 地址；
- **router 地址**：对外暴露的分发入口。

这两层地址可以相同，但语义不能混。

#### 备注

`verl` 的代码提醒一个很重要的细节：在多节点环境里，真正负责对外注册和资源协调的，通常是 `node_rank=0` 或者类似的主节点角色，而不是每个 rank 都去抢注册。这一点在 slime 里要特别明确，否则多 model / 多 router 时会非常乱。

#### 结论

| 主题 | 已确认 |
| --- | --- |
| engine 形态 | **RayActor + 本地 vLLM server subprocess**，不把推理逻辑直接塞进训练进程。 |

#### 待讨论

- vLLM engine 启动策略：`native`、`reload`、`sleep/wake`、`pause/resume` 哪些是必备？
- 单个 engine 的注册粒度是"整个 engine"还是"每个 rank"？

---

### 4.2 运行时控制语义

#### 背景知识

控制面操作的语义（endpoint 清单见 §1.3）：

- `pause_generation` / `continue_generation`
- `flush_cache` / `reset_prefix_cache`
- `sleep` / `wake_up`（带 tag 语义）
- `health` / `health_generate`
- `start_profile` / `stop_profile`
- `init_weight_transfer_engine` / `update_weights`
- `abort_request`

#### 结论

| 主题 | 暂定 | 仍需确认 |
| --- | --- | --- |
| 权重热更新 | 优先 native / reload 双路并存 | 哪些部署形态必须支持 native，哪些只需要 reload。 |

#### 待讨论

- `pause_generation` 和 `continue_generation` 是否只影响 server，不影响 router？
- `flush_cache` 只清 KV cache，还是还要清 prefix cache / sleep state？
- `sleep/wake_up` 的 tag 语义是否保留？
- `health_generate` 是单纯 health check，还是必须做一次真实最小生成？

---

## 5. 参数层 (arguments.py)

### 5.1 vLLM server args 注入策略

#### 背景知识

**Calvin 新增的两个最容易混的开关：**

| 参数 | 它控制什么 | 直观理解 |
| --- | --- | --- |
| `--vllm-weight-sync-mode {auto,reload,native}` | vLLM 权重同步的主策略。 | "这一轮训练更新完以后，vLLM server 到底怎么拿到新权重？" |
| `--vllm-try-native-weight-update` | 在不是强制 `native` 的情况下，也先尝试 vLLM 原生权重更新接口。失败后可以回退。 | "能不能先试试更高级的原生热更新？不行就别硬撑。" |

Calvin 分支里这两个参数的调用链：

```text
arguments.py
  --vllm-weight-sync-mode
  --vllm-try-native-weight-update
      ↓
VLLMEngine.__init__()
  self._sync_mode = args.vllm_weight_sync_mode
  self._try_native_update = args.vllm_try_native_weight_update
      ↓
VLLMEngine.init_weights_update_group()
  if _try_native_update or _sync_mode == "native":
    POST /init_weight_transfer_engine
      ↓
VLLMEngine.update_weights_from_distributed()
  if native init succeeded:
    POST /update_weights
  else:
    mark _pending_reload_version
      ↓
VLLMEngine.continue_generation()
  if _pending_reload_version is not None:
    restart local vLLM server
```

三种 `--vllm-weight-sync-mode` 的语义：

| mode | 行为 | 失败时 |
| --- | --- | --- |
| `reload` | 默认兼容路径。训练侧更新后不真正热更新 vLLM 内存里的权重，而是标记版本，等 `continue_generation()` 时重启 vLLM server。 | 不涉及 native 更新失败。代价是慢，但简单。 |
| `auto` | 代码注释里说当前为了 vLLM 0.15.1 兼容，基本等同 `reload`。 | 一般走 reload。 |
| `native` | 强制使用 vLLM 原生权重传输接口：`/init_weight_transfer_engine` + `/update_weights`。 | 失败就报错，不允许回退 reload。 |

`--vllm-try-native-weight-update` 的语义：

```text
mode=reload + try_native=false
  → 直接走 reload。

mode=reload + try_native=true
  → 先尝试 native。
  → 如果 native 初始化或更新失败，回退 reload。

mode=native
  → 强制 native。
  → 不管 try_native 是否设置，失败都报错。
```

这两个参数是"权重同步策略"，不是 rollout 采样策略。它们决定的是训练 step 后 rollout engine 的权重如何变新。

**Samit 的 `--vllm-weight-sync-packed`：**

- `True`：偏向"一次性打包传输"的权重同步路径，适合 non-colocate / distributed，减少逐 bucket 往返。
- `False`：偏向 per-bucket / per-parameter 的同步路径，兼容性更强，但同步更碎。

只影响"训练权重如何同步到 vLLM"，不影响 rollout 采样。

#### 备注

**Samit 分支处理 sglang 依赖的方式（在 `parse_args` 中）：**

- lazy import `RouterArgs`
- `_pre_parse_mode()` 提前解析 `--rollout-backend`
- vLLM backend 下跳过 `sglang_parse_args()`
- vLLM backend 下跳过 `sglang_validate_args()`
- 手动补 `sglang_dp_size` / `sglang_pp_size` / `sglang_ep_size` / `sglang_tp_size`

其中 `elif getattr(args, "rollout_backend", "sglang") == "vllm"` 这个分支不是"启动了 sglang 的另一种模式"，而是：

1. 不跑 `sglang_validate_args()`，因为这条链路里没有 sglang server。
2. 给旧代码补 `sglang_*` 别名，避免上层逻辑到处加 backend 判断。
3. 让后续代码继续按"有 dp / pp / ep / tp 这些字段"的习惯工作。

**为什么这里没有像 SGLang 那样把所有 server args 都批量注入？**

- SGLang 之所以能批量注入，是因为它有 `ServerArgs.add_cli_args(parser)` 这类统一入口，slime 只要包一层 `parser.add_argument` 就能自动前缀化。
- vLLM 这条线没有把完整的 vLLM CLI surface 接进 slime，也没有做一层等价的 `add_cli_args` 包装器。
- 所以这里的 vLLM 参数只保留 slime 真正要控制的少数开关，比如 memory util、eager、sleep mode、weight sync。
- 本质上这是一个"最小适配层"，不是"把 vLLM 的全部 serverargs 复刻进 slime"。

#### 待讨论

- 哪些参数第一阶段要保留？
- 目标是"全量注入 vLLM server args"还是"少数开关"？
- 哪些参数应该走"顶层统一注入"，哪些可以只走 engine 内部默认值？
- 权重同步参数等读完 weight sync 后再定。

---

### 5.2 rollout function 路由

#### 背景知识

Calvin 在 `slime_validate_args()` 里：

- `rollout_backend == "vllm"` 时切换 `rollout_function_path`
- `eval_function_path` 也切到 `slime.rollout.vllm_rollout.generate_rollout`
- 禁止 `prefill_num_servers`

#### 结论

| 主题 | 暂定 | 仍需确认 |
| --- | --- | --- |
| rollout 函数入口 | 倾向于直接切到 vLLM rollout 入口 | 是否彻底替换 `sglang_rollout.py`，还是保留一个很薄的兼容壳。 |

#### 待讨论

- `rollout_function_path` 是否真的应该切到 `vllm_rollout`？
- 如果采用 sidecar 复用 `sglang_rollout.py`，这里可能不应该照搬。
- 决定 vLLM backend 是否切换 rollout function：如果采用直连 vLLM `/v1/completions`，就应该切到单独 `vllm_rollout.py`。如果某个历史兼容分支仍要保留 `/generate`，那也只能是临时兼容，不应作为最终主路径。

---

### 5.3 backend-agnostic 参数命名

#### 背景知识

Samit 新增了 `--rollout-server-concurrency`（替代 `--sglang-server-concurrency`），把 router 与 HTTP client 的并发参数从 `sglang_*` 改名成 backend 无关的形态。

#### 待讨论

- `rollout-server-concurrency` 是否应该替换 `sglang-server-concurrency`？
- 如果要做 backend-agnostic 参数，可以加。如果第一阶段少改主线，可以继续复用 `--sglang-server-concurrency`。

---

### 5.4 validate 规则与禁用项

#### 待讨论

- vLLM 模式是否真的要跳过全部 sglang 参数？
- 如果仍然使用 `sglang_router`，哪些 sglang 参数其实还需要保留？
- 决定 vLLM backend 是否允许 PD / EPD：第一阶段推荐不允许。遇到 `prefill_num_servers` 或 `sglang_config` 中 `prefill/decode/encoder`，直接报错更清楚。

---

### 5.5 最小 patch 方案

#### 待讨论

输出 `arguments.py` 的最小 patch 方案：

- 新增参数列表
- validate 规则
- 不引入 FSDP
- 不重写 rollout function，除非后面放弃 sidecar 路线
