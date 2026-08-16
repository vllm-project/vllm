# 第2章：核心数据结构

> 一句话：SamplingParams、Request、Outputs 等数据类型定义了请求从进入到产出结果的完整生命周期，是理解引擎调度和输出处理的基础。

## 涉及文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/sampling_params.py` | ~1149 | SamplingParams：温度、top_p、repetition_penalty 等采样参数 |
| `vllm/v1/request.py` | ~393 | V1 Request 对象：请求在 Engine Core 中的内部表示 |
| `vllm/outputs.py` | ~362 | RequestOutput / CompletionOutput：返回给用户的输出结构 |
| `vllm/inputs/engine.py` | ~387 | 引擎侧的输入预处理：token 化、多模态数据绑定 |
| `vllm/tasks.py` | ~43 | 任务类型定义：generate / embed / classify 等 |

## 关键问题（带着这些问题读）

1. 一个请求从 API 进入到 Engine Core 内部，经历了哪些数据结构的转换？（API protocol → inputs → Request → Outputs）
2. SamplingParams 中哪些参数会影响调度决策（而非仅影响采样层）？
3. V1 Request 与旧版 Sequence / SequenceGroup 有什么本质区别？为什么 V1 简化了这一层？
4. RequestOutput 如何承载流式输出（streaming）的增量数据？

## 调用链概览

```
用户请求 (JSON)
  → entrypoints: 解析为 SamplingParams + prompt
    → inputs/: tokenize → EngineInputs
      → v1/engine/input_processor.py: 转换为 EngineCoreRequest
        → v1/request.py: 创建内部 Request 对象
          → ... 调度、执行 ...
        → v1/engine/output_processor.py: 组装 RequestOutput
      → 返回给用户
```

## 详细笔记

> 阅读会话记录：
> - 2026-08-15 session: `first` — 已核实"OpenAI Completion 请求的数据结构生命周期"两条调用链（入站链 + 出站链）并产出三行格式

### 调用链：OpenAI /v1/completions 请求 —— 从 JSON 到 RequestOutput 的数据结构变换

本案例追踪一条非流式 Completion 请求的完整生命周期。入站链从 `OpenAIServingCompletion._create_completion` 开始，经 `CompletionRequest.to_sampling_params()` 构造 `SamplingParams`，再经 `AsyncLLM.generate()` → `add_request()` → `InputProcessor.process_inputs()` 构造 `EngineCoreRequest`，最终交给 `OutputProcessor.add_request()` 创建前端侧的 `RequestState`。此后请求被序列化发送到 EngineCore 进程（跨进程边界），在那里由 `Request.from_engine_core_request()` 构造调度侧的 `Request` 对象。出站链由后台异步任务 `output_handler` 驱动，从 EngineCore 拉取 `EngineCoreOutput`，经 `OutputProcessor.process_outputs()` 做 detokenize 和 logprobs 计算，最终组装为 `CompletionOutput` + `RequestOutput` 推入队列。两条链之间隔着异步等待（EngineCore 推理完成后才有输出），无法用函数调用串成一条。

#### 源码调用栈（入站链：构造请求数据结构）

```
入口：serving.py → _create_completion() →（非 beam_search 分支）

_create_completion()                                      serving.py:131
 ├→ request.to_sampling_params(max_tokens, ...)           serving.py:178
 │    └→ CompletionRequest.to_sampling_params              protocol.py:307（定义）
 │         └→ SamplingParams.from_optional(...)            protocol.py:363
 │              └→ SamplingParams(...)                     sampling_params.py:421  ← 构造 + __post_init__
 │                   └→ __post_init__()                   sampling_params.py:457
 │                        └→ _verify_args()               sampling_params.py:497
 │
 ├→ self.engine_client.generate(engine_input, sampling_params, ...)   serving.py:209
 │    └→ AsyncLLM.generate                                async_llm.py:550（定义）
 │         │  :586 处调用 ↓
 │         └→ self.add_request(request_id, prompt, sampling_params, ...)  async_llm.py:586
 │              └→ AsyncLLM.add_request                   async_llm.py:283（定义）
 │                   │  :356 处调用 ↓（已渲染 EngineInput 分支）
 │                   ├→ self.input_processor.process_inputs(...)           async_llm.py:356
 │                   │    └→ InputProcessor.process_inputs                input_processor.py:244（定义）
 │                   │         ├→ _validate_params(params, ...)           input_processor.py:259
 │                   │         │    └→ params.verify(...)                 input_processor.py:104
 │                   │         ├→ split_enc_dec_input(processed_inputs)   input_processor.py:301
 │                   │         ├→ params.clone()                          input_processor.py:318
 │                   │         ├→ sampling_params.update_from_generation_config(...)  input_processor.py:326
 │                   │         └→ EngineCoreRequest(...)                  input_processor.py:373  ← 构造
 │                   │
 │                   ├→ self.input_processor.assign_request_id(request)   async_llm.py:392
 │                   │
 │                   └→ self._add_request(request, prompt_text, None, 0, queue)  async_llm.py:406
 │                        └→ AsyncLLM._add_request                       async_llm.py:424（定义）
 │                             ├→ self.output_processor.add_request(...)  async_llm.py:433
 │                             │    └→ OutputProcessor.add_request        output_processor.py:525（定义）
 │                             │         └→ RequestState.from_new_request(...)  output_processor.py:539
 │                             │              └→ RequestState(...)        output_processor.py:253  ← 构造
 │                             │
 │                             └→ engine_core.add_request_async(request)  async_llm.py:436
 │                                  ╌╌ 跨进程边界：序列化 EngineCoreRequest 发往 EngineCore ╌╌
```

```
EngineCore 进程侧（接收 EngineCoreRequest 后）：

preprocess_add_request(request)                          core.py:965（定义）
 └→ Request.from_engine_core_request(request, ...)       core.py:979
      └→ Request.__init__(...)                           request.py:60（定义）  ← 调度侧 Request 构造
```

#### 源码调用栈（出站链：EngineCoreOutput → RequestOutput）

```
╌╌ 异步等待：EngineCore 完成推理后推送 EngineCoreOutputs ╌╌

output_handler()                                         async_llm.py:684（闭包）
 │  :688 拉取输出 ↓
 ├→ engine_core.get_output_async()                       async_llm.py:688
 │  :703 处理输出 ↓
 └→ output_processor.process_outputs(outputs_slice, ...)  async_llm.py:703
      └→ OutputProcessor.process_outputs                  output_processor.py:589（定义）
           │  对每个 engine_core_output 循环（:619）
           ├→ req_state.detokenizer.update(new_token_ids, ...)  output_processor.py:656
           ├→ req_state.logprobs_processor.update_from_output(...)  output_processor.py:665
           └→ req_state.make_request_output(...)           output_processor.py:668
                └→ RequestState.make_request_output        output_processor.py:276（定义）
                     ├→ _new_completion_output(...)        output_processor.py:324
                     │    └→ CompletionOutput(...)         output_processor.py:414  ← 构造
                     └→ _new_request_output(...)           output_processor.py:334
                          └→ RequestOutput(...)            output_processor.py:373  ← 构造
                               → req_state.queue.put(request_output)  output_processor.py:681
```

#### 三行格式（入站链：每步做什么 / 调用链 / 补充）

```
① to_sampling_params：将 OpenAI 格式的 CompletionRequest 字段转换为 SamplingParams
   entrypoints/openai/completion/serving.py:178 to_sampling_params → protocol.py:363 SamplingParams.from_optional → sampling_params.py:421 SamplingParams()
   ── SamplingParams 登场（msgspec.Struct + PydanticMsgspecMixin）；__post_init__(:457) 做参数校验与 greedy 归一化

② generate → add_request：API server 调用 AsyncLLM.generate 发起请求，内部委托 add_request 
   serving.py:209 engine_client.generate → async_llm.py:586 self.add_request
   ── OpenAIServingCompletion的engine_client 在 vllm/entrypoints/openai/api_server.py:778传入 AsyncLLM 类型

③ process_inputs：InputProcessor 校验参数、clone SamplingParams、拆分 encoder/decoder 输入，构造 EngineCoreRequest
   async_llm.py:356 self.input_processor.process_inputs → input_processor.py:259 _validate_params → :301 split_enc_dec_input → :318 params.clone → :373 EngineCoreRequest()
   ── EngineCoreRequest 登场（msgspec.Struct, array_like）；SamplingParams 在此被 clone 并 update_from_generation_config 后送入 EngineCoreRequest()

④ _add_request：将 EngineCoreRequest 注册到 OutputProcessor（前端侧）并发送到 EngineCore（跨进程）
   async_llm.py:406 _add_request → :433 output_processor.add_request → output_processor.py:539 RequestState.from_new_request
   ── RequestState 登场；内含 IncrementalDetokenizer 和 LogprobsProcessor，用于后续输出处理

⑤ from_engine_core_request：EngineCore 进程收到序列化的 EngineCoreRequest，构造调度侧 Request
   async_llm.py:436 engine_core.add_request_async → core_client.py:1148 AsyncMPClient._send_input(ADD, request)
     → :1116 self.encoder.encode(request) msgspec 编码 → :1117 _send_input_message → ZMQ 发送到 EngineCore 进程
     ── EngineCore 进程（core.py:1697 input_processing_thread 的 poller.poll 循环）──
     → core.py:1710 add_request_decoder.decode(data_frames) 反序列化得到 EngineCoreRequest
     → core.py:1712 self.preprocess_add_request(req)
     → core.py:979 Request.from_engine_core_request → request.py:60 Request.__init__
   ── Request 登场（调度侧内部表示）；含 RequestStatus 状态机、StructuredOutputRequest、block_hashes 等调度专用字段
   ╌╌ 跨进程桥梁：前端 AsyncMPClient 通过 ZMQ socket 发送 msgspec 序列化的 EngineCoreRequest，
      EngineCore 进程的 input_processing_thread 在 poll 循环中接收并反序列化 ╌╌
```

#### EngineCoreProc::process_input_sockets 方法调用逻辑

从 `main()` 到 `process_input_sockets` 接收请求的完整调用链：

```
# ── main.py → EngineCoreProc.__init__ → process_input_sockets 启动链 ──
main()                                                        # cli/main.py:17
→ ServeSubcommand.cmd(args)                                   # cli/serve.py:50
  → run_server(args)                                          # api_server.py:751
    → run_server_worker(...)                                   # api_server.py:767
      → build_async_engine_client(args)                        # api_server.py:778→110
        → AsyncEngineArgs.from_cli_args(args)                  # :127
        → build_async_engine_client_from_engine_args(...)      # :132→141
          → AsyncLLM.from_vllm_config(vllm_config, ...)        # :168, async_llm.py:206
            → AsyncLLM.__init__(vllm_config, executor_class)   # async_llm.py:75
              → EngineCoreClient.make_async_mp_client(...)      # :149, core_client.py:116
                → AsyncMPClient.__init__(...)                   # core_client.py:980
                  → super().__init__() → MPClient.__init__      # :990→516
                    → launch_core_engines(vllm_config, ...)     # :609, utils.py:1070
                      → CoreEngineProcManager(...)              # utils.py:1187→120
                        → context.Process(                      # utils.py:165
                            target=EngineCoreProc.run_engine_core)
                        → proc.start()                          # utils.py:204
                          ─── 新进程 ───
                          → EngineCoreProc.run_engine_core()    # core.py:1268
                            → EngineCoreProc.__init__(...)      # core.py:1312→1010
                              → Thread(target=self.process_input_sockets).start  # :1094-1104
                                → process_input_sockets()       # :1639 (daemon thread)
                                  → ZMQ DEALER connect + ready  # :1655-1686
                                  → while True: poller.poll()   # :1696-1697

# ── 跨进程 ZMQ：请求到达后 ──
process_input_sockets poll 收到消息后分发：
  → ADD:     decode → preprocess_add_request → Request.from_engine_core_request  # :1710-1712→979
  → UTILITY: decode → dispatch method call              # :1716-1718
  → ABORT:   decode → 取消请求                           # :1727
```

设计意图（:1088-1092）：独立线程做 ZMQ IO，释放 GIL，与 GPU 模型前向传播重叠执行。

#### AsyncMPClient <---> EngineCoreProc
注册阶段:
- build_async_engine_client
  #vllm/v1/engine/core_client.py:589
  - AsyncMPClient:  绑定addresses.inputs[0]
  - EngineCoreProc: launch_core_engines 传递 addresses，创建 local_engine_manager
    - CoreEngineLaunch 关联 local_engine_manager 和addresses 
    - wait_for_engine_startup
- build_and_serve # POST API 注册

请求阶段:
- AsyncMPClient._send_input(ADD, request)  # core_client.py:1148 
- EngineCoreProc.process_input_sockets poll循环接收到请求

#### 三行格式（出站链：每步做什么 / 调用链 / 补充）

```
⑥ process_outputs：后台 output_handler 拉取 EngineCoreOutputs，逐请求 detokenize + 计算 logprobs
   async_llm.py:397 _run_output_handler -> :703 output_processor.process_outputs → output_processor.py:656 detokenizer.update → :665 logprobs_processor.update_from_output
   ── EngineCoreOutput 登场（msgspec.Struct）；携带 new_token_ids、finish_reason、logprobs 等原始数据

⑦ make_request_output：将处理后的文本和 logprobs 组装为 CompletionOutput + RequestOutput，推入队列
   output_processor.py:668 make_request_output → :324 _new_completion_output → :414 CompletionOutput() → :334 _new_request_output → :373 RequestOutput()
   ── CompletionOutput 和 RequestOutput 登场；RequestOutput 通过 queue.put(:681) 传回 generate() 的 async 循环
```

#### 处理output 与 使用 generate流程
async_llm.py:703 output_processor.process_outputs
```
POST v1/completion -> 
OpenAIServingCompletion._create_completion  # vllm/entrypoints/openai/completion/serving.py:131 
serving.py:209 engine_client.generate 
  - async_llm.py:586 self.add_request     # 
    - request = self.input_processor.process_inputs()
    - self._run_output_handler()          # 处理engine_core_output
      - process_output 
        - req_state.queue.put(request_output) # 启动协程 获取engine_core输出，put进
    - await self._add_request(request, prompt_text, None, 0, queue) # 传递queue 注册赋值
  out = q.get_nowait() or await q.get()   # 取结果
```

### 数据结构转换小结

请求生命周期中出现的核心数据结构及其所在边界：

| 数据结构 | 文件 | 所在边界 | 角色 |
|----------|------|----------|------|
| `CompletionRequest` | `entrypoints/openai/completion/protocol.py` | API server | Pydantic 模型，OpenAI 协议映射 |
| `SamplingParams` | `sampling_params.py` | 全程携带 | 采样参数容器，msgspec.Struct |
| `EngineInput` (TokensInput 等) | `inputs/engine.py` | API server → InputProcessor | TypedDict，已 tokenize 的引擎输入 |
| `EngineCoreRequest` | `v1/engine/__init__.py` | 前端 → EngineCore | msgspec.Struct，跨进程序列化载体 |
| `Request` | `v1/request.py` | EngineCore 内部 | 调度侧内部对象，含状态机和 KV cache 元数据 |
| `EngineCoreOutput` | `v1/engine/__init__.py` | EngineCore → 前端 | msgspec.Struct，推理结果的序列化载体 |
| `RequestState` | `v1/engine/output_processor.py` | 前端 OutputProcessor | 前端侧请求状态，含 detokenizer 和 logprobs |
| `CompletionOutput` | `outputs.py` | 前端 | 单条补全结果（text + token_ids + logprobs） |
| `RequestOutput` | `outputs.py` | 前端 → API server | 最终输出，包含 prompt 信息 + CompletionOutput 列表 |

#### POST /v1/completions 初始化

从 CLI 入口到路由注册 + Handler 实例化的完整调用流：

```
vllm serve <model>
  → ServeSubcommand.cmd(args)                              # cli/serve.py:50
    → uvloop.run(run_server(args))                         # cli/serve.py:152

run_server(args)                                           # api_server.py:751
  → run_server_worker(listen_address, sock, args)          # api_server.py:767
    → build_async_engine_client(args) as engine_client     # api_server.py:778
    → build_and_serve(engine_client, ...)                  # api_server.py:658
        │
        │  ① 路由注册 (build_app)
        ├─ app = build_app(args, supported_tasks, ...)     # api_server.py:679→189
        │    └─ if "generate" in supported_tasks:          # api_server.py:235
        │         register_generate_api_routers(app)       # generate/api_router.py:21
        │           └─ register_completion_api_router(app) # generate/api_router.py:34
        │                └─ app.include_router(router)     # completion/api_router.py:69
        │                     └─ @router.post("/v1/completions")   # completion/api_router.py:34
        │                          绑定 → create_completion()      # completion/api_router.py:46
        │
        │  ② Handler 实例化 (init_app_state)
        └─ init_app_state(engine_client, app.state, args)  # api_server.py:680
             └─ init_generate_state(...)                   # generate/api_router.py:57
                  └─ state.openai_serving_completion =      # generate/api_router.py:181
                       OpenAIServingCompletion(engine_client, ...)
```

请求到达时的衔接：`create_completion()` 通过 `request.app.state.openai_serving_completion`（completion/api_router.py:30-31）取到 Handler，调用 `handler.create_completion(request, raw_request)` 进入入站链。

**关键点**：
- 路由注册（`build_app`）和 Handler 实例化（`init_app_state`）是两条平行线，前者绑定 URL→函数，后者创建实际处理对象挂到 `app.state`

## 反馈给规划Agent

- [ ] `vllm/inputs/engine.py` 主要是 TypedDict 类型别名定义，不含复杂逻辑；骨架中标注行数 ~387 但实际无需深入追踪。章节可考虑将其与 `InputProcessor`（`v1/engine/input_processor.py`）合并讲解，因为 `EngineInput` 的消费者就是 `InputProcessor`。
- [ ] `vllm/tasks.py` 仅 43 行，只包含 `Literal` 类型别名和一个移除检查函数，无调用链可追。建议降级为附录或合并到第1章（配置与启动）的任务类型部分。
- [ ] 建议补充 `vllm/v1/engine/__init__.py` 到涉及文件列表，因为 `EngineCoreRequest`、`EngineCoreOutput`、`FinishReason` 等跨进程数据结构均定义于此，是理解请求生命周期的关键。
- [ ] 出站链涉及 `vllm/v1/engine/output_processor.py`（骨架已提及）和 `vllm/v1/engine/detokenizer.py`、`vllm/v1/engine/logprobs.py`，后两者未列入涉及文件但在调用链中被使用。
