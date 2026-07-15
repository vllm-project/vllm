# 执行层深度解析：Executor、Worker 与 ModelRunner

## 概述

执行层是 vLLM 中负责"真正干活"的部分——把调度器的决策转化为 GPU 上的实际计算。它由三个核心角色组成：

```
EngineCore
  └── Executor（进程编排）
        └── Worker（设备管理）
              └── ModelRunner（模型执行）
```

简单类比：
- **Executor** 是"工厂厂长"——决定有几个车间、如何分派任务、如何收集结果
- **Worker** 是"车间主任"——管理一块 GPU 的初始化、内存、模型加载
- **ModelRunner** 是"技术工人"——构造输入张量、调用模型 forward、采样输出 token

---

## 1. 三者的职责边界

| 组件 | 职责 | 关键文件 |
|------|------|----------|
| Executor | 管理 worker 进程的生命周期；通过 RPC 将方法调用广播给所有 worker；支持单机/多机/Ray 等部署方式 | `vllm/v1/executor/abstract.py` |
| Worker | 持有一块 GPU 的设备状态；初始化分布式环境；加载模型；处理 Pipeline Parallel 的中间张量通信；管理显存（sleep/wake_up）| `vllm/v1/worker/gpu_worker.py` |
| ModelRunner | 维护输入 batch 状态；将 SchedulerOutput 转为模型输入张量；调用模型 forward；执行采样得到 next token | `vllm/v1/worker/gpu_model_runner.py` |

---

## 2. Executor：进程编排层

### 2.1 抽象接口

```python
# vllm/v1/executor/abstract.py
class Executor(ABC):
    """管理 worker 进程，提供统一的 RPC 调用接口"""

    def execute_model(self, scheduler_output, non_block=False):
        # 通过 collective_rpc 调用所有 worker 的 execute_model 方法
        output = self.collective_rpc("execute_model", args=(scheduler_output,), non_block=non_block)
        return output[0]  # 只返回 driver worker 的结果

    @abstractmethod
    def collective_rpc(self, method, args=(), kwargs=None, non_block=False):
        """向所有 worker 广播一个方法调用"""
        raise NotImplementedError
```

核心设计：Executor 本身不做任何计算。它只负责把方法名和参数"广播"给底下的所有 Worker，然后收集返回值。`collective_rpc` 是所有 Executor 子类必须实现的核心方法。

### 2.2 具体实现

vLLM 提供了多种 Executor 实现，适配不同部署场景：

| Executor | 场景 | 进程模型 |
|----------|------|----------|
| `UniProcExecutor` | 单 GPU（TP=1, PP=1） | worker 就在当前进程内，直接调用 |
| `MultiprocExecutor` | 多 GPU 单机（TP>1 或 PP>1） | 每个 worker 一个独立进程，通过共享内存消息队列通信 |
| `RayDistributedExecutor` | 多机分布式 | 使用 Ray Actor 管理远程 worker |

### 2.3 UniProcExecutor 的简洁实现

单 GPU 场景最简单，可以帮我们理解核心调用链：

```python
# vllm/v1/executor/uniproc_executor.py
class UniProcExecutor(Executor):
    def _init_executor(self):
        # 创建 WorkerWrapper，内部实例化真正的 Worker
        self.driver_worker = WorkerWrapperBase(rpc_rank=0)
        self.driver_worker.init_worker(all_kwargs=[kwargs])
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def collective_rpc(self, method, args=(), kwargs=None, ...):
        # 单进程，直接在当前进程内调用 worker 的方法
        target = self.driver_worker
        func = getattr(target, method)
        output = run_method(func, args=args, kwargs=kwargs)
        return [output]  # 包装成列表保持接口一致
```

### 2.4 MultiprocExecutor 的消息队列模型

多 GPU 场景下，每个 Worker 运行在独立进程中：

```
                   ┌─────────────┐
                   │  Executor   │
                   │ (主进程)     │
                   └──────┬──────┘
                          │ MessageQueue (共享内存)
               ┌──────────┼──────────┐
               ▼          ▼          ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │WorkerProc│ │WorkerProc│ │WorkerProc│
        │  rank 0  │ │  rank 1  │ │  rank 2  │
        └──────────┘ └──────────┘ └──────────┘
```

WorkerProc 的主循环持续从消息队列中取出方法调用并执行：

```python
# vllm/v1/executor/multiproc_executor.py
class WorkerProc:
    def worker_busy_loop(self):
        while True:
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue()
            func = getattr(self.worker, method)  # self.worker 是 WorkerWrapperBase
            output = func(*args, **kwargs)
            self.handle_output(output)
```

---

## 3. Worker：设备管理层

### 3.1 WorkerBase 与 WorkerWrapperBase

有两个容易混淆的类：

- **`WorkerBase`**（`worker_base.py`）：Worker 的抽象接口，定义了 `init_device`、`load_model`、`execute_model` 等方法
- **`WorkerWrapperBase`**（`worker_base.py`）：Worker 的"壳"，负责延迟初始化真正的 Worker 实例（根据配置选择 `gpu_worker.Worker`、`cpu_worker.Worker` 等）

```python
# worker_base.py
class WorkerWrapperBase:
    def init_worker(self, all_kwargs):
        # 根据平台选择 worker 类（GPU/CPU/XPU/TPU）
        worker_module = resolve_obj_by_qualname(parallel_config.worker_cls)
        self.worker = worker_module(**kwargs)  # 实例化真正的 Worker
```

### 3.2 GPU Worker 的核心职责

```python
# vllm/v1/worker/gpu_worker.py
class Worker(WorkerBase):
    def init_device(self):
        """初始化 GPU 设备和分布式环境"""
        # 1. 设置 CUDA 设备
        torch.accelerator.set_device_index(self.device)
        # 2. 初始化 NCCL 分布式通信
        init_worker_distributed_environment(...)
        # 3. 创建 ModelRunner
        self.model_runner = GPUModelRunner(self.vllm_config, self.device)

    def load_model(self):
        """加载模型权重到 GPU"""
        self.model_runner.load_model()

    def execute_model(self, scheduler_output):
        """执行一步推理"""
        # 处理 Pipeline Parallel 的中间张量接收
        if not get_pp_group().is_first_rank:
            intermediate_tensors = self._recv_from_prev_stage()

        # 委托给 ModelRunner 执行
        output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)

        # 如果不是最后一个 PP stage，发送中间结果给下一个 stage
        if isinstance(output, IntermediateTensors):
            self._send_to_next_stage(output)
            return None

        return output
```

Worker 的核心价值：
1. **设备初始化**：设置 CUDA 设备、初始化分布式通信（NCCL）
2. **Pipeline Parallel 通信**：处理 PP 各 stage 之间的中间张量传输
3. **显存管理**：`sleep`/`wake_up` 机制，支持显存动态释放和恢复
4. **生命周期管理**：加载模型、性能分析、健康检查

### 3.3 Worker 是 ModelRunner 的"管家"

Worker 自己几乎不做计算，大部分方法都是简单委托给 ModelRunner：

```python
def load_model(self):
    self.model_runner.load_model()

def determine_available_memory(self):
    self.model_runner.profile_run()  # 让 ModelRunner 做 profiling
    # ... 计算可用显存

def compile_or_warm_up_model(self):
    self.model_runner.compile_or_warm_up_model()
```

---

## 4. ModelRunner：模型执行层

ModelRunner 是真正的"计算引擎"，负责把调度器的抽象决策转化为具体的 GPU 计算。

### 4.1 核心职责

```python
# vllm/v1/worker/gpu_model_runner.py
class GPUModelRunner:
    def __init__(self, vllm_config, device):
        self.input_batch = ...    # 持久化的输入 batch 状态
        self.model = None         # 模型实例（load_model 后赋值）
        self.kv_caches = []       # KV cache 张量

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        """一步推理的完整流程"""
        # 1. 更新 batch 状态（添加/移除请求）
        self._update_states(scheduler_output)

        # 2. 准备输入张量（token_ids, positions, block_table 等）
        #    这一步根据 scheduler_output 构建模型需要的所有输入

        # 3. 调用模型 forward
        hidden_states = self.model(
            input_ids=...,
            positions=...,
            kv_caches=self.kv_caches,
            attn_metadata=...,
        )

        # 4. 采样 next token
        logits = self.model.compute_logits(hidden_states)
        output = self.sampler(logits, sampling_metadata)

        return ModelRunnerOutput(...)
```

### 4.2 关键设计：持久化 InputBatch

ModelRunner 维护一个持久化的 `InputBatch`，不会每步重新构建：

```
SchedulerOutput 告诉 ModelRunner：
  - 哪些新请求加入（prefill）
  - 哪些请求继续（decode）
  - 哪些请求结束或被抢占

ModelRunner 据此增量更新 InputBatch，避免重复构建张量
```

### 4.3 CUDA Graph 和编译优化

ModelRunner 负责 CUDA Graph 捕获和 torch.compile 优化：

```python
def compile_or_warm_up_model(self):
    """预热/编译模型，捕获 CUDA Graph"""
    # 对常见 batch size 预先捕获 CUDA Graph
    # 运行时直接 replay，避免 kernel launch overhead
```

---

## 5. 完整调用链

一次 `step()` 的完整调用链：

```
EngineCore.step()
│
├── scheduler.schedule()  → SchedulerOutput
│
├── executor.execute_model(scheduler_output)
│   │
│   ├── collective_rpc("execute_model", args=(scheduler_output,))
│   │   │
│   │   │  [UniProc: 直接调用]
│   │   │  [Multiproc: 通过 MessageQueue 发送到 worker 进程]
│   │   │
│   │   └── worker.execute_model(scheduler_output)
│   │       │
│   │       ├── [PP 非首 rank] recv intermediate_tensors
│   │       │
│   │       ├── model_runner.execute_model(scheduler_output, intermediate_tensors)
│   │       │   │
│   │       │   ├── _update_states()      ← 更新 batch 状态
│   │       │   ├── 构建输入张量           ← token_ids, positions, attn_metadata
│   │       │   ├── model.forward()        ← GPU 计算
│   │       │   └── sampler()             ← 采样 next token
│   │       │
│   │       └── [PP 非末 rank] send intermediate_tensors to next stage
│   │
│   └── return output[0]  ← driver worker 的结果
│
└── scheduler.update_from_output(model_output)  → 更新请求状态
```

---

## 6. 为什么要分三层？

### 6.1 关注点分离

- **Executor 不关心硬件**：它只知道"我有 N 个 worker，我要广播 RPC"
- **Worker 不关心调度**：它只知道"给我 SchedulerOutput，我返回 ModelRunnerOutput"
- **ModelRunner 不关心通信**：它只知道"给我输入张量，我跑 forward 返回结果"

### 6.2 支持多种部署拓扑

同一套 Worker + ModelRunner 代码，搭配不同 Executor 即可运行在：
- 单 GPU（UniProcExecutor）
- 单机多 GPU（MultiprocExecutor）
- 多机分布式（RayDistributedExecutor）

### 6.3 Tensor Parallel vs Pipeline Parallel 的处理分工

| 并行方式 | 谁负责 | 机制 |
|----------|--------|------|
| Tensor Parallel | ModelRunner 内部透明处理 | NCCL all-reduce 在模型层自动插入 |
| Pipeline Parallel | Worker 层处理 | Worker 在 execute_model 前后做 send/recv |
| Data Parallel | Executor/EngineCore 层处理 | 多个 EngineCore 实例各自独立 |

---

## 7. 关键数据流

```
SchedulerOutput (调度器 → 执行层)
├── num_scheduled_tokens: dict[req_id → token_count]
├── scheduled_new_reqs: 新请求列表
├── scheduled_cached_reqs: 继续执行的请求
├── preempted_req_ids: 被抢占的请求
└── kv_connector_metadata: KV 传输元数据

ModelRunnerOutput (执行层 → 调度器)
├── sampled_token_ids: 每个请求采样到的 token
├── logprobs: token 的对数概率（可选）
└── spec_decode_tokens: 投机解码的草稿 token（可选）
```

---

## 8. 常见疑问

### Q: ModelRunner 和 Worker 为什么不合并？

因为 Pipeline Parallel。当 PP > 1 时，非首/末 rank 的 Worker 需要在 ModelRunner 执行前后做张量通信。Worker 是这个通信逻辑的承载者。如果只有 TP，Worker 确实只是一个薄薄的壳。

### Q: Executor 的 `collective_rpc` 和 NCCL 通信有什么区别？

- `collective_rpc` 是**控制面**通信：Executor 告诉 Worker "执行 execute_model 方法"
- NCCL 是**数据面**通信：Worker 之间交换张量数据（all-reduce, send/recv）

两者是正交的。`collective_rpc` 确保所有 Worker 同步执行相同的操作，NCCL 则在操作内部完成张量同步。

### Q: WorkerWrapperBase 为什么存在？

延迟初始化。Worker 的具体类型（GPU/CPU/XPU/TPU）由配置决定，需要在正确的进程和设备上下文中才能实例化。WorkerWrapperBase 先占位，等 `init_worker` 被调用时才真正创建 Worker。

---

## 9. 后续深入方向

- ModelRunner 如何管理 CUDA Graph（捕获/重放的策略）
- 多模态模型的 Encoder 如何在 ModelRunner 中执行
- 投机解码（Speculative Decoding）如何与 ModelRunner 协作
- Data Parallel 下多个 EngineCore + Executor 的协调
- `sleep`/`wake_up` 显存管理的细节
