# 模型 Forward 深度拆解：以 Llama 为例

## 概述

本节以 LLaMA 模型为代表，深入拆解 vLLM 中一个模型类的代码结构、Attention 如何调用 PagedAttention kernel、TP 间通信发生在何处、以及 CUDA Graph 如何与模型执行交互。

核心代码位置：`vllm/model_executor/models/llama.py`

---

## 1. 模型类层次结构

```
LlamaForCausalLM                      ← ModelRunner 持有的顶层对象
  ├── model: LlamaModel               ← Transformer backbone
  │     ├── embed_tokens: VocabParallelEmbedding     ← Embedding 查表
  │     ├── layers: [LlamaDecoderLayer × N]          ← N 层 Transformer
  │     │     ├── input_layernorm: RMSNorm
  │     │     ├── self_attn: LlamaAttention
  │     │     │     ├── qkv_proj: QKVParallelLinear        ← Q/K/V 投影
  │     │     │     ├── rotary_emb: RotaryEmbedding        ← RoPE
  │     │     │     ├── attn: Attention                    ← PagedAttention
  │     │     │     └── o_proj: RowParallelLinear          ← 输出投影
  │     │     ├── post_attention_layernorm: RMSNorm
  │     │     └── mlp: LlamaMLP
  │     │           ├── gate_up_proj: MergedColumnParallelLinear
  │     │           ├── act_fn: SiluAndMul
  │     │           └── down_proj: RowParallelLinear
  │     └── norm: RMSNorm                            ← 最终 LayerNorm
  ├── lm_head: ParallelLMHead                        ← hidden → vocab 投影
  └── logits_processor: LogitsProcessor              ← logits 后处理
```

---

## 2. 各组件详解

### 2.1 LlamaForCausalLM — 顶层入口

```python
# vllm/model_executor/models/llama.py
class LlamaForCausalLM:
    def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
        # 直接透传给 LlamaModel
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states):
        # ModelRunner 在 forward 之后单独调用
        return self.logits_processor(self.lm_head, hidden_states)
```

顶层类职责很薄：`forward` 透传，`compute_logits` 负责把最后一层的 hidden_states 映射到词表维度。

### 2.2 LlamaModel — Transformer Backbone

```python
class LlamaModel:
    def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
        # 1. 获取初始 hidden_states
        if is_first_pp_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds       # 多模态模型已预处理
            else:
                hidden_states = self.embed_tokens(input_ids)  # 查 embedding 表
            residual = None
        else:
            # Pipeline Parallel 中间 rank，从上游接收
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # 2. 逐层 Transformer（只跑本 PP rank 分到的层）
        for layer in self.layers[start_layer:end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        # 3. 最终 LayerNorm（只有最后 PP rank 做）
        if is_last_pp_rank:
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states
        else:
            return IntermediateTensors({"hidden_states": hidden_states,
                                        "residual": residual})
```

关键设计：
- 用 `start_layer` / `end_layer` 切片实现 Pipeline Parallel 层分配
- residual 连接不存在独立的加法，而是由 fused RMSNorm 内部处理

### 2.3 LlamaDecoderLayer — 单层 Transformer

```python
class LlamaDecoderLayer:
    def forward(self, positions, hidden_states, residual):
        # Pre-Norm + Self-Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # Pre-Norm + FFN
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
```

经典的 Pre-LayerNorm + Residual 结构。`RMSNorm` 的 fused 实现接收 `(hidden_states, residual)` 两个输入，内部做 `residual += hidden_states` 再 normalize，避免多余的显存读写。

### 2.4 LlamaAttention — 注意力层

```python
class LlamaAttention:
    def forward(self, positions, hidden_states):
        # (1) QKV 投影：一次 GEMM 得到 q, k, v
        qkv, _ = self.qkv_proj(hidden_states)  # [num_tokens, q_size + 2*kv_size]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # (2) RoPE 位置编码：对 q 和 k 应用旋转
        q, k = self.rotary_emb(positions, q, k)

        # (3) Attention 计算（含 KV cache 读写）
        attn_output = self.attn(q, k, v)

        # (4) 输出投影
        output, _ = self.o_proj(attn_output)
        return output
```

### 2.5 LlamaMLP — FFN 层

```python
class LlamaMLP:
    def forward(self, x):
        x, _ = self.gate_up_proj(x)   # [num_tokens, 2 * intermediate_size]
        x = self.act_fn(x)            # SiLU(gate) * up，fused kernel
        x, _ = self.down_proj(x)      # [num_tokens, hidden_size]
        return x
```

`gate_up_proj` 是 `MergedColumnParallelLinear`，一次矩阵乘法同时计算 gate 和 up 两个分支（拼接在一起）。`SiluAndMul` 是一个 fused CUDA kernel，将 `silu(gate) * up` 合并为一步。

---

## 3. Attention 如何调用 PagedAttention Kernel

### 3.1 调用链

```
LlamaAttention.forward()
  → self.attn(q, k, v)                              # Attention 类的 forward
    → unified_kv_cache_update(key, value, layer_name) # 写入 KV cache
    → unified_attention_with_output(q, k, v, output, layer_name)  # 读 KV cache + 计算
      → self.impl.forward(...)                       # 具体 backend 实现
```

### 3.2 Attention 类的 forward 逻辑

```python
# vllm/model_executor/layers/attention/attention.py
class Attention(nn.Module):
    def forward(self, query, key, value):
        # reshape 为 [num_tokens, num_heads, head_dim]
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        # Step 1: 将本步的 k, v 写入 KV cache
        unified_kv_cache_update(key, value, self.layer_name)

        # Step 2: 执行 attention（从 KV cache 读取所有历史 k, v）
        unified_attention_with_output(query, key, value, output, self.layer_name)

        return output.view(-1, hidden_size)
```

### 3.3 KV Cache 的读写机制

`unified_kv_cache_update` 和 `unified_attention_with_output` 通过 `get_forward_context()` 获取：
- **attn_metadata** — attention 计算的元信息（seq_lens, block_table 等）
- **kv_cache** — 当前层的 KV cache tensor
- **slot_mapping** — 本步每个 token 的 KV 写入位置（物理槽号）

```python
def unified_kv_cache_update(key, value, layer_name):
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    # 将 key, value 写入 kv_cache 的指定 slot 位置
    attn_layer.impl.do_kv_cache_update(attn_layer, key, value, kv_cache, layer_slot_mapping)

def unified_attention_with_output(query, key, value, output, layer_name):
    attn_metadata, self, kv_cache, _ = get_attention_context(layer_name)
    # 调用具体 backend（FlashAttention / FlashInfer / Triton）的 forward
    self.impl.forward(self, query, key, value, kv_cache, attn_metadata, output=output)
```

### 3.4 Forward Context 机制

为什么 KV cache 和 attn_metadata 不是作为参数传递的？

因为模型的 `forward` 签名需要保持简洁（尤其是 `torch.compile` 需要稳定的签名）。vLLM 使用了一个全局上下文管理器：

```python
# ModelRunner 在调用模型 forward 前设置
with set_forward_context(attn_metadata, vllm_config, slot_mapping=slot_mappings, ...):
    outputs = self.model(input_ids, positions, ...)
```

模型内部每一层的 `Attention` 通过 `get_forward_context()` 从这个全局上下文中取到自己层的 metadata、kv_cache 和 slot_mapping。

---

## 4. Tensor Parallel 通信时机

### 4.1 TP 切分策略

| 层 | 类型 | 切分维度 | 通信 |
|------|------|----------|------|
| `embed_tokens` | VocabParallelEmbedding | 词表维度 | all-reduce |
| `qkv_proj` | QKVParallelLinear (Column) | 按 head 数切分输出 | 无 |
| `o_proj` | RowParallelLinear | 按 head 数切分输入 | **all-reduce** |
| `gate_up_proj` | MergedColumnParallelLinear | 按 intermediate 维度切分输出 | 无 |
| `down_proj` | RowParallelLinear | 按 intermediate 维度切分输入 | **all-reduce** |
| `lm_head` | ParallelLMHead | 词表维度 | all-gather |

### 4.2 通信模式：Column + Row 配对

核心模式是 **ColumnParallel → 本地计算 → RowParallel + all-reduce**：

```
ColumnParallelLinear:
  - 权重按输出维度切分，每个 GPU 只存一部分列
  - 输入不切分（所有 GPU 拿到相同输入）
  - 输出是部分结果（每个 GPU 有不同的输出切片）
  - 无通信

RowParallelLinear:
  - 权重按输入维度切分，每个 GPU 只存一部分行
  - 输入已经是切分的（来自上游 ColumnParallel 的输出）
  - 本地 GEMM 得到部分和
  - all-reduce 将所有 GPU 的部分和求和 → 完整输出
```

### 4.3 每层的通信次数

**每个 Transformer 层有 2 次 all-reduce：**

```
hidden_states (完整)
  → qkv_proj (Column, 无通信)
  → RoPE + Attention (本地计算)
  → o_proj (Row, ★ all-reduce)            ← 第 1 次
  → LayerNorm + gate_up_proj (Column, 无通信)
  → SiLU*Mul (本地计算)
  → down_proj (Row, ★ all-reduce)         ← 第 2 次
  → 输出 (完整)
```

### 4.4 RowParallelLinear 的 forward 代码

```python
# vllm/model_executor/layers/linear.py
class RowParallelLinear:
    def forward(self, input_):
        # 输入已经是按 TP 切分的（来自上游 ColumnParallel 的输出）
        input_parallel = input_

        # 本地矩阵乘法
        output_parallel = self.quant_method.apply(self, input_parallel, bias_)

        # ★ all-reduce：将所有 TP rank 的部分和相加
        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        return output
```

### 4.5 通信开销分析

对于一个 N 层的模型，TP=T 时：
- 每步推理需要 **2N 次 all-reduce**
- 每次 all-reduce 的数据量 = `num_tokens × hidden_size × dtype_size`
- all-reduce 使用 NCCL，走 NVLink（同机）或 RDMA（跨机）

这就是为什么 TP 通常只用在单机内（NVLink 带宽高），跨机用 PP 或 DP。

---

## 5. CUDA Graph 与模型执行的交互

### 5.1 为什么需要 CUDA Graph

每次 decode 只处理少量 token（每个请求 1 个），GPU 计算量小但 CPU 调度开销大（kernel launch、内存分配等）。CUDA Graph 通过预录制整个计算图来消除这些开销：

- **首次**：录制（capture）所有 kernel 调用和内存操作
- **后续**：一次 replay 完成所有计算，CPU 只需发一条命令

### 5.2 两种 CUDA Graph 模式

```python
class CUDAGraphMode(enum.Enum):
    NONE = 0        # 不使用 CUDA Graph（eager 模式）
    PIECEWISE = 1   # 分段捕获（attention kernel 在段间 eager 执行）
    FULL = 2        # 完整捕获（整个 forward 一个 graph）
```

**FULL 模式**：
- 整个 `model.forward()` 被捕获为一个 CUDA Graph
- 最快（zero CPU overhead）
- 限制：batch 中所有请求必须是 uniform decode（每个请求调度 1 个 token）
- 不支持 cascade attention、encoder-decoder 有 encoder 输出时

**PIECEWISE 模式**：
- 模型被 `@support_torch_compile` 标记
- `torch.compile` 将模型编译为多个子图
- Attention 操作标记为 "break point"（`@eager_break_during_capture`）
- 结果：Transformer 层的计算图被分成多段，每段各自捕获为 CUDA Graph，段与段之间 eager 执行 attention

**NONE 模式**：
- Prefill 阶段（token 数量不固定，难以复用 graph）
- 有 encoder 输入时
- 特殊场景（cascade attention、LoRA 等）

### 5.3 CUDA Graph 的生命周期

```
初始化阶段（启动时一次性完成）：
  1. profile_run()        → 探测最大显存占用
  2. capture_model()      → 针对预定义的 batch size 列表逐一捕获

推理阶段（每步执行）：
  3. _determine_batch_execution_and_padding()  → 根据当前 batch 决定用哪个模式
  4. dispatch_cudagraph() → 从已捕获的 graph 中选择匹配的（按 num_tokens 查找）
  5. 如果找到 → replay（重放已捕获的 graph）
     如果没找到 → 退化为 eager 执行
```

### 5.4 Capture 过程

```python
def capture_model(self):
    # 按 num_tokens 从大到小捕获（大的先分配，小的复用显存）
    with graph_capture(device=self.device):
        for runtime_mode, batch_descs in self.cudagraph_dispatcher.get_capture_descs():
            self._capture_cudagraphs(
                batch_descriptors=batch_descs,
                cudagraph_runtime_mode=runtime_mode,
            )
```

捕获的 batch size 列表通常是：`[1, 2, 4, 8, 16, 32, 64, 128, ...]`。运行时如果实际 batch 不在列表中，会 padding 到最近的已捕获大小。

### 5.5 Runtime Dispatch 决策逻辑

```python
def _determine_batch_execution_and_padding(self, num_tokens, num_reqs, ...):
    # 判断是否是 uniform decode（所有请求都只调度 1 个 token）
    uniform_decode = self._is_uniform_decode(max_num_scheduled_tokens, ...)

    # 调度 CUDA Graph 模式
    cudagraph_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
        num_tokens=num_tokens,
        uniform_decode=uniform_decode,       # FULL 模式要求 uniform
        has_lora=has_lora,                   # LoRA 限制 graph 复用
        ...
    )
    # batch_descriptor 包含 padding 后的 num_tokens
```

决策优先级：
1. 如果是 uniform decode 且有对应捕获 → **FULL**
2. 如果不满足 FULL 条件但有 piecewise 支持 → **PIECEWISE**
3. 否则 → **NONE**（eager）

### 5.6 为什么 Attention 是 "break point"

Attention 无法直接被完整捕获进 CUDA Graph（在 PIECEWISE 模式下），因为：
- 不同步骤的 seq_len 不同，attention kernel 的参数变化
- KV cache 的 slot_mapping 每步变化
- block_table 动态更新

因此 `unified_attention_with_output` 被标记为 `@eager_break_during_capture`——在 piecewise 捕获时作为"断点"，其前后的计算分别被捕获为独立的 CUDA Graph 段：

```
[Graph 段 1: embed + LayerNorm + QKV proj + RoPE]
    ↓ eager: attention kernel（动态参数）
[Graph 段 2: o_proj + LayerNorm + MLP]
    ↓ eager: attention kernel
[Graph 段 3: ...]
```

在 FULL 模式下，attention kernel 的元数据在 padding 后是固定的，所以整个 forward 可以作为一个完整的 graph。

### 5.7 CUDA Graph 中的固定缓冲区

CUDA Graph replay 要求输入/输出 tensor 的地址不变。因此 ModelRunner 在初始化时预分配固定大小的缓冲区：

```python
self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
self.positions = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=device)
```

每步推理时，不创建新 tensor，而是往这些固定缓冲区里填写数据，然后把缓冲区的 slice 传给模型。

---

## 6. 完整调用链（一步 Decode 推理）

```
ModelRunner.execute_model(scheduler_output)
│
├── _update_states()                      # 更新 batch 持久状态
├── _prepare_inputs()                     # 填充 input_ids, positions, slot_mapping
├── _build_attention_metadata()           # 构造 attn_metadata
├── _determine_batch_execution_and_padding()  # 决定 CUDA Graph 模式
│
├── set_forward_context(attn_metadata, slot_mapping, ...)  # 注入全局上下文
│
├── LlamaForCausalLM.forward(input_ids, positions)
│   └── LlamaModel.forward()
│       ├── embed_tokens(input_ids)                # [num_tokens] → [num_tokens, hidden]
│       │
│       └── for layer in layers:
│           ├── input_layernorm(hidden, residual)   # fused RMSNorm + residual add
│           ├── LlamaAttention.forward()
│           │   ├── qkv_proj(hidden)               # GEMM, 无通信
│           │   ├── rotary_emb(positions, q, k)    # RoPE
│           │   ├── attn(q, k, v)                  # KV cache write + PagedAttention
│           │   └── o_proj(attn_output)            # GEMM + ★ all-reduce
│           ├── post_attention_layernorm            # fused RMSNorm + residual add
│           └── LlamaMLP.forward()
│               ├── gate_up_proj(hidden)           # GEMM, 无通信
│               ├── SiluAndMul                     # fused activation
│               └── down_proj(x)                   # GEMM + ★ all-reduce
│
│       └── norm(hidden, residual)                 # 最终 RMSNorm
│
├── hidden_states[logits_indices]                  # 只取需要采样的位置
├── compute_logits(sample_hidden_states)           # lm_head GEMM → [num_sample, vocab]
│
└── sample_tokens(logits, grammar_output)          # 采样 → token_ids
```

---

## 7. 关键设计总结

| 设计决策 | 原因 |
|----------|------|
| Embedding 查表在模型内部 | 保持 ModelRunner 和模型解耦；多模态时 ModelRunner 提前处理 |
| QKV 合并为一个 GEMM | 减少 kernel launch 开销 |
| gate + up 合并为一个 GEMM | 同上 |
| RMSNorm fused residual | 减少一次显存读写 |
| SiluAndMul fused | 减少中间 tensor 分配 |
| ColumnParallel + RowParallel 配对 | 最小化通信次数（每层只需 2 次 all-reduce） |
| Attention 通过全局上下文获取 metadata | 保持模型签名稳定，兼容 torch.compile |
| CUDA Graph 固定缓冲区 | replay 要求地址不变 |
| Piecewise Graph + Attention eager | 兼顾动态 attention 参数和 graph 加速 |

---

## 8. 常见疑问

### Q: 为什么 compute_logits 不在 forward 内部？

为了节省计算。forward 输出所有 token 的 hidden_states，但只有一部分需要采样（decode 时每个请求的最后一个 token）。先用 `logits_indices` 选出需要的位置，再过 lm_head，避免对无用位置做大矩阵乘法。

### Q: FULL Graph 和 PIECEWISE Graph 性能差多少？

FULL 略快（单次 replay，无 eager 断点开销），但差距不大（< 5%）。FULL 的限制是只能用于 uniform decode，而 PIECEWISE 更通用。

### Q: 如果 TP=1，all-reduce 还会执行吗？

不会。`RowParallelLinear.forward` 中有判断：`if self.reduce_results and self.tp_size > 1`。TP=1 时跳过通信，等同于普通线性层。

### Q: RoPE 在哪执行？

在 `self.attn()` 之前，由 `LlamaAttention.forward` 中的 `self.rotary_emb(positions, q, k)` 完成。positions tensor 提供每个 token 的绝对位置，RoPE 根据位置对 q 和 k 施加旋转变换。
