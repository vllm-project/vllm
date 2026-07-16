# 模型层深度解析

## 概述

模型层是 vLLM 中负责定义和加载模型结构的部分。它回答了一个核心问题：**如何把 HuggingFace 上的模型权重转化为能在 vLLM 中高效运行的 `nn.Module`？**

模型层包含三个关键子系统：

```
模型层
├── 模型注册表 (registry)      ← 架构名 → 实现类的映射
├── 模型实现 (models/)         ← 每种模型的具体 nn.Module 定义
├── 基础算子层 (layers/)       ← 可复用的高性能算子（Linear, Attention, RMSNorm...）
└── 模型加载器 (model_loader/) ← 从磁盘读取权重并填充到模型中
```

---

## 1. 模型注册表：从架构名到实现类

当用户指定一个 HuggingFace 模型（如 `meta-llama/Llama-3-8B`），vLLM 需要知道该用哪个类来实例化它。

### 1.1 注册表结构

```python
# vllm/model_executor/models/registry.py
_TEXT_GENERATION_MODELS = {
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "DeepseekV3ForCausalLM": ("deepseek_v2", "DeepseekV3ForCausalLM"),
    "MistralForCausalLM": ("llama", "LlamaForCausalLM"),  # 共享实现
    ...
}
```

映射格式是 `"HF架构名": ("模块名", "类名")`。很多架构共享同一个实现（如 Mistral 和 Llama 结构相同）。

### 1.2 查找流程

```
HF config.json 中的 architectures 字段
  → 在 _TEXT_GENERATION_MODELS / _MULTIMODAL_MODELS 中查找
  → 动态 import 对应模块
  → 返回模型类
```

目前注册表中有约 290 个模型实现文件，覆盖了主流的开源模型。

---

## 2. 模型实现：以 Llama 为例

每个模型实现文件的结构高度统一，遵循一套标准化的模式。以 `llama.py` 为例：

### 2.1 层级结构

```
LlamaForCausalLM (顶层，对接 ModelRunner)
├── LlamaModel (模型主体)
│   ├── embed_tokens (词嵌入)
│   ├── layers[0..N] (Decoder 层)
│   │   ├── LlamaAttention
│   │   │   ├── qkv_proj (QKVParallelLinear)
│   │   │   ├── o_proj (RowParallelLinear)
│   │   │   ├── rotary_emb (RoPE)
│   │   │   └── attn (Attention 后端)
│   │   ├── LlamaMLP
│   │   │   ├── gate_up_proj (MergedColumnParallelLinear)
│   │   │   ├── act_fn (SiluAndMul)
│   │   │   └── down_proj (RowParallelLinear)
│   │   ├── input_layernorm (RMSNorm)
│   │   └── post_attention_layernorm (RMSNorm)
│   └── norm (最终 RMSNorm)
├── lm_head (ParallelLMHead)
└── logits_processor (LogitsProcessor)
```

### 2.2 模型实现的三个核心方法

每个模型必须实现：

**1. `__init__`：构建模型结构**

```python
class LlamaForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # 从 HF config 读取参数
        config = vllm_config.model_config.hf_config
        # 构建模型结构
        self.model = LlamaModel(vllm_config=vllm_config, prefix="model")
        # 输出 head
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config.vocab_size)
```

**2. `forward`：前向推理**

```python
def forward(self, input_ids, positions, intermediate_tensors, ...):
    hidden_states = self.model(input_ids, positions, intermediate_tensors)
    return hidden_states
```

注意：`forward` 不负责采样，只返回 hidden_states。采样由 ModelRunner 调用 `compute_logits` + Sampler 完成。

**3. `load_weights`：加载权重**

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    stacked_params_mapping = [
        # (vllm内部名, HF原始名, shard_id)
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]
    # 遍历权重，按映射规则加载到对应参数
    for name, loaded_weight in weights:
        param = params_dict[name]
        weight_loader = param.weight_loader
        weight_loader(param, loaded_weight, shard_id)
```

`load_weights` 的关键职责是处理**权重合并**：vLLM 为了高效推理，会把分离的 q/k/v 投影合并为一个 `qkv_proj`，把 gate/up 投影合并为一个 `gate_up_proj`。

### 2.3 Pipeline Parallel 支持

模型通过 `make_layers` 函数自动支持 PP：

```python
self.start_layer, self.end_layer, self.layers = make_layers(
    config.num_hidden_layers,
    lambda prefix: LlamaDecoderLayer(vllm_config=vllm_config, prefix=prefix),
    prefix="model.layers",
)
```

每个 PP rank 只实例化属于自己的那部分层。不属于当前 rank 的组件用 `PPMissingLayer()` 占位。

---

## 3. 基础算子层（layers/）

模型实现不会直接使用 PyTorch 原生的 `nn.Linear` 等，而是使用 vLLM 提供的高性能替代品。

### 3.1 Linear 层家族

```
LinearBase (抽象基类)
├── ReplicatedLinear        ← 每个 TP rank 持有完整权重的副本
├── ColumnParallelLinear    ← 权重按列切分，输出拼接
│   ├── MergedColumnParallelLinear  ← 多个 Column 合并（gate_up_proj）
│   └── QKVParallelLinear           ← Q/K/V 三个投影合并
└── RowParallelLinear       ← 权重按行切分，输出 all-reduce
```

Tensor Parallel 的核心在这里：

```python
# ColumnParallelLinear: 输出维度按 TP 切分
# 每个 rank 持有 output_size / tp_size 列的权重
# 各 rank 独立计算后拼接（或在后续 RowParallel 中 all-reduce）

# RowParallelLinear: 输入维度按 TP 切分
# 每个 rank 持有 input_size / tp_size 行的权重
# 各 rank 结果需要 all-reduce 求和
```

经典的 Transformer MLP 模式（Megatron-LM 风格）：
```
Column Parallel (gate_up_proj)  →  激活函数  →  Row Parallel (down_proj, with all-reduce)
```

### 3.2 Attention 层

```python
# vllm/model_executor/layers/attention/attention.py
class Attention(nn.Module):
    """统一的 Attention 接口"""
    def forward(self, query, key, value):
        # 1. 将 K/V 写入 KV cache
        # 2. 根据后端（FlashAttention / FlashInfer / Triton 等）执行注意力
        # 3. 返回 attention 输出
```

Attention 层的关键设计：
- **不自己实现注意力计算**，而是委托给可插拔的 attention backend（`vllm/v1/attention/backends/`）
- **自动管理 KV cache 的读写**（通过 ForwardContext 获取当前 cache 状态）
- 支持多种注意力变体：MHA、MQA、GQA、MLA、Sliding Window

### 3.3 其他核心层

| 层 | 文件 | 作用 |
|----|------|------|
| `RMSNorm` | `layernorm.py` | 使用 custom CUDA kernel 的归一化 |
| `VocabParallelEmbedding` | `vocab_parallel_embedding.py` | 词表按 TP 切分的嵌入层 |
| `RotaryEmbedding` | `rotary_embedding.py` | RoPE 位置编码 |
| `SiluAndMul` | `activation.py` | 融合的激活函数（减少 kernel launch） |
| `FusedMoE` | `fused_moe/` | MoE 专家混合层 |
| `LogitsProcessor` | `logits_processor.py` | 最终 logits 投影 |

### 3.4 量化透明接入

每个 Linear 层通过 `quant_method` 实现量化透明：

```python
class LinearBase:
    def __init__(self, ..., quant_config=None):
        if quant_config is None:
            self.quant_method = UnquantizedLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self)
```

量化配置（GPTQ、AWQ、FP8、BitsAndBytes 等）会替换 Linear 的计算方式，但对模型定义层面完全透明。支持的量化方式约 18 种。

---

## 4. 模型加载器（model_loader/）

### 4.1 加载流程

```
ModelRunner.load_model()
│
├── get_model_loader(load_config)  ← 根据配置选择加载器
│   ├── DefaultModelLoader    ← 标准 safetensors/pt 文件
│   ├── ShardedStateLoader    ← 预切分的分布式 checkpoint
│   ├── BitsAndBytesLoader    ← BnB 量化加载
│   ├── TensorizerLoader      ← Tensorizer 格式
│   └── RunAIStreamerLoader   ← RunAI 流式加载
│
├── loader.load_model()
│   ├── 1. 实例化模型（空权重 / meta device）
│   ├── 2. 迭代权重文件
│   ├── 3. 对每个权重调用 model.load_weights()
│   └── 4. 权重填充完成
│
└── 返回完整模型
```

### 4.2 权重加载的关键设计

**懒加载 + 流式处理**：不会一次把所有权重读入内存，而是逐文件、逐张量流式加载：

```python
# DefaultModelLoader 的核心逻辑
for name, loaded_weight in weights_iterator(model_path):
    # 逐个张量送给模型的 load_weights 方法
    model.load_weights([(name, loaded_weight)])
```

**weight_loader 机制**：每个参数可以有自定义的 `weight_loader` 函数，处理 TP 切分、量化反量化等：

```python
# 参数知道自己应该加载权重的哪个 shard
param = params_dict["model.layers.0.self_attn.qkv_proj.weight"]
param.weight_loader(param, loaded_weight, shard_id="q")
# 内部：只加载到 param 的 Q 对应的行区间
```

---

## 5. Attention Backend 系统

Attention 后端是性能的关键，vLLM 支持多种实现：

### 5.1 后端选择

```python
# vllm/v1/attention/selector.py
def get_attn_backend(head_size, dtype, kv_cache_dtype, block_size, ...):
    # 根据硬件平台、头维度、数据类型等条件选择最优后端
```

### 5.2 主要后端

| 后端 | 场景 | 优势 |
|------|------|------|
| FlashAttention | NVIDIA GPU，主流选择 | 高性能，memory-efficient |
| FlashInfer | NVIDIA GPU，备选 | 对 decode 阶段优化好 |
| Triton | 通用 GPU | 可移植，支持自定义 |
| MLA | DeepSeek 系列 | Multi-head Latent Attention |
| CPU Attention | CPU 推理 | 无 GPU 依赖 |

### 5.3 统一接口

```python
# vllm/v1/attention/backend.py
class AttentionBackend(ABC):
    @abstractmethod
    def forward(self, query, key, value, kv_cache, attn_metadata, output):
        """所有后端实现统一的 forward 签名"""
```

---

## 6. 模型接口协议

vLLM 通过接口 mixin 声明模型的能力：

```python
class LlamaForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle):
    ...
```

| 接口 | 含义 |
|------|------|
| `SupportsPP` | 支持 Pipeline Parallel |
| `SupportsLoRA` | 支持 LoRA 适配器 |
| `SupportsMultiModal` | 支持多模态输入 |
| `SupportsEagle` | 支持 EAGLE 投机解码 |
| `VllmModelForPooling` | 是嵌入/池化模型 |
| `MixtureOfExperts` | 是 MoE 模型 |

ModelRunner 根据这些接口调整行为（如是否启用 LoRA 路径、是否需要编码多模态输入等）。

---

## 7. 添加新模型的流程

```
1. 在 vllm/model_executor/models/ 下创建 my_model.py
2. 定义模型类，实现 __init__ / forward / load_weights
3. 在 registry.py 的 _TEXT_GENERATION_MODELS 中注册
4. 使用 vLLM 提供的 layers（QKVParallelLinear, Attention 等）
5. 如需量化支持，无需额外工作（layers 自动处理）
6. 如需 PP 支持，使用 make_layers + PPMissingLayer
```

关键原则：
- 使用 vLLM 的 Attention 层而非自己实现 —— 自动获得 KV cache、各种 backend 支持
- 使用 ColumnParallelLinear / RowParallelLinear —— 自动获得 TP 支持
- 实现 `load_weights` 处理 HF 权重名到 vLLM 内部名的映射

---

## 8. 关键设计决策总结

| 设计点 | 选择 | 原因 |
|--------|------|------|
| 权重合并（Q/K/V → QKV） | 合并存储 | 减少 kernel launch，提高 TP 效率 |
| Attention 可插拔后端 | 策略模式 | 不同硬件最优实现不同 |
| 量化在 Linear 层透明注入 | `quant_method` 抽象 | 模型定义代码无需感知量化 |
| PP 通过 `make_layers` 自动切分 | 按层切分 | 模型代码最小改动 |
| 统一的 `load_weights` 协议 | 模型自己映射权重名 | 灵活处理各种命名差异 |

---

## 9. 后续深入方向

- MoE 模型中 `FusedMoE` 层的 Expert Parallel 实现
- MLA（Multi-head Latent Attention）的压缩 KV cache 机制
- 多模态模型的 Vision Encoder 集成方式
- `torch.compile` / CUDA Graph 对模型结构的约束
- 权重的在线更新（LoRA hot-swap、权重热迁移）
