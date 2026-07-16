# 芯片适配插件精度对齐排查指南

## 背景

当使用不同芯片适配插件（如 NPU、DCU、MLU 等）替换默认 CUDA 后端时，最终输出结果可能与基准不一致。本文档提供一套系统化的方法，逐层对比推理过程中的中间值，定位精度 divergence 的具体位置。

---

## 排查前提

在开始对比前，先排除非计算因素：

1. **Tokenization 一致性** — 同一 prompt 在两个插件下产生相同的 token ids
2. **Sampling 确定性** — 使用 greedy decoding（`temperature=0`），排除随机性
3. **权重加载一致性** — 模型加载后逐层对比权重 checksum
4. **数据类型一致** — 确认两端使用相同的 dtype（fp16 / bf16 / fp32）

```python
# 验证权重一致性
import torch

def compare_weights(model_baseline, model_experiment):
    for name, param in model_baseline.named_parameters():
        other = dict(model_experiment.named_parameters())[name]
        diff = (param.data.float() - other.data.float()).abs().max().item()
        if diff > 0:
            print(f"权重不一致: {name}, max_diff={diff:.6e}")
```

---

## 核心方法：逐层 Hook + 二分定位

### 思路

在模型 forward 过程中，通过 PyTorch 的 `register_forward_hook` 机制抓取每一层的输出张量，分别在基准插件和实验插件下运行同一条输入，然后逐层对比。

### 对比粒度（由粗到细）

```
Embedding 输出
  → Layer[i] 整体输出
    → Layer[i].self_attn 输出
      → Q/K/V 投影输出
      → Attention Score
      → Attention Output (o_proj)
    → Layer[i].mlp 输出
      → gate_up_proj 输出
      → activation 输出
      → down_proj 输出
    → LayerNorm 输出
  → 最终 LM Head logits
    → Top-K token 及概率
```

---

## 实现步骤

### Step 1: 注册 Hook 并收集激活值

```python
import torch
from collections import OrderedDict


class ActivationCollector:
    """收集模型推理过程中各层的激活值"""

    def __init__(self):
        self.activations: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._hooks = []

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach().cpu().float()
            elif isinstance(output, tuple) and len(output) > 0:
                self.activations[name] = output[0].detach().cpu().float()
        return hook_fn

    def register(self, model: torch.nn.Module):
        """对模型的关键节点注册 hook"""

        # Embedding
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            self._hooks.append(
                model.model.embed_tokens.register_forward_hook(
                    self._make_hook("embed_tokens")
                )
            )

        # 每个 Transformer Layer
        layers = None
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h

        if layers is not None:
            for i, layer in enumerate(layers):
                self._hooks.append(
                    layer.register_forward_hook(
                        self._make_hook(f"layer_{i}")
                    )
                )
                # Attention 子模块
                attn = getattr(layer, 'self_attn', None) or getattr(layer, 'attention', None)
                if attn is not None:
                    self._hooks.append(
                        attn.register_forward_hook(
                            self._make_hook(f"layer_{i}.attn")
                        )
                    )
                    # QKV 投影
                    qkv = getattr(attn, 'qkv_proj', None)
                    if qkv is not None:
                        self._hooks.append(
                            qkv.register_forward_hook(
                                self._make_hook(f"layer_{i}.attn.qkv_proj")
                            )
                        )
                    # Output 投影
                    o_proj = getattr(attn, 'o_proj', None)
                    if o_proj is not None:
                        self._hooks.append(
                            o_proj.register_forward_hook(
                                self._make_hook(f"layer_{i}.attn.o_proj")
                            )
                        )

                # MLP 子模块
                mlp = getattr(layer, 'mlp', None)
                if mlp is not None:
                    self._hooks.append(
                        mlp.register_forward_hook(
                            self._make_hook(f"layer_{i}.mlp")
                        )
                    )
                    gate_up = getattr(mlp, 'gate_up_proj', None)
                    if gate_up is not None:
                        self._hooks.append(
                            gate_up.register_forward_hook(
                                self._make_hook(f"layer_{i}.mlp.gate_up_proj")
                            )
                        )
                    down = getattr(mlp, 'down_proj', None)
                    if down is not None:
                        self._hooks.append(
                            down.register_forward_hook(
                                self._make_hook(f"layer_{i}.mlp.down_proj")
                            )
                        )

                # LayerNorm
                for norm_name in ['input_layernorm', 'post_attention_layernorm']:
                    norm = getattr(layer, norm_name, None)
                    if norm is not None:
                        self._hooks.append(
                            norm.register_forward_hook(
                                self._make_hook(f"layer_{i}.{norm_name}")
                            )
                        )

        # LM Head
        lm_head = getattr(model, 'lm_head', None)
        if lm_head is not None:
            self._hooks.append(
                lm_head.register_forward_hook(self._make_hook("lm_head"))
            )

    def clear(self):
        self.activations.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
```

### Step 2: 定义 Diff 指标

```python
import torch
import torch.nn.functional as F


def compute_diff(baseline: torch.Tensor, experiment: torch.Tensor, name: str) -> dict:
    """计算两个张量之间的多维度 diff 指标"""
    baseline = baseline.float().flatten()
    experiment = experiment.float().flatten()

    abs_diff = (baseline - experiment).abs()
    rel_diff = abs_diff / (baseline.abs() + 1e-8)

    result = {
        "name": name,
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_rel_diff": rel_diff.mean().item(),
        "cosine_sim": F.cosine_similarity(
            baseline.unsqueeze(0), experiment.unsqueeze(0)
        ).item(),
        "rmse": torch.sqrt((abs_diff ** 2).mean()).item(),
    }
    return result


def print_diff_report(diffs: list[dict]):
    """打印对比报告，按 max_abs_diff 降序排列"""
    diffs_sorted = sorted(diffs, key=lambda x: x["max_abs_diff"], reverse=True)

    print(f"{'Layer':<40} {'MaxAbs':<12} {'MeanAbs':<12} {'MaxRel':<12} {'Cosine':<10}")
    print("-" * 86)
    for d in diffs_sorted:
        print(f"{d['name']:<40} {d['max_abs_diff']:<12.6e} {d['mean_abs_diff']:<12.6e} "
              f"{d['max_rel_diff']:<12.6e} {d['cosine_sim']:<10.8f}")
```

### Step 3: 运行对比

```python
def run_comparison(model_baseline, model_experiment, input_ids, positions):
    """在两个模型上运行同一输入，收集并对比激活值"""

    collector_base = ActivationCollector()
    collector_exp = ActivationCollector()

    collector_base.register(model_baseline)
    collector_exp.register(model_experiment)

    # Forward pass
    with torch.no_grad():
        model_baseline(input_ids, positions)
        model_experiment(input_ids, positions)

    # 对比所有共同的激活点
    diffs = []
    common_keys = set(collector_base.activations.keys()) & set(collector_exp.activations.keys())

    for key in sorted(common_keys):
        base_act = collector_base.activations[key]
        exp_act = collector_exp.activations[key]

        if base_act.shape != exp_act.shape:
            print(f"[WARNING] Shape mismatch at {key}: {base_act.shape} vs {exp_act.shape}")
            continue

        diff = compute_diff(base_act, exp_act, key)
        diffs.append(diff)

    print_diff_report(diffs)

    # 清理
    collector_base.remove_hooks()
    collector_exp.remove_hooks()

    return diffs
```

---

## 在 vLLM 中的具体注入点

vLLM 的模型执行由 `ModelRunner` 驱动，核心调用链为：

```
ModelRunner.execute_model()
  → model.forward(input_ids, positions, intermediate_tensors, ...)
```

### 方式一：修改 ModelRunner（推荐用于快速调试）

在 `vllm/v1/worker/gpu_model_runner.py` 的 `execute_model` 方法中插入 hook：

```python
# vllm/v1/worker/gpu_model_runner.py
# 在 execute_model() 中，model forward 调用前后插入

class GPUModelRunner:
    def execute_model(self, scheduler_output, ...):
        ...
        # === 插入点：注册 hook ===
        collector = ActivationCollector()
        collector.register(self.model)

        hidden_states = self.model(...)

        # === 插入点：保存激活值 ===
        torch.save(collector.activations, f"/tmp/activations_{self.device}.pt")
        collector.remove_hooks()
        ...
```

### 方式二：使用独立脚本加载模型（推荐用于精细对比）

绕开 vLLM 引擎，直接加载模型进行对比：

```python
from vllm import LLM
from vllm.config import VllmConfig

# 加载基准模型（CUDA 后端）
llm_baseline = LLM(
    model="your-model-path",
    dtype="float16",
    enforce_eager=True,  # 关闭 CUDA Graph，方便 hook
    max_model_len=512,
)

# 加载实验模型（目标芯片插件）
llm_experiment = LLM(
    model="your-model-path",
    dtype="float16",
    enforce_eager=True,
    max_model_len=512,
    # 通过环境变量或参数启用目标插件
)

# 获取内部模型引用
model_baseline = llm_baseline.llm_engine.model_executor.driver_worker.model_runner.model
model_experiment = llm_experiment.llm_engine.model_executor.driver_worker.model_runner.model
```

> **注意**：`enforce_eager=True` 是必需的，CUDA Graph 模式下 hook 无法正常工作。

---

## 判断标准

| 指标 | 正常范围 (fp16) | 需关注 | 明确有问题 |
|------|-----------------|--------|-----------|
| max_abs_diff | < 1e-3 | 1e-3 ~ 1e-1 | > 1e-1 |
| cosine_sim | > 0.9999 | 0.999 ~ 0.9999 | < 0.999 |
| mean_abs_diff | < 1e-4 | 1e-4 ~ 1e-2 | > 1e-2 |

典型的精度误差模式：

- **突然跳变**：某层 diff 突然增大几个数量级 → 该层的算子实现有 bug
- **逐层累积**：diff 随层数线性/指数增长 → dtype 精度不足或某个通用 kernel 有系统性偏差
- **从头就有**：embedding 输出已有 diff → 权重加载或 embedding 查找实现有问题

---

## Logits 层面的快速对比

如果只需要对比最终 logits 而不关心中间层，可以用更轻量的方式：

```python
from vllm import LLM, SamplingParams

prompt = "The capital of France is"
sampling_params = SamplingParams(temperature=0, max_tokens=1, logprobs=20)

# 分别跑基准和实验
output_base = llm_baseline.generate([prompt], sampling_params)[0]
output_exp = llm_experiment.generate([prompt], sampling_params)[0]

# 对比 top token
print(f"Baseline top token: {output_base.outputs[0].token_ids[0]}")
print(f"Experiment top token: {output_exp.outputs[0].token_ids[0]}")

# 对比 logprobs 分布
logprobs_base = output_base.outputs[0].logprobs[0]
logprobs_exp = output_exp.outputs[0].logprobs[0]

for token_id, logprob in logprobs_base.items():
    exp_logprob = logprobs_exp.get(token_id)
    if exp_logprob is not None:
        diff = abs(logprob.logprob - exp_logprob.logprob)
        print(f"  token={logprob.decoded_token!r:<10} "
              f"base={logprob.logprob:.6f} exp={exp_logprob.logprob:.6f} diff={diff:.6e}")
```

---

## 排查流程总结

```
1. 确认前提条件（tokenization / weights / dtype / sampling 一致）
       │
2. 对比最终 logits → 确认是否存在 diff
       │
3. 注册逐层 hook，对比各层输出
       │
4. 定位 diff 首次出现或突然放大的层
       │
5. 对该层内部细化对比（attn vs mlp, 具体哪个 proj）
       │
6. 定位到具体算子后，编写单元测试用随机输入验证
       │
7. 修复算子实现 or 调整精度策略
```

---

## 常见根因

| 现象 | 可能原因 |
|------|---------|
| 所有层都有微小 diff，逐层累积 | fp16 精度不足，考虑 bf16 或混合精度 |
| 某一层 Attention 突然跳变 | FlashAttention / 自定义 attention kernel 实现差异 |
| MLP 层 diff 较大 | 激活函数（SiLU）或 fused kernel 精度问题 |
| RMSNorm 后 diff 放大 | 归一化实现的数值稳定性差异（rsqrt 精度） |
| Embedding 就有 diff | 权重加载时的类型转换有损，或 embedding 查表实现不同 |
| Decode 阶段 diff 比 Prefill 大 | KV Cache 存储/读取的精度损失 |
