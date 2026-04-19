## Phase A 共享接口契约

### 共享数据类型约定

| 名称 | 类型 | shape | dtype | 说明 |
|------|------|-------|-------|------|
| pairs | Tensor | (N, 2) | int64 | 通道索引对，0 <= idx < num_channels |
| angles | Tensor | (N,) | float32 | 旋转角度（弧度） |
| risk_scores | Tensor | (C,) | float32 | 每通道风险分数 |
| RotationPlan | dataclass | — | — | 见下方定义 |

空 pairs 约定：`torch.empty(0, 2, dtype=torch.int64)` + `torch.empty(0, dtype=torch.float32)`。RotationApplier 对空 pairs 返回输入不变。

### Device 约定
- 函数不主动搬移 tensor device
- 输出 tensor 与输入在同一 device 上
- risk_scores 始终 float32，可在 CPU 上

### Channel 维度约定
- 统一为 tensor 最后一维 (dim=-1)
- weight shape: (out_features, in_features)，channels = in_features
- activation shape: (..., in_features)，channels = last dim

### 错误处理约定
- 参数校验: ValueError
- 计算失败: RuntimeError
- 不做静默 fallback

### RotationPlan 定义

```python
@dataclass
class RotationPlan:
    mode: str              # "weight_only" | "activation_only" | "joint"
    layer_index: str       # e.g. "model.layers.0.self_attn.qkv_proj"
    pairs: torch.Tensor    # (N, 2), int64
    angles: torch.Tensor   # (N,), float32
    pair_meta: dict        # {"policy": "high_high", "top_ratio": 0.05}
    angle_meta: dict       # {"solver": "heuristic"}
```

### 各模块函数签名

#### rotation_applier.py
```python
def apply_givens_rotation(
    tensor: torch.Tensor,   # (..., C) 任意前导 batch 维
    pairs: torch.Tensor,    # (N, 2), int64
    angles: torch.Tensor,   # (N,), float32
    inverse: bool = False,
) -> torch.Tensor:          # 同 shape/dtype
```

#### channel_monitor.py
```python
def compute_risk_scores(
    tensor: torch.Tensor,       # weight: (O, I) 或 activation: (..., I)
    method: str = "max_abs",    # "max_abs" | "dynamic_range"
    channel_dim: int = -1,
) -> torch.Tensor:              # (C,), float32

def save_risk_scores(
    scores: torch.Tensor,       # (C,), float32
    path: str,
    metadata: dict,             # layer_index, target, method, shard_id, num_channels
) -> None:

def load_risk_scores(path: str) -> tuple[torch.Tensor, dict]:
```

#### pair_constructor.py
```python
def construct_pairs(
    risk_scores: torch.Tensor,     # (C,), float32 — 主 risk scores
    policy: str = "high_high",     # "adjacent_sorted"|"high_high"|"high_low"|"random_baseline"|"joint_compatible"
    top_ratio: float = 0.05,
    secondary_risk_scores: torch.Tensor | None = None,  # joint 模式下另一侧 risk
) -> torch.Tensor:                 # (N, 2), int64
```

#### angle_solver.py
```python
def solve_angles(
    pairs: torch.Tensor,            # (N, 2), int64
    solver: str = "heuristic",      # "heuristic" | "small_search"
    weight: torch.Tensor | None = None,
    activation: torch.Tensor | None = None,
    risk_scores: torch.Tensor | None = None,  # (C,), float32
) -> torch.Tensor:                   # (N,), float32
```

#### fp4_quant_policy.py
```python
def quantize_weight_to_fp4(
    weight: torch.Tensor,            # (O, I), bfloat16/float16
    global_scale: torch.Tensor,      # scalar, float32
    block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Returns (packed_weight [O, I//2] uint8,
    #          block_scales [O, I//block_size] fp8-e4m3,
    #          global_scale scalar float32)

def estimate_global_scale(
    weight: torch.Tensor,            # (O, I)
    block_size: int = 16,
) -> torch.Tensor:                   # scalar, float32

def pack_fp4_to_uint8(
    fp4_float: torch.Tensor,         # (O, I), float32 with 8 discrete FP4 values
) -> torch.Tensor:                   # (O, I//2), uint8
```

#### rotation_plan.py
```python
class RotationPlanBuilder:
    def __init__(self, config: dict): ...
    
    def build(
        self,
        layer_index: str,
        mode: str,
        weight: torch.Tensor | None = None,
        activation: torch.Tensor | None = None,
    ) -> RotationPlan: ...

def save_plan(plan: RotationPlan, path: str) -> None: ...
def load_plan(path: str) -> RotationPlan: ...
```

### JSON 序列化格式

risk_scores 文件:
```json
{
  "layer_index": "model.layers.0.self_attn.qkv_proj",
  "target": "weight",
  "method": "max_abs",
  "shard_id": 0,
  "num_channels": 896,
  "scores": [0.123, 0.456, ...]
}
```

RotationPlan 文件:
```json
{
  "mode": "weight_only",
  "layer_index": "model.layers.0.self_attn.qkv_proj",
  "num_pairs": 22,
  "pairs": [[0, 1], [2, 3], ...],
  "angles": [0.123, 0.456, ...],
  "pair_meta": {"policy": "high_high", "top_ratio": 0.05},
  "angle_meta": {"solver": "heuristic"}
}
```

## Phase B 交付记录 (2026-04-18)

### Step 7: `config.py` ✅
- `PairwiseFP4Config(QuantizationConfig)` — 10 个配置字段 + 验证
- `get_quant_method()` 用 lazy import 返回 `PairwiseFP4LinearMethod`
- `get_plan_builder_config()` 便利方法

### Step 8: `linear_method.py` ✅
- `PairwiseFP4LinearMethod(QuantizeMethodBase)`
- `create_weights`: 创建 BF16 weight + partition metadata
- `process_weights_after_loading`: build plan → rotate weight → quantize FP4 → kernel format → store rotation buffers
- `apply`: optional activation rotation → `apply_nvfp4_linear`
- 不使用 `uses_meta_device`; 权重以 BF16 加载后在线量化
- `input_global_scale` 使用默认值 1.0 (v1 无校准)

### Step 9: 注册 ✅
- `QuantizationMethods` Literal 添加 `"pairwise_fp4"`
- `get_quantization_config` 添加映射
- `WEIGHT_LOADER_V2_SUPPORTED` 添加 `PairwiseFP4LinearMethod`

### 端到端验证 ✅
- CPU: config创建, import, create_weights, process_weights_after_loading 全通过
- GPU (CUDA): 完整 create→process→apply 流程通过, FlashInfer CUTLASS backend
- output shape/dtype 正确: (2, 128) bfloat16

## Phase C: 测试验证 (Steps 10-11) — 已完成 2026-04-19

### Step 10: 单元测试 ✅ (52 tests, all passed)
File: `tests/quantization/test_pairwise_fp4_rotation.py`

运行: `pytest tests/quantization/test_pairwise_fp4_rotation.py -v`

| 测试类 | 测试数 | 覆盖模块 |
|--------|--------|----------|
| TestGivensRotation | 7 | rotation_applier: forward/inverse, identity, norm, empty, known_value, batch, validation |
| TestChannelMonitor | 5 | channel_monitor: max_abs/dynamic_range known values, shape, save/load, metadata validation |
| TestPairConstructor | 7 | pair_constructor: 4 policies parametrized, joint_compatible, top_ratio edge cases, scaling |
| TestAngleSolver | 6 | angle_solver: heuristic shape/symmetric/bounded, small_search shape/bounded, empty |
| TestFP4QuantPolicy | 6 | fp4_quant_policy: global_scale +/zero, quantize shape, pack known values, invalid dims/divisibility |
| TestRotationPlan | 7 | utils.RotationPlan: validate ok/bad_mode/shape_mismatch, serialization roundtrip, save/load, empty plan |
| TestRotationPlanBuilder | 6 | rotation_plan: weight_only/joint build, missing tensor errors, cache create+reuse, small_search solver |
| TestPairwiseFP4Config | 6 | config: defaults, invalid mode, from_config, plan_builder_config, filenames, dtypes |

### Step 11: 集成测试 ✅ (12 tests, all passed, CUDA required)
File: `tests/quantization/test_pairwise_fp4_integration.py`

运行: `pytest tests/quantization/test_pairwise_fp4_integration.py -v`

| 测试类 | 测试数 | 覆盖场景 |
|--------|--------|----------|
| TestWeightOnlyForward | 3 | forward runs, bias, partitioned output [64,32,32] |
| TestActivationOnlyForward | 3 | placeholder, empty plan (prebuilt), with rotation (prebuilt) |
| TestJointForward | 1 | joint with prebuilt plan |
| TestIdentityPlanEquivalence | 1 | top_ratio=0 → empty plan → 两次执行结果一致 |
| TestOutputDeterminism | 1 | 固定 plan + 固定 seed → 3 次输出完全一致 |

---

## Phase D: 最小验收实验 ✅

### Step 12: 基础设施修改

1. **`config.py`**: `get_config_filenames()` 改为返回 `[]`，允许 `LLM(quantization="pairwise_fp4")` 无需配置文件直接使用默认参数
2. **`config.py`**: 新增 `from_config_dict_json()` 方法，支持通过 `hf_overrides={"quantization_config_dict_json": ...}` 传递自定义参数
3. **`linear_method.py`**: `create_weights()` 从 plain `Parameter` + `default_weight_loader` 改为 `ModelWeightParameter`（支持 QKV 等 merged linear 层的分片加载）

### Step 13: 实验脚本
File: `scripts/eval_pairwise_fp4_gsm8k.py`

支持 5 组实验（A: BF16 baseline, B: FP4 no-rotation, C: weight_only, D: activation_only, E: joint），
CLI 参数控制 `--group`, `--num-samples`, `--model`, `--data`, `--diagnose`。

### Step 14: 实验执行与结果

5 组实验全部跑通（RTX 5090, emulation backend, enforce_eager=True）。
详细结果见 `pairwise_online_rotation_quant_experiment_log.md`。

关键数据（100 samples）：
- BF16: 39.0%
- FP4 no-rotation: 1.0%
- FP4 weight_only: 1.0%

旋转生效验证：plan builder 产生 45 对旋转（top_ratio=0.1），90/896 通道受影响，
但 heuristic solver 角度极小（max ~0.017 rad），量化误差改善仅 0.01%。
FP4 固有精度损失（无 input scale 校准）是准确率下降的主因。
| TestKernelAttributes | 3 | all required attributes exist, alpha=input_gs*weight_gs, weight halved |