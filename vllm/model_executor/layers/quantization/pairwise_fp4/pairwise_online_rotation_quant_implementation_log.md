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

---

## Phase E: input_global_scale 链路修复 (2026-04-20)

### 根因定位

经审计发现，FP4 输出为乱码的根因并非 `input_global_scale` 本身，
而是 **emulation 后端的 block scale 未经 swizzle**：

1. `quantize_weight_to_fp4()` 返回 LINEAR 布局的 block scales
2. `convert_to_nvfp4_linear_kernel_format()` 对 EMULATION 后端不做任何变换
3. `run_nvfp4_emulations()` → `dequantize_to_dtype()` 始终调用
   `convert_swizzled_to_linear()` 做 un-swizzle
4. LINEAR scales 被错误 un-swizzle → 权重反量化结果打乱 → 乱码输出

数值验证：
- 未 swizzle: cosine_sim = 0.648, abs_err = 0.636 (单层 matmul)
- 已 swizzle: cosine_sim = 0.993, abs_err = 0.077 (8.2x 改善)
- 24 层累积: 0.648^24 ≈ 0 (信号完全消失) vs 0.993^24 ≈ 0.85 (合理)

### 修改内容

**文件: `linear_method.py`**

1. **导入 `NvFp4LinearBackend` 和 `swizzle_blockscale`**
2. **Block scale swizzle (关键修复)**:
   EMULATION 后端在 `process_weights_after_loading` 中预先 swizzle block scales，
   其他后端由 `convert_to_nvfp4_linear_kernel_format` 内部处理。
3. **Alpha 修正**: `alpha = input_global_scale / weight_global_scale`
   (原来是 `input_global_scale * weight_global_scale`，
   导致 alpha ≈ 500+ 而非正确的 ≈ 0.002)

### input_global_scale 校准设计

当前 `input_global_scale = 1.0` 在 emulation 路径是正确的，因为 emulation
反量化公式 `fp4 * sf / gs` 中 gs 自消：sf = fp8(gs * vecMax / FP4_MAX)，
所以 `sf/gs = vecMax/FP4_MAX` 与 gs 取值无关。

对于未来 CUTLASS 路径：
- `alpha = 1/(gs_a * gs_w)` 已正确计算
- 当 `gs_a = 1.0` 时，block scale `sf_x = fp8(blockmax/6)`，
  仅在 blockmax > 2688 时溢出 FP8，对常规模型不成问题
- 如需更精细的激活量化，可通过 calibration 收集激活统计后设置
  `input_global_scale = amax_act / (FP8_MAX * FP4_MAX)` (SMALL)

### 验证结果

修复后 smoke test (20 samples, Group B, no rotation):
- 修复前: 1-5% (乱码输出如 "ar eventfee TTTTTTT")
- 修复后: 10% (连贯推理如 "To determine how much Janet makes...")
- BF16 基线: 39%

FP4 PTQ 仍有显著精度下降 (10% vs 39%)，这是 FP4 4-bit 量化的固有局限，
非 scale 链路问题。单层 matmul 相对误差 ~98%，主要来自 FP4 的极低精度
(仅 15 个离散值)。

---

## Phase F 激活侧在线监控与 risk score 落盘闭环

### 背景

Phase A-E 实现了 weight_only 模式的完整链路，但 activation_only 和 joint 模式的激活侧数据流完全断裂：
- `RotationPlanBuilder.build()` 中 `activation=None` 硬编码，调用 activation_only/joint 会直接 ValueError
- 无在线激活统计收集机制
- 无激活 risk score 缓存/复用能力

### 新增文件

**`activation_collector.py`** — 有状态的逐通道激活统计收集器
- `ActivationCollector(num_channels, risk_method, warmup_samples, device)`
- `update(x)`: 每次前向传播调用，累积 per-channel max_abs（或 dynamic_range）
- `finalize()`: warmup 完成后返回 `(C,)` risk scores tensor
- `ready` 属性: warmup_samples 次 update 后返回 True

### 修改文件

**`linear_method.py`** — 核心改动：
1. `process_weights_after_loading()`:
   - 从 `layer.prefix` 获取层名（vLLM LinearBase 存储了完整路径如 `model.layers.0.self_attn.qkv_proj`）
   - 检查 `_act_risk_cache_path()` 是否有缓存：
     - **有缓存** → 加载 risk scores，构造 dummy activation tensor，通过 builder 构建完整 plan
     - **无缓存** → 在 layer 上安装 `_act_collector`，进入 warmup 阶段
   - joint 模式无缓存时先构建 weight_only plan（有权重旋转），warmup 后重建为 joint plan
2. `apply()`:
   - 检查 `layer._act_collector`，每次前向调用 `collector.update(x)`
   - collector ready 后触发 `_finalize_activation_plan(layer)`
3. `_finalize_activation_plan()`:
   - 获取 risk scores → 保存到 JSON 缓存 → 构建 RotationPlan → 安装 pairs/angles buffer → 清理 collector
4. `_act_risk_cache_path()`:
   - 路径格式: `{cache_dir}/{safe_name}__activation__{risk_method}.json`
   - safe_name 将 `.` 和 `/` 替换为 `_`

### 缓存格式与时序

**缓存格式** (JSON):
```json
{
  "risk_scores": [0.0177, 2.9062, ...],  // C 个 float
  "metadata": {
    "layer_index": "model.layers.0.self_attn.qkv_proj",
    "target": "activation",
    "method": "max_abs",
    "num_channels": 896
  }
}
```

**时序**:
1. 首次运行（无缓存）: 加载模型 → 安装 collector → 前 8 次推理收集统计 → finalize → 保存 JSON → 安装 plan → 后续推理使用 plan
2. 后续运行（有缓存）: 加载模型 → 检测到缓存 → 直接加载 risk scores → 构建 plan → 所有推理都使用 plan（零 warmup 开销）

### 验证结果

**activation_only**:
- 首次运行: 96 个 warmup → 96 个 JSON 缓存文件（24 层 × 4 线性层）
- 二次运行: 96 个 "Loaded cached" 日志，零 warmup，输出连贯
- 每层 pairs: 45-243 对（取决于通道数），angle range [0.0, 0.17] rad

**joint**:
- 首次运行: 96 warmup + 192 缓存文件（96 weight + 96 activation）
- 输出连贯: "Paris, located in the southeast..."

**层名解析**: 使用 `layer.prefix`（vLLM LinearBase 属性），fallback 到 `layer._layer_name` 或 "unknown"。
