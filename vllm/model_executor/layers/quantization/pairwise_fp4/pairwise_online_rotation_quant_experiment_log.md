# Pairwise FP4 — 最小验收实验日志

## 实验环境

| 项目 | 值 |
|------|----|
| GPU | NVIDIA GeForce RTX 5090 (SM 12.0, Blackwell) |
| Python | 3.12.13 |
| vLLM | v0.1.dev15259+g290809456.d20260404 |
| Model | Qwen2.5-0.5B-Instruct |
| Dataset | GSM8K test (1319 total samples) |
| FP4 GEMM Backend | EMULATION (`VLLM_USE_NVFP4_CT_EMULATIONS=1`) |
| CUDA Graph | 关闭 (`enforce_eager=True`) |

### 关于 GEMM 后端

RTX 5090 (SM 12.0) 上 FlashInfer CUTLASS FP4 GEMM 存在 bug：
```
RuntimeError: [FP4 gemm Runner] Failed to run cutlass FP4 gemm on sm120/sm121. Error: Error Internal
```
因此所有 FP4 实验均使用 EMULATION 后端。此后端在数值上等价但速度较慢（~67 toks/s vs BF16 的 ~2273 toks/s），速度数据不具参考性。

---

## 实验组定义

| 组 | 名称 | quantization | mode | top_ratio | 备注 |
|----|------|-------------|------|-----------|------|
| A | BF16 baseline | 无 | - | - | 原始 BF16 模型 |
| B | FP4 no-rotation | pairwise_fp4 | weight_only | 0.0 | FP4 量化，无旋转 |
| C | FP4 weight_only | pairwise_fp4 | weight_only | 0.1 | FP4 + 10% 通道旋转 |
| D | FP4 activation_only | pairwise_fp4 | activation_only | 0.1 | FP4 + 空预构建 plan |
| E | FP4 joint | pairwise_fp4 | joint | 0.1 | FP4 + 空预构建 plan |

D/E 组使用空预构建 plan（因为当前无校准数据用于激活侧旋转），
实际效果等同于 B 组。

---

## 实验结果

### Smoke test (20 samples)

| 组 | 准确率 | 正确/总计 | 耗时 | 状态 |
|----|--------|-----------|------|------|
| A | 30.0% | 6/20 | 15.3s | OK |
| B | 5.0% | 1/20 | 37.3s | OK |
| C | 0.0% | 0/20 | 36.8s | OK |
| D | 5.0% | 1/20 | 37.6s | OK |
| E | 5.0% | 1/20 | 36.8s | OK |

### 100-sample test (Groups A/B/C)

| 组 | 准确率 | 正确/总计 | 耗时 |
|----|--------|-----------|------|
| A | 39.0% | 39/100 | 14s |
| B | 1.0% | 1/100 | 36s |
| C | 1.0% | 1/100 | 36s |

---

## 旋转生效验证

### 1. RotationPlanBuilder 功能验证

```
top_ratio=0.0: pairs.shape=[0, 2], is_empty=True   → 正确生成空 plan
top_ratio=0.1: pairs.shape=[45, 2], is_empty=False  → 正确生成 45 对旋转
  angles range: [0.0000, 0.0167] (弧度)
  sample pairs: [[822, 82], [877, 308], [470, 329], ...]
```

### 2. 权重旋转确实改变了权重

```
weight diff: max=0.080318, mean=0.000083
channels affected: 90/896 (10% ≈ top_ratio=0.1)
```

### 3. 旋转对量化误差的影响

```
Quantization error without rotation: 1.559641
Quantization error with rotation:    1.559451
Improvement: 0.01%
```

改善极其微小，原因分析：
- heuristic solver 的角度公式 θ = (π/4)|r_i-r_j|/(r_i+r_j+eps) 在随机分布权重上产生的角度极小（最大 ~0.017 rad ≈ 1°）
- 0.5B 模型的权重分布相对均匀，risk score 差异不大
- `input_global_scale=1.0`（无校准），激活量化精度差是主要降级来源

### 4. D/E 预构建 plan 验证

```
D/E prebuilt plan: mode=activation_only, is_empty=True
  pairs=[0, 2], angles=[0]
  → 无激活旋转（预期：当前无校准数据）
```

---

## 关键发现与结论

### 基础设施完整性 ✅
1. **5 组实验全部跑通**，无运行时错误
2. **eval 脚本** `scripts/eval_pairwise_fp4_gsm8k.py` 支持单组/多组/全部运行
3. **参数传递** 通过 `hf_overrides` + `quantization_config_dict_json` 正常工作
4. **旋转机制已集成** 到 vLLM 推理路径中

### 准确率分析
- BF16 → FP4 的准确率下降巨大（39% → 1%），**主要原因不是旋转不生效，而是 FP4 量化本身的精度损失**
- `input_global_scale=1.0` 是关键问题：没有校准数据，激活量化使用默认 scale，导致信息丢失严重
- 旋转确实在生效（权重被改变，90/896 通道受影响），但改善量级太小无法弥补 FP4 的固有精度损失
- B 和 C 准确率相同（1%），说明当前 rotation 方案的改善淹没在 FP4 噪声中

### 需要后续改进的方向
1. **input_global_scale 校准**: 当前硬编码为 1.0，需要用少量校准数据计算合适的 scale
2. **angle solver 优化**: heuristic solver 产生的角度太小（< 0.02 rad），需要更aggressive 的角度搜索
3. **FlashInfer CUTLASS SM 12.0 兼容性**: CUTLASS FP4 kernel 在 RTX 5090 上有 bug，限制了推理速度

---

## 运行方式

```bash
# 环境准备
conda activate vllm

# Smoke test (20 samples, single group)
VLLM_USE_NVFP4_CT_EMULATIONS=1 python scripts/eval_pairwise_fp4_gsm8k.py \
  --group A --num-samples 20

# standard test(100 samples)
VLLM_USE_NVFP4_CT_EMULATIONS=1 python scripts/eval_pairwise_fp4_gsm8k.py \
  --group C --num-samples 100 --output results_C_100.json

# 全部 5 组，100 samples
VLLM_USE_NVFP4_CT_EMULATIONS=1 python scripts/eval_pairwise_fp4_gsm8k.py \
  --group all --num-samples 100 --output results_all.json

# 指定组，指定模型路径
VLLM_USE_NVFP4_CT_EMULATIONS=1 python scripts/eval_pairwise_fp4_gsm8k.py \
  --group B,C --num-samples 50 --model /path/to/model
```
