# 任务书：在 vLLM 中实现静态 pairwise Givens + FP4 推理原型（带内置 risk score 监控）

## 研究背景（简要）

我在阅读 ParoQuant 后，关注到 **pairwise Givens rotation** 这类变换不仅可以作用于权重，也可能通过逆变换作用于激活，从而缓解低比特推理中的 outlier 问题。当前目标不是接入 RL 训练闭环，也不是实现最终高性能 kernel，而是先在 **vLLM 推理路径** 内实现一个**静态的 pairwise Givens + FP4 原型**，验证以下三种模式能否正确接入并运行：

* `weight_only`
* `activation_only`
* `joint`

实现上借鉴 ParoQuant 的**推理边界**：
**前置旋转 + 复用现有低比特 matmul 路径**，而不是完整移植其离线 PTQ 优化器。ParoQuant 的真实推理本质上也是“在线旋转输入激活，再乘已在旋转域量化好的低比特权重”。

---

## 一、任务目标

在 **vLLM** 中新增一种研究型量化/推理原型，暂命名为：

* `pairwise_fp4`
* 或 `givens_pairwise_fp4`

该原型需要支持在 **FP4 推理路径** 上引入**静态 pairwise Givens rotation**，用于比较三种模式：

1. `weight_only`
2. `activation_only`
3. `joint`

此外，本任务还需要在 vLLM 内部实现一套**risk score 监控与缓存机制**，用于为权重侧和激活侧构造静态 pairwise plan。

要求：

* 能跑通 dense LLM 的推理；
* 能复用现有 FP4 quant + matmul 路径；
* 数值逻辑正确；
* 能在首次请求中自动监控并落盘 risk score；
* 后续请求可直接复用已有统计结果；
* 模块边界清晰，便于后续替换为动态 risk-score 更新与 fused kernel。

---

## 二、明确非目标

本任务**不要做**以下事情：

1. 不接入 verl / PPO / RL 训练闭环；
2. 不实现 step 级动态 risk score 更新；
3. 不完整移植 ParoQuant 的 layer-wise PTQ 优化器；
4. 不重写 MXFP4 / NVFP4 kernel；
5. 不做最终性能优化或 speedup 结论；
6. 不支持 MoE，仅处理 dense LLM。ParoQuant 的 MoE 路径是单独的一套状态与导出逻辑，本任务显式排除。

---

## 三、总体实现思路

请按下面的边界实现，不要偏离：

### 核心边界

借鉴 ParoQuant 的推理设计，只保留这条主线：

1. 在 vLLM 推理过程中监控权重与激活的 risk score；
2. 将 risk score 以**推理侧可直接消费**的格式按层/分片落盘；
3. 由 `PairConstructor` 按指定层索引读取 risk score，构造 pair；
4. 由 `AngleSolver` 计算角度；
5. 对权重和/或激活应用 pairwise Givens 旋转；
6. 调用现有 FP4 quant + matmul 路径完成推理。

### 不要照搬的部分

* 不要移植 ParoQuant 的完整离线优化器；
* 不要照搬其随机 pair 初始化逻辑；
* 不要把其 INT4 affine quantizer 的 `channel_scales` 直接平移到当前设计中。当前任务面向 MXFP4 / NVFP4，应把量化抽象成统一的 `FP4QuantPolicy`。

---

## 四、支持的三种模式

### 1. `weight_only`

* 通过 `ChannelMonitor` 对权重做静态监控，得到 risk score；
* `PairConstructor` 从对应层索引读取 weight risk score，构造 pair；
* 对权重侧应用变换；
* 激活侧不参与 pair 构造。

### 2. `activation_only`

* 在 vLLM 推理过程中直接监控激活，得到 activation risk score；
* `PairConstructor` 从对应层索引读取 activation risk score，构造 pair；
* 仅激活侧应用 pairwise 逻辑。

### 3. `joint`

* 同时监控权重与激活；
* `PairConstructor` 综合两侧 risk score 构造 pair；
* 同时对权重和激活应用对应变换。

---

## 五、静态监控与缓存机制

### 基本要求

虽然本任务会在 vLLM 推理时实时拿到激活值和权重值来计算 risk score，但在方法论上仍然把这件事视为**静态监控**，实现方式如下：

1. **首次启动某模型、某配置推理时**：

   * 检查是否存在对应的静态 risk score 文件；
   * 若不存在，则在本次推理过程中一边执行、一边监控并累计 risk score；
   * 推理完成后将结果保存到磁盘。

2. **后续请求时**：

   * 若存在对应静态文件，则直接复用；
   * 不再重新构建，除非显式要求刷新。

### 文件命名建议

建议文件命名至少包含以下信息：

* 模型名称
* risk score 计算方式
* 目标类型（`weight` / `activation`）
* 量化格式（可选）
* 其他必要配置哈希（可选）

例如仅示意：

```text id="03as7r"
{model_name}__{target}__{risk_method}__{extra_config}.json
```

### 记录粒度要求

不要强行把信息聚合成“原始全局层”再保存。
应优先按 **vLLM 推理侧能直接使用的执行单元** 保存，也就是：

* 模块名
* 层编号
* shard / partition 信息
* 通道维度信息

换句话说，保存格式应服务于“后续直接构造 pairwise”，而不是服务于离线展示。
ParoQuant 在 vLLM 后端里也是按 partition 处理旋转元数据，这一点应保持一致。

---

## 六、risk score 设计要求

### 1. 统一抽象

risk score 的计算逻辑应统一服务于：

* 权重监控
* 激活监控

### 2. 第一版至少支持的 risk score 方法

请至少预留以下几种 risk score 计算方式：

* `max_abs`
* `dynamic_range`

其中：

* `max_abs`：通道最大绝对值
* `dynamic_range`：通道最大值与最小值之比的绝对值，具体实现上请合理处理数值稳定性与零值

### 3. 可扩展性

risk score 方法必须可配置，便于后续扩展更多统计方式。

---

## 七、覆盖范围

### 层类型

目标是覆盖 **FP4 路径中的所有线性层**，保持“全层量化”的设定。

实现上：

* 请沿 vLLM 现有量化方法的接入方式，对 `LinearBase` 体系中的线性层统一接入；
* 如果第一版个别特殊层路径不兼容，可临时 fallback，但不要把设计写死成只支持少数层。

### 模型范围

仅支持：

* dense decoder-only LLM

不支持：

* MoE
* encoder-decoder
* 其他特殊结构

---

## 八、核心模块设计

算法代码必须和 vLLM glue code 解耦。至少拆成以下五个模块。

---

### 1. `ChannelMonitor`

职责：在 vLLM 推理路径中监控权重或激活，并计算/保存 risk score。
这是对旧版 `ChannelSelector` 的替换。

#### 输入

* `layer_index: str`
* `target: str` (`weight` / `activation`)
* `tensor: Tensor`
* `risk_method: str`
* `cache_path: Optional[str]`
* 其他必要配置

#### 输出

* `risk_scores: dict`
* 或按统一格式写入静态文件

#### 要求

* 能用于权重侧监控；
* 能用于激活侧监控；
* 首次无文件时负责生成并保存；
* 有文件时支持直接读取；
* 记录内容应面向后续推理侧直接消费，不要求聚合成全局层视图。

---

### 2. `PairConstructor`

职责：按指定层索引读取 risk score，并构造若干 pair。

#### 输入

* `layer_index: str`
* `mode: str`
* `policy_name: str`
* `top_ratio_or_k: float | int`
* `weight_risk_source: Optional[str | dict]`
* `activation_risk_source: Optional[str | dict]`

#### 输出

* `pairs: list[tuple[int, int]]`

#### 必须支持的策略名

至少预留：

* `adjacent_sorted`
* `high_high`
* `high_low`
* `random_baseline`
* `joint_compatible`

#### 要求

* 直接读取对应层索引下的 risk score 记录；
* 根据 `mode` 和 `policy` 构造 pair；
* `top_ratio_or_k` 从原先 selector 侧移动到这里。

#### 备注

ParoQuant 当前仓库的 pair 初始化本质上是随机/贪心，而不是数据驱动 pair 搜索，所以本模块是后续方法改进的关键承载点。

---

### 3. `AngleSolver`

职责：为每个 pair 计算旋转角。

#### 输入

* `layer_index: str`
* `pairs: list[tuple[int, int]]`
* `mode: str`
* `weight_tensor: Optional[Tensor]`
* `activation_tensor: Optional[Tensor]`
* `solver_name: str`

#### 输出

* `angles: list[float]`
* 可选：`solver_meta: dict`

#### 要求

* 第一版不要求理论最优；
* 至少支持：

  * `heuristic`
  * `small_search`

---

### 4. `RotationApplier`

职责：执行 pairwise Givens 旋转。

#### 输入

* `tensor: Tensor`
* `pairs: list[tuple[int, int]]`
* `angles: list[float]`
* `target: str` (`weight` / `activation`)
* `inverse: bool`

#### 输出

* `rotated_tensor: Tensor`

#### 要求

* 必须支持：

  * 对权重应用
  * 对激活应用
  * 正旋
  * 逆旋
* 第一版可以先用 PyTorch 实现，优先保证数值正确性；
* 不要求 fused kernel。

---

### 5. `FP4QuantPolicy`

职责：统一管理 FP4 量化策略。
这是受控背景模块，不是当前任务的创新核心。

#### 输入

* `format: str` (`mxfp4` / `nvfp4`)
* `block_size`
* `rounding_mode`
* `tensor`
* `target: str` (`weight` / `activation`)

#### 输出

* 量化后的 FP4 表示
* block/group scale
* 必要时的 packed representation

#### 要求

* 第一版尽量复用现有实现；
* 不要重新发明 FP4 quantizer；
* rounding / 量化细节只需预留接口，不作为第一版主变量。

---

## 九、计划装配层

### 1. `RotationPlan`

定义一个静态数据结构，至少包含：

* `mode`
* `layer_index`
* `pairs`
* `angles`
* `pair_meta`
* `angle_meta`

### 2. `RotationPlanBuilder`

职责：把 `ChannelMonitor + PairConstructor + AngleSolver` 串起来，为每层生成静态 plan。

#### 输入

* `layer_index`
* `mode`
* `weight_tensor`
* `activation_tensor`
* `config`

#### 输出

* `RotationPlan`

#### 要求

* 当前只做**静态 plan 驱动**；
* 推理时不进行 step 级动态更新；
* plan 可以在首次推理后保存并复用；
* 当已有缓存文件时，应优先直接读取。

---

## 十、vLLM 集成要求

### 集成原则

必须优先走 **vLLM 官方量化扩展路径**，不要随意散改。

### 建议实现形态

新增一个 quantization config，例如：

* `PairwiseFP4Config`

新增一个线性层执行方法，例如：

* `PairwiseFP4LinearMethod`

Agent 需要在阅读 vLLM 代码后，沿现有量化方法的模式完成接入。不要执着于我这里的类名，重点是职责边界。

### 职责建议

#### `create_weights(...)`

* 创建需要的参数/缓冲区；
* 注册与本方法相关的 metadata。

#### `process_weights_after_loading(...)`

* 对权重侧执行静态监控，或加载已缓存的权重 risk score；
* 在 `weight_only` / `joint` 模式下，对权重做 pairwise 变换；
* 再调用既有 FP4 quant policy 对权重做量化/打包。

#### `apply(layer, x, bias)`

* 在推理时对激活执行监控，或加载已缓存的 activation risk score；
* 在 `activation_only` / `joint` 模式下，对输入 `x` 应用静态 pairwise 旋转；
* 调用现有 FP4 matmul 路径；
* 第一版只要求逻辑正确，不要求融合优化。

### 重要提醒

ParoQuant 在 vLLM 后端里，旋转参数是和 **partition 后的 weight shard** 强绑定的；若线性层被分片，每个 partition 可能需要自己的旋转参数。实现时必须考虑这一点，不要假设一层只对应一份全局 plan。

---

## 十一、目录结构建议

建议把算法部分与 vLLM glue code 分开。

### 算法模块

```text id="1f8o0g"
pairwise_fp4/
  channel_monitor.py
  pair_constructor.py
  angle_solver.py
  rotation_applier.py
  fp4_quant_policy.py
  rotation_plan.py
  utils.py
```

### vLLM 集成部分

```text id="humtye"
vllm/.../quantization/pairwise_fp4/
  config.py
  linear_method.py
  loader.py
  registry.py
```

文件名可以调整，但职责边界必须保持。

---

## 十二、静态缓存文件要求

### 总要求

缓存文件应尽量按**推理侧直接可用**的形式保存，不要求先聚合成“全局层”。

### 至少应包含的信息

* 模型名称
* 目标类型（`weight` / `activation`）
* risk score 方法
* 层索引 / 分片索引
* 通道维度
* 每个通道的 risk score

### 读取原则

* `PairConstructor` 必须能通过 `layer_index` 直接定位到对应 risk score；
* 不要求用户额外手工整理文件；
* 首次构建后自动保存，后续自动复用。

---

## 十三、建议配置字段

至少支持以下字段：

```yaml id="40w7ab"
quant_method: pairwise_fp4
fp4_format: nvfp4 | mxfp4
mode: weight_only | activation_only | joint

risk_method: max_abs | dynamic_range
top_ratio: 0.05
pair_policy: high_high
angle_solver: heuristic

risk_cache_dir: ...
rotation_plan_path: ...
use_prebuilt_plan: true

block_size: 16 | 32
rounding_mode: default
```

### 说明

* `risk_method` 第一版至少支持 `max_abs / dynamic_range`
* `top_ratio` 默认支持 `0.05 / 0.10`
* `use_prebuilt_plan=true` 时优先加载现有 plan
* 无缓存时自动监控并构建

---

## 十四、实现顺序

请按下面顺序推进，不要一开始就把所有模式和所有逻辑糅在一起。

### Step 1：数值单元测试

目标：

* 验证 `RotationApplier` 的 Givens 实现正确；
* 验证正旋/逆旋互逆；
* 验证维度、dtype、数值行为稳定。

### Step 2：实现 `ChannelMonitor`

目标：

* 支持权重 risk score 监控；
* 支持激活 risk score 监控；
* 支持缓存文件生成与读取；
* 生成推理侧可直接消费的记录格式。

### Step 3：实现 `PairConstructor` + `AngleSolver`

目标：

* 能按 `layer_index` 读取 risk score；
* 能构造 pair；
* 能生成角度；
* 能保存/读取 `RotationPlan`。

### Step 4：接入 vLLM 的 `weight_only`

目标：

* 先接最简单模式；
* 跑通权重监控、pair 构造、权重量化与推理。

### Step 5：接入 `activation_only`

目标：

* 在推理中监控激活；
* 生成静态 risk score 缓存；
* 对输入激活做旋转；
* 跑通基本推理。

### Step 6：接入 `joint`

目标：

* 同时支持权重与激活两侧 risk score；
* 跑通完整路径。

---

## 十五、最小实验集合

实现完成后，至少支持以下最小比较组：

1. BF16 baseline
2. FP4 no rotation
3. FP4 + `weight_only`
4. FP4 + `activation_only`
5. FP4 + `joint`

可选附加组：

* FP4 + random pairwise
  用于验证配对策略不是纯噪声。

---

## 十六、验收标准

### 代码交付

* 一个可在 vLLM 中启用的 `pairwise_fp4` 原型；
* 五个核心模块独立成文件/类；
* 至少一个可运行配置示例。

### 功能验收

* `weight_only / activation_only / joint` 三种模式都能被识别并运行；
* 能自动生成和读取 risk score 缓存文件；
* 能在至少一个 dense LLM 上完成推理。

### 数值验收

* 单元测试证明 Givens 旋转正逆一致；
* no rotation 与 identity plan 行为一致；
* 同一输入在固定 plan 下输出稳定。

### 实验验收

* 给出一份最小实验结果表；
* 说明各模式是否出现可观测差异；
* 记录已知限制与后续接口点。

---

## 十七、特别提醒

1. **优先阅读 vLLM 中现有量化方法的接入范式**，不要自创集成路径。
2. **借鉴 ParoQuant 的推理边界，不要移植其完整优化器。**
   最值得复用的是“在线前置旋转 + 复用现有低比特 matmul”的边界。
3. **不要把 ParoQuant 当前随机 pair 初始化当成最终方案。**
   当前任务要把 pair 选择、角度求解做成独立模块，便于替换与迭代。
4. risk score 监控虽然发生在推理过程中，但当前任务中应被实现成**静态缓存构建机制**，而不是训练闭环中的动态更新。
5. 当前重点是**研究原型可验证性**，不是性能最优实现。